from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from matplotlib.lines import Line2D

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections



count_dict = utils.get_otu_dict()

transfers = [12,18]

mean_rel_abund_all_treatments_dict = {}

for treatment in ['No_migration.4.T12', 'No_migration.40.T12', 'Global_migration.4.T12', 'Parent_migration.4.T12', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.4.T18', 'Parent_migration.NA.T0']:

    #mean_abundance_dict[treatment] = {}
    #samples_to_keep = [sample for sample in samples if treatment in sample]
    count_dict_to_keep = {key: value for key, value in count_dict.items() if treatment in key}
    abundance_dict = {}
    n_samples = len(count_dict_to_keep)
    for sample, asv_dict in count_dict_to_keep.items():
        N = sum(asv_dict.values())
        for asv, abundance in asv_dict.items():
            if asv not in abundance_dict:
                abundance_dict[asv] = []
            abundance_dict[asv].append(abundance/N)


    for asv, rel_abundance_list in abundance_dict.items():

        #if len(rel_abundance_list) < 3:
        #    continue

        mean_rel_abundance = np.mean(rel_abundance_list)
        if asv not in mean_rel_abund_all_treatments_dict:
            mean_rel_abund_all_treatments_dict[asv] = {}
        mean_rel_abund_all_treatments_dict[asv][treatment] = {}
        mean_rel_abund_all_treatments_dict[asv][treatment]['mean'] = mean_rel_abundance
        mean_rel_abund_all_treatments_dict[asv][treatment]['cv'] = np.std(rel_abundance_list)/mean_rel_abundance

asv_all = list(mean_rel_abund_all_treatments_dict.keys())
for treatment in ['No_migration.4.T12', 'No_migration.40.T12', 'Global_migration.4.T12', 'Parent_migration.4.T12', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.4.T18', 'Parent_migration.NA.T0']:
    means_all = []
    for asv in asv_all:
        if treatment in mean_rel_abund_all_treatments_dict[asv]:
            means_all.append(mean_rel_abund_all_treatments_dict[asv][treatment]['mean'])

    means_all = np.asarray(means_all)
    for asv in asv_all:
        if treatment in mean_rel_abund_all_treatments_dict[asv]:
            mean_rel_abund_all_treatments_dict[asv][treatment]['mean_norm'] = mean_rel_abund_all_treatments_dict[asv][treatment]['mean']/sum(means_all)


#fig, ax = plt.subplots(figsize=(4,4))

def parent_migration_fig():

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for transfer_idx, transfer in enumerate(transfers):

        ax_compare = plt.subplot2grid((2, 2), (0, transfer_idx), colspan=1)
        ax_parent = plt.subplot2grid((2, 2), (1, transfer_idx), colspan=1)

        migration_treatment = 'Parent_migration.4.T%d' % transfer
        no_migration_treatment = 'No_migration.4.T%d' % transfer

        migration_treatment_all = []
        no_migration_treatment_all = []

        ratios_all = []
        parent_all = []
        no_migration_norm_all = []

        for asv, mean_rel_abundance_dict in mean_rel_abund_all_treatments_dict.items():

            if (no_migration_treatment in mean_rel_abundance_dict) and (migration_treatment in mean_rel_abundance_dict):

                no_migration = mean_rel_abundance_dict[no_migration_treatment]['mean']
                migration = mean_rel_abundance_dict[migration_treatment]['mean']

                no_migration_norm = mean_rel_abundance_dict[no_migration_treatment]['mean_norm']
                #migration_norm = mean_rel_abundance_dict[migration_treatment]['mean_norm']

                no_migration_treatment_all.append(no_migration)
                migration_treatment_all.append(migration)

                if ('Parent_migration.NA.T0' in mean_rel_abundance_dict):
                    ratios_all.append(migration/no_migration_norm)
                    no_migration_norm_all.append(no_migration_norm)
                    parent_all.append(mean_rel_abundance_dict['Parent_migration.NA.T0']['mean'])


        ax_compare.scatter(no_migration_treatment_all, migration_treatment_all, alpha=0.8, c=utils.color_dict_range[('Parent_migration', 4)][12], zorder=2)
        ax_compare_min = min(no_migration_treatment_all + migration_treatment_all)
        ax_compare_max = max(no_migration_treatment_all + migration_treatment_all)

        ax_compare.plot([ax_compare_min*0.5, 1.1],[ax_compare_min*0.5, 1.1], c='k', ls=':', lw=2, label='1:1')
        ax_compare.set_xlim(ax_compare_min*0.5,1.1)
        ax_compare.set_ylim(ax_compare_min*0.5, 1.1)
        ax_compare.set_title('Transfer %s' % transfer, fontsize=14 )

        #rho = np.corrcoef(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))[0,1]

        rho, p_value = utils.run_permutation_corr(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))
        p_value_to_plot = utils.get_p_value(p_value)

        ax_compare.text(0.2,0.92, r'$\rho=$' + str(round(rho,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes )
        ax_compare.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes)


        ax_compare.set_xlabel('Mean rel. abundance, no migration, ' + r'$\left< x \right>_{\mathrm{no\, mig}}$', fontsize=10)
        ax_compare.set_ylabel('Mean rel. abundance, parent, ' + r'$\left< x \right>_{\mathrm{parent}}$', fontsize=10)

        ax_compare.set_xscale('log', basex=10)
        ax_compare.set_yscale('log', basey=10)
        ax_compare.legend(loc="lower right", fontsize=8)

        ax_parent.scatter(parent_all, ratios_all, alpha=0.8, c=utils.color_dict_range[('Parent_migration', 4)][12], zorder=2)
        ax_parent.axhline(1, lw=1.5, ls=':',color='k', zorder=1)
        #ax_parent.set_xlim(0.6*min(obs), 2*max(obs))
        #ax_parent.set_ylim(0.6*min(obs), 2*max(obs))

        ax_parent_abs_max_y = max(np.absolute(np.log10(ratios_all)))
        ax_parent.set_ylim(10**(-1.1*ax_parent_abs_max_y), 10**(ax_parent_abs_max_y*1.1))

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(parent_all), np.log10(ratios_all))
        x_log10_range =  np.linspace(min(np.log10(parent_all)) , max(np.log10(parent_all)), 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        ax_parent.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="OLS regression")

        if p_value < 0.05:
            p_value_text = r'$P < 0.05$'
        else:
            p_value_text = r'$P \nless 0.05$'

        ax_parent.text(0.2,0.9, p_value_text, fontsize=10, color='k', ha='center', va='center', transform=ax_parent.transAxes )

        ax_parent.set_xscale('log', basex=10)
        ax_parent.set_yscale('log', basey=10)

        ax_parent.set_xlabel('Rel. abundance in parent community', fontsize=10)
        ax_parent.set_ylabel('Mean rel. abundance ratio, ' + r'$\left< x \right>_{\mathrm{parent}}/\left< x \right>_{\mathrm{no\, mig}}$',  fontsize=10)

        no_migration_norm_all = np.asarray(no_migration_norm_all)
        no_migration_norm_all_log10 = np.log10(no_migration_norm_all)
        parent_all = np.asarray(parent_all)
        parent_all_log10 = np.log10(parent_all)
        #parent_all = np.asarray(parent_all)
        #prediction = np.sqrt(parent_all/no_migration_norm_all)
        #idx_ = parent_all.argsort()
        #parent_all = parent_all[idx_]
        #prediction = prediction[idx_]

        #prediction = np.sqrt((10**x_log10_range)/1)

        hist, bin_edges = np.histogram(parent_all_log10, bins=7, density=True)

        parent_prediction_all = []
        ratio_predicton_all = []
        for i in range(len(hist)):

            idx_ = (parent_all_log10 >= bin_edges[i]) & (parent_all_log10 < bin_edges[i+1])

            parent_prediction = np.mean(parent_all[idx_])
            parent_prediction_all.append(parent_prediction)
            ratio_predicton_all.append(np.mean(np.sqrt(parent_all[idx_]/no_migration_norm_all[idx_])))

        ax_parent.plot(parent_prediction_all, ratio_predicton_all, c='k', lw=2.5, linestyle=':', zorder=2, label="SLMm prediction")

        ax_parent.legend(loc="lower right", fontsize=8)
    #ratios_all_12 = np.asarray(ratios_all_12)
    ##ratios_all_18 = np.asarray(ratios_all_18)

    #parent_migration_12 = np.asarray(parent_migration_12)
    #parent_migration_18 = np.asarray(parent_migration_18)

    #rho_12 = np.corrcoef(np.log10(parent_migration_12), np.log10(ratios_all_12))[0,1]
    #rho_18 = np.corrcoef(np.log10(parent_migration_18), np.log10(ratios_all_18))[0,1]

    #custom_lines = [Line2D([0], [0], marker='o', color='w',  markerfacecolor='lightblue', markersize=12),
    #                Line2D([0], [0], marker='o', color='w',  markerfacecolor='dodgerblue', markersize=12)]
    #ax.legend(custom_lines, ['Transfer 12', 'Transfer 18'], loc="lower right")



    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_parent.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def global_migration_fig():

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for transfer_idx, transfer in enumerate(transfers):

        ax_mean = plt.subplot2grid((2, 2), (0, transfer_idx), colspan=1)
        ax_cv = plt.subplot2grid((2, 2), (1, transfer_idx), colspan=1)

        migration_treatment = 'Global_migration.4.T%d' % transfer
        no_migration_treatment = 'No_migration.4.T%d' % transfer

        no_migration_treatment_mean_all = []
        migration_treatment_mean_all = []

        no_migration_treatment_cv_all = []
        migration_treatment_cv_all = []

        for asv, mean_rel_abundance_dict in mean_rel_abund_all_treatments_dict.items():

            if (no_migration_treatment in mean_rel_abundance_dict) and (migration_treatment in mean_rel_abundance_dict):

                no_migration_mean = mean_rel_abundance_dict[no_migration_treatment]['mean']
                migration_mean = mean_rel_abundance_dict[migration_treatment]['mean']

                no_migration_cv = mean_rel_abundance_dict[no_migration_treatment]['cv']
                migration_cv = mean_rel_abundance_dict[migration_treatment]['cv']

                no_migration_treatment_mean_all.append(no_migration_mean)
                migration_treatment_mean_all.append(migration_mean)

                no_migration_treatment_cv_all.append(no_migration_cv)
                migration_treatment_cv_all.append(migration_cv)


        ax_mean.scatter(no_migration_treatment_mean_all, migration_treatment_mean_all, alpha=0.8, c=utils.color_dict_range[('Global_migration', 4)][12], zorder=2)

        ax_mean_min = min(no_migration_treatment_mean_all + migration_treatment_mean_all)
        ax_mean_max = max(no_migration_treatment_mean_all + migration_treatment_mean_all)

        ax_mean.plot([ax_mean_min*0.5, 1.1],[ax_mean_min*0.5, 1.1], c='k', ls=':', lw=2)
        ax_mean.set_xlim(ax_mean_min*0.5, 1.1)
        ax_mean.set_ylim(ax_mean_min*0.5, 1.1)
        ax_mean.set_title('Transfer %s' % transfer, fontsize=14 )

        #rho = np.corrcoef(np.log10(no_migration_treatment_mean_all), np.log10(migration_treatment_mean_all))[0,1]

        rho, p_value_rho = utils.run_permutation_corr(np.log10(no_migration_treatment_mean_all), np.log10(migration_treatment_mean_all))

        p_value_to_plot = utils.get_p_value(p_value_rho)

        ax_mean.text(0.2,0.9, r'$\rho=$' + str(round(rho,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_mean.transAxes)
        ax_mean.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_mean.transAxes)

        ax_mean.set_xlabel('Mean rel. abundance, no migration, ' + r'$\left< x \right>_{\mathrm{no\, mig}}$', fontsize=10)
        ax_mean.set_ylabel('Mean rel. abundance, global, ' + r'$\left< x \right>_{\mathrm{global}}$', fontsize=10)

        ax_mean.set_xscale('log', basex=10)
        ax_mean.set_yscale('log', basey=10)


        no_migration_treatment_cv_all = np.asarray(no_migration_treatment_cv_all)
        migration_treatment_cv_all = np.asarray(migration_treatment_cv_all)
        idx_to_keep = (no_migration_treatment_cv_all>0) & (migration_treatment_cv_all>0)
        no_migration_treatment_cv_all = no_migration_treatment_cv_all[idx_to_keep]
        migration_treatment_cv_all = migration_treatment_cv_all[idx_to_keep]

        ax_cv.scatter(no_migration_treatment_cv_all, migration_treatment_cv_all, alpha=0.8, c=utils.color_dict_range[('Global_migration', 4)][12], zorder=2)
        ax_cv_min = min(no_migration_treatment_cv_all + migration_treatment_cv_all)
        ax_cv_max = max(no_migration_treatment_cv_all + migration_treatment_cv_all)

        ax_cv.plot([ax_cv_min*0.1, ax_cv_max*1.3],[ax_cv_min*0.1, ax_cv_max*1.3], c='k', ls=':', lw=2)
        ax_cv.set_xlim(ax_cv_min*0.1, ax_cv_max*1.3)
        ax_cv.set_ylim(ax_cv_min*0.1, ax_cv_max*1.3)
        ax_cv.set_title('Transfer %s' % transfer, fontsize=14 )

        #rho = np.corrcoef(np.log10(no_migration_treatment_cv_all), np.log10(migration_treatment_cv_all))[0,1]
        rho, p_value_rho = utils.run_permutation_corr(np.log10(no_migration_treatment_cv_all), np.log10(migration_treatment_cv_all))

        print(rho, p_value_rho)
        p_value_to_plot = utils.get_p_value(p_value_rho)

        ax_cv.text(0.2,0.9, r'$\rho=$' + str(round(rho,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_cv.transAxes )
        ax_cv.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_cv.transAxes )

        ax_cv.set_xlabel('CV rel. abundance, no migration, ' + r'$\left< x \right>_{\mathrm{no\, mig}}$', fontsize=10)
        ax_cv.set_ylabel('CV rel. abundance, global, ' + r'$\left< x \right>_{\mathrm{global}}$', fontsize=10)

        ax_cv.set_xscale('log', basex=10)
        ax_cv.set_yscale('log', basey=10)



    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_global.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()


parent_migration_fig()
#global_migration_fig()
