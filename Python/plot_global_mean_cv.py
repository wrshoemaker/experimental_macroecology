from __future__ import division
import os, sys, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from matplotlib.lines import Line2D
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import slm_simulation_utils
import plot_utils



n_iter = 10000


count_dict = utils.get_otu_dict()

mean_rel_abund_all_treatments_dict = {}
for treatment in ['No_migration.4.T%s', 'No_migration.40.T%s', 'Global_migration.4.T%s', 'Parent_migration.4.T%s', 'Parent_migration.NA.T0%s']:

    #if ('No_migration.4.' in treatment) or ('Global_migration.4.' in treatment):
    #    transfers = utils.transfers_all

    if 'Parent_migration.NA.T0' in treatment:
        # empty string
        transfers = ['']
    else:
        transfers = utils.transfers

    for transfer in transfers:

        treatment_transfer = treatment % str(transfer)
        count_dict_to_keep = {key: value for key, value in count_dict.items() if treatment_transfer in key}
        abundance_dict = {}
        n_samples = len(count_dict_to_keep.keys())
        for sample, asv_dict in count_dict_to_keep.items():
            N = sum(asv_dict.values())
            for asv, abundance in asv_dict.items():
                if asv not in abundance_dict:
                    abundance_dict[asv] = []
                abundance_dict[asv].append(abundance/N)


        for asv, rel_abundance_list in abundance_dict.items():

            if len(rel_abundance_list) < 3:
                continue

            # add zeros back in
            if n_samples > len(rel_abundance_list):
                rel_abundance_list.extend([0]*(n_samples-len(rel_abundance_list)))

            mean_rel_abundance = np.mean(rel_abundance_list)
            if asv not in mean_rel_abund_all_treatments_dict:
                mean_rel_abund_all_treatments_dict[asv] = {}
            mean_rel_abund_all_treatments_dict[asv][treatment_transfer] = {}
            mean_rel_abund_all_treatments_dict[asv][treatment_transfer]['mean'] = mean_rel_abundance
            mean_rel_abund_all_treatments_dict[asv][treatment_transfer]['cv'] = np.std(rel_abundance_list)/mean_rel_abundance


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







#fig = plt.figure(figsize = (8, 8)) #
#fig.subplots_adjust(bottom= 0.15)

fig = plt.figure(figsize = (8, 16)) #
fig.subplots_adjust(bottom= 0.15)




log_dict = {}
log_dict['CV'] = {}
log_dict['mean'] = {}
for transfer_idx, transfer in enumerate(utils.transfers):

    ax_mean = plt.subplot2grid((4, 2), (transfer_idx, 0), colspan=1)
    ax_cv = plt.subplot2grid((4, 2), (transfer_idx, 1), colspan=1)

    migration_treatment = 'Global_migration.4.T%d' % transfer
    no_migration_treatment = 'No_migration.4.T%d' % transfer

    no_migration_treatment_mean_all = []
    migration_treatment_mean_all = []

    no_migration_treatment_cv_all = []
    migration_treatment_cv_all = []

    #log_dict['CV'][transfer] = {}
    #log_dict['mean'][transfer] = {}

    asv_to_keep = []

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

            #asv_to_keep.append(asv)

            if asv not in log_dict['CV']:
                log_dict['CV'][asv] = {}
                log_dict['CV'][asv]['no_migration'] = {}
                log_dict['CV'][asv]['global_migration'] = {}

            if asv not in log_dict['mean']:
                log_dict['mean'][asv] = {}
                log_dict['mean'][asv]['no_migration'] = {}
                log_dict['mean'][asv]['global_migration'] = {}

            log_dict['CV'][asv]['global_migration'][transfer] = migration_cv
            log_dict['CV'][asv]['no_migration'][transfer] = no_migration_cv

            log_dict['mean'][asv]['global_migration'][transfer] = migration_mean
            log_dict['mean'][asv]['no_migration'][transfer] = no_migration_mean

            

    color = utils.color_dict_range[('Global_migration', 4)][transfer-1].reshape(1,-1)

    ax_mean.scatter(no_migration_treatment_mean_all, migration_treatment_mean_all, alpha=0.8, c=color, zorder=2)

    ax_mean_min = min(no_migration_treatment_mean_all + migration_treatment_mean_all)
    ax_mean_max = max(no_migration_treatment_mean_all + migration_treatment_mean_all)

    ax_mean.plot([ax_mean_min*0.5, 1.1],[ax_mean_min*0.5, 1.1], c='k', ls=':', lw=2)
    ax_mean.set_xlim(ax_mean_min*0.5, 1.1)
    ax_mean.set_ylim(ax_mean_min*0.5, 1.1)
    ax_mean.set_title('Transfer %s' % transfer, fontsize=14 )

    #rho = np.corrcoef(np.log10(no_migration_treatment_mean_all), np.log10(migration_treatment_mean_all))[0,1]

    rho, p_value_rho = utils.run_permutation_corr(np.log10(no_migration_treatment_mean_all), np.log10(migration_treatment_mean_all))
    p_value_to_plot = utils.get_p_value(p_value_rho)

    print(transfer, 'Mean', rho, p_value_rho)

    ax_mean.text(0.2,0.9, r'$\rho=$' + str(round(rho,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_mean.transAxes)
    ax_mean.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_mean.transAxes)

    ax_mean.set_xlabel('No migration mean rel. abundance, ' + r'$\left< x \right>_{\mathrm{no\, mig}}$', fontsize=10)
    ax_mean.set_ylabel('Global migration mean rel. abundance, ' + r'$\left< x \right>_{\mathrm{global}}$', fontsize=9)

    ax_mean.set_xscale('log', basex=10)
    ax_mean.set_yscale('log', basey=10)


    no_migration_treatment_cv_all = np.asarray(no_migration_treatment_cv_all)
    migration_treatment_cv_all = np.asarray(migration_treatment_cv_all)
    idx_to_keep = (no_migration_treatment_cv_all>0) & (migration_treatment_cv_all>0)
    no_migration_treatment_cv_all = no_migration_treatment_cv_all[idx_to_keep]
    migration_treatment_cv_all = migration_treatment_cv_all[idx_to_keep]

    ax_cv.scatter(no_migration_treatment_cv_all, migration_treatment_cv_all, alpha=0.8, c=color, zorder=2)
    ax_cv_min = min(no_migration_treatment_cv_all + migration_treatment_cv_all)
    ax_cv_max = max(no_migration_treatment_cv_all + migration_treatment_cv_all)

    ax_cv.plot([ax_cv_min*0.1, ax_cv_max*1.3],[ax_cv_min*0.1, ax_cv_max*1.3], c='k', ls=':', lw=2)
    ax_cv.set_xlim(ax_cv_min*0.1, ax_cv_max*1.3)
    ax_cv.set_ylim(ax_cv_min*0.1, ax_cv_max*1.3)
    ax_cv.set_title('Transfer %s' % transfer, fontsize=14 )

    #rho = np.corrcoef(np.log10(no_migration_treatment_cv_all), np.log10(migration_treatment_cv_all))[0,1]
    rho, p_value_rho = utils.run_permutation_corr(np.log10(no_migration_treatment_cv_all), np.log10(migration_treatment_cv_all))

    print(transfer, 'CV', rho, p_value_rho)
    p_value_to_plot = utils.get_p_value(p_value_rho)

    ax_cv.text(0.2,0.9, r'$\rho=$' + str(round(rho,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_cv.transAxes )
    ax_cv.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_cv.transAxes )

    ax_cv.set_xlabel('No migration coefficient of variation, ' + r'$\mathrm{CV}_{x_{i}}(t)$', fontsize=10)
    ax_cv.set_ylabel('Global migration coefficient of variation, ' + r'$\mathrm{CV}_{x_{i}}(t)$', fontsize=9)

    ax_cv.set_xscale('log', basex=10)
    ax_cv.set_yscale('log', basey=10)


    ax_mean.text(-0.1, 1.04, plot_utils.sub_plot_labels[transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mean.transAxes)
    ax_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[4 + transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv.transAxes)







measure_dict= {}
for measure in ['mean', 'CV']:

    # make list for each ASV 
    asv_ = log_dict[measure].keys()

    no_migration_all = []
    global_migration_all = []

    for a in asv_:

        if (len(log_dict[measure][a]['no_migration']) == 2) and (len(log_dict[measure][a]['global_migration']) == 2):
            
            no_migration_all.append([log_dict[measure][a]['no_migration'][12], log_dict[measure][a]['no_migration'][18]])
            global_migration_all.append([log_dict[measure][a]['global_migration'][12], log_dict[measure][a]['global_migration'][18]])
    
    measure_no_migration_12 = np.log10([i[0] for i in no_migration_all])
    measure_migration_12 = np.log10([i[0] for i in global_migration_all])
    measure_no_migration_18 = np.log10([i[1] for i in no_migration_all])
    measure_migration_18 = np.log10([i[1] for i in global_migration_all])

    rho_18, rho_12, z_rho = utils.compare_rho_fisher_z(measure_no_migration_18, measure_migration_18, measure_no_migration_12, measure_migration_12)

    n_measure_no_migration_12 = len(measure_no_migration_12)
    n_measure_no_migration_18 = len(measure_no_migration_18)

    measure_12_merged = np.concatenate((measure_no_migration_12, measure_migration_12))
    measure_18_merged = np.concatenate((measure_no_migration_18, measure_migration_18))

    null_z_rho_all = []
    for i in range(n_iter):

        np.random.shuffle(measure_12_merged)
        np.random.shuffle(measure_18_merged)

        rho_18_null, rho_12_null, z_rho_null = utils.compare_rho_fisher_z(measure_18_merged[:n_measure_no_migration_18], measure_18_merged[n_measure_no_migration_18:], measure_12_merged[:n_measure_no_migration_12], measure_12_merged[n_measure_no_migration_12:])
        null_z_rho_all.append(z_rho_null)

    null_z_rho_all = np.asarray(null_z_rho_all)

    p_z_rho = sum(null_z_rho_all < z_rho)/n_iter
    print(measure + ' Z-test', z_rho, p_z_rho)

    measure_dict[measure] = {}
    measure_dict[measure]['z_rho'] = z_rho
    measure_dict[measure]['p_z_rho'] = p_z_rho


# plot error rates

simulation_global_rho_dict = slm_simulation_utils.load_simulation_global_rho_dict()

tau_all = np.asarray(list(simulation_global_rho_dict.keys()))
sigma_all = np.asarray(list(simulation_global_rho_dict[tau_all[0]].keys()))

np.sort(tau_all)
np.sort(sigma_all)


tau_delta_mean_all = []
tau_delta_cv_all = []
tau_delta_mean_error_all = []
tau_delta_cv_error_all = []
for tau in tau_all:

    tau_delta_mean = []
    tau_delta_cv = []

    tau_delta_mean_error = []
    tau_delta_cv_error = []

    for sigma in sigma_all:

        z_mean = np.asarray(simulation_global_rho_dict[tau][sigma]['z_rho']['mean_log10']['z_mean'])
        z_cv = np.asarray(simulation_global_rho_dict[tau][sigma]['z_rho']['cv_log10']['z_cv'])

        mean_z_mean = np.mean(z_mean)
        mean_z_cv = np.mean(z_cv)

        mean_error_z_mean = np.mean(np.absolute((z_mean - measure_dict['mean']['z_rho'])/measure_dict['mean']['z_rho']))
        mean_error_z_cv = np.mean(np.absolute((z_cv - measure_dict['CV']['z_rho'])/measure_dict['CV']['z_rho']))

        tau_delta_mean.append(mean_z_mean)
        tau_delta_cv.append(mean_z_cv)

        tau_delta_mean_error.append(mean_error_z_mean)
        tau_delta_cv_error.append(mean_error_z_cv)


    tau_delta_mean_all.append(tau_delta_mean)
    tau_delta_cv_all.append(tau_delta_cv)

    tau_delta_mean_error_all.append(tau_delta_mean_error)
    tau_delta_cv_error_all.append(tau_delta_cv_error)



tau_delta_mean_all = np.asarray(tau_delta_mean_all)
tau_delta_cv_all = np.asarray(tau_delta_cv_all)

tau_delta_mean_error_all = np.asarray(tau_delta_mean_error_all)
tau_delta_cv_error_all = np.asarray(tau_delta_cv_error_all)


ax_simulation_mean = plt.subplot2grid((4, 2), (2, 0), colspan=1)
ax_simulation_cv = plt.subplot2grid((4, 2), (2, 1), colspan=1)

ax_simulation_mean_error = plt.subplot2grid((4, 2), (3, 0), colspan=1)
ax_simulation_cv_error = plt.subplot2grid((4, 2), (3, 1), colspan=1)

x_axis = sigma_all
y_axis = tau_all

x_axis_log10 = np.log10(x_axis)




# ax_simulation_mean
z_rho_mean = measure_dict['mean']['z_rho']
delta_range = max([z_rho_mean - np.amin(tau_delta_mean_all),  np.amax(tau_delta_mean_all) - z_rho_mean])
pcm_rho = ax_simulation_mean.pcolor(x_axis_log10, y_axis, tau_delta_mean_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=z_rho_mean - delta_range, vcenter=z_rho_mean, vmax=z_rho_mean + delta_range))
clb_rho = plt.colorbar(pcm_rho, ax=ax_simulation_mean)
clb_rho.set_label(label='Change in ' + r'$\rho$'  + ' after cessation of migration, ' +  r'$Z_{\rho}$', fontsize=9)
ax_simulation_mean.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_simulation_mean.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_simulation_mean.xaxis.set_major_formatter(plot_utils.fake_log)
# Set observed marking and label
clb_rho.ax.axhline(y=z_rho_mean, c='k')
clb_rho.set_ticks([-0.6, -0.4, 0, 0.2])
clb_rho.set_ticklabels(['-0.6', '-0.4', '0.0', '0.2'])
original_ticks = list(clb_rho.get_ticks())
clb_rho.set_ticks(original_ticks + [z_rho_mean])
clb_rho.set_ticklabels(original_ticks + ['Obs.'])





# ax_simulation_cv
z_rho_cv = measure_dict['CV']['z_rho']
delta_range = max([z_rho_cv - np.amin(tau_delta_cv_all),  np.amax(tau_delta_cv_all) - z_rho_cv])
pcm_slope_rho = ax_simulation_cv.pcolor(x_axis_log10, y_axis, tau_delta_cv_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=z_rho_cv - delta_range, vcenter=z_rho_cv, vmax=z_rho_cv + delta_range))
clb_slope_rho = plt.colorbar(pcm_slope_rho, ax=ax_simulation_cv)
clb_slope_rho.set_label(label='Change in ' + r'$\rho$'  + ' after cessation of migration, ' +  r'$Z_{\rho}$', fontsize=9)
ax_simulation_cv.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_simulation_cv.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_simulation_cv.xaxis.set_major_formatter(plot_utils.fake_log)
# Set observed marking and label
clb_slope_rho.ax.axhline(y=z_rho_cv, c='k')
clb_slope_rho.set_ticks([-0.2, 2.0, 0.4, 0.8])
clb_slope_rho.set_ticklabels(['-0.2', '2.0', '0.4', '0.8'])
original_ticks = list(clb_slope_rho.get_ticks())
clb_slope_rho.set_ticks(original_ticks + [z_rho_cv])
clb_slope_rho.set_ticklabels(original_ticks + ['Obs.'])



# error heatmapts
pcm_rho_error = ax_simulation_mean_error.pcolor(x_axis_log10, y_axis, tau_delta_mean_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(tau_delta_mean_error_all), vcenter=np.median(np.ndarray.flatten(tau_delta_mean_error_all)), vmax=np.amax(tau_delta_mean_error_all)))
clb_rho_error = plt.colorbar(pcm_rho_error, ax=ax_simulation_mean_error)
clb_rho_error.set_label(label='Relative error of ' + r'$Z_{\rho}$'  + ' from simulated data', fontsize=9)
ax_simulation_mean_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_simulation_mean_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_simulation_mean_error.xaxis.set_major_formatter(plot_utils.fake_log)

pcm_slope_rho_error = ax_simulation_cv_error.pcolor(x_axis_log10, y_axis, tau_delta_cv_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(tau_delta_cv_error_all), vcenter=np.median(np.ndarray.flatten(tau_delta_cv_error_all)), vmax=np.amax(tau_delta_cv_error_all)))
clb_slope_rho_error = plt.colorbar(pcm_slope_rho_error, ax=ax_simulation_cv_error)
clb_slope_rho_error.set_label(label='Relative error of ' + r'$Z_{\rho}$' + ' from simulated data', fontsize=9)
ax_simulation_cv_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_simulation_cv_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_simulation_cv_error.xaxis.set_major_formatter(plot_utils.fake_log)

ax_simulation_mean.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_mean.transAxes)
ax_simulation_mean_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_mean_error.transAxes)

ax_simulation_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[6], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_cv.transAxes)
ax_simulation_cv_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[7], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_cv_error.transAxes)



fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_global.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()

