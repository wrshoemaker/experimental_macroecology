from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import pickle

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import slm_simulation_utils
import plot_utils

np.random.seed(123456789)


mad_dict_path = utils.directory + "/data/mad_dict.pickle"




def make_mad_dict():

    count_dict = utils.get_otu_dict()

    mean_rel_abund_all_treatments_dict = {}
    for treatment in ['No_migration.4.T%s', 'No_migration.40.T%s', 'Global_migration.4.T%s', 'Parent_migration.4.T%s', 'Parent_migration.NA.T0%s']:

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


    mad_dict = {}
    mad_dict['data'] = {}
    mad_dict['stats'] = {}

    for transfer_idx, transfer in enumerate(utils.transfers):

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


        mad_dict['data'][transfer] = {}
        mad_dict['data'][transfer]['log10_no_migration_treatment_all'] = np.log10(no_migration_treatment_all).tolist()
        mad_dict['data'][transfer]['log10_migration_treatment_all'] = np.log10(migration_treatment_all).tolist()
        mad_dict['data'][transfer]['log10_ratios_all'] = np.log10(ratios_all).tolist()
        mad_dict['data'][transfer]['log10_parent_all'] = np.log10(parent_all).tolist()
        mad_dict['data'][transfer]['no_migration_norm'] = no_migration_norm


    iter_ = 10000
    # test for correlation change 

    log10_no_migration_treatment_all_12 = np.asarray(mad_dict['data'][12]['log10_no_migration_treatment_all'])
    log10_migration_treatment_all_12 = np.asarray(mad_dict['data'][12]['log10_migration_treatment_all'])

    log10_no_migration_treatment_all_18 = np.asarray(mad_dict['data'][18]['log10_no_migration_treatment_all'])
    log10_migration_treatment_all_18 = np.asarray(mad_dict['data'][18]['log10_migration_treatment_all'])

    n_no_migration_12 = len(mad_dict['data'][12]['log10_no_migration_treatment_all'])
    n_no_migration_18 = len(mad_dict['data'][18]['log10_no_migration_treatment_all'])
    rho_18, rho_12, z = utils.compare_rho_fisher_z(log10_no_migration_treatment_all_18, log10_migration_treatment_all_18, log10_no_migration_treatment_all_12, log10_migration_treatment_all_12)

    merged_mad_no_migration = np.concatenate((log10_no_migration_treatment_all_12, log10_no_migration_treatment_all_18))
    merged_mad_parent_migration = np.concatenate((log10_migration_treatment_all_12, log10_migration_treatment_all_18))

    # permute time labels for paired data!
    idx_mad = np.arange(n_no_migration_12 + n_no_migration_18)

    z_null_all = []
    for i in range(iter_):

        np.random.shuffle(idx_mad)

        merged_mad_no_migration_12 = merged_mad_no_migration[idx_mad[:n_no_migration_12]]
        merged_mad_no_migration_18 = merged_mad_no_migration[idx_mad[n_no_migration_12:]]

        merged_mad_parent_migration_12 = merged_mad_parent_migration[idx_mad[:n_no_migration_12]]
        merged_mad_parent_migration_18 = merged_mad_parent_migration[idx_mad[n_no_migration_12:]]

        rho_12_null, rho_18_null, z_null = utils.compare_rho_fisher_z(merged_mad_no_migration_18, merged_mad_parent_migration_18, merged_mad_no_migration_12, merged_mad_parent_migration_12)

        z_null_all.append(z_null)


    z_null_all = np.asarray(z_null_all)
    p_z = sum(z_null_all > z)/iter_

    mad_dict['stats']['z'] = z
    mad_dict['stats']['p_value_z'] = p_z

    print('Change in rho', z, p_z)

    
    # test for change of slope
    n_parent_12 = len(mad_dict['data'][12]['log10_parent_all'])
    n_parent_18 = len(mad_dict['data'][18]['log10_parent_all'])

    log10_parent_all_12 = np.asarray(mad_dict['data'][12]['log10_parent_all'])
    log10_ratios_all_12 = np.asarray(mad_dict['data'][12]['log10_ratios_all'])

    log10_parent_all_18 = np.asarray(mad_dict['data'][18]['log10_parent_all'])
    log10_ratios_all_18 = np.asarray(mad_dict['data'][18]['log10_ratios_all'])

    merged_parent = np.concatenate((log10_parent_all_12, log10_parent_all_18))
    merged_ratio = np.concatenate((log10_ratios_all_12, log10_ratios_all_18))

    idx_slope = np.arange(n_parent_12 + n_parent_18)

    slope_18, slope_12, t_slope, intercept_18, intercept_12, t_intercept, r_value_18, r_value_12 = utils.t_statistic_two_slopes(log10_parent_all_18, log10_ratios_all_18, log10_parent_all_12, log10_ratios_all_12)

    t_slope_null_all = []
    for i in range(iter_):

        np.random.shuffle(idx_slope)

        merged_parent_12 = merged_parent[idx_slope[:n_parent_12]]
        merged_parent_18 = merged_parent[idx_slope[n_parent_12:]]

        merged_ratio_12 = merged_ratio[idx_slope[:n_parent_12]]
        merged_ratio_18 = merged_ratio[idx_slope[n_parent_12:]]


        slope_18_null, slope_12_null, t_slope_null, intercept_18_null, intercept_12_null, t_intercept_null, r_value_18_null, r_value_12_null = utils.t_statistic_two_slopes(merged_parent_18, merged_ratio_18, merged_parent_12, merged_ratio_12)
        t_slope_null_all.append(t_slope_null)


    t_slope_null_all = np.asarray(t_slope_null_all)
    p_t_slope = sum(t_slope_null_all < t_slope)/iter_

    mad_dict['stats']['t_slope'] = t_slope
    mad_dict['stats']['p_value_t_slope'] = p_t_slope

    print('t-statistic', t_slope, p_t_slope)

    sys.stderr.write("Saving dictionary...\n")
    with open(mad_dict_path, 'wb') as handle:
        pickle.dump(mad_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



########################
# plot the simulations #
########################

def run_best_parameter_simulations():

    with open(mad_dict_path, 'rb') as handle:
        mad_dict = pickle.load(handle)

    # identify parameter regime with lowest error
    simulation_parent_rho_abc_dict = slm_simulation_utils.load_simulation_parent_rho_abc_dict()

    tau_all = np.asarray(simulation_parent_rho_abc_dict['tau_all'])
    sigma_all = np.asarray(simulation_parent_rho_abc_dict['sigma_all'])

    z = mad_dict['data']['z']
    t_slope = mad_dict['data']['t_slope']


    # change in correlation
    z_all = np.asarray(simulation_parent_rho_abc_dict['rho_12_vs_18']['Z'])
    t_slope_all = np.asarray(simulation_parent_rho_abc_dict['slope_12_vs_18']['mad_slope_t_test'])

    obs = np.asarray([z, t_slope])
    pred = np.asarray([z_all, t_slope_all])

    best_tau, best_sigma = utils.weighted_euclidean_distance(tau_all, sigma_all, obs, pred)

    # run simulations
    sys.stderr.write("Running simulation with optimal parameters...\n")
    #slm_simulation_utils.run_simulation_parent_rho_fixed_parameters(best_tau, best_sigma, n_iter=1000)
    slm_simulation_utils.run_simulation_parent_rho_abc(tau=best_tau, sigma=best_sigma, n_iter=1000)
    sys.stderr.write("Done!\n")




def make_plot():

    with open(mad_dict_path, 'rb') as handle:
        mad_dict = pickle.load(handle)

    z = mad_dict['stats']['z']
    t_slope = mad_dict['stats']['t_slope']

    fig = plt.figure(figsize=(9, 12))
    fig.subplots_adjust(top=0.15)

    fig.text(0.29, 1.02 , "Change in MAD\ncorrelation over time", va='center', ha='center', fontsize=20)
    fig.text(0.89, 1.02, "Progenitor abundance vs.\nassembled community MADs", va='center',  ha='center', fontsize=20)
    
    fig.text(0.58, 1.06 , "Regional migration statistics", ha='center', fontweight='bold', fontsize=23)


    delta_w = 0.55

    # over reps per-unit time
    outergs_z = gridspec.GridSpec(1, 1)
    outergs_z.update(bottom=-0.025, left=0.01, top=1-0.01, right=0.01+delta_w)
    outerax_z = fig.add_subplot(outergs_z[0])
    outerax_z.tick_params(axis='both' ,which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    outerax_z.set_facecolor('none')

    gs_left = 0.1
    gs_right_offset = 0.11
    gs_z = gridspec.GridSpec(3, 1)
    #gs_delta_w = 
    gs_z.update(bottom=0.05, left=gs_left, top=1-0.05, right=gs_left+delta_w-gs_right_offset, wspace=0.15, hspace=0.25)


    outergs_t = gridspec.GridSpec(1, 1)
    outergs_t.update(bottom=-0.025, left=0.61, top=0.99, right=0.61+delta_w)
    outerax_t = fig.add_subplot(outergs_t[0])
    outerax_t.tick_params(axis='both' ,which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    outerax_t.set_facecolor('none')

    gs_t = gridspec.GridSpec(3, 1)
    gs_t.update(bottom=0.05, left=0.61+gs_left, top=1-0.05, right=0.61+gs_left+delta_w-gs_right_offset, wspace=0.15, hspace=0.25)


    # axis widths 
    for axis in ['top','bottom','left','right']:
        outerax_z.spines[axis].set_linewidth(3)
        outerax_t.spines[axis].set_linewidth(3)
    

    for transfer_idx, transfer in enumerate(utils.transfers):

        ax_z_transfer = fig.add_subplot(gs_z[transfer_idx, 0])
        ax_t_transfer = fig.add_subplot(gs_t[transfer_idx, 0])


        color = utils.color_dict_range[('Parent_migration', 4)][transfer-1].reshape(1,-1)

        log10_no_migration_treatment_all = np.asarray(mad_dict['data'][transfer]['log10_no_migration_treatment_all'])
        log10_migration_treatment_all = np.asarray(mad_dict['data'][transfer]['log10_migration_treatment_all'])
        log10_ratios_all = np.asarray(mad_dict['data'][transfer]['log10_ratios_all'])
        log10_parent_all = np.asarray(mad_dict['data'][transfer]['log10_parent_all'])
        #no_migration_norm = np.asarray(mad_dict['data'][transfer]['no_migration_norm'])

        no_migration_treatment_all = 10**log10_no_migration_treatment_all
        migration_treatment_all = 10**log10_migration_treatment_all
        ratios_all = 10**log10_ratios_all
        parent_all = 10**log10_parent_all


        ax_z_transfer.scatter(no_migration_treatment_all, migration_treatment_all, alpha=0.8, c=color, zorder=2, label='One ASV')
        ax_compare_min = min(np.concatenate([no_migration_treatment_all, migration_treatment_all]))


        ax_z_transfer.plot([ax_compare_min*0.5, 1.1],[ax_compare_min*0.5, 1.1], c='k', ls=':', lw=2, label='1:1')
        ax_z_transfer.set_xlim(ax_compare_min*0.5,1.1)
        ax_z_transfer.set_ylim(ax_compare_min*0.5, 1.1)
        ax_z_transfer.set_title('Transfer %s' % transfer, fontsize=14)

        rho, p_value = utils.run_permutation_corr(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))
        p_value_to_plot = utils.get_p_value(p_value)

        ax_z_transfer.text(0.8,0.29, r'$\rho^{2}=$' + str(round(rho**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_z_transfer.transAxes )
        ax_z_transfer.text(0.8,0.2, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_z_transfer.transAxes)


        ax_z_transfer.set_xlabel('Mean rel. abundance, no migration, ' + r'$\left< x \right>_{\mathrm{no}}$', fontsize=12)
        ax_z_transfer.set_ylabel('Mean rel. abundance, regional, ' + r'$\left< x \right>_{\mathrm{reg.}}$', fontsize=12)

        ax_z_transfer.set_xscale('log', basex=10)
        ax_z_transfer.set_yscale('log', basey=10)

        ax_t_transfer.scatter(parent_all, ratios_all, alpha=0.8, c=color, zorder=2, label='One ASV')
        ax_t_transfer.axhline(1, lw=1.5, ls=':',color='k', zorder=1, label='Null')


        ax_parent_abs_max_y = max(np.absolute(np.log10(ratios_all)))
        ax_t_transfer.set_ylim(10**(-1.1*ax_parent_abs_max_y), 10**(ax_parent_abs_max_y*1.1))
        ax_t_transfer.set_title('Transfer %s' % transfer, fontsize=14)

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(parent_all), np.log10(ratios_all))
        x_log10_range =  np.linspace(min(np.log10(parent_all)) , max(np.log10(parent_all)), 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        ax_t_transfer.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="OLS regression")

        if p_value < 0.05:
            p_value_text = r'$P < 0.05$'
        else:
            p_value_text = r'$P \geq 0.05$'


        ax_t_transfer.text(0.8,0.29, r'$\rho^{2}=$' + str(round(r_value**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_t_transfer.transAxes )
        ax_t_transfer.text(0.8,0.2, p_value_text, fontsize=10, color='k', ha='center', va='center', transform=ax_t_transfer.transAxes )

        ax_t_transfer.set_xscale('log', basex=10)
        ax_t_transfer.set_yscale('log', basey=10)

        ax_t_transfer.set_xlabel('Rel. abundance in progenitor community', fontsize=12)
        ax_t_transfer.set_ylabel('Mean rel. abundance ratio, ' + r'$\left< x \right>_{\mathrm{reg.}}/\left< x \right>_{\mathrm{no}}$',  fontsize=12)

        
        if transfer_idx == 0:
            ax_z_transfer.legend(loc="lower right", fontsize=8) 
            ax_t_transfer.legend(loc="lower left", fontsize=8)


        ax_z_transfer.text(-0.1, 1.04, plot_utils.sub_plot_labels[transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_z_transfer.transAxes)
        ax_t_transfer.text(-0.1, 1.04, plot_utils.sub_plot_labels[3 + transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_transfer.transAxes)

    

    ax_z_simulation = fig.add_subplot(gs_z[2, 0])
    ax_t_slope_simulation = fig.add_subplot(gs_t[2, 0])

    sys.stderr.write("Loading simulation dictionary...\n")
    simulation_parent_rho_fixed_parameters_dict = slm_simulation_utils.load_simulation_parent_rho_fixed_parameters_dict()

    best_tau = simulation_parent_rho_fixed_parameters_dict['tau_all'][0]
    best_sigma = simulation_parent_rho_fixed_parameters_dict['sigma_all'][0]

    z_simulation_best_parameters = simulation_parent_rho_fixed_parameters_dict['rho_12_vs_18']['Z']
    t_slope_simulation_best_parameters = simulation_parent_rho_fixed_parameters_dict['slope_12_vs_18']['mad_slope_t_test']

    z_simulation_best_parameters = np.sort(z_simulation_best_parameters)
    lower_ci_z = z_simulation_best_parameters[int(0.025*len(z_simulation_best_parameters))]
    upper_ci_z = z_simulation_best_parameters[int(0.975*len(z_simulation_best_parameters))]

    ax_z_simulation.hist(z_simulation_best_parameters, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Parent_migration',4)], histtype='stepfilled', density=True, zorder=2)
    ax_z_simulation.axvline(x=z, ls='--', lw=3, c='k', label='Observed ' +  r'$Z_{\rho}$')
    ax_z_simulation.set_xlabel('Predicted ' + r'$Z_{\rho}$' + ' from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=12)
    ax_z_simulation.set_ylabel('Probability density',  fontsize=12)
    ax_z_simulation.axvline(x=lower_ci_z, ls=':', lw=3, c='k', label='95% CIs')
    ax_z_simulation.axvline(x=upper_ci_z, ls=':', lw=3, c='k')
    ax_z_simulation.legend(loc="upper right", fontsize=8)


    t_slope_simulation_best_parameters = np.sort(t_slope_simulation_best_parameters)
    lower_ci_t = t_slope_simulation_best_parameters[int(0.025*len(t_slope_simulation_best_parameters))]
    upper_ci_t = t_slope_simulation_best_parameters[int(0.975*len(t_slope_simulation_best_parameters))]

    ax_t_slope_simulation.hist(t_slope_simulation_best_parameters, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Parent_migration',4)], histtype='stepfilled', density=True, zorder=2)
    ax_t_slope_simulation.axvline(x=t_slope, ls='--', lw=3, c='k', label='Observed ' +  r'$t_{\mathrm{slope}}$')
    ax_t_slope_simulation.set_xlabel('Predicted ' + r'$t_{\mathrm{slope}}$' + ' from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=12)
    ax_t_slope_simulation.set_ylabel('Probability density',  fontsize=12)
    ax_z_simulation.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_z_simulation.transAxes)
    ax_t_slope_simulation.text(-0.1, 1.04, plot_utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_slope_simulation.transAxes)
    ax_t_slope_simulation.axvline(x=lower_ci_t, ls=':', lw=3, c='k', label='95% CIs')
    ax_t_slope_simulation.axvline(x=upper_ci_t, ls=':', lw=3, c='k')
    ax_t_slope_simulation.legend(loc="upper right", fontsize=8)



    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_parent.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def make_plot_old():

    with open(mad_dict_path, 'rb') as handle:
        mad_dict = pickle.load(handle)

    z = mad_dict['stats']['z']
    t_slope = mad_dict['stats']['t_slope']


    #############
    # make plot #
    #############

    fig = plt.figure(figsize = (8, 12)) #
    fig.subplots_adjust(bottom= 0.15)

    for transfer_idx, transfer in enumerate(utils.transfers):

        ax_compare = plt.subplot2grid((3, 2), (transfer_idx, 0), colspan=1)
        ax_parent = plt.subplot2grid((3, 2), (transfer_idx, 1), colspan=1)

        color = utils.color_dict_range[('Parent_migration', 4)][transfer-1].reshape(1,-1)


        log10_no_migration_treatment_all = np.asarray(mad_dict['data'][transfer]['log10_no_migration_treatment_all'])
        log10_migration_treatment_all = np.asarray(mad_dict['data'][transfer]['log10_migration_treatment_all'])
        log10_ratios_all = np.asarray(mad_dict['data'][transfer]['log10_ratios_all'])
        log10_parent_all = np.asarray(mad_dict['data'][transfer]['log10_parent_all'])
        no_migration_norm = np.asarray(mad_dict['data'][transfer]['no_migration_norm'])

        no_migration_treatment_all = 10**log10_no_migration_treatment_all
        migration_treatment_all = 10**log10_migration_treatment_all
        ratios_all = 10**log10_ratios_all
        parent_all = 10**log10_parent_all


        ax_compare.scatter(no_migration_treatment_all, migration_treatment_all, alpha=0.8, c=color, zorder=2, label='One ASV')
        ax_compare_min = min(no_migration_treatment_all + migration_treatment_all)
        ax_compare_max = max(no_migration_treatment_all + migration_treatment_all)

        ax_compare.plot([ax_compare_min*0.5, 1.1],[ax_compare_min*0.5, 1.1], c='k', ls=':', lw=2, label='1:1')
        ax_compare.set_xlim(ax_compare_min*0.5,1.1)
        ax_compare.set_ylim(ax_compare_min*0.5, 1.1)
        ax_compare.set_title('Transfer %s' % transfer, fontsize=12)

        #rho = np.corrcoef(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))[0,1]

        rho, p_value = utils.run_permutation_corr(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))
        p_value_to_plot = utils.get_p_value(p_value)


        #ax_compare.text(0.2,0.92, r'$\rho^{2}=$' + str(round(rho**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes )
        #ax_compare.text(0.18,0.8, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes)
        ax_compare.text(0.8,0.29, r'$\rho^{2}=$' + str(round(rho**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes )
        ax_compare.text(0.8,0.2, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_compare.transAxes)


        ax_compare.set_xlabel('Mean rel. abundance, no migration, ' + r'$\left< x \right>_{\mathrm{no\, mig}}$', fontsize=10)
        ax_compare.set_ylabel('Mean rel. abundance, regional, ' + r'$\left< x \right>_{\mathrm{regional}}$', fontsize=10)

        ax_compare.set_xscale('log', basex=10)
        ax_compare.set_yscale('log', basey=10)
        #ax_compare.legend(loc="lower right", fontsize=8)

        ax_parent.scatter(parent_all, ratios_all, alpha=0.8, c=color, zorder=2, label='One ASV')
        ax_parent.axhline(1, lw=1.5, ls=':',color='k', zorder=1, label='Null')
        #ax_parent.set_xlim(0.6*min(obs), 2*max(obs))
        #ax_parent.set_ylim(0.6*min(obs), 2*max(obs))


        ax_parent_abs_max_y = max(np.absolute(np.log10(ratios_all)))
        ax_parent.set_ylim(10**(-1.1*ax_parent_abs_max_y), 10**(ax_parent_abs_max_y*1.1))
        ax_parent.set_title('Transfer %s' % transfer, fontsize=12)

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(parent_all), np.log10(ratios_all))
        x_log10_range =  np.linspace(min(np.log10(parent_all)) , max(np.log10(parent_all)), 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        ax_parent.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="OLS regression")

        if p_value < 0.05:
            p_value_text = r'$P < 0.05$'
        else:
            p_value_text = r'$P \geq 0.05$'

        #ax_parent.text(0.2,0.92, r'$\rho^{2}=$' + str(round(r_value**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_parent.transAxes )
        #ax_parent.text(0.2,0.8, p_value_text, fontsize=10, color='k', ha='center', va='center', transform=ax_parent.transAxes )

        ax_parent.text(0.8,0.29, r'$\rho^{2}=$' + str(round(r_value**2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_parent.transAxes )
        ax_parent.text(0.8,0.2, p_value_text, fontsize=10, color='k', ha='center', va='center', transform=ax_parent.transAxes )


        ax_parent.set_xscale('log', basex=10)
        ax_parent.set_yscale('log', basey=10)

        ax_parent.set_xlabel('Rel. abundance in progenitor community', fontsize=10)
        ax_parent.set_ylabel('Mean rel. abundance ratio, ' + r'$\left< x \right>_{\mathrm{regional}}/\left< x \right>_{\mathrm{no\, mig}}$',  fontsize=10)

        no_migration_norm_all = np.asarray(no_migration_norm_all)
        no_migration_norm_all_log10 = np.log10(no_migration_norm_all)
        parent_all = np.asarray(parent_all)
        parent_all_log10 = np.log10(parent_all)
        
        if transfer_idx == 0:
            ax_compare.legend(loc="lower right", fontsize=8) 
            ax_parent.legend(loc="lower left", fontsize=8)



        ax_compare.text(-0.1, 1.04, plot_utils.sub_plot_labels[transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_compare.transAxes)
        ax_parent.text(-0.1, 1.04, plot_utils.sub_plot_labels[3 + transfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_parent.transAxes)



    sys.stderr.write("Loading simulation dictionary...\n")
    simulation_parent_rho_fixed_parameters_dict = slm_simulation_utils.load_simulation_parent_rho_fixed_parameters_dict()

    best_tau = simulation_parent_rho_fixed_parameters_dict['tau_all'][0]
    best_sigma = simulation_parent_rho_fixed_parameters_dict['sigma_all'][0]

    z_simulation_best_parameters = simulation_parent_rho_fixed_parameters_dict['rho_12_vs_18']['Z']
    t_slope_simulation_best_parameters = simulation_parent_rho_fixed_parameters_dict['slope_12_vs_18']['mad_slope_t_test']


    ax_z_simulation = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    ax_t_slope_simulation = plt.subplot2grid((3, 2), (2, 1), colspan=1)

    z_simulation_best_parameters = np.sort(z_simulation_best_parameters)
    lower_ci_z = z_simulation_best_parameters[int(0.025*len(z_simulation_best_parameters))]
    upper_ci_z = z_simulation_best_parameters[int(0.975*len(z_simulation_best_parameters))]

    ax_z_simulation.hist(z_simulation_best_parameters, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Parent_migration',4)], histtype='stepfilled', density=True, zorder=2)
    ax_z_simulation.axvline(x=z, ls='--', lw=3, c='k', label='Observed ' +  r'$Z_{\rho}$')
    #ax_z_simulation.set_xlabel('Simulated ' + r'$Z_{\rho}$' + '\nfrom optimal ' + r'$\tau$' + ' and ' + r'$\sigma$', fontsize=11)
    #ax_z_simulation.set_xlabel('Simulated ' + r'$Z_{\rho}$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_z_simulation.set_xlabel('Predicted ' + r'$Z_{\rho}$' + ' from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_z_simulation.set_ylabel('Probability density',  fontsize=11)
    ax_z_simulation.axvline(x=lower_ci_z, ls=':', lw=3, c='k', label='95% CIs')
    ax_z_simulation.axvline(x=upper_ci_z, ls=':', lw=3, c='k')
    ax_z_simulation.legend(loc="upper right", fontsize=8)



    t_slope_simulation_best_parameters = np.sort(t_slope_simulation_best_parameters)
    lower_ci_t = t_slope_simulation_best_parameters[int(0.025*len(t_slope_simulation_best_parameters))]
    upper_ci_t = t_slope_simulation_best_parameters[int(0.975*len(t_slope_simulation_best_parameters))]

    ax_t_slope_simulation.hist(t_slope_simulation_best_parameters, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Parent_migration',4)], histtype='stepfilled', density=True, zorder=2)
    ax_t_slope_simulation.axvline(x=t_slope, ls='--', lw=3, c='k', label='Observed ' +  r'$t_{\mathrm{slope}}$')
    #ax_t_slope_simulation.set_xlabel('Simulated ' + r'$t_{\mathrm{slope}}$' + 'from optimal ' + r'$\tau$' + ' and ' + r'$\sigma$', fontsize=11)
    #ax_t_slope_simulation.set_xlabel('Simulated ' + r'$t_{\mathrm{slope}}$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_t_slope_simulation.set_xlabel('Predicted ' + r'$t_{\mathrm{slope}}$' + ' from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_t_slope_simulation.set_ylabel('Probability density',  fontsize=11)
    ax_z_simulation.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_z_simulation.transAxes)
    ax_t_slope_simulation.text(-0.1, 1.04, plot_utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_slope_simulation.transAxes)
    ax_t_slope_simulation.axvline(x=lower_ci_t, ls=':', lw=3, c='k', label='95% CIs')
    ax_t_slope_simulation.axvline(x=upper_ci_t, ls=':', lw=3, c='k')
    ax_t_slope_simulation.legend(loc="upper right", fontsize=8)


    fig.text(0.27, 0.925, "Regional migration statistics", va='center', fontsize=20)


    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_parent.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    #fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_parent.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




if __name__=='__main__':

    #make_mad_dict()

    #run_best_parameter_simulations()


    make_plot()

