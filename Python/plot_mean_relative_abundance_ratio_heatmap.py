from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import pickle

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import slm_simulation_utils
import plot_utils

np.random.seed(123456789)


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

        #mean_abundance_dict[treatment] = {}
        #samples_to_keep = [sample for sample in samples if treatment in sample]
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



#############
# make plot #
#############

fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)

mad_dict = {}

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


    color = utils.color_dict_range[('Parent_migration', 4)][transfer-1].reshape(1,-1)
    rho, p_value = utils.run_permutation_corr(np.log10(no_migration_treatment_all), np.log10(migration_treatment_all))
    p_value_to_plot = utils.get_p_value(p_value)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(parent_all), np.log10(ratios_all))
    x_log10_range =  np.linspace(min(np.log10(parent_all)) , max(np.log10(parent_all)), 10000)
    y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)

    if p_value < 0.05:
        p_value_text = r'$P < 0.05$'
    else:
        p_value_text = r'$P \nless 0.05$'


    mad_dict[transfer] = {}
    mad_dict[transfer]['log10_no_migration_treatment_all'] = np.log10(no_migration_treatment_all)
    mad_dict[transfer]['log10_migration_treatment_all'] = np.log10(migration_treatment_all)
    mad_dict[transfer]['log10_parent_all'] = np.log10(parent_all)
    mad_dict[transfer]['log10_ratios_all'] = np.log10(ratios_all)

    no_migration_norm_all = np.asarray(no_migration_norm_all)
    no_migration_norm_all_log10 = np.log10(no_migration_norm_all)
    parent_all = np.asarray(parent_all)
    parent_all_log10 = np.log10(parent_all)







iter_ = 10000
# test for correlation change 

n_no_migration_12 = len(mad_dict[12]['log10_no_migration_treatment_all'])
n_no_migration_18 = len(mad_dict[18]['log10_no_migration_treatment_all'])
rho_18, rho_12, z = utils.compare_rho_fisher_z(mad_dict[18]['log10_no_migration_treatment_all'], mad_dict[18]['log10_migration_treatment_all'], mad_dict[12]['log10_no_migration_treatment_all'], mad_dict[12]['log10_migration_treatment_all'])


#merged_mad_12 = np.concatenate((mad_dict[12]['log10_no_migration_treatment_all'], mad_dict[12]['log10_migration_treatment_all']))
#merged_mad_18 = np.concatenate((mad_dict[18]['log10_no_migration_treatment_all'], mad_dict[18]['log10_migration_treatment_all']))

merged_mad_no_migration = np.concatenate((mad_dict[12]['log10_no_migration_treatment_all'], mad_dict[18]['log10_no_migration_treatment_all']))
merged_mad_parent_migration = np.concatenate((mad_dict[12]['log10_migration_treatment_all'],  mad_dict[18]['log10_migration_treatment_all']))

# permute time labels for paired data!
idx_mad = np.arange(n_no_migration_12 + n_no_migration_18)


z_null_all = []
for i in range(iter_):

    #np.random.shuffle(merged_mad_12)
    #np.random.shuffle(merged_mad_18)

    np.random.shuffle(idx_mad)

    merged_mad_no_migration_12 = merged_mad_no_migration[idx_mad[:n_no_migration_12]]
    merged_mad_no_migration_18 = merged_mad_no_migration[idx_mad[n_no_migration_12:]]

    merged_mad_parent_migration_12 = merged_mad_parent_migration[idx_mad[:n_no_migration_12]]
    merged_mad_parent_migration_18 = merged_mad_parent_migration[idx_mad[n_no_migration_12:]]

    rho_12_null, rho_18_null, z_null = utils.compare_rho_fisher_z(merged_mad_no_migration_18, merged_mad_parent_migration_18, merged_mad_no_migration_12, merged_mad_parent_migration_12)

    #rho_12_null, rho_18_null, z_null = utils.compare_rho_fisher_z(merged_mad_18[:n_no_migration_18], merged_mad_18[n_no_migration_18:], merged_mad_12[:n_no_migration_12], merged_mad_12[n_no_migration_12:])
    z_null_all.append(z_null)


z_null_all = np.asarray(z_null_all)
p_z = sum(z_null_all > z)/iter_


print('Change in rho', z, p_z)




# test for change of slope
n_parent_12 = len(mad_dict[12]['log10_parent_all'])
n_parent_18 = len(mad_dict[18]['log10_parent_all'])

merged_parent = np.concatenate((mad_dict[12]['log10_parent_all'], mad_dict[18]['log10_parent_all']))
merged_ratio = np.concatenate((mad_dict[12]['log10_ratios_all'], mad_dict[18]['log10_ratios_all']))

idx_slope = np.arange(n_parent_12 + n_parent_18)
#merged_mad_12 = np.concatenate((mad_dict[12]['log10_no_migration_treatment_all'], mad_dict[12]['log10_migration_treatment_all']))
#merged_mad_18 = np.concatenate((mad_dict[18]['log10_no_migration_treatment_all'], mad_dict[18]['log10_migration_treatment_all']))


slope_18, slope_12, t_slope, intercept_18, intercept_12, t_intercept, r_value_18, r_value_12 = utils.t_statistic_two_slopes(mad_dict[18]['log10_parent_all'], mad_dict[18]['log10_ratios_all'], mad_dict[12]['log10_parent_all'], mad_dict[12]['log10_ratios_all'])

t_slope_null_all = []
for i in range(iter_):

    #np.random.shuffle(merged_ratio_12)
    #np.random.shuffle(merged_ratio_18)

    np.random.shuffle(idx_slope)

    merged_parent_12 = merged_parent[idx_slope[:n_parent_12]]
    merged_parent_18 = merged_parent[idx_slope[n_parent_12:]]

    merged_ratio_12 = merged_ratio[idx_slope[:n_parent_12]]
    merged_ratio_18 = merged_ratio[idx_slope[n_parent_12:]]


    slope_18_null, slope_12_null, t_slope_null, intercept_18_null, intercept_12_null, t_intercept_null, r_value_18_null, r_value_12_null = utils.t_statistic_two_slopes(merged_parent_18, merged_ratio_18, merged_parent_12, merged_ratio_12)
    t_slope_null_all.append(t_slope_null)


t_slope_null_all = np.asarray(t_slope_null_all)
p_t_slope = sum(t_slope_null_all < t_slope)/iter_


print('Change in t-statistic', t_slope, p_t_slope)



########################
# plot the simulations #
########################





simulation_parent_rho_dict = slm_simulation_utils.load_simulation_parent_rho_dict()

tau_all = np.asarray(list(simulation_parent_rho_dict.keys()))
sigma_all = np.asarray(list(simulation_parent_rho_dict[tau_all[0]].keys()))

np.sort(tau_all)
np.sort(sigma_all)

delta_rho_all = []
delta_slope_rho_all = []
delta_rho_error_all = []
delta_slope_rho_error_all = []
for tau in tau_all:

    tau_delta_rho = []
    tau_delta_slope_rho = []

    tau_delta_rho_error = []
    tau_delta_rho_slope_error = []

    for sigma in sigma_all:

        z_rho = simulation_parent_rho_dict[tau][sigma]['rho_12_vs_18']['Z']
        mad_slope_t_test = simulation_parent_rho_dict[tau][sigma]['slope_12_vs_18']['mad_slope_t_test']

        z_rho = np.asarray(z_rho)
        mad_slope_t_test = np.asarray(mad_slope_t_test)

        error_z_rho = np.absolute((z_rho - z) / z)
        error_mad_slope_t_test = np.absolute((mad_slope_t_test - t_slope) / t_slope)

        tau_delta_rho.append(np.mean(z_rho))
        tau_delta_slope_rho.append(np.mean(mad_slope_t_test))

        tau_delta_rho_error.append(np.mean(error_z_rho))
        tau_delta_rho_slope_error.append(np.mean(error_mad_slope_t_test))


    delta_rho_all.append(tau_delta_rho)
    delta_slope_rho_all.append(tau_delta_slope_rho)

    delta_rho_error_all.append(tau_delta_rho_error)
    delta_slope_rho_error_all.append(tau_delta_rho_slope_error)


# rows = tau
# columns = sigma
delta_rho_all = np.asarray(delta_rho_all)
delta_slope_rho_all = np.asarray(delta_slope_rho_all)

delta_rho_error_all = np.asarray(delta_rho_error_all)
delta_slope_rho_error_all = np.asarray(delta_slope_rho_error_all)



ax_rho = plt.subplot2grid((2, 2), (0, 0), colspan=1)
ax_slope_rho = plt.subplot2grid((2, 2), (0, 1), colspan=1)

ax_rho_error = plt.subplot2grid((2, 2), (1, 0), colspan=1)
ax_slope_rho_error = plt.subplot2grid((2, 2), (1, 1), colspan=1)


x_axis = sigma_all
y_axis = tau_all

x_axis_log10 = np.log10(x_axis)


delta_range = max([z - np.amin(delta_rho_all),  np.amax(delta_rho_all) - z])
pcm_rho = ax_rho.pcolor(x_axis_log10, y_axis, delta_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=z-delta_range, vcenter=z, vmax=z+delta_range))
clb_rho = plt.colorbar(pcm_rho, ax=ax_rho)
clb_rho.set_label(label='Change in ' + r'$\rho$'  + ' after cessation of migration, ' +  r'$Z_{\rho}$', fontsize=9)
ax_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_rho.xaxis.set_major_formatter(plot_utils.fake_log)


# Set observed marking and label
clb_rho.ax.axhline(y=z, c='k')
original_ticks = list(clb_rho.get_ticks())
clb_rho.set_ticks(original_ticks + [z])
clb_rho.set_ticklabels(original_ticks + ['Obs.'])




delta_range = max([t_slope - np.amin(delta_slope_rho_all),  np.amax(delta_slope_rho_all) - t_slope])

pcm_slope_rho = ax_slope_rho.pcolor(x_axis_log10, y_axis, delta_slope_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=t_slope-delta_range, vcenter=t_slope, vmax=t_slope+delta_range))
clb_slope_rho = plt.colorbar(pcm_slope_rho, ax=ax_slope_rho)
clb_slope_rho.set_label(label='Change in slope after cessation of migration, ' +  r'$t_{\mathrm{slope}}$', fontsize=9)
ax_slope_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_slope_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_slope_rho.xaxis.set_major_formatter(plot_utils.fake_log)
# Set observed marking and label
clb_slope_rho.ax.axhline(y=t_slope, c='k')
original_ticks = list(clb_slope_rho.get_ticks())
clb_slope_rho.set_ticks(original_ticks + [t_slope])
clb_slope_rho.set_ticklabels(original_ticks + ['Obs.'])




pcm_rho_error = ax_rho_error.pcolor(x_axis_log10, y_axis, delta_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(delta_rho_error_all), vcenter=np.median(np.ndarray.flatten(delta_slope_rho_error_all)), vmax=np.amax(delta_rho_error_all)))
clb_rho_error = plt.colorbar(pcm_rho_error, ax=ax_rho_error)
clb_rho_error.set_label(label='Relative error of ' + r'$Z_{\rho}$'  + ' from simulated data', fontsize=9)
ax_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_rho_error.xaxis.set_major_formatter(plot_utils.fake_log)




pcm_slope_rho_error = ax_slope_rho_error.pcolor(x_axis_log10, y_axis, delta_slope_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(delta_slope_rho_error_all), vcenter=np.median(np.ndarray.flatten(delta_slope_rho_error_all)), vmax=np.amax(delta_slope_rho_error_all)))
clb_slope_rho_error = plt.colorbar(pcm_slope_rho_error, ax=ax_slope_rho_error)
clb_slope_rho_error.set_label(label='Relative error of ' + r'$t_{\mathrm{slope}}$' + ' from simulated data', fontsize=9)
ax_slope_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
ax_slope_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
ax_slope_rho_error.xaxis.set_major_formatter(plot_utils.fake_log)




ax_rho.text(-0.1, 1.04, plot_utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_rho.transAxes)
ax_slope_rho.text(-0.1, 1.04, plot_utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_slope_rho.transAxes)

ax_rho_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_rho_error.transAxes)
ax_slope_rho_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_slope_rho_error.transAxes)


fig.text(0.2, 0.95, "Regional migration statistics", va='center', fontsize=25)



fig.subplots_adjust(wspace=0.4, hspace=0.3)
fig.savefig(utils.directory + "/figs/mean_relative_abundance_comparison_parent_heatmap.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
