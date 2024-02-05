from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma
import scipy.special as special
import utils
import pickle
import plot_utils




color_dict = {'Parent_migration.4.T18': utils.color_dict_range[('Parent_migration', 4)][13],
                'No_migration.4.T18': utils.color_dict_range[('No_migration',4)][13],
                'No_migration.40.T18': utils.color_dict_range[('No_migration',40)][13],
                'Global_migration.4.T18': utils.color_dict_range[('Global_migration',4)][13],
                'Parent_migration.NA.T0': 'k'}



count_dict = utils.get_otu_dict()


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
        mean_rel_abundance = sum(rel_abundance_list)/n_samples
        if asv not in mean_rel_abund_all_treatments_dict:
            mean_rel_abund_all_treatments_dict[asv] = {}
        mean_rel_abund_all_treatments_dict[asv][treatment] = mean_rel_abundance





mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
#mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

species_in_descendants = []


fig = plt.figure(figsize = (12, 12.5)) #
fig.subplots_adjust(bottom= 0.15)

transfers = [12,18]

ax_no_4 = plt.subplot2grid((2, 2), (0,0), colspan=1)
ax_no_40 = plt.subplot2grid((2, 2), (0,1), colspan=1)
ax_global = plt.subplot2grid((2, 2), (1,0), colspan=1)
ax_parent = plt.subplot2grid((2, 2), (1,1), colspan=1)


ax_no_4.text(-0.1, 1.04, plot_utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_no_4.transAxes)
ax_no_40.text(-0.1, 1.04, plot_utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_no_40.transAxes)
ax_global.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_global.transAxes)
ax_parent.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_parent.transAxes)

ax_all = [ax_no_4, ax_no_40, ax_global, ax_parent]


# exclude parent migration
migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]
for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula): #utils.migration_innocula:

    s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
    means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=True)
    #species_in_descendants.extend(ESVs)


    species_in_descendants = list(set(ESVs))

    mean_rel_abundances_parent_present = [mean_rel_abundances_parent[s_idx] for (s_idx,s) in enumerate(species_parent) if s in species_in_descendants]
    mean_rel_abundances_parent_absent = [mean_rel_abundances_parent[s_idx] for (s_idx,s) in enumerate(species_parent) if s not in species_in_descendants]
    mean_rel_abundances_parent_present_log10 = np.log10(mean_rel_abundances_parent_present)
    mean_rel_abundances_parent_absent_log10 = np.log10(mean_rel_abundances_parent_absent)


    ks_statistic, p_value = utils.run_permutation_ks_test(mean_rel_abundances_parent_present_log10, mean_rel_abundances_parent_absent_log10)

    #ks_statistic=0.023

    ax = ax_all[migration_innoculum_idx]

    color_ = utils.color_dict_range[migration_innoculum][-2]

    ax.hist(mean_rel_abundances_parent_present_log10, lw=3, alpha=0.8, bins= 12, color=color_, ls='-', histtype='step', density=True, label='Present in descendents')
    ax.hist(mean_rel_abundances_parent_absent_log10, lw=3, alpha=0.8, bins= 12, color=color_, ls=':', histtype='step', density=True, label='Absent in descendents')

    ax.set_xlabel('Relative abundance in ' + utils.titles_dict_no_caps[migration_innoculum]  + ', ' +  r'$\mathrm{log}_{10}$', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.legend(loc="upper right", fontsize=8)


    #ax.text(0.8,0.81, r'$D = {{{}}}$'.format(str(round(ks_statistic, 3))), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.8,0.81, r'$\mathrm{KS} = $' + str(round(ks_statistic, 3)), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes)
    
    ax.text(0.8,0.73, r'$P < 0.05$', fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes)





#fig_name = utils.directory + '/figs/abundance_hist_parent_vs_descendant.png'
fig.savefig(utils.directory + '/figs/abundance_hist_parent_vs_descendant.png', format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
fig.savefig(utils.directory + '/figs/abundance_hist_parent_vs_descendant.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
