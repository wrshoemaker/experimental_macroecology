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
        mean_rel_abundance = sum(rel_abundance_list)/n_samples
        if asv not in mean_rel_abund_all_treatments_dict:
            mean_rel_abund_all_treatments_dict[asv] = {}
        mean_rel_abund_all_treatments_dict[asv][treatment] = mean_rel_abundance



fig, ax = plt.subplots(figsize=(4,4))
ratios_all_12 = []
ratios_all_18 = []
for asv, mean_rel_abundance_dict in mean_rel_abund_all_treatments_dict.items():

    if ('Parent_migration.NA.T0' not in mean_rel_abundance_dict):
        continue

    # plot transfer 12
    parent = mean_rel_abundance_dict['Parent_migration.NA.T0']

    if ('No_migration.4.T12' in mean_rel_abundance_dict) and ('Parent_migration.4.T12' in mean_rel_abundance_dict):
        no_migration = mean_rel_abundance_dict['No_migration.4.T12']
        parent_migration = mean_rel_abundance_dict['Parent_migration.4.T12']

        ax.scatter(parent, parent_migration/no_migration, alpha=0.8, c='lightblue')

        ratios_all_12.append(parent_migration/no_migration)

    if ('No_migration.4.T18' in mean_rel_abundance_dict) and ('Parent_migration.4.T18' in mean_rel_abundance_dict):
        no_migration = mean_rel_abundance_dict['No_migration.4.T18']
        parent_migration = mean_rel_abundance_dict['Parent_migration.4.T18']

        ax.scatter(parent, parent_migration/no_migration, alpha=0.8, c='dodgerblue')

        ratios_all_18.append(parent_migration/no_migration)


ratios_all_12 = np.asarray(ratios_all_12)
ratios_all_18 = np.asarray(ratios_all_18)
print(sum(ratios_all_12>1)/len(ratios_all_12))
print(sum(ratios_all_18>1)/len(ratios_all_18))

ax.axhline(1, lw=1.5, ls=':',color='k', zorder=1)

ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)


custom_lines = [Line2D([0], [0], marker='o', color='w',  markerfacecolor='lightblue', markersize=12),
                Line2D([0], [0], marker='o', color='w',  markerfacecolor='dodgerblue', markersize=12)]
ax.legend(custom_lines, ['Transfer 12', 'Transfer 18'], loc="lower right")

ax.set_xlabel('Relative abundance in parent community', fontsize=12)
ax.set_ylabel('Ratio of mean relative abundance in\nparent and no migration treatments',  fontsize=12)



fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/mean_relative_abundance_ratio.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
