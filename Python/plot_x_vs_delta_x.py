from __future__ import division
import os
import sys
import itertools
import random


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm
import slm_simulation_utils
import plot_utils

np.random.seed(123456789)
random.seed(123456789)


experiments = [('No_migration', 4), ('Global_migration', 4)]


fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)


for experiment_idx, experiment in enumerate(experiments):


    communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
    communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]

    #ax_mean = plt.subplot2grid((3,4), (0, experiment_idx), colspan=1)
    #ax_cv = plt.subplot2grid((3,4), (0, 2+experiment_idx), colspan=1)

    #ax_mean.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx*3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mean.transAxes)
    #ax_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[6+experiment_idx*3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv.transAxes)

    species_relative_abundances_dict = {}
    for transfer in range(1, 18+1):

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities=communities_keep)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        for afd_idx, afd in enumerate(rel_s_by_s):

            species_i = species[afd_idx]
            if species_i not in species_relative_abundances_dict:
                species_relative_abundances_dict[species_i] = {}

            for comm_rep_list_i_idx, comm_rep_list_i in enumerate(comm_rep_list):
                if comm_rep_list_i not in species_relative_abundances_dict[species_i]:
                    species_relative_abundances_dict[species_i][comm_rep_list_i] = {}

                if afd[comm_rep_list_i_idx] > 0:

                    species_relative_abundances_dict[species_i][comm_rep_list_i][transfer] = afd[comm_rep_list_i_idx]


    ax_before = plt.subplot2grid((2,2), (experiment_idx, 0), colspan=1)
    ax_after = plt.subplot2grid((2,2), (experiment_idx, 1), colspan=1)

    species_all = list(species_relative_abundances_dict.keys())

    relative_abundance_before_all = []
    relative_abundance_after_all = []
    delta_relative_abundance_before_all = []
    delta_relative_abundance_after_all = []
    for species_i in species_all:

        log_abundance_ratio_dict = {}

        log_abundance_ratio_before_all = []
        log_abundance_ratio_after_all = []
        log_abundance_ratio_all = []
        transfers_all = []

        for community_j, community_dict in species_relative_abundances_dict[species_i].items():

            transfers_t = list(community_dict.keys())
            transfers_t.sort()

            tuples_t = list(zip(transfers_t[:-1], transfers_t[1:]))

            if len(tuples_t) == 0:
                continue

            tuples_t_filter = [t for t in tuples_t if ((t[0]+1) == t[1]) and (t[0] >= 6) ]

            if len(tuples_t_filter) == 0:
                continue
            
            t_filter = np.asarray([t[0] for t in tuples_t_filter])
            relative_abundance = np.asarray([community_dict[t[1]] for t in tuples_t_filter])
            delta_relative_abundance = np.absolute([community_dict[t[1]] - community_dict[t[0]] for t in tuples_t_filter])
            
            relative_abundance_before = relative_abundance[t_filter < 12]
            relative_abundance_after = relative_abundance[t_filter >= 12]

            delta_relative_abundance_before = delta_relative_abundance[t_filter < 12]
            delta_relative_abundance_after = delta_relative_abundance[t_filter >= 12]

            colors = np.asarray([utils.color_dict_range[experiment][t] for t in t_filter])
            colors_before = colors[t_filter < 12]
            colors_after = colors[t_filter >= 12]

            if len(relative_abundance_before) > 0:
                ax_before.scatter(relative_abundance_before, delta_relative_abundance_before, c=colors_before, alpha=0.8, edgecolors='k', zorder=2)  # , c='#87CEEB')
                
                relative_abundance_before_all.append(relative_abundance_before)
                delta_relative_abundance_before_all.append(delta_relative_abundance_before)

            if len(relative_abundance_after) > 0:
                ax_after.scatter(relative_abundance_after, delta_relative_abundance_after, c=colors_after, alpha=0.8, edgecolors='k', zorder=2)  # , c='#87CEEB')
                
                relative_abundance_after_all.append(relative_abundance_after)
                delta_relative_abundance_after_all.append(delta_relative_abundance_after)

        
   


    ax_before.set_xscale('log', basex=10)
    ax_before.set_yscale('log', basey=10)

    ax_after.set_xscale('log', basex=10)
    ax_after.set_yscale('log', basey=10)


    relative_abundance_before_all = np.concatenate(relative_abundance_before_all).ravel()
    delta_relative_abundance_before_all = np.concatenate(delta_relative_abundance_before_all).ravel()

    relative_abundance_after_all = np.concatenate(relative_abundance_after_all).ravel()
    delta_relative_abundance_after_all = np.concatenate(delta_relative_abundance_after_all).ravel()

    slope_before, intercept_before, r_value_before, p_value_before, std_err_before = stats.linregress(np.log10(relative_abundance_before_all), np.log10(delta_relative_abundance_before_all))
    slope_after, intercept_after, r_value_after, p_value_after, std_err_after = stats.linregress(np.log10(relative_abundance_after_all), np.log10(delta_relative_abundance_after_all))

    print(experiment, slope_before, slope_after)


fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/delta_relative_abundance.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()


