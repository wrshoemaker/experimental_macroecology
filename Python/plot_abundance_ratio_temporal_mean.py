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

n_iter = 1000

fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)


ks_test_dict = {}

for experiment_idx, experiment in enumerate(experiments):

    delta_cv_all = []

    cv_before_all = []
    cv_after_all = []

    mean_mean_dict = {}
    mean_cv_dict = {}

    cv_delta_null_dict = {}

    cv_over_time_dict = {}
    cv_over_time_dict['cv_delta'] = {}
    cv_over_time_dict['cv'] = {}

    communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
    communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]

    ax_mean = plt.subplot2grid((1,2), (0, experiment_idx), colspan=1)

    ax_mean.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mean.transAxes)
    


    species_relative_abundances_dict = {}
    for transfer in range(1, 18+1):

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities=communities_keep)

        comm_rep_array = np.asarray(comm_rep_list)

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        for afd_idx, afd in enumerate(rel_s_by_s):

            species_i = species[afd_idx]
            if species_i not in species_relative_abundances_dict:
                species_relative_abundances_dict[species_i] = {}

            for comm_rep_list_i_idx, comm_rep_list_i in enumerate(comm_rep_list):
                if comm_rep_list_i not in species_relative_abundances_dict[species_i]:
                    species_relative_abundances_dict[species_i][comm_rep_list_i] = {}

                species_relative_abundances_dict[species_i][comm_rep_list_i][transfer] = afd[comm_rep_list_i_idx]


    species_all = list(species_relative_abundances_dict.keys())
    for species_i in species_all:

        log_abundance_ratio_dict = {}

        log_abundance_ratio_before_all = []
        log_abundance_ratio_after_all = []
        log_abundance_ratio_all = []
        transfers_all = []

        for community_j, community_dict in species_relative_abundances_dict[species_i].items():

            transfers_t = list(species_relative_abundances_dict[species_i][community_j].keys())
            transfers_t.sort()

            tuples_t = list(zip(transfers_t[:-1], transfers_t[1:]))

            if len(tuples_t) == 0:
                continue

            tuples_t_filter = [t for t in tuples_t if (t[0]+1) == t[1]]

            if len(tuples_t_filter) < 10:
                continue

            abundance_ratio = [species_relative_abundances_dict[species_i][community_j][t[1]]/species_relative_abundances_dict[species_i][community_j][t[0]] for t in tuples_t_filter]

            # remove nan and zero
            tuples_t_filter = np.asarray(tuples_t_filter)
            abundance_ratio = np.asarray(abundance_ratio)

            idx_to_keep = ((~np.isinf(abundance_ratio)) & (~np.isnan(abundance_ratio)) & (abundance_ratio>0))

            if sum(idx_to_keep) < 10:
                continue

            tuples_t_filter = tuples_t_filter[idx_to_keep]
            abundance_ratio = abundance_ratio[idx_to_keep]

            tuples_t_filter_first_timepoints = [t_[0] for t_ in tuples_t_filter]
            tuples_t_filter_first_timepoints = np.asarray(tuples_t_filter_first_timepoints)

            log_abundance_ratio = np.log10(abundance_ratio)

            log_abundance_ratio_after_5 = log_abundance_ratio[tuples_t_filter_first_timepoints>5]

            tuples_t_filter_first_timepoints_after_5 = tuples_t_filter_first_timepoints[tuples_t_filter_first_timepoints>5]

            log_abundance_ratio_before = log_abundance_ratio_after_5[(tuples_t_filter_first_timepoints_after_5<12)]
            log_abundance_ratio_after = log_abundance_ratio_after_5[(tuples_t_filter_first_timepoints_after_5>=12)]

            log_abundance_ratio_before_all.append(log_abundance_ratio_before)
            log_abundance_ratio_after_all.append(log_abundance_ratio_after)

            log_abundance_ratio_all.append(log_abundance_ratio_after_5)
            transfers_all.append(tuples_t_filter_first_timepoints_after_5)

            for t_idx, t in enumerate(tuples_t_filter_first_timepoints):
                if t not in log_abundance_ratio_dict:
                    log_abundance_ratio_dict[t] = []

                log_abundance_ratio_dict[t].append(log_abundance_ratio[t_idx])


        if (len(log_abundance_ratio_before_all)>0) and (len(log_abundance_ratio_after_all)>0):

            log_abundance_ratio_before_flat = np.concatenate(log_abundance_ratio_before_all).ravel()
            log_abundance_ratio_after_flat = np.concatenate(log_abundance_ratio_after_all).ravel()

            log_abundance_ratio_flat = np.concatenate(log_abundance_ratio_all).ravel()
            transfers_flat = np.concatenate(transfers_all).ravel()

            if (len(log_abundance_ratio_before_flat) >= 10) and (len(log_abundance_ratio_after_flat) >= 10):

                cv_before =  np.std(log_abundance_ratio_before_flat)/np.absolute(np.mean(log_abundance_ratio_before_flat))
                cv_after = np.std(log_abundance_ratio_after_flat)/np.absolute(np.mean(log_abundance_ratio_after_flat))
                delta_cv = cv_after - cv_before

                cv_before_all.append(cv_before)
                cv_after_all.append(cv_after)
                delta_cv_all.append(delta_cv)


                # get null CV  before/after
                if species_i not in cv_delta_null_dict:
                    cv_delta_null_dict[species_i] = []

  
        trasnfers_ratio = list(log_abundance_ratio_dict.keys())
        trasnfers_ratio.sort()

        # pools observations across species
        mean_log_abundance_ratio = [np.mean(log_abundance_ratio_dict[t]) for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]
        cv_log_abundance_ratio = [np.std(log_abundance_ratio_dict[t])/np.absolute(np.mean(log_abundance_ratio_dict[t])) for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]
        trasnfers_ratio = [t for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]


        if len(cv_log_abundance_ratio) < 5:
            continue

        for t_idx, t_ in enumerate(trasnfers_ratio):

            if t_ not in mean_mean_dict:
                mean_mean_dict[t_] = {}
                mean_mean_dict[t_]['measure'] = []
                mean_mean_dict[t_]['species'] = []

            if t_ not in mean_cv_dict:
                mean_cv_dict[t_] = {}
                mean_cv_dict[t_]['measure'] = []
                mean_cv_dict[t_]['species'] = []

            mean_mean_dict[t_]['species'].append(species_i)
            mean_cv_dict[t_]['species'].append(species_i)

            mean_mean_dict[t_]['measure'].append(mean_log_abundance_ratio[t_idx])
            mean_cv_dict[t_]['measure'].append(cv_log_abundance_ratio[t_idx])



        mean_log_abundance_ratio = np.asarray(mean_log_abundance_ratio)

        ax_mean.plot(trasnfers_ratio, mean_log_abundance_ratio, alpha=0.6, c=utils.color_dict_range[experiment][7], zorder=2)


    # ks test for the distribution of CVs over *species* before/after manipulation.
    # Each CV is over all replicates and all transfers before/after manipulation for a given species

    cv_before_all = np.asarray(cv_before_all)
    cv_after_all = np.asarray(cv_after_all)
    delta_cv_all = np.asarray(delta_cv_all)
    mean_delta_cv = np.mean(delta_cv_all)

    # get null distribution
    #cv_delta_null_dict_keys = list(cv_delta_null_dict.keys())
    #mean_delta_cv_null = []
    #for i in range(10000):
    #    mean_delta_cv_null.append(np.mean([cv_delta_null_dict[s][i] for s in cv_delta_null_dict_keys]))

    #mean_delta_cv_null = np.asarray(mean_delta_cv_null)
    #p_mean_delta_cv = sum(mean_delta_cv>mean_delta_cv_null)/10000
    #print(mean_delta_cv, p_mean_delta_cv)


    #cv_before_after_all = np.column_stack((cv_before_all, cv_after_all))
    #mean_delta_cv = np.mean(delta_cv_all)
    #mean_delta_cv_null_all = []
    #for i in range(1000):

    #    cv_before_after_all_null = np.apply_along_axis(np.random.permutation, axis=1, arr=cv_before_after_all)
    #    mean_delta_cv_null = cv_before_after_all_null[:,1] - cv_before_after_all_null[:,0]
    #    mean_delta_cv_null_all.append(np.mean(mean_delta_cv_null))

    #mean_delta_cv_null_all = np.asarray(mean_delta_cv_null_all)
    #p_mean_delta_cv = sum(mean_delta_cv_null_all<mean_delta_cv)/1000
    #print(mean_delta_cv, p_mean_delta_cv)



    transfers_mean_mean = list(mean_mean_dict.keys())
    transfers_mean_mean.sort()

    transfers_mean_cv = list(mean_cv_dict.keys())
    transfers_mean_cv.sort()

    mean_mean_to_plot = [np.mean(mean_mean_dict[t]['measure']) for t in transfers_mean_mean]
    cv_mean_to_plot = [np.mean(np.log10(mean_cv_dict[t]['measure'])) for t in transfers_mean_cv]

    mean_mean_to_plot = np.asarray(mean_mean_to_plot)
    cv_mean_to_plot = np.asarray(cv_mean_to_plot)

    # ks test for the distribution of CVs over time for all species before/after end of manipulation
    mean_over_all_species_t_before = [mean_mean_dict[t]['measure'] for t in transfers_mean_mean if (t <= 12) and (t > 7)]
    mean_over_all_species_t_after = [mean_mean_dict[t]['measure'] for t in transfers_mean_mean if t > 12]

    mean_over_all_species_t_before = np.asarray(list(itertools.chain(*mean_over_all_species_t_before)))
    mean_over_all_species_t_after = np.asarray(list(itertools.chain(*mean_over_all_species_t_after)))

    cv_over_all_species_t_before = [mean_cv_dict[t]['measure'] for t in transfers_mean_cv if (t <= 12) and (t > 5)]
    cv_over_all_species_t_after = [mean_cv_dict[t]['measure'] for t in transfers_mean_cv if (t > 12) and (t > 5)]

    species_over_all_species_t_before = [mean_cv_dict[t]['species'] for t in transfers_mean_cv if (t <= 12) and (t > 5)]
    species_over_all_species_t_after = [mean_cv_dict[t]['species'] for t in transfers_mean_cv if (t > 12) and (t > 5)]

    cv_over_all_species_t_before = np.asarray(list(itertools.chain(*cv_over_all_species_t_before)))
    cv_over_all_species_t_after = np.asarray(list(itertools.chain(*cv_over_all_species_t_after)))

    #print(len(cv_over_all_species_t_before), len(cv_over_all_species_t_after))


    ax_mean.plot(transfers_mean_mean, mean_mean_to_plot, alpha=1, c=utils.color_dict_range[experiment][13], zorder=3)

    ax_mean.set_xlabel('Transfer, ' + r'$t$', fontsize=12)
    ax_mean.set_ylabel('Mean relative abundance ratio, ' + r'$\left<  \Delta l \right>$', fontsize=11)
    ax_mean.set_title(utils.titles_no_inocula_dict[experiment], fontsize=13)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #ax.set_ylim([-2.5, 2.5])
    ax_mean.set_xlim([1, 17])
    #ax_mean.set_ylim([10**(-2.5), 10**2.5])
    ax_mean.set_ylim([(-2.5), 2.5])

    #ax_mean.set_yscale('log', basey=10)

    legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                        Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean of ASVs'),
                        Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration')]

    if experiment_idx == 0:
        ax_mean.legend(handles=legend_elements, fontsize=9, loc='upper left')

    #if experiment_idx == 1:
    ax_mean.axvline(x=12, color='k', linestyle=':', lw = 3, zorder=1)






fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfe_mean.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
