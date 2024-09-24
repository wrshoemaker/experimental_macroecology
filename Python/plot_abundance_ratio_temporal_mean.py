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

    mean_mean_dict = {}

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

            mean_mean_dict[t_]['species'].append(species_i)
            mean_mean_dict[t_]['measure'].append(mean_log_abundance_ratio[t_idx])


        mean_log_abundance_ratio = np.asarray(mean_log_abundance_ratio)
        ax_mean.plot(trasnfers_ratio, mean_log_abundance_ratio, alpha=0.6, c=utils.color_dict_range[experiment][7], zorder=2)



    transfers_mean_mean = list(mean_mean_dict.keys())
    transfers_mean_mean.sort()

    mean_mean_to_plot = np.asarray([np.mean(mean_mean_dict[t]['measure']) for t in transfers_mean_mean])

    # ks test for the distribution of CVs over time for all species before/after end of manipulation
    mean_over_all_species_t_before = [mean_mean_dict[t]['measure'] for t in transfers_mean_mean if (t <= 12) and (t > 7)]
    mean_over_all_species_t_after = [mean_mean_dict[t]['measure'] for t in transfers_mean_mean if t > 12]

    # t-test per treatment

    asv_mean_dict = {}
    for t_ in mean_mean_dict.keys():

        if t_ <= 6:
            continue

        for asv_idx, asv in enumerate(mean_mean_dict[t_]['species']):

            if asv not in asv_mean_dict:
                asv_mean_dict[asv] = {}
                asv_mean_dict[asv]['before'] = []
                asv_mean_dict[asv]['after'] = []

            if (t_ <= 12) and (t_ > 7):
                asv_mean_dict[asv]['before'].append(mean_mean_dict[t_]['measure'][asv_idx])

            elif (t_ > 12):
                asv_mean_dict[asv]['after'].append(mean_mean_dict[t_]['measure'][asv_idx])

            else:
                continue



    asv_keys = list(asv_mean_dict.keys())
    for asv in asv_keys:

        if (len(asv_mean_dict[asv]['before']) < 5) or (len(asv_mean_dict[asv]['after']) < 5):
            del asv_mean_dict[asv]

    #for asv in asv_mean_dict.keys():

    #    before_asv = asv_mean_dict[asv]['before']
    #    after_asv = asv_mean_dict[asv]['after']

    #    print(len(before_asv), len(after_asv))

    #    t_test_all.append()


    mean_t_test = np.mean([stats.ttest_ind(asv_mean_dict[asv]['after'], asv_mean_dict[asv]['before'], equal_var=True)[0] for asv in asv_mean_dict.keys()])

    mean_t_test_null_all = []
    for n in range(n_iter):

        t_test_null_all = []
        for asv in asv_mean_dict.keys():

            before_asv = asv_mean_dict[asv]['before']
            after_asv = asv_mean_dict[asv]['after']

            merged_asv = before_asv + after_asv
            random.shuffle(merged_asv)

            t_test_null_all.append(stats.ttest_ind(merged_asv[:len(after_asv)], merged_asv[len(after_asv):], equal_var=True)[0])

        mean_t_test_null_all.append(np.mean(t_test_null_all))

    mean_t_test_null_all = np.asarray(mean_t_test_null_all)

    p_value_mean_t_test = sum(mean_t_test > np.absolute(mean_t_test_null_all))/n_iter

    print(experiment, mean_t_test, p_value_mean_t_test)


    mean_over_all_species_t_before = np.asarray(list(itertools.chain(*mean_over_all_species_t_before)))
    mean_over_all_species_t_after = np.asarray(list(itertools.chain(*mean_over_all_species_t_after)))

    ax_mean.plot(transfers_mean_mean, mean_mean_to_plot, alpha=1, c=utils.color_dict_range[experiment][13], zorder=3)
    ax_mean.set_xlabel('Transfer, ' + r'$k$', fontsize=12)
    ax_mean.set_ylabel('Mean relative abundance ratio, ' + r'$\left<  \Delta l^{(k)} \right>$', fontsize=11)
    ax_mean.set_title(utils.titles_no_inocula_dict[experiment], fontsize=13)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #ax.set_ylim([-2.5, 2.5])
    ax_mean.set_xlim([1, 17])
    #ax_mean.set_ylim([10**(-2.5), 10**2.5])
    ax_mean.set_ylim([(-2.5), 2.5])

    #ax_mean.set_yscale('log', basey=10)

    legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                        Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean of ASVs'),
                        Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration'),
                        Line2D([0], [0], color='k', ls='--', lw=1.5, label='Stationarity')]

    if experiment_idx == 0:
        ax_mean.legend(handles=legend_elements, fontsize=9, loc='upper left')

    ax_mean.axhline(y=0, color='k', linestyle='--', lw = 3, zorder=1)

    if experiment_idx == 1:
        ax_mean.axvline(x=12, color='k', linestyle=':', lw = 3, zorder=1)

    
    ax_mean.text(0.2 , 0.25, r'$\bar{t} = $' + str(round(mean_t_test, 3)), fontsize=10, ha='center', va='center', transform=ax_mean.transAxes)
    ax_mean.text(0.2, 0.15, r'$P = $' + str(round(p_value_mean_t_test, 3)), fontsize=10, ha='center', va='center', transform=ax_mean.transAxes)






fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfer_mean.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
#fig.savefig(utils.directory + '/figs/abundance_ratio_per_transfer_mean.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
