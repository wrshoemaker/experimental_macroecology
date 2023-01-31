from __future__ import division
import os
import sys
import itertools
import random


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
import mle_utils

from matplotlib import cm

np.random.seed(123456789)

experiments = [('No_migration', 4), ('Global_migration', 4)]



cv_over_time_dict = {}
cv_over_time_dict['cv'] = {}
cv_over_time_dict['delta_cv'] = {}

for experiment_idx, experiment in enumerate(experiments):

    cv_over_time_dict['cv'][experiment] = {}
    cv_over_time_dict['delta_cv'][experiment] = {}

    communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
    communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]

    species_abundances_dict = {}
    species_relative_abundances_dict = {}
    for transfer in range(1, 18+1):

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities=communities_keep)
        N_reads = s_by_s.sum(axis=0)

        comm_rep_array = np.asarray(comm_rep_list)

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        for afd_idx, afd in enumerate(rel_s_by_s):

            species_i = species[afd_idx]
            if species_i not in species_relative_abundances_dict:
                species_abundances_dict[species_i] = {}
                species_relative_abundances_dict[species_i] = {}

            for comm_rep_list_i_idx, comm_rep_list_i in enumerate(comm_rep_list):
                if comm_rep_list_i not in species_relative_abundances_dict[species_i]:
                    species_abundances_dict[species_i][comm_rep_list_i] = {}
                    species_abundances_dict[species_i][comm_rep_list_i]['n_reads'] = {}
                    species_abundances_dict[species_i][comm_rep_list_i]['N_reads'] = {}

                    species_relative_abundances_dict[species_i][comm_rep_list_i] = {}


                #species_abundances_dict[species_i][comm_rep_list_i][transfer] = s_by_s[afd_idx,:][comm_rep_list_i_idx]
                species_abundances_dict[species_i][comm_rep_list_i]['n_reads'][transfer] = s_by_s[afd_idx,:][comm_rep_list_i_idx]
                species_abundances_dict[species_i][comm_rep_list_i]['N_reads'][transfer] = int(N_reads[comm_rep_list_i_idx])

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

            abundance = np.asarray([species_abundances_dict[species_i][community_j]['n_reads'][t] for t in transfers_t if t > 5])
            N_reads = np.asarray([species_abundances_dict[species_i][community_j]['N_reads'][t] for t in transfers_t if t > 5])

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

            if len(log_abundance_ratio_after_5) >= 5:

                cv = np.std(log_abundance_ratio_after_5)/np.absolute(np.mean(log_abundance_ratio_after_5))

                if species_i not in cv_over_time_dict['cv'][experiment]:
                    cv_over_time_dict['cv'][experiment][species_i] = {}
                    cv_over_time_dict['cv'][experiment][species_i]['observed'] = []
                    cv_over_time_dict['cv'][experiment][species_i]['gamma'] = []


                # fit model on abundances after transfer five
                mean_x = np.mean(abundance/N_reads)
                var_x = np.var(abundance/N_reads)
                beta_x = (mean_x**2)/var_x

                gamma_sampling_model = mle_utils.mle_gamma_sampling(N_reads, abundance)
                gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", disp = False, start_params=[mean_x,var_x], bounds= [(mean_x*0.1, 1), (var_x*0.01, 1)])
                #gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)
                gamma_mean, gamma_var = gamma_sampling_result.params
                gamma_beta = (gamma_mean**2)/gamma_var

                cv_log_ratio_gamma_all = []
                while len(cv_log_ratio_gamma_all) < 100:

                    # draw gamma
                    #s = np.random.gamma(shape, scale, 1000)
                    # scale = 1/rate
                    x_gamma = np.random.gamma(gamma_beta, gamma_mean/gamma_beta, len(N_reads))
                    #x_gamma = np.random.gamma(beta_x, mean_x/beta_x, len(N_reads))

                    # a few observations have relative abundances >1, set those to 1
                    x_gamma[x_gamma>1] = 1

                    # sample reads
                    n_gamma_sample = np.random.binomial(N_reads, x_gamma)
                    x_gamma_sample = n_gamma_sample/N_reads

                    # log ratio
                    ratio_x_gamma_sample = x_gamma_sample[1:] / x_gamma_sample[:-1]

                    # remove inf
                    idx_to_keep = ((~np.isinf(ratio_x_gamma_sample)) & (~np.isnan(ratio_x_gamma_sample)) & (ratio_x_gamma_sample>0))
                    ratio_x_gamma_sample = ratio_x_gamma_sample[idx_to_keep]

                    # same as criteria used to keep observed CV
                    if len(ratio_x_gamma_sample) >= 5:

                        log_ratio_x_gamma_sample = np.log(ratio_x_gamma_sample)
                        cv_log_ratio_gamma = np.std(log_ratio_x_gamma_sample)/np.absolute(np.mean(log_ratio_x_gamma_sample))
                        cv_log_ratio_gamma_all.append(cv_log_ratio_gamma)


                cv_log_ratio_gamma_all = np.asarray(cv_log_ratio_gamma_all)
                cv_log_ratio_gamma_all = np.sort(cv_log_ratio_gamma_all)


                cv_over_time_dict['cv'][experiment][species_i]['observed'].append(cv)
                cv_over_time_dict['cv'][experiment][species_i]['gamma'].append(np.mean(cv_log_ratio_gamma_all))


                if (len(log_abundance_ratio_before) >= 5) and (len(log_abundance_ratio_after) >= 5):
                    cv_before =  np.std(log_abundance_ratio_before)/np.absolute(np.mean(log_abundance_ratio_before))
                    cv_after = np.std(log_abundance_ratio_after)/np.absolute(np.mean(log_abundance_ratio_after))

                    delta_cv = cv_after - cv_before

                    if species_i not in cv_over_time_dict['delta_cv'][experiment]:
                        cv_over_time_dict['delta_cv'][experiment][species_i] = []

                    cv_over_time_dict['delta_cv'][experiment][species_i].append(delta_cv)


        #if species_i in cv_over_time_dict['delta_cv'][experiment]:
        #    print(experiment, np.mean(cv_over_time_dict['delta_cv'][experiment][species_i]))

# permute delta_cv
species_intersection = set(cv_over_time_dict['delta_cv'][experiments[0]].keys()).intersection(set(cv_over_time_dict['delta_cv'][experiments[1]].keys()))
iter = 1000
for s in species_intersection:

    cv_no = np.asarray(cv_over_time_dict['delta_cv'][experiments[0]][s])
    cv_global = np.asarray(cv_over_time_dict['delta_cv'][experiments[1]][s])

    if (len(cv_no) >= 5) and (len(cv_global) >= 5):

        ks, p_value = stats.ks_2samp(cv_global, cv_no)

        cv_merged = np.concatenate([cv_global, cv_no])

        ks_null_all = []
        for i in range(iter):
            np.random.shuffle(cv_merged)
            ks_null, p_value_null = stats.ks_2samp(cv_merged[:len(cv_global)], cv_merged[len(cv_global):])
            ks_null_all.append(ks_null)

        ks_null_all = np.asarray(ks_null_all)

        p_value_perm = sum(ks>ks_null_all)/iter

        #print(np.mean(cv_global), np.mean(cv_no), ks, p_value_perm)






#fig = plt.figure(figsize = (8, 4)) #
#fig.subplots_adjust(bottom= 0.15)

#for experiment_idx, experiment in enumerate(experiments):

#    ax = plt.subplot2grid((1, 2), (0, experiment_idx))

#    for species, cv_dict in cv_over_time_dict['cv'][experiment].items():

#        observed = cv_dict['observed']
#        gamma = cv_dict['gamma']

#        ax.scatter(observed, gamma, alpha=0.8, s=10)


#    #cv_over_time_dict['cv'][experiment][species_i]['observed'].append(cv)
#    ax.set_xscale('log', basex=10)
#    ax.set_yscale('log', basey=10)

#    ax.plot([0.01,500],[0.01,500], lw=3,ls='--',c='k',zorder=1)

#    ax.set_xlabel('Observed CV of log-ratio', fontsize=12)
#    ax.set_ylabel('Predicted CV of log-ratio', fontsize=12)
#    ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )



fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)



ax_1 = plt.subplot2grid((1, 2), (0, 0))
ax_2 = plt.subplot2grid((1, 2), (0, 1))

ax_all = [ax_1, ax_2]

species_to_plot = set(cv_over_time_dict['cv'][experiments[0]].keys()).intersection(set(cv_over_time_dict['cv'][experiments[1]].keys()))

for e_idx, e in enumerate(experiments):

    ax = ax_all[e_idx]

    cv_no_all = []
    cv_gamma_all = []

    for s in species_to_plot:

        cv_no = cv_over_time_dict['cv'][e][s]['observed']
        cv_gamma = cv_over_time_dict['cv'][e][s]['gamma']

        print(cv_no, cv_gamma)


        cv_no_all.extend(cv_no)
        cv_gamma_all.extend(cv_gamma)


        ax.scatter(np.mean(cv_no), np.mean(cv_global), alpha=0.8, s=10)


    #cv_over_time_dict['cv'][experiment][species_i]['observed'].append(cv)
    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)

    min_ = min(cv_no_all + cv_gamma_all)
    max_ = max(cv_no_all + cv_gamma_all)

    ax.plot([0.5*min_,2*max_],[0.5*min_,2*max_], lw=2, ls='--', c='k', zorder=1)

    ax.set_xlabel('Observed CV of log-ratio', fontsize=12)
    ax.set_ylabel('Predicted CV of log-ratio', fontsize=12)
    ax.set_title(utils.titles_dict[e], fontsize=12, fontweight='bold' )






fig.subplots_adjust(wspace=0.3)
fig.savefig(utils.directory + "/figs/cv_observed_vs_null.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
