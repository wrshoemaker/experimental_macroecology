from __future__ import division
import os
import sys
import itertools
import random
import pickle


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

import plot_abundance_ratio_temporal_cv

np.random.seed(123456789)
random.seed(123456789)


experiments = [('No_migration', 4), ('Global_migration', 4)]


n_iter = 1000



def make_old_dict():

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

                    for i in range(10000):

                        np.random.shuffle(log_abundance_ratio_flat)

                        log_abundance_ratio_before_null = log_abundance_ratio_flat[(transfers_flat<12)]
                        log_abundance_ratio_after_null = log_abundance_ratio_flat[(transfers_flat>=12)]

                        cv_before_null =  np.std(log_abundance_ratio_before_null)/np.absolute(np.mean(log_abundance_ratio_before_null))
                        cv_after_null = np.std(log_abundance_ratio_after_null)/np.absolute(np.mean(log_abundance_ratio_after_null))
                        delta_cv_null = cv_after_null - cv_before_null

                        cv_delta_null_dict[species_i].append(delta_cv_null)



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

        ks_statistic_mean_over_all_species_t, p_value_mean_over_all_species_t = utils.run_permutation_ks_test(mean_over_all_species_t_before, mean_over_all_species_t_after)

        cv_over_all_species_t_before = [mean_cv_dict[t]['measure'] for t in transfers_mean_cv if (t <= 12) and (t > 5)]
        cv_over_all_species_t_after = [mean_cv_dict[t]['measure'] for t in transfers_mean_cv if (t > 12) and (t > 5)]

        species_over_all_species_t_before = [mean_cv_dict[t]['species'] for t in transfers_mean_cv if (t <= 12) and (t > 5)]
        species_over_all_species_t_after = [mean_cv_dict[t]['species'] for t in transfers_mean_cv if (t > 12) and (t > 5)]

        cv_over_all_species_t_before = np.asarray(list(itertools.chain(*cv_over_all_species_t_before)))
        cv_over_all_species_t_after = np.asarray(list(itertools.chain(*cv_over_all_species_t_after)))

        #print(len(cv_over_all_species_t_before), len(cv_over_all_species_t_after))

        ks_statistic_cv_over_all_species_t, p_value_cv_over_all_species_t = utils.run_permutation_ks_test(cv_over_all_species_t_before, cv_over_all_species_t_after, n=2)
        

        treatment_str = experiment[0].lower()
        ks_test_dict[treatment_str] = {}
        ks_test_dict[treatment_str]['mean'] = ks_statistic_mean_over_all_species_t
        ks_test_dict[treatment_str]['cv'] = ks_statistic_cv_over_all_species_t

        print('Mean ', experiment[0], ks_statistic_mean_over_all_species_t, p_value_mean_over_all_species_t)
        print('CV ', experiment[0], ks_statistic_cv_over_all_species_t, p_value_cv_over_all_species_t)

        def ks_test_constrain_species(iter=10000):

            # try permuting tranfer labels constrained on species identity
            species_cv_dict = {}
            for t in transfers_mean_cv:

                if t <= 5:
                    continue

                for s_idx, s in enumerate(mean_cv_dict[t]['species']):

                    if s not in species_cv_dict:
                        species_cv_dict[s] = {}
                        species_cv_dict[s]['measure'] = []
                        species_cv_dict[s]['transfer'] = []

                    #np.append(species_cv_dict[s]['measure'], mean_cv_dict[t]['measure'][s_idx])
                    #np.append(species_cv_dict[s]['transfer'], t)

                    species_cv_dict[s]['measure'].append(mean_cv_dict[t]['measure'][s_idx])
                    species_cv_dict[s]['transfer'].append(t)


            for s in species_cv_dict.keys():

                transfer = np.asarray(species_cv_dict[s]['transfer'])
                n_geq_12 = sum(transfer<=12)
                species_cv_dict[s]['n_leq_12'] = n_geq_12


            ks_statistic_cv_over_all_species_t_null_all = []
            for i in range(n_iter):

                cv_over_all_species_t_before_null = []
                cv_over_all_species_t_after_null = []

                for s in species_cv_dict.keys():

                    measure = species_cv_dict[s]['measure']
                    random.shuffle(measure)
                    cv_over_all_species_t_before_null.extend(measure[:species_cv_dict[s]['n_leq_12']])
                    cv_over_all_species_t_after_null.extend(measure[species_cv_dict[s]['n_leq_12']:])

                cv_over_all_species_t_before_null = np.asarray(cv_over_all_species_t_before_null)
                cv_over_all_species_t_after_null = np.asarray(cv_over_all_species_t_after_null)

                #ks_statistic_cv_over_all_species_t_null, p_value_cv_over_all_species_t_null = utils.run_permutation_ks_test(cv_over_all_species_t_before_null, cv_over_all_species_t_after_null)
                ks_statistic_cv_over_all_species_t_null, p_value_cv_over_all_species_t_null = stats.ks_2samp(cv_over_all_species_t_before_null, cv_over_all_species_t_after_null)
                ks_statistic_cv_over_all_species_t_null_all.append(ks_statistic_cv_over_all_species_t_null)


            ks_statistic_cv_over_all_species_t_null_all = np.asarray(ks_statistic_cv_over_all_species_t_null_all)

            p_perm = sum(ks_statistic_cv_over_all_species_t_null_all > ks_statistic_cv_over_all_species_t)/n_iter

            return ks_statistic_cv_over_all_species_t, p_perm

        ks_statistic_cv_over_all_species_t, p_perm = ks_test_constrain_species()


        # no migration
        #0.1 0.775
        # global migration
        #0.2845138055222089 0.009



fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)


# plot simulation results
simulation_global_rho_dict = slm_simulation_utils.load_simulation_global_rho_dict()

#print(simulation_global_rho_dict.keys())

tau_all = np.asarray(list(simulation_global_rho_dict.keys()))
sigma_all = np.asarray(list(simulation_global_rho_dict[tau_all[0]].keys()))

np.sort(tau_all)
np.sort(sigma_all)

with open(plot_abundance_ratio_temporal_cv.abundance_ratio_temporal_cv_dict_path, 'rb') as handle:
        delta_cv_dict = pickle.load(handle)

#for treatment_idx, treatment in enumerate(['no_migration', 'global_migration']):


clb_ticks = [[-0.05, 0, 0.05, 0.15, 0.2, 0.25], [0, 0.5, 1, 1.5, 2, 2.5]]

for experiment_idx, experiment in enumerate(experiments):

    treatment = utils.migration_status_dict[experiment]

    tau_delta_mean_all = []
    tau_delta_cv_all = []
    #tau_delta_mean_error_all = []
    tau_delta_cv_error_all = []

    #observed_ks_mean = ks_test_dict[treatment]['mean']
    observed_ks_cv = delta_cv_dict[experiment]['t_stat']['mean_t_stat']

    for tau in tau_all:

       #tau_delta_mean = []
        tau_delta_cv = []

        #tau_delta_mean_error = []
        tau_delta_cv_error = []

        for sigma in sigma_all:

                #ks_mean = np.asarray(simulation_global_rho_dict[tau][sigma]['ratio_stats'][treatment]['ks_mean'])
                ks_cv = np.asarray(simulation_global_rho_dict[tau][sigma]['ratio_stats'][treatment]['mean_t_stat'])

                ks_cv = ks_cv[~np.isnan(ks_cv)]

                #tau_delta_mean.append(np.mean(ks_mean))
                tau_delta_cv.append(np.mean(ks_cv))

                #mean_error_ks_mean = np.mean(np.absolute((ks_mean - observed_ks_mean)/observed_ks_mean))
                mean_error_ks_cv = np.mean(np.absolute((ks_cv - observed_ks_cv)/observed_ks_cv ))

                #tau_delta_mean_error.append(mean_error_ks_mean)
                tau_delta_cv_error.append(mean_error_ks_cv)

        
        #tau_delta_mean_all.append(tau_delta_mean)
        tau_delta_cv_all.append(tau_delta_cv)

        #tau_delta_mean_error_all.append(tau_delta_mean_error)
        tau_delta_cv_error_all.append(tau_delta_cv_error)


    #tau_delta_mean_all = np.asarray(tau_delta_mean_all)
    tau_delta_cv_all = np.asarray(tau_delta_cv_all)

    #tau_delta_mean_error_all = np.asarray(tau_delta_mean_error_all)
    tau_delta_cv_error_all = np.asarray(tau_delta_cv_error_all)

    

    ax_simulation_cv = plt.subplot2grid((2, 2), (0, experiment_idx), colspan=1)
    ax_simulation_cv_error = plt.subplot2grid((2, 2), (1, experiment_idx), colspan=1)

    x_axis = sigma_all
    y_axis = tau_all

    x_axis_log10 = np.log10(x_axis)

    tau_delta_cv_all_flat = tau_delta_cv_all.flatten()
    tau_delta_cv_all_flat = tau_delta_cv_all_flat[~np.isnan(tau_delta_cv_all_flat)]

    tau_delta_cv_error_all_flat = tau_delta_cv_error_all.flatten()
    tau_delta_cv_error_all_flat = tau_delta_cv_error_all_flat[~np.isnan(tau_delta_cv_error_all_flat)]


    # ax_simulation_cv
    print(tau_delta_cv_all)
    #delta_range = max([observed_ks_cv - np.amin(tau_delta_cv_all),  np.amax(tau_delta_cv_all) - observed_ks_cv])
    delta_range = max([observed_ks_cv - np.min(tau_delta_cv_all_flat),  np.max(tau_delta_cv_all_flat) - observed_ks_cv])
    pcm_slope_rho = ax_simulation_cv.pcolor(x_axis_log10, y_axis, tau_delta_cv_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=observed_ks_cv - delta_range, vcenter=observed_ks_cv, vmax=observed_ks_cv + delta_range))
    clb_slope_rho = plt.colorbar(pcm_slope_rho, ax=ax_simulation_cv)

    #clb_slope_rho.set_bad(color='K')

    #clb_slope_rho.set_label(label='Simulated KS distance', fontsize=9)
    clb_slope_rho.set_label(label='Simulated ' + r'$\bar{t}$', fontsize=9)
    ax_simulation_cv.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
    ax_simulation_cv.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
    ax_simulation_cv.xaxis.set_major_formatter(plot_utils.fake_log)
    # Set observed marking and label
    
    clb_slope_rho.ax.axhline(y=observed_ks_cv, c='k')
    
    clb_slope_rho.set_ticks(clb_ticks[experiment_idx])
    clb_slope_rho.set_ticklabels([str(l) for l in clb_ticks[experiment_idx]])
    
    original_ticks = list(clb_slope_rho.get_ticks())
    clb_slope_rho.set_ticks(original_ticks + [observed_ks_cv])
    clb_slope_rho.set_ticklabels(original_ticks + ['Obs.'])
    ax_simulation_cv.set_title(utils.titles_str_no_inocula_dict[treatment], fontsize=14)

    # plot error
    pcm_slope_rho_error = ax_simulation_cv_error.pcolor(x_axis_log10, y_axis, tau_delta_cv_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.min(tau_delta_cv_error_all_flat), vcenter=np.median(tau_delta_cv_error_all_flat), vmax=np.max(tau_delta_cv_error_all_flat)))
    clb_slope_rho_error = plt.colorbar(pcm_slope_rho_error, ax=ax_simulation_cv_error)
    #clb_slope_rho_error.set_label(label='Relative error of ' + r'$D$' + ' from simulated data', fontsize=9)
    clb_slope_rho_error.set_label(label='Relative error of ' + r'$\mathrm{KS}$' + ' from simulated data', fontsize=9)
    ax_simulation_cv_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
    ax_simulation_cv_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
    ax_simulation_cv_error.xaxis.set_major_formatter(plot_utils.fake_log)

    
    ax_simulation_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_cv.transAxes)
    ax_simulation_cv_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx + 2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_simulation_cv_error.transAxes)


fig.text(0.16, 0.94, "Global migration simulation statistics", va='center', fontsize=20)



fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfer_cv_heatmap.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
#fig.savefig(utils.directory + '/figs/abundance_ratio_per_transfer_cv_heatmap.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
