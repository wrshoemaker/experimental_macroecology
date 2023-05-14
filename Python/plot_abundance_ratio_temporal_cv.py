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

fig = plt.figure(figsize = (8.5, 8)) #
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

    ax_cv = plt.subplot2grid((2, 2), (0, experiment_idx), colspan=1)
    ax_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv.transAxes)

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

        ax_cv.plot(trasnfers_ratio, cv_log_abundance_ratio, alpha=0.6, c=utils.color_dict_range[experiment][7], zorder=2)


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

    print('KS test constrained on ASV identity')
    print(experiment[0], ks_statistic_cv_over_all_species_t, p_perm)

    ax_cv.text(0.2  , 0.9, r'$\mathrm{KS} = $' + str(round(ks_statistic_cv_over_all_species_t, 3)), fontsize=10, ha='center', va='center', transform=ax_cv.transAxes)
    ax_cv.text(0.2, 0.8, r'$P = $' + str(round(p_perm, 3)), fontsize=10, ha='center', va='center', transform=ax_cv.transAxes)


    # no migration
    #0.1 0.775
    # global migration
    #0.2845138055222089 0.009


    ax_cv.plot(transfers_mean_cv, 10**cv_mean_to_plot, alpha=1, c=utils.color_dict_range[experiment][13], zorder=3)

    ax_cv.set_xlabel('Transfer, ' + r'$t$', fontsize=12)
    ax_cv.set_ylabel('CV of relative abundance ratio, ' + r'$\mathrm{CV}_{\Delta l}$', fontsize=11)
    ax_cv.set_title(utils.titles_no_inocula_dict[experiment], fontsize=13)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #ax.set_ylim([-2.5, 2.5])
    ax_cv.set_xlim([1, 17])
    ax_cv.set_ylim([0.04, 700])
    ax_cv.set_yscale('log', basey=10)


    legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                        Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean of ASVs'),
                        Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration')]

    if experiment_idx == 0:
        ax_cv.legend(handles=legend_elements, fontsize=9, loc='lower right')

    ax_cv.axvline(x=12, color='k', linestyle=':', lw = 3, zorder=1)




########################
# plot the simulations #
########################

# identify parameter regime with lowest error
simulation_global_rho_abc_dict = slm_simulation_utils.load_simulation_global_rho_abc_dict()

tau_all = np.asarray(simulation_global_rho_abc_dict['tau_all'])
sigma_all = np.asarray(simulation_global_rho_abc_dict['sigma_all'])



ks_cv_no_migration = np.asarray(simulation_global_rho_abc_dict['ratio_stats']['no_migration']['ks_cv'])
ks_cv_global_migration = np.asarray(simulation_global_rho_abc_dict['ratio_stats']['global_migration']['ks_cv'])


obs = np.asarray([0.1, 0.2845138055222089])
pred = np.asarray([ks_cv_no_migration, ks_cv_global_migration])

best_tau, best_sigma = utils.weighted_euclidean_distance(tau_all, sigma_all, obs, pred)

# run simulations
sys.stderr.write("Running simulation with optimal parameters...\n")
#slm_simulation_utils.run_simulation_global_rho_fixed_parameters(best_tau, best_sigma, n_iter=1000)
sys.stderr.write("Done!\n")


simulation_global_rho_fixed_parameters_dict = slm_simulation_utils.load_simulation_global_rho_fixed_parameters_dict()

ks_cv_simulation_no_migration = simulation_global_rho_fixed_parameters_dict['ratio_stats']['no_migration']['ks_cv']
ks_cv_simulation_global_migration = simulation_global_rho_fixed_parameters_dict['ratio_stats']['global_migration']['ks_cv']



ax_ks_cv_simulation_no_migration = plt.subplot2grid((2, 2), (1, 0), colspan=1)
ax_ks_cv_simulation_global_migration = plt.subplot2grid((2, 2), (1, 1), colspan=1)


ax_ks_cv_simulation_no_migration.hist(ks_cv_simulation_no_migration, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('No_migration',4)], histtype='stepfilled', density=True, zorder=2)
#ax_ks_cv_simulation_no_migration.axvline(x=0.1, ls='--', lw=3, c='k', label='Observed ' +  r'$D$')
ax_ks_cv_simulation_no_migration.axvline(x=0.1, ls='--', lw=3, c='k', label='Observed ' +  r'$\mathrm{KS}$')

ax_ks_cv_simulation_no_migration.legend(loc="upper right", fontsize=8)
#ax_ks_cv_simulation_no_migration.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
ax_ks_cv_simulation_no_migration.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)

ax_ks_cv_simulation_no_migration.set_ylabel('Probability density',  fontsize=11)

ax_ks_cv_simulation_global_migration.hist(ks_cv_simulation_global_migration, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Global_migration',4)], histtype='stepfilled', density=True, zorder=2)
ax_ks_cv_simulation_global_migration.axvline(x=0.2845138055222089, ls='--', lw=3, c='k', label='Observed ' +  r'$D$')
ax_ks_cv_simulation_global_migration.legend(loc="upper right", fontsize=8)
#ax_ks_cv_simulation_global_migration.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
ax_ks_cv_simulation_global_migration.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
ax_ks_cv_simulation_global_migration.set_ylabel('Probability density',  fontsize=11)



ax_ks_cv_simulation_no_migration.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks_cv_simulation_no_migration.transAxes)
ax_ks_cv_simulation_global_migration.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks_cv_simulation_global_migration.transAxes)


fig.text(0.19, 0.94, "Global migration simulation statistics", va='center', fontsize=20)



fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfer_cv.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
