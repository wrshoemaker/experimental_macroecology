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
import matplotlib.gridspec as gridspec


import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm
import slm_simulation_utils
import plot_utils

import plot_cv_ratio_per_replicate



np.random.seed(123456789)
random.seed(123456789)


experiments = [('No_migration', 4), ('Global_migration', 4)]

n_iter = 1000



asv_genus_dict = {'GCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGTGTAGGCGGTTCGGAAAGAAAGATGTGAAATCCCAGGGCTCAACCTTGGAACTGCATTTTTAACTGCCGAGCTAGAGTATGTCAGAGGGGGGTAGAATTCCACGTGTAGCAGTGAAATGCGTAGATATGTGGAGGAATACCGATGGCGAAGGCAGCCCCCTGGGATAATACTGACGCTCAGACACGAAAGCGTGGGG': 'Alcaligenaceae',
                'GCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTCTGTCAAGTCGGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATTCGAAACTGGCAGGCTGGAGTCTTGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACAAAGACTGACGCTCAGGTGCGAAAGCGTGGGG': 'Enterobacteriaceae',
                'GCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTCTGTCAAGTCGGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATTCGAAACTGGCAGGCTGGAGTCTTGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGCCCCCCTGGACAAAGACTGACGCTCAGGTGCGAAAGCGTGGGG': 'Enterobacteriaceae',
                'GCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTCTGTCAAGTCGGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATTCGAAACTGGCAGGCTAGAGTCTTGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACAAAGACTGACGCTCAGGTGCGAAAGCGTGGGG': 'Enterobacteriaceae'}

abundance_ratio_temporal_cv_dict_path = utils.directory + "/data/abundance_ratio_temporal_cv_dict.pickle"
taxonomy_dict = utils.make_taxonomy_dict()


def make_t_stat_dict():

    delta_cv_dict = {}

    for experiment_idx, experiment in enumerate(experiments):

        delta_cv_dict[experiment] = {}
        delta_cv_dict[experiment]['species'] = {}
        delta_cv_dict[experiment]['t_stat'] = {}

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


                    for i in range(10000):

                        np.random.shuffle(log_abundance_ratio_flat)

                        log_abundance_ratio_before_null = log_abundance_ratio_flat[(transfers_flat<12)]
                        log_abundance_ratio_after_null = log_abundance_ratio_flat[(transfers_flat>=12)]

                        cv_before_null =  np.std(log_abundance_ratio_before_null)/np.absolute(np.mean(log_abundance_ratio_before_null))
                        cv_after_null = np.std(log_abundance_ratio_after_null)/np.absolute(np.mean(log_abundance_ratio_after_null))
                        delta_cv_null = cv_after_null - cv_before_null


            trasnfers_ratio = list(log_abundance_ratio_dict.keys())
            trasnfers_ratio.sort()

            # pools observations across species
            mean_log_abundance_ratio = [np.mean(log_abundance_ratio_dict[t]) for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]
            cv_log_abundance_ratio = [np.std(log_abundance_ratio_dict[t])/np.absolute(np.mean(log_abundance_ratio_dict[t])) for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]
            trasnfers_ratio = [t for t in trasnfers_ratio if len(log_abundance_ratio_dict[t]) >= 5]

            if len(cv_log_abundance_ratio) == 0:
                continue

            delta_cv_dict[experiment]['species'][species_i] = {}
            delta_cv_dict[experiment]['species'][species_i]['trasnfers_ratio_to_plot'] = trasnfers_ratio
            delta_cv_dict[experiment]['species'][species_i]['mean_log_abundance_ratio_to_plot'] = mean_log_abundance_ratio
            delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_to_plot'] = cv_log_abundance_ratio

            mean_log_abundance_ratio = np.asarray(mean_log_abundance_ratio)
            cv_log_abundance_ratio = np.asarray(cv_log_abundance_ratio)
            trasnfers_ratio = np.asarray(trasnfers_ratio)

            trasnfers_ratio_before_idx = (trasnfers_ratio>=5) & (trasnfers_ratio<12)
            trasnfers_ratio_after_idx = (trasnfers_ratio>=12)

            # observed at every time point
            if sum(trasnfers_ratio >= 5) == 13:

                delta_cv_dict[experiment]['species'][species_i]['mean_log_abundance_ratio_before_to_test'] = (mean_log_abundance_ratio[trasnfers_ratio_before_idx]).tolist()
                delta_cv_dict[experiment]['species'][species_i]['mean_log_abundance_ratio_after_to_test'] = (mean_log_abundance_ratio[trasnfers_ratio_after_idx]).tolist()

                delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_before_to_test'] = (cv_log_abundance_ratio[trasnfers_ratio_before_idx]).tolist()
                delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_after_to_test'] = (cv_log_abundance_ratio[trasnfers_ratio_after_idx]).tolist()

                delta_cv_dict[experiment]['species'][species_i]['mean_log_abundance_ratio_to_test'] = (mean_log_abundance_ratio[(trasnfers_ratio>=5)]).tolist()
                delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_to_test'] = (cv_log_abundance_ratio[(trasnfers_ratio>=5)]).tolist()

                t_species, p_value_t_species = stats.ttest_ind(cv_log_abundance_ratio[trasnfers_ratio_after_idx], cv_log_abundance_ratio[trasnfers_ratio_before_idx], equal_var=True)

                delta_cv_dict[experiment]['species'][species_i]['t_species'] = t_species
                delta_cv_dict[experiment]['species'][species_i]['p_value_t_species'] = p_value_t_species


        # do the test
        cv_before_all = []
        cv_after_all = []
        #species_id_all = []
        species_count = 0
        species_to_test = [k for k in delta_cv_dict[experiment]['species'].keys() if 'cv_log_abundance_ratio_to_test' in delta_cv_dict[experiment]['species'][k]]
        for species_i in species_to_test:

            if 'cv_log_abundance_ratio_before_to_test' not in delta_cv_dict[experiment]['species'][species_i]:
                continue

            #species_id_all.extend([species_count]*len(delta_cv_dict[experiment][species_i]['cv_log_abundance_ratio_after_to_test']))
            cv_before_all.extend(delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_before_to_test'])
            cv_after_all.extend(delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_after_to_test'])

            species_count +=1 



        t_null_pooled_all = []
        for n in range(n_iter):

            cv_log_abundance_ratio_before_to_test_null = []
            cv_log_abundance_ratio_after_to_test_null = []

            for species_i in species_to_test:

                cv_log_abundance_ratio_to_test = np.asarray(delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_to_test'])

                np.random.shuffle(cv_log_abundance_ratio_to_test)

                cv_log_abundance_ratio_before_to_test_null.append(cv_log_abundance_ratio_to_test[:6])
                cv_log_abundance_ratio_after_to_test_null.append(cv_log_abundance_ratio_to_test[6:])


            cv_log_abundance_ratio_before_to_test_null = np.concatenate(cv_log_abundance_ratio_before_to_test_null, axis=0)
            cv_log_abundance_ratio_after_to_test_null = np.concatenate(cv_log_abundance_ratio_after_to_test_null, axis=0)


            t_null, p_value_t_null = stats.ttest_ind(cv_log_abundance_ratio_after_to_test_null, cv_log_abundance_ratio_before_to_test_null, equal_var=True)

            t_null_pooled_all.append(t_null)


        pooled_t_stat, p_value_pooled_t_stat = stats.ttest_ind(cv_after_all, cv_before_all, equal_var=True)

        t_null_pooled_all = np.asarray(t_null_pooled_all)
        pooled_t_stat_p_value = sum(t_null_pooled_all > t)/n_iter

        mean_t_stat = np.mean([delta_cv_dict[experiment]['species'][s]['t_species'] for s in species_to_test])

        t_null_mean_all = []
        for n in range(n_iter):

            t_null_species = []
            for species_i in species_to_test:

                cv_log_abundance_ratio_to_test = np.asarray(delta_cv_dict[experiment]['species'][species_i]['cv_log_abundance_ratio_to_test'])

                np.random.shuffle(cv_log_abundance_ratio_to_test)
                t_null_species.append(stats.ttest_ind(cv_log_abundance_ratio_to_test[6:], cv_log_abundance_ratio_to_test[:6], equal_var=True))

            t_null_mean_all.append(np.mean(t_null_species))


        t_null_mean_all = np.asarray(t_null_mean_all)
        mean_t_stat_p_value = sum(t_null_mean_all > mean_t_stat)/n_iter
        

        print(mean_t_stat, mean_t_stat_p_value)



        
        delta_cv_dict[experiment]['t_stat']['pooled_t_stat'] = pooled_t_stat
        delta_cv_dict[experiment]['t_stat']['pooled_t_stat_p_value'] = pooled_t_stat_p_value
        delta_cv_dict[experiment]['t_stat']['mean_t_stat'] = mean_t_stat
        delta_cv_dict[experiment]['t_stat']['mean_t_stat_p_value'] = mean_t_stat_p_value



    sys.stderr.write("Saving dictionary...\n")
    with open(abundance_ratio_temporal_cv_dict_path, 'wb') as handle:
        pickle.dump(delta_cv_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        

def run_best_parameter_simulations():

    with open(abundance_ratio_temporal_cv_dict_path, 'rb') as handle:
        delta_cv_dict = pickle.load(handle)

    simulation_global_rho_abc_dict = slm_simulation_utils.load_simulation_global_rho_abc_dict()

    tau_all = np.asarray(simulation_global_rho_abc_dict['tau_all'])
    sigma_all = np.asarray(simulation_global_rho_abc_dict['sigma_all'])


    #for migration_status in ['global_migration', 'no_migration']:

    for experiment_idx, experiment in enumerate(experiments):

        if experiment == ('No_migration', 4):
            migration_status = 'no_migration'
        else:
            migration_status = 'global_migration'

        mean_t_stat = delta_cv_dict[experiment]['t_stat']['mean_t_stat']
        mean_t_stat_sim = np.asarray(simulation_global_rho_abc_dict['ratio_stats'][migration_status]['mean_t_stat'])

        dist = np.sqrt(((mean_t_stat-mean_t_stat_sim)/mean_t_stat)**2)
        min_dist_idx = np.argmin(dist)

        best_tau = tau_all[min_dist_idx]
        best_sigma = sigma_all[min_dist_idx]

        print(best_tau, best_sigma, dist[min_dist_idx])

        sys.stderr.write("Running simulation with optimal parameters...\n")
        slm_simulation_utils.run_simulation_global_rho_fixed_parameters(best_tau, best_sigma, label = migration_status, n_iter=1000)
        sys.stderr.write("Done!\n")




def make_plot():

    fig = plt.figure(figsize = (8.5, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    with open(abundance_ratio_temporal_cv_dict_path, 'rb') as handle:
        delta_cv_dict = pickle.load(handle)


    for experiment_idx, experiment in enumerate(experiments):

        #communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
        #communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]

        ax_cv = plt.subplot2grid((2, 2), (0, experiment_idx), colspan=1)
        ax_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv.transAxes)

        cv_per_t_dict = {}
        for asv in delta_cv_dict[experiment]['species'].keys():

            trasnfers_ratio_to_plot_asv = delta_cv_dict[experiment]['species'][asv]['trasnfers_ratio_to_plot']
            cv_log_abundance_ratio_to_plot_asv = delta_cv_dict[experiment]['species'][asv]['cv_log_abundance_ratio_to_plot']

            ax_cv.plot(trasnfers_ratio_to_plot_asv, cv_log_abundance_ratio_to_plot_asv, alpha=0.6, c=utils.color_dict_range[experiment][7], zorder=2)

            for t_idx, t in enumerate(trasnfers_ratio_to_plot_asv):

                if t not in cv_per_t_dict:
                    cv_per_t_dict[t] = []

                cv_per_t_dict[t].append(cv_log_abundance_ratio_to_plot_asv[t_idx])

        
        # mean of log CV over ASVs
        mean_log_cv = np.asarray([np.mean(np.log10(cv_per_t_dict[t])) for t in cv_per_t_dict.keys()])
        transfes_to_plot = list(cv_per_t_dict.keys())
        transfes_to_plot.sort()

        ax_cv.plot(transfes_to_plot, 10**mean_log_cv, alpha=1, c=utils.color_dict_range[experiment][13], zorder=3)

        ax_cv.set_xlabel('Transfer, ' + r'$k$', fontsize=12)
        ax_cv.set_ylabel('CV of log-fold abundance ratio, ' + r'$\mathrm{CV}_{\Delta l}^{(k)}$', fontsize=11)
        ax_cv.set_title(utils.titles_no_inocula_dict[experiment], fontsize=13)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_cv.set_xlim([1, 17])
        ax_cv.set_ylim([0.04, 700])
        ax_cv.set_yscale('log', basey=10)


        mean_t_stat_p_value = delta_cv_dict[experiment]['t_stat']['mean_t_stat_p_value']

        if mean_t_stat_p_value == float(0):
            p_value_label = r'$P < 10^{-3}$'
        else:
            p_value_label = r'$P = $' + str(round(mean_t_stat_p_value, 3))


        ax_cv.text(0.2  , 0.9, r'$\bar{t} = $' + str(round(delta_cv_dict[experiment]['t_stat']['mean_t_stat'], 3)), fontsize=10, ha='center', va='center', transform=ax_cv.transAxes)
        ax_cv.text(0.2, 0.8, p_value_label, fontsize=10, ha='center', va='center', transform=ax_cv.transAxes)



        legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                            Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean over ASVs'),
                            Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration')]

        if experiment_idx == 0:
            ax_cv.legend(handles=legend_elements, fontsize=9, loc='lower right')

        if experiment_idx == 1:
            ax_cv.axvline(x=12, color='k', linestyle=':', lw = 3, zorder=1)


        # plot simulation
        if experiment == ('No_migration', 4):
            migration_status = 'no_migration'
        else:
            migration_status = 'global_migration'

        print(experiment)

        sim_dict = slm_simulation_utils.load_simulation_global_rho_fixed_parameters_dict(migration_status)

        
        mean_t_stat = delta_cv_dict[experiment]['t_stat']['mean_t_stat']
        mean_t_stat_sim = np.sort(sim_dict['ratio_stats'][migration_status]['mean_t_stat'])

        best_tau = sim_dict['tau_all'][0]
        best_sigma = sim_dict['sigma_all'][0]

        ax_sim = plt.subplot2grid((2, 2), (1, experiment_idx), colspan=1)


        ax_sim.hist(mean_t_stat_sim, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiment], histtype='stepfilled', density=True, zorder=2)
        ax_sim.axvline(x=mean_t_stat, ls='--', lw=3, c='k', label='Observed mean ' + r'$t$' + '-statistic')


        ax_sim.set_xlabel('Predicted mean ' + r'$t$' + '-statistic from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
        ax_sim.set_ylabel('Probability density',  fontsize=11)

        lower_ci = mean_t_stat_sim[int(0.025*len(mean_t_stat_sim))]
        upper_ci = mean_t_stat_sim[int(0.975*len(mean_t_stat_sim))]
        ax_sim.axvline(x=lower_ci, ls=':', lw=3, c='k', label='95% CIs')
        ax_sim.axvline(x=upper_ci, ls=':', lw=3, c='k')

        if experiment_idx == 0:
            ax_sim.legend(loc="upper left", fontsize=8)

        ax_sim.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim.transAxes)



    fig.text(0.3, 0.94, "Global migration statistics", va='center', fontsize=20)
    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfer_cv.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def make_plot_across_reps_and_within_rep_plot():

    axis_label_fontsize=14
    legend_fontsize=11


    with open(abundance_ratio_temporal_cv_dict_path, 'rb') as handle:
        delta_cv_dict = pickle.load(handle)

    #fig = plt.figure(figsize=(19, 8.5))
    fig = plt.figure(figsize=(18, 8.5))
    fig.subplots_adjust(top=0.15)
    #fig.suptitle(' title ', fontsize=12, bbox={'facecolor':'none', 'alpha':0.5, 'pad':5})
    fig.text(0.23, 1.01, r'$\mathrm{CV}_{\Delta \ell}$' + " across communities, per-transfer", va='center', ha='center', fontsize=20)
    fig.text(0.74, 1.01, r'$\mathrm{CV}_{\Delta \ell}$' + " across transfers, per-community", va='center',  ha='center', fontsize=20)
    fig.text(0.49, 1.05, "Global migration statistics", ha='center', fontweight='bold', fontsize=23)

    
    #########################
    # across community stats
    #########################

    # over reps per-unit time
    outergs_across = gridspec.GridSpec(1, 1)
    outergs_across.update(bottom=0.01, left=0.01, top=1-0.01, right=0.5-0.02)
    outerax_across = fig.add_subplot(outergs_across[0])
    outerax_across.tick_params(axis='both' ,which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    outerax_across.set_facecolor('none')

    gs_across = gridspec.GridSpec(2, 2)
    gs_across.update(bottom=0.1, left=+0.08, top=1-0.05, right=0.5-0.05, wspace=0.34, hspace=0.25)

    for experiment_idx, experiment in enumerate(experiments):

        ax_cv = fig.add_subplot(gs_across[0, experiment_idx])
        ax_cv.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv.transAxes)

        cv_per_t_dict = {}
        for asv in delta_cv_dict[experiment]['species'].keys():

            trasnfers_ratio_to_plot_asv = delta_cv_dict[experiment]['species'][asv]['trasnfers_ratio_to_plot']
            cv_log_abundance_ratio_to_plot_asv = delta_cv_dict[experiment]['species'][asv]['cv_log_abundance_ratio_to_plot']
            ax_cv.plot(trasnfers_ratio_to_plot_asv, cv_log_abundance_ratio_to_plot_asv, alpha=0.6, c=utils.color_dict_range[experiment][7], zorder=2)

            for t_idx, t in enumerate(trasnfers_ratio_to_plot_asv):

                if t not in cv_per_t_dict:
                    cv_per_t_dict[t] = []

                cv_per_t_dict[t].append(cv_log_abundance_ratio_to_plot_asv[t_idx])

        
        # mean of log CV over ASVs
        mean_log_cv = np.asarray([np.mean(np.log10(cv_per_t_dict[t])) for t in cv_per_t_dict.keys()])
        transfes_to_plot = list(cv_per_t_dict.keys())
        transfes_to_plot.sort()

        ax_cv.plot(transfes_to_plot, 10**mean_log_cv, alpha=1, lw=1, c=utils.color_dict_range[experiment][13], zorder=3)

        ax_cv.set_xlabel('Transfer, ' + r'$k$', fontsize=axis_label_fontsize)
        ax_cv.set_ylabel('CV of log-fold abundance ratio, ' + r'$\mathrm{CV}_{\Delta l}^{(k)}$', fontsize=axis_label_fontsize)
        ax_cv.set_title(utils.titles_no_inocula_dict[experiment], fontsize=axis_label_fontsize+1)

        ax_cv.set_xlim([1, 17])
        ax_cv.set_ylim([0.04, 700])
        ax_cv.set_yscale('log', basey=10)

        mean_t_stat_p_value = delta_cv_dict[experiment]['t_stat']['mean_t_stat_p_value']

        if mean_t_stat_p_value == float(0):
            p_value_label = r'$P < 10^{-3}$'
        else:
            p_value_label = r'$P = $' + str(round(mean_t_stat_p_value, 3))


        ax_cv.text(0.2  , 0.9, r'$\bar{t} = $' + str(round(delta_cv_dict[experiment]['t_stat']['mean_t_stat'], 3)), fontsize=12, ha='center', va='center', transform=ax_cv.transAxes)
        ax_cv.text(0.2, 0.8, p_value_label, fontsize=12, ha='center', va='center', transform=ax_cv.transAxes)

        legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                            Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean over ASVs'),
                            Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration')]

        if experiment_idx == 0:
            ax_cv.legend(handles=legend_elements, fontsize=legend_fontsize, loc='lower right')

        if experiment_idx == 1:
            ax_cv.axvline(x=12, color='k', linestyle=':', lw = 3, zorder=1)


        # plot simulation
        sim_dict = slm_simulation_utils.load_simulation_global_rho_fixed_parameters_dict(utils.migration_status_dict[experiment])
        mean_t_stat = delta_cv_dict[experiment]['t_stat']['mean_t_stat']
        mean_t_stat_sim = np.sort(sim_dict['ratio_stats'][utils.migration_status_dict[experiment]]['mean_t_stat'])
        
        ax_sim = fig.add_subplot(gs_across[1, experiment_idx])
        ax_sim.hist(mean_t_stat_sim, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiment], histtype='stepfilled', density=True, zorder=2)
        ax_sim.axvline(x=mean_t_stat, ls='--', lw=3, c='k', label='Observed mean ' + r'$t$' + '-statistic')

        ax_sim.set_xlabel('Predicted mean ' + r'$t$' + '-statistic from optimal\nparameters, ' + r'$\tau = $' + str(round(sim_dict['tau_all'][0], 2)) + ' and ' + r'$\sigma = $' + str(round(sim_dict['sigma_all'][0], 3)), fontsize=axis_label_fontsize-1)
        ax_sim.set_ylabel('Probability density',  fontsize=axis_label_fontsize)

        lower_ci = mean_t_stat_sim[int(0.025*len(mean_t_stat_sim))]
        upper_ci = mean_t_stat_sim[int(0.975*len(mean_t_stat_sim))]
        ax_sim.axvline(x=lower_ci, ls=':', lw=3, c='k', label='95% CIs')
        ax_sim.axvline(x=upper_ci, ls=':', lw=3, c='k')

        if experiment_idx == 0:
            ax_sim.legend(loc="upper left", fontsize=legend_fontsize)

        ax_sim.text(-0.1, 1.04, plot_utils.sub_plot_labels[2+experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim.transAxes)



    #########################
    # within community stats
    #########################

    outergs_within = gridspec.GridSpec(1, 1)
    outergs_within.update(bottom=0.01, left=0.51, top=0.99, right=0.98)
    outerax_within = fig.add_subplot(outergs_within[0])
    outerax_within.tick_params(axis='both' ,which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    outerax_within.set_facecolor('none')

    #for axis in ['top','bottom','left','right']:
    #    outergs_within.spines[axis].set_linewidth(3)
    

    gs_within = gridspec.GridSpec(2, 2)
    gs_within.update(bottom=0.1, left=0.58, top=1-0.05, right=1-0.05, wspace=0.34, hspace=0.25)

    ax_f_dist = fig.add_subplot(gs_within[1, 0])
    ax_f_rank = fig.add_subplot(gs_within[1, 1])

    with open(plot_cv_ratio_per_replicate.cv_ratio_per_replicate_dict_path, 'rb') as handle:
        cv_within_dict = pickle.load(handle)

    f_rank_dict = {}
    for experiment_idx, experiment in enumerate(experiments):

        ax_cv_compare = fig.add_subplot(gs_within[0, experiment_idx])
        ax_cv_compare.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx+4], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv_compare.transAxes)
        ax_cv_compare.set_title(utils.titles_no_inocula_dict[experiment], fontsize=axis_label_fontsize+1)

        cv_before_all = []
        cv_after_all = []
        f_dist_all = []
        for asv_idx, asv in enumerate(cv_within_dict['asv'].keys()):

            if len(cv_within_dict['asv'][asv]) < 2:
                continue

            #print(cv_within_dict[asv].keys())

            migration_reps = list(cv_within_dict['asv'][asv][utils.migration_status_dict[experiment]].keys())

            cv_before = [cv_within_dict['asv'][asv][utils.migration_status_dict[experiment]][r]['cv_log_ratio_before'] for r in migration_reps]
            cv_after = [cv_within_dict['asv'][asv][utils.migration_status_dict[experiment]][r]['cv_log_ratio_after'] for r in migration_reps]
            f_dist = [cv_within_dict['asv'][asv][utils.migration_status_dict[experiment]][r]['F_cv'] for r in migration_reps]

            cv_before_all.extend(cv_before)
            cv_after_all.extend(cv_after)
            f_dist_all.extend(f_dist)

            if (len(cv_within_dict['asv'][asv]['no_migration']) < 10) or len(cv_within_dict['asv'][asv]['global_migration']) < 3:
                continue

            if asv not in f_rank_dict:
                f_rank_dict[asv] = {}
            
            f_rank_dict[asv][experiment] = f_dist


        ax_cv_compare.scatter(cv_before_all, cv_after_all, color=utils.color_dict[experiment], s=12, alpha=0.5, zorder=2)
        ax_cv_compare.set_xscale('log', basex=10)
        ax_cv_compare.set_yscale('log', basey=10)

        min_ = min(cv_before_all+cv_after_all)
        max_ = max(cv_before_all+cv_after_all)

        ax_cv_compare.plot([min_, max_], [min_, max_], color='k', lw=3, ls=':', label='1:1', zorder=1)
        ax_cv_compare.set_xlim([min_*0.9, max_*1.1])
        ax_cv_compare.set_ylim([min_*0.9, max_*1.1])

        if experiment_idx == 0:
            ax_cv_compare.legend(loc="upper left", fontsize=legend_fontsize)

        ax_cv_compare.set_xlabel(r'$\mathrm{CV}^{<}_{\Delta \ell}$' + ' before cessation of migration',  fontsize=axis_label_fontsize)
        ax_cv_compare.set_ylabel(r'$\mathrm{CV}^{>}_{\Delta \ell}$' + ' after cessation of migration',  fontsize=axis_label_fontsize)
        ax_cv_compare.text(-0.1, 1.04, plot_utils.sub_plot_labels[4+experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_cv_compare.transAxes)


        # plot F distribution
        ax_f_dist.hist(f_dist_all, lw=3, alpha=0.4, bins=10, color=utils.color_dict[experiment], histtype='stepfilled', density=True, zorder=2)
    
    
    ax_f_dist.axvline(x=1, ls=':', lw=3, c='k', label='No change', zorder=3)
    ax_f_dist.legend(loc="upper right", fontsize=legend_fontsize)
    ax_f_dist.set_xlabel('Change in ' + r'$\mathrm{CV}^{<}_{\Delta \ell}$' + '\nafter cessation of migration, ' + r'$F$',  fontsize=axis_label_fontsize)
    ax_f_dist.set_ylabel('Probability density',  fontsize=axis_label_fontsize)
    ax_f_dist.text(-0.1, 1.04, plot_utils.sub_plot_labels[5+experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_f_dist.transAxes)
    ax_f_dist.text(0.7  , 0.83, r'$\mathrm{KS} = $' + str(round(cv_within_dict['stats']['ks_stat_constrain_on_asv'], 3)), fontsize=12, ha='center', va='center', transform=ax_f_dist.transAxes)
    ax_f_dist.text(0.7, 0.73, r'$P = $' + str(round(cv_within_dict['stats']['p_value_ks_stat_constrain_on_asv'], 3)) , fontsize=12, ha='center', va='center', transform=ax_f_dist.transAxes)
   
    # plot ranks
    ax_f_rank.text(-0.1, 1.04, plot_utils.sub_plot_labels[6+experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_f_rank.transAxes)


    asv_to_plot = list(f_rank_dict.keys())
    mean_f_no_migration = [np.mean(f_rank_dict[a][('No_migration', 4)]) for a in asv_to_plot]

    zipped_pairs = zip(mean_f_no_migration, asv_to_plot)
    asv_to_plot_sorted = [x for _, x in sorted(zipped_pairs)]

    # family level labels
    family_labels = [asv_genus_dict[a] for a in asv_to_plot_sorted]
    family_labels_idx = []
    
    idx_count = 0
    for asv in asv_to_plot_sorted:

        for experiment_idx, experiment in enumerate(experiments):

            f_ = f_rank_dict[asv][experiment]
            ax_f_rank.scatter(f_, [idx_count]*len(f_), color=utils.color_dict[experiment], s=20, alpha=0.5, zorder=2)

            idx_count += 1

        family_labels_idx.append(idx_count - 1.5)


    ax_f_rank.set_xlabel('Change in ' + r'$\mathrm{CV}^{<}_{\Delta \ell}$' + '\nafter cessation of migration, ' + r'$F$',  fontsize=axis_label_fontsize)
    ax_f_rank.axvline(x=1, ls=':', lw=3, c='k', label='No change', zorder=1)
    
    ax_f_rank.set_yticks(family_labels_idx)
    ax_f_rank.set_yticklabels(family_labels, fontsize=7.5, rotation=45)


    # axis widths 
    for axis in ['top','bottom','left','right']:
        outerax_across.spines[axis].set_linewidth(3)
        outerax_within.spines[axis].set_linewidth(3)
    

    #fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/cv_logfold_across_and_within_reps.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()









def old_plot():


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

            cv_log_abundance_ratio_before = cv_log_abundance_ratio[6:12]
            cv_log_abundance_ratio_after = cv_log_abundance_ratio[12:]

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

        species_over_all_species_t_before_flat = np.asarray(list(itertools.chain(*species_over_all_species_t_before)))
        species_over_all_species_t_after_flat = np.asarray(list(itertools.chain(*species_over_all_species_t_after)))

        n_paired_events = 0


        # t-test
        #t, p_value_t, df_t = stats.ttest_ind(cv_over_all_species_t_before, cv_over_all_species_t_after, equal_var=True)


        ks_statistic_cv_over_all_species_t, p_value_cv_over_all_species_t = utils.run_permutation_ks_test(cv_over_all_species_t_before, cv_over_all_species_t_after, n=2)
        #(experiment, len(cv_over_all_species_t_before), len(cv_over_all_species_t_after), len(np.intersect1d(species_over_all_species_t_before_flat, species_over_all_species_t_after_flat ) ))

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

        ax_cv.set_xlabel('Transfer, ' + r'$k$', fontsize=12)
        ax_cv.set_ylabel('CV of relative abundance ratio, ' + r'$\mathrm{CV}_{\Delta l}^{(k)}$', fontsize=11)
        ax_cv.set_title(utils.titles_no_inocula_dict[experiment], fontsize=13)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        #ax.set_ylim([-2.5, 2.5])
        ax_cv.set_xlim([1, 17])
        ax_cv.set_ylim([0.04, 700])
        ax_cv.set_yscale('log', basey=10)


        legend_elements = [Line2D([0], [0], color=utils.color_dict_range[experiment][7], lw=1.5, label='One ASV'),
                            Line2D([0], [0], color=utils.color_dict_range[experiment][13], lw=1.5, label='Mean over ASVs'),
                            Line2D([0], [0], color='k', ls=':', lw=1.5, label='End of migration')]

        if experiment_idx == 0:
            ax_cv.legend(handles=legend_elements, fontsize=9, loc='lower right')

        if experiment_idx == 1:
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
    #pred = np.asarray([ks_cv_no_migration, ks_cv_global_migration])
    pred = np.hstack([ks_cv_no_migration, ks_cv_global_migration])


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


    #ax_ks_cv_simulation_no_migration.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    #ax_ks_cv_simulation_no_migration.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' statistic from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_ks_cv_simulation_no_migration.set_xlabel('Predicted ' + r'$\mathrm{KS}$' + ' statistic from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_ks_cv_simulation_no_migration.set_ylabel('Probability density',  fontsize=11)

    ax_ks_cv_simulation_global_migration.hist(ks_cv_simulation_global_migration, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('Global_migration',4)], histtype='stepfilled', density=True, zorder=2)
    ax_ks_cv_simulation_global_migration.axvline(x=0.2845138055222089, ls='--', lw=3, c='k', label='Observed ' + r'$\mathrm{KS}$')
    #ax_ks_cv_simulation_global_migration.legend(loc="upper right", fontsize=8)
    #ax_ks_cv_simulation_global_migration.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_ks_cv_simulation_global_migration.set_xlabel('Predicted ' + r'$\mathrm{KS}$' + ' statistic from optimal\nparameters, ' + r'$\tau = $' + str(round(best_tau, 2)) + ' and ' + r'$\sigma = $' + str(round(best_sigma, 3)), fontsize=11)
    ax_ks_cv_simulation_global_migration.set_ylabel('Probability density',  fontsize=11)


    ks_cv_simulation_no_migration = np.sort(ks_cv_simulation_no_migration)
    lower_ci_no_migration = ks_cv_simulation_no_migration[int(0.025*len(ks_cv_simulation_no_migration))]
    upper_ci_no_migration = ks_cv_simulation_no_migration[int(0.975*len(ks_cv_simulation_no_migration))]
    ax_ks_cv_simulation_no_migration.axvline(x=lower_ci_no_migration, ls=':', lw=3, c='k', label='95% CIs')
    ax_ks_cv_simulation_no_migration.axvline(x=upper_ci_no_migration, ls=':', lw=3, c='k')


    ks_cv_simulation_global_migration = np.sort(ks_cv_simulation_global_migration)
    lower_ci_global_migration = ks_cv_simulation_global_migration[int(0.025*len(ks_cv_simulation_global_migration))]
    upper_ci_global_migration = ks_cv_simulation_global_migration[int(0.975*len(ks_cv_simulation_global_migration))]
    ax_ks_cv_simulation_global_migration.axvline(x=lower_ci_global_migration, ls=':', lw=3, c='k', label='95% CIs')
    ax_ks_cv_simulation_global_migration.axvline(x=upper_ci_global_migration, ls=':', lw=3, c='k')


    ax_ks_cv_simulation_no_migration.legend(loc="upper right", fontsize=8)


    ax_ks_cv_simulation_no_migration.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks_cv_simulation_no_migration.transAxes)
    ax_ks_cv_simulation_global_migration.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks_cv_simulation_global_migration.transAxes)


    fig.text(0.3, 0.94, "Global migration statistics", va='center', fontsize=20)



    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/abundance_ratio_per_transfer_cv.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    #fig.savefig(utils.directory + '/figs/abundance_ratio_per_transfer_cv.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()









if __name__=='__main__':

    print("Running per-transfer log-ratio CV analysis....")

    #make_t_stat_dict()
    #run_best_parameter_simulations()

    #make_plot()

    make_plot_across_reps_and_within_rep_plot()

