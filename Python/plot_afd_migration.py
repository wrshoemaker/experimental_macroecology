from __future__ import division
import os, sys
import numpy as np
import pickle

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from itertools import combinations
#import statsmodels.stats.multitest as multitest
import plot_utils
import slm_simulation_utils


#afd_migration_transfer_12

ks_dict_path = "%s/data/afd_ks_dict.pickle" %  utils.directory


transfers = np.asarray([12, 18])

experiments = [('No_migration',4), ('Parent_migration', 4) , ('Global_migration',4) ]
treatment_combinations = list(combinations(experiments,2))
treatments_no_innoculum = ['no_migration', 'parent_migration', 'global_migration']
rescaled_status = 'afd_rescaled'

if rescaled_status == 'afd_rescaled':
    rescaled_status_sim = 'ks_rescaled_12_vs_18'
else:
    rescaled_status_sim = 'ks_12_vs_18'

rescaled_status_all = ['afd', 'afd_rescaled']


min_occupancy_afd = 0.8



def make_afd_dict():

    afd_dict = {}

    for experiment in experiments:

        afd_dict[experiment] = {}

        for transfer in transfers:

            #relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])
            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
            afd_log10_all, afd_log10_rescaled = utils.get_flat_rescaled_afd(s_by_s)

            afd_dict[experiment][transfer] = {}
            afd_dict[experiment][transfer]['afd'] = afd_log10_all
            afd_dict[experiment][transfer]['afd_rescaled'] = afd_log10_rescaled

            # only occupancy of one
            afd_log10_all_occupancy_one, afd_log10_rescaled_occupancy_one = utils.get_flat_rescaled_afd(s_by_s, min_occupancy=1)
            afd_dict[experiment][transfer]['afd_occupancy_one'] = afd_log10_all_occupancy_one
            afd_dict[experiment][transfer]['afd_rescaled_occupancy_one'] = afd_log10_rescaled_occupancy_one



    afd_paired_dict = {}
    for experiment in experiments:

        afd_paired_dict[experiment] = {}

        s_by_s_dict = {}
        for transfer in transfers:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            s_by_s_dict[transfer] = {}
            s_by_s_dict[transfer]['rel_s_by_s'] = rel_s_by_s
            s_by_s_dict[transfer]['species'] = np.asarray(species)
            s_by_s_dict[transfer]['comm_rep_list'] = np.asarray(comm_rep_list)

        #communities_to_keep = list( set(s_by_s_dict[12]['comm_rep_list'].tolist() ) &  set(s_by_s_dict[18]['comm_rep_list'].tolist() ))
        # only use communities present in both time points
        communities_to_keep = np.intersect1d( s_by_s_dict[12]['comm_rep_list'], s_by_s_dict[18]['comm_rep_list'] )
        
        paired_rel_abundances_all = []
        for community in communities_to_keep:

            community_12_idx = np.where(s_by_s_dict[12]['comm_rep_list'] == community)[0][0]
            community_18_idx = np.where(s_by_s_dict[18]['comm_rep_list'] == community)[0][0]

            community_asv_union = np.union1d(s_by_s_dict[12]['species'], s_by_s_dict[18]['species'])

            for community_asv_j in community_asv_union:

                if community_asv_j in s_by_s_dict[12]['species']:
                    
                    community_asv_j_12_idx = np.where(s_by_s_dict[12]['species'] == community_asv_j)[0][0]
                    rel_abundance_12 = s_by_s_dict[12]['rel_s_by_s'][community_asv_j_12_idx, community_12_idx]

                else:
                    rel_abundance_12 = float(0)  

                
                if community_asv_j in s_by_s_dict[18]['species']:
                    
                    community_asv_j_18_idx = np.where(s_by_s_dict[18]['species'] == community_asv_j)[0][0]
                    rel_abundance_18 = s_by_s_dict[18]['rel_s_by_s'][community_asv_j_18_idx, community_18_idx]

                else:
                    rel_abundance_18 = float(0)


                if (rel_abundance_12 == float(0)) and (rel_abundance_18 == float(0)):
                    continue

                paired_rel_abundances_all.append([rel_abundance_12, rel_abundance_18])

        afd_paired_dict[experiment] = paired_rel_abundances_all

        #print(paired_rel_abundances_all)
    
    return afd_dict, afd_paired_dict





def make_per_asv_ks_dict():

    distances_dict = {}

    for experiment in experiments:

        s_by_s_dict = {}
        for transfer in transfers:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            for species_idx, species in enumerate(species):

                if species not in s_by_s_dict:
                    s_by_s_dict[species] = {}
                
                s_by_s_dict[species][transfer] = {}
                s_by_s_dict[species][transfer]['rel_s_by_s'] = rel_s_by_s[species_idx,:]
                s_by_s_dict[species][transfer]['species'] = np.asarray(species)
                s_by_s_dict[species][transfer]['comm_rep_list'] = np.asarray(comm_rep_list)



        ks_statistic_all = []
        ks_statistic_rescaled_all = []
        mad_12 = []
        mad_18 = []

        distances_dict[experiment] = {}
        distances_dict[experiment]['per_species_ks'] = {}
        for species, afd_dict in s_by_s_dict.items():

            if len(afd_dict) != 2:
                continue

            afd_12 = afd_dict[12]['rel_s_by_s']
            afd_18 = afd_dict[18]['rel_s_by_s']

            n_zero_12 = sum(afd_12==0)
            n_zero_18 = sum(afd_18==0)

            # not sampled in either timepoint
            if (n_zero_12>0) or (n_zero_18 > 0):
                continue


            afd_12_log = np.log10(afd_12)
            afd_18_log = np.log10(afd_18)

            mad_12.append(np.mean(afd_12))
            mad_18.append(np.mean(afd_18))

            afd_12_rescaled = (afd_12_log - np.mean(afd_12_log))/np.std(afd_12_log)
            afd_18_rescaled = (afd_18_log - np.mean(afd_18_log))/np.std(afd_18_log)

            ks_statistic, p_value = stats.ks_2samp(afd_12_log, afd_18_log)
            ks_statistic_all.append(ks_statistic)

            ks_statistic_rescaled, p_value = stats.ks_2samp(afd_12_rescaled, afd_18_rescaled)
            scaled_ks_statistic_rescaled = np.sqrt((len(afd_12_rescaled)*len(afd_18_rescaled)) / (len(afd_12_rescaled)+ len(afd_18_rescaled)) ) * ks_statistic_rescaled

            ks_statistic_rescaled_all.append(scaled_ks_statistic_rescaled)

            distances_dict[experiment]['per_species_ks'][species] = {}
            distances_dict[experiment]['per_species_ks'][species]['afd'] = {}
            distances_dict[experiment]['per_species_ks'][species]['afd'][12] = afd_12
            distances_dict[experiment]['per_species_ks'][species]['afd'][18] = afd_18

            distances_dict[experiment]['per_species_ks'][species]['afd_rescaled'] = {}
            distances_dict[experiment]['per_species_ks'][species]['afd_rescaled'][12] = afd_12_rescaled
            distances_dict[experiment]['per_species_ks'][species]['afd_rescaled'][18] = afd_18_rescaled


        
        distances_dict[experiment]['mean_over_asv_ks_12_vs_18'] = np.mean(ks_statistic_all)
        distances_dict[experiment]['mean_over_asv_ks_rescaled_12_vs_18'] = np.mean(ks_statistic_rescaled_all)

        print(experiment, np.mean(ks_statistic_rescaled_all))

        #print(experiment, np.mean(ks_statistic_rescaled_all), len(ks_statistic_rescaled_all))

        # ('No_migration', 4) 0.37391304347826093 0.2963768115942029 3
        #('Parent_migration', 4) 0.5688405797101449 0.22826086956521738 3
        #('Global_migration', 4) 0.28136200716845877 0.11827956989247312 6
        #print(len(rel_s_by_s[species_idx,:]), len(comm_rep_list))

 
        # KS statistic rescaled by number of observations
        #('No_migration', 4) 0.37391304347826093 1.2012796547156197 3
        #('Parent_migration', 4) 0.5688405797101449 1.5481405396264198 3
        #('Global_migration', 4) 0.28136200716845877 0.8065591326174429 6


    return distances_dict


# make_per_asv_ks_dict_attractor():




def make_ks_dict():

    afd_dict, afd_paired_dict = make_afd_dict()

    distances_dict = {}

    for combo in treatment_combinations:

        if combo not in distances_dict:
            distances_dict[combo] = {}

        for transfer in transfers:

            distances_dict[combo][transfer] = {}

            for rescaled_status in rescaled_status_all:

                afd_experiment_1 = afd_dict[combo[0]][transfer][rescaled_status]
                afd_experiment_2 = afd_dict[combo[1]][transfer][rescaled_status]

                ks_statistic, p_value = utils.run_permutation_ks_test(afd_experiment_1, afd_experiment_2, n=1000)

                distances_dict[combo][transfer][rescaled_status] = {}
                distances_dict[combo][transfer][rescaled_status]['D'] = ks_statistic
                distances_dict[combo][transfer][rescaled_status]['pvalue'] = p_value
                distances_dict[combo][transfer][rescaled_status]['n1'] = len(afd_experiment_1)
                distances_dict[combo][transfer][rescaled_status]['n2'] = len(afd_experiment_2)


    for experiment_idx, experiment in enumerate(experiments):

        distances_dict[experiment] = {}

        for rescaled_status in rescaled_status_all:

            ks_statistic, p_value = utils.run_permutation_ks_test(afd_dict[experiment][transfers[0]][rescaled_status], afd_dict[experiment][transfers[1]][rescaled_status], n=1000)

            distances_dict[experiment][rescaled_status] = {}
            distances_dict[experiment][rescaled_status]['D'] = ks_statistic
            distances_dict[experiment][rescaled_status]['pvalue'] = p_value
            distances_dict[experiment][rescaled_status]['n1'] = len(afd_dict[experiment][transfers[0]][rescaled_status])
            distances_dict[experiment][rescaled_status]['n2'] = len(afd_dict[experiment][transfers[1]][rescaled_status])


        ks_statistic, p_value = utils.run_permutation_ks_test_control(afd_paired_dict[experiment])

        distances_dict[experiment]['afd_rescaled_and_paired'] = {}
        distances_dict[experiment]['afd_rescaled_and_paired']['D'] = ks_statistic
        distances_dict[experiment]['afd_rescaled_and_paired']['pvalue'] = p_value
        distances_dict[experiment]['afd_rescaled_and_paired']['n'] = len((afd_paired_dict[experiment]))


    with open(ks_dict_path, 'wb') as outfile:
        pickle.dump(distances_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)



def load_ks_dict():

    dict_ = pickle.load(open(ks_dict_path, "rb"))
    return dict_





def make_attractor_afd_dict():

    #experiments = [('No_migration',4)]

    #afd_dict[experiment][transfer]['afd_rescaled_occupancy_one']
    attractor_afd_dict = {}
    attractor_afd_dict['merged'] = {}
    attractor_afd_dict['per_asv'] = {}

    attractor_dict = utils.get_attractor_status(migration='No_migration', inocula=4)
    #n_Alcaligenaceae = len(attractor_dict['Alcaligenaceae'])
    #n_Pseudomonadaceae = len(attractor_dict['Pseudomonadaceae'])

    afd_dict_merged_attractors = {}

    for transfer in [12, 18]:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration='No_migration',inocula=4)
        afd, rescaled_afd = utils.get_flat_rescaled_afd(s_by_s)

        afd_dict_merged_attractors[transfer] = rescaled_afd


    for attractor_idx, attractor in enumerate(attractor_dict.keys()):

        attractor_afd_dict['merged'][attractor] = {}

        attractor_afd_dict['per_asv'][attractor] = {}
        attractor_afd_dict['per_asv'][attractor]['asv'] = {}

        for transfer in [12, 18]:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration='No_migration',inocula=4)
            species = np.asarray(species)

            relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(relative_s_by_s_attractor == 0, axis=1)][0]
            attractor_species = species[attractor_species_idx]

            #species = species[attractor_species_idx]
            relative_s_by_s_attractor = relative_s_by_s_attractor[attractor_species_idx]

            afd, rescaled_afd = utils.get_flat_rescaled_afd(relative_s_by_s_attractor)
            afd_occupancy_one, rescaled_afd_occupancy_one = utils.get_flat_rescaled_afd(relative_s_by_s_attractor, min_occupancy=1)

            attractor_afd_dict['merged'][attractor][transfer] = {}
            attractor_afd_dict['merged'][attractor][transfer]['afd'] = afd
            attractor_afd_dict['merged'][attractor][transfer]['afd_rescaled'] = rescaled_afd

            attractor_afd_dict['merged'][attractor][transfer]['afd_occupancy_one'] = afd_occupancy_one
            attractor_afd_dict['merged'][attractor][transfer]['afd_rescaled_occupancy_one'] = rescaled_afd_occupancy_one


            #s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration='No_migration', inocula=4)
            #relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            #attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            #relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            
            # occupancy of one
            asv_to_keep_idx = (np.sum(relative_s_by_s_attractor>0, axis=1) == relative_s_by_s_attractor.shape[1])

            asv_to_keep = attractor_species[asv_to_keep_idx]
            relative_s_by_s_attractor_to_keep = relative_s_by_s_attractor[asv_to_keep_idx,:]

            for asv_idx, asv in enumerate(asv_to_keep):
                if asv not in attractor_afd_dict['per_asv'][attractor]['asv']:
                    attractor_afd_dict['per_asv'][attractor]['asv'][asv] = {}

                attractor_afd_dict['per_asv'][attractor]['asv'][asv][transfer] = relative_s_by_s_attractor_to_keep[asv_idx,:]



        asv_attractor = list(attractor_afd_dict['per_asv'][attractor]['asv'].keys())
        ks_stat_all = []
        for asv in asv_attractor:
            
            # occupancy of one at both timepoints
            if len(attractor_afd_dict['per_asv'][attractor]['asv'][asv]) != 2:
                continue

            afd_12 = attractor_afd_dict['per_asv'][attractor]['asv'][asv][12]
            afd_18 = attractor_afd_dict['per_asv'][attractor]['asv'][asv][18]

            afd_log10_12 = np.log10(afd_12)
            afd_log10_18 = np.log10(afd_18)

            rescaled_afd_log10_12 = (afd_log10_12 - np.mean(afd_log10_12))/np.std(afd_log10_12)
            rescaled_afd_log10_18 = (afd_log10_18 - np.mean(afd_log10_18))/np.std(afd_log10_18)


            D, pvalue = stats.ks_2samp(rescaled_afd_log10_12, rescaled_afd_log10_18)
            scaled_d = D*np.sqrt(len(rescaled_afd_log10_12)*len(rescaled_afd_log10_18)/ (len(rescaled_afd_log10_12) + len(rescaled_afd_log10_18)))

            ks_stat_all.append(scaled_d)

            # mean scaled ks 
            # Alcaligenaceae 0.7786122309834377
            # Pseudomonadaceae 0.554024732920621

        #print(attractor, len(asv_attractor), np.mean(ks_stat_all))


        attractor_afd_dict['per_asv'][attractor]['ks_stats'] = {}
        attractor_afd_dict['per_asv'][attractor]['ks_stats']['mean_ks_over_rescaled_afds'] = np.mean(ks_stat_all)

    
    return attractor_afd_dict





def run_best_parameter_simulations():

    ks_dict = load_ks_dict()

    per_asv_ks_dict = make_per_asv_ks_dict()

    attractor_afd_dict = make_attractor_afd_dict()

    #print(per_asv_ks_dict.keys())

    # plot simulation
    #simulation_dict = slm_simulation_utils.load_simulation_all_migration_abc_dict()
    simulation_dict = slm_simulation_utils.load_simulation_all_migration_abc_afd_dict()

    tau_all = np.asarray(simulation_dict['tau_all'])
    sigma_all = np.asarray(simulation_dict['sigma_all'])

    # mean_over_asv_ks_12_vs_18, mean_over_asv_ks_rescaled_12_vs_18

    for treatment_idx, treatment in enumerate(treatments_no_innoculum):

        #ks_rescaled_12_vs_18_simulation = np.asarray(simulation_dict[rescaled_status_sim][treatment])
        #ks_rescaled_12_vs_18 =  ks_dict[experiments[treatment_idx]][rescaled_status]['D']  
        ks_rescaled_12_vs_18_simulation = np.asarray(simulation_dict['mean_over_asv_ks_rescaled_12_vs_18'][treatment])
        ks_rescaled_12_vs_18 =  per_asv_ks_dict[experiments[treatment_idx]]['mean_over_asv_ks_rescaled_12_vs_18']

        ks_rescaled_12_vs_18_attractor_simulation = np.asarray(simulation_dict['mean_over_asv_ks_rescaled_12_vs_18_attractor'][treatment])

        #ks_12_vs_18_simulation = np.asarray(simulation_dict['mean_over_asv_ks_12_vs_18'][treatment])  
        #ks_12_vs_18 =  per_asv_ks_dict[experiments[treatment_idx]]['mean_over_asv_ks_12_vs_18']  

        euc_dist = np.sqrt(((ks_rescaled_12_vs_18 - ks_rescaled_12_vs_18_simulation)/ks_rescaled_12_vs_18) **2)
        if treatment == 'no_migration':

            ks_rescaled_12_vs_18_alcaligenaceae = attractor_afd_dict['per_asv']['Alcaligenaceae']['ks_stats']['mean_ks_over_rescaled_afds']
            euc_dist_attractor = np.sqrt(((ks_rescaled_12_vs_18_alcaligenaceae - ks_rescaled_12_vs_18_attractor_simulation)/ks_rescaled_12_vs_18_alcaligenaceae) **2)
            #print('Attractor', min(euc_dist), tau_all[np.argmin(euc_dist)], sigma_all[np.argmin(euc_dist)])

            label_attractor = '%s_afd_attractor' % treatment
            #slm_simulation_utils.run_simulation_all_migration_afd_abc(n_iter=1000, tau=tau_all[np.argmin(euc_dist_attractor)], sigma=sigma_all[np.argmin(euc_dist_attractor)], label=label_attractor)
            slm_dict_attractor = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_afd_dict(label_attractor)
            ks_array_sim_attractor = np.sort(slm_dict_attractor['mean_over_asv_ks_rescaled_12_vs_18_attractor']['no_migration'])
            lower_ci_attractor = ks_array_sim_attractor[int(0.025*len(ks_array_sim_attractor))]
            upper_ci_attractor = ks_array_sim_attractor[int(0.975*len(ks_array_sim_attractor))]

            print(ks_rescaled_12_vs_18_alcaligenaceae, lower_ci_attractor, upper_ci_attractor)


        # index of best parameter
        min_parameter_idx = np.argmin(euc_dist)

        tau_best = tau_all[min_parameter_idx]
        sigma_best = sigma_all[min_parameter_idx]

        if sum(euc_dist==float(0)) > 1:
            tau_best = np.mean(tau_all[euc_dist==float(0)])
            sigma_best = np.mean(sigma_all[euc_dist==float(0)])
            #print(tau_all[euc_dist==float(0)], sigma_all[euc_dist==float(0)])


        #if treatment != 'global_migration':
        #    continue

        #print(treatment, ks_rescaled_12_vs_18, ks_rescaled_12_vs_18_simulation[min_parameter_idx], tau_best, sigma_best,  euc_dist[min_parameter_idx])

        label = '%s_afd' % treatment

        slm_simulation_utils.run_simulation_all_migration_afd_abc(n_iter=1000, tau=tau_best, sigma=sigma_best, label=label)
        slm_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_afd_dict(label)

        ks_array_sim = np.sort(slm_dict['mean_over_asv_ks_rescaled_12_vs_18'][treatment])
        lower_ci = ks_array_sim[int(0.025*len(ks_array_sim))]
        upper_ci = ks_array_sim[int(0.975*len(ks_array_sim))]

        print(ks_rescaled_12_vs_18, lower_ci, upper_ci)





def make_plot():

    afd_dict, afd_paired_dict = make_afd_dict()
    ks_dict = load_ks_dict()
    per_asv_ks_dict = make_per_asv_ks_dict()

    attractor_afd_dict = make_attractor_afd_dict()

    fig = plt.figure(figsize = (12, 8))
    fig.subplots_adjust(bottom= 0.15)

    x_label = 'Rescaled ' + r'$\mathrm{log}_{10}$' + ' relative abundance'

    for experiment_idx, experiment in enumerate(experiments):

        ax = plt.subplot2grid((2, 3), (0, experiment_idx), colspan=1)

        for transfer in transfers:

            colors_experiment_transfer = utils.color_dict_range[experiment][transfer-1]
            #afd = afd_dict[experiment][transfer]['afd_rescaled_occupancy_one']
            afd = afd_dict[experiment][transfer]['afd_rescaled_occupancy_one']

            #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
            #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
            ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

            #for asv in per_asv_ks_dict[experiment]['per_species_ks'].keys():

            #    afd_rescaled = per_asv_ks_dict[experiment]['per_species_ks'][asv]['afd_rescaled'][transfer]
            #    ax.hist(afd_rescaled, lw=3, alpha=0.8, bins= 8, color=colors_experiment_transfer, histtype='step', label='Transfer %d' % transfer,  density=True)
        

        if experiment == ('No_migration', 4):

            #ax_attractor = inset_axes(ax, width="40%", height="40%", loc=2)
            ax_attractor = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0,0.5,0.4,0.4), bbox_transform=ax.transAxes, loc='upper left')
            ax_attractor.tick_params(labelleft=False, labelbottom=True)
            ax_attractor.xaxis.set_tick_params(labelsize=6)
            #axi.tick_params(labelleft=False, labelbottom=False)
            ax_attractor.set_title("Major attractor\n(Alcaligenaceae)", fontsize=8, fontweight='bold')

            for transfer in transfers:

                afd = attractor_afd_dict['merged']['Alcaligenaceae'][transfer]['afd_rescaled_occupancy_one'] 
                ax_attractor.hist(afd, lw=2, alpha=0.8, bins= 15, color=utils.color_dict_range[experiment][transfer-1], histtype='step', density=True)


        #ks_statistic = ks_dict[experiment][rescaled_status]['D']
        #p_value = ks_dict[experiment][rescaled_status]['pvalue']
        #ax.text(0.70,0.7, '$D=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        #ax.text(0.70,0.7, '$\mathrm{KS}=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        #ax.text(0.68,0.62, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

        ax.set_title(utils.titles_no_inocula_dict[experiment], fontsize=12, fontweight='bold' )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)

        ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


    for treatment_idx, treatment in enumerate(treatments_no_innoculum):

        label = '%s_afd' % treatment
        #simulation_all_migration_fixed_parameters_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_dict(label)
        simulation_all_migration_fixed_parameters_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_afd_dict(label)

        #ks_simulated = np.asarray(simulation_all_migration_fixed_parameters_dict[rescaled_status_sim][treatment])
        ks_simulated = np.sort(simulation_all_migration_fixed_parameters_dict['mean_over_asv_ks_rescaled_12_vs_18'][treatment])
        #ks_observed = ks_dict[experiments[treatment_idx]][rescaled_status]['D']
        ks_observed = per_asv_ks_dict[experiments[treatment_idx ]]['mean_over_asv_ks_rescaled_12_vs_18']

        tau_best = simulation_all_migration_fixed_parameters_dict['tau_all'][0]
        sigma_best = simulation_all_migration_fixed_parameters_dict['sigma_all'][0]

        ax = plt.subplot2grid((2, 3), (1, treatment_idx), colspan=1)

        #p_value = sum(ks_simulated > ks_observed)/len(ks_simulated)

        lower_ci = ks_simulated[int(0.025*len(ks_simulated))]
        upper_ci = ks_simulated[int(0.975*len(ks_simulated))]


        ax.hist(ks_simulated, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiments[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
        #ax.axvline(x=0, ls=':', lw=3, c='k', label='Null')
        #ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$D$')
        ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$\mathrm{KS}$')

        #ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$D$')
        #ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$\mathrm{KS}$')

        ax.axvline(x=lower_ci, ls=':', lw=3, c='k', label='95% CIs ')
        ax.axvline(x=upper_ci, ls=':', lw=3, c='k')

        #ax.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)
        #ax.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)
        ax.set_xlabel('Predicted mean ' + r'$\mathrm{KS}$' + ' from optimal\nparameters, ' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)
        ax.set_ylabel('Probability density',  fontsize=11)
        ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[3 + treatment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


        if treatment_idx == 0:
            ax.legend(loc="upper right", fontsize=8)


        # inset axis for dominant attractor
        if treatment == 'no_migration':

            #ax_attractor = inset_axes(ax, width="40%", height="40%", loc=2)
            ax_attractor = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.55,0.1,0.4,0.4), bbox_transform=ax.transAxes, loc='upper right')
            ax_attractor.tick_params(labelleft=False, labelbottom=True)
            ax_attractor.xaxis.set_tick_params(labelsize=6)


            label_attractor = '%s_afd_attractor' % treatment
            #slm_simulation_utils.run_simulation_all_migration_afd_abc(n_iter=1000, tau=tau_all[np.argmin(euc_dist_attractor)], sigma=sigma_all[np.argmin(euc_dist_attractor)], label=label_attractor)
            slm_dict_attractor = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_afd_dict(label_attractor)

            ks_simulated_attractor = np.sort(slm_dict_attractor['mean_over_asv_ks_rescaled_12_vs_18'][treatment])
            ks_observed_attractor = attractor_afd_dict['per_asv']['Alcaligenaceae']['ks_stats']['mean_ks_over_rescaled_afds']

            lower_ci_attractor = ks_simulated_attractor[int(0.025*len(ks_simulated_attractor))]
            upper_ci_attractor = ks_simulated_attractor[int(0.975*len(ks_simulated_attractor))]

            ax_attractor.hist(ks_simulated_attractor, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiments[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
            ax_attractor.axvline(x=lower_ci_attractor, ls=':', lw=3, c='k')
            ax_attractor.axvline(x=upper_ci_attractor, ls=':', lw=3, c='k')
            ax_attractor.axvline(x=ks_observed_attractor, ls='--', lw=3, c='k')
            ax_attractor.set_title("Major attractor\n(Alcaligenaceae)", fontsize=8, fontweight='bold')




    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(utils.directory + "/figs/afd_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    #fig.savefig(utils.directory + "/figs/afd_migration.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



if __name__=='__main__':


    #make_per_asv_ks_dict()
    #run_best_parameter_simulations()
    make_plot()

    
    #print(attractor_afd_dict)

    #per_asv_ks_dict = make_per_asv_ks_dict()



