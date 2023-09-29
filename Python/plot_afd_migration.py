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


from itertools import combinations
#import statsmodels.stats.multitest as multitest
import plot_utils
import slm_simulation_utils


#afd_migration_transfer_12

ks_dict_path = "%s/data/afd_ks_dict.pickle" %  utils.directory

afd_dict = {}
transfers = np.asarray([12, 18])

experiments = [('No_migration',4), ('Parent_migration', 4) , ('Global_migration',4) ]
treatment_combinations = list(combinations(experiments,2))

rescaled_status_all = ['afd', 'afd_rescaled']

for experiment in experiments:

    afd_dict[experiment] = {}

    for transfer in transfers:

        #relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])
        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=experiment[0],inocula=experiment[1])
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        afd = rel_s_by_s.flatten()
        afd = afd[afd>0]
        afd = np.log10(afd)

        afd_rescaled = (afd - np.mean(afd))/np.std(afd)

        afd_dict[experiment][transfer] = {}
        afd_dict[experiment][transfer]['afd'] = afd
        afd_dict[experiment][transfer]['afd_rescaled'] = afd_rescaled



#
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
    communities_to_keep = np.intersect1d( s_by_s_dict[12]['comm_rep_list'], s_by_s_dict[18]['comm_rep_list'] )
    
    paired_rel_abundances_all = []
    for community in communities_to_keep:

        community_12_idx = np.where( s_by_s_dict[12]['comm_rep_list'] == community)[0][0]
        community_18_idx = np.where( s_by_s_dict[18]['comm_rep_list'] == community)[0][0]

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








def make_ks_dict():

    distances_dict = {}

    for combo in treatment_combinations:

        if combo not in distances_dict:
            distances_dict[combo] = {}

        for transfer in transfers:

            distances_dict[combo][transfer] = {}

            for rescaled_status in rescaled_status_all:

                print(combo, transfer, rescaled_status)

                afd_experiment_1 = afd_dict[combo[0]][transfer][rescaled_status]
                afd_experiment_2 = afd_dict[combo[1]][transfer][rescaled_status]

                ks_statistic, p_value = utils.run_permutation_ks_test(afd_experiment_1, afd_experiment_2, n=1000)

                distances_dict[combo][transfer][rescaled_status] = {}
                distances_dict[combo][transfer][rescaled_status]['D'] = ks_statistic
                distances_dict[combo][transfer][rescaled_status]['pvalue'] = p_value



    for experiment_idx, experiment in enumerate(experiments):

        distances_dict[experiment] = {}

        for rescaled_status in rescaled_status_all:

            #print(experiment, rescaled_status)

            ks_statistic, p_value = utils.run_permutation_ks_test(afd_dict[experiment][transfers[0]][rescaled_status], afd_dict[experiment][transfers[1]][rescaled_status], n=1000)

            distances_dict[experiment][rescaled_status] = {}
            distances_dict[experiment][rescaled_status]['D'] = ks_statistic
            distances_dict[experiment][rescaled_status]['pvalue'] = p_value



        ks_statistic, p_value = utils.run_permutation_ks_test_control(afd_paired_dict[experiment])

        distances_dict[experiment]['afd_rescaled_and_paired'] = {}
        distances_dict[experiment]['afd_rescaled_and_paired']['D'] = ks_statistic
        distances_dict[experiment]['afd_rescaled_and_paired']['pvalue'] = p_value


    with open(ks_dict_path, 'wb') as outfile:
        pickle.dump(distances_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_ks_dict():

    dict_ = pickle.load(open(ks_dict_path, "rb"))
    return dict_




#make_ks_dict()

ks_dict = load_ks_dict()

fig = plt.figure(figsize = (12, 8))
fig.subplots_adjust(bottom= 0.15)


rescaled_status = 'afd_rescaled'

x_label = 'Rescaled ' + r'$\mathrm{log}_{10}$' + ' relative abundance'

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((2, 3), (0, experiment_idx), colspan=1)

    for transfer in transfers:

        colors_experiment_transfer = utils.color_dict_range[experiment][transfer-1]
        afd = afd_dict[experiment][transfer][rescaled_status]
        #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)


    ks_statistic = ks_dict[experiment][rescaled_status]['D']
    p_value = ks_dict[experiment][rescaled_status]['pvalue']

    #ax.text(0.70,0.7, '$D=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.70,0.7, '$\mathrm{KS}=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.text(0.68,0.62, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.set_title(utils.titles_no_inocula_dict[experiment], fontsize=12, fontweight='bold' )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)



treatments_no_innoculum = ['no_migration', 'parent_migration', 'global_migration']

def run_best_parameter_simulations():

    # plot simulation
    simulation_dict = slm_simulation_utils.load_simulation_all_migration_abc_dict()

    tau_all = np.asarray(simulation_dict['tau_all'])
    sigma_all = np.asarray(simulation_dict['sigma_all'])

    for treatment_idx, treatment in enumerate(treatments_no_innoculum):

        ks_rescaled_12_vs_18_simulation = np.asarray(simulation_dict['ks_rescaled_12_vs_18'][treatment])

        ks_rescaled_12_vs_18 =  ks_dict[experiments[treatment_idx]][rescaled_status]['D']  

        euc_dist = np.sqrt((ks_rescaled_12_vs_18 - ks_rescaled_12_vs_18_simulation)**2)
        min_parameter_idx = np.argmin(euc_dist)

        tau_best = tau_all[min_parameter_idx]
        sigma_best = sigma_all[min_parameter_idx]

        label = '%s_afd' % treatment
        slm_simulation_utils.run_simulation_all_migration_fixed_parameters(tau_best, sigma_best, label, n_iter=1000)





for treatment_idx, treatment in enumerate(treatments_no_innoculum):

    label = '%s_afd' % treatment
    simulation_all_migration_fixed_parameters_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_dict(label)

    ks_simulated = np.asarray(simulation_all_migration_fixed_parameters_dict['ks_12_vs_18'][treatment])
    ks_observed = ks_dict[experiments[treatment_idx]]['afd_rescaled']['D']

    tau_best = simulation_all_migration_fixed_parameters_dict['tau_all']
    sigma_best = simulation_all_migration_fixed_parameters_dict['sigma_all']

    ax = plt.subplot2grid((2, 3), (1, treatment_idx), colspan=1)

    p_value = sum(ks_simulated > ks_observed)/len(ks_simulated)

    print(np.median(ks_simulated), p_value)


    ax.hist(ks_simulated, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiments[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
    #ax.axvline(x=0, ls=':', lw=3, c='k', label='Null')
    #ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$D$')
    ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$\mathrm{KS}$')

    #ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$D$')
    ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$\mathrm{KS}$')

    #ax.set_xlabel('Simulated ' + r'$D$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)
    ax.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)

    ax.set_ylabel('Probability density',  fontsize=11)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[3 + treatment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


    if treatment_idx == 0:
        ax.legend(loc="upper right", fontsize=8)





fig.subplots_adjust(wspace=0.35, hspace=0.3)
#fig.savefig(utils.directory + "/figs/afd_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/afd_migration.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
