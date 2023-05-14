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
import slm_simulation_utils
import plot_utils


#afd_migration_transfer_12

ks_dict_path = "%s/data/afd_ks_dict.pickle" %  utils.directory

afd_dict = {}
transfers = np.asarray([12, 18])

experiments = [('No_migration',4), ('Parent_migration', 4), ('Global_migration',4) ]
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


fig = plt.figure(figsize = (12, 8.5))
fig.subplots_adjust(bottom= 0.15)

rescaled_status = 'afd_rescaled'

x_label = 'Rescaled ' + r'$\mathrm{log}_{10}$' + ' relative abundance'



# plot simulation
simulation_dict = slm_simulation_utils.load_simulation_all_migration_dict()
tau_all = np.asarray(list(simulation_dict.keys()))
sigma_all = np.asarray(list(simulation_dict[tau_all[0]].keys()))

x_axis = sigma_all
y_axis = tau_all
x_axis_log10 = np.log10(x_axis)

for treatment_idx, treatment in enumerate(['no_migration', 'parent_migration', 'global_migration' ]):

    

    observed = ks_dict[(treatment.capitalize(), 4)][rescaled_status]['D']

    print(observed)

    ks_rescaled_12_vs_18_all = []
    ks_rescaled_12_vs_18_error_all = []

    for tau in tau_all:

        ks_rescaled_12_vs_18 = []
        ks_rescaled_12_vs_18_error = []

        for sigma in sigma_all:

            ks_rescaled_12_vs_18_i = np.asarray(simulation_dict[tau][sigma]['ks_rescaled_12_vs_18'][treatment])

            mean_ks_rescaled_12_vs_18 = np.mean(ks_rescaled_12_vs_18_i)

            mean_error_ks_rescaled_12_vs_18 = np.mean(np.absolute((ks_rescaled_12_vs_18_i - observed ) / observed ))

            ks_rescaled_12_vs_18.append(mean_ks_rescaled_12_vs_18)
            ks_rescaled_12_vs_18_error.append(mean_error_ks_rescaled_12_vs_18)

        ks_rescaled_12_vs_18_all.append(ks_rescaled_12_vs_18)
        ks_rescaled_12_vs_18_error_all.append(ks_rescaled_12_vs_18_error)

    ks_rescaled_12_vs_18_all = np.asarray(ks_rescaled_12_vs_18_all)
    ks_rescaled_12_vs_18_error_all = np.asarray(ks_rescaled_12_vs_18_error_all)


    ax_ks = plt.subplot2grid((2, 3), (0, treatment_idx), colspan=1)
    ax_ks_error = plt.subplot2grid((2, 3), (1, treatment_idx), colspan=1)

    delta_range = max([observed  - np.amin(ks_rescaled_12_vs_18_all),  np.amax(ks_rescaled_12_vs_18_all) - observed])
    pcm_slope = ax_ks.pcolor(x_axis_log10, y_axis, ks_rescaled_12_vs_18_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=observed-delta_range, vcenter=observed, vmax=observed+delta_range))
    #fmt = lambda x, pos: '{:.1%}'.format(x)
    clb_slope = plt.colorbar(pcm_slope, ax=ax_ks)
    #clb_slope.set_label(label='KS distance between AFDs, ' + r'$D$' , fontsize=9)
    clb_slope.set_label(label='KS distance between AFDs' , fontsize=9)
    ax_ks.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
    ax_ks.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
    ax_ks.xaxis.set_major_formatter(plot_utils.fake_log)
    # Set observed marking and label
    clb_slope.ax.axhline(y=observed, c='k')
    #original_ticks = list(clb_slope.get_ticks())
    #clb_slope.set_ticks(original_ticks + [observed])
    #clb_slope.set_ticklabels(original_ticks + ['Obs.'])

    # Set observed marking and label
    clb_slope .ax.axhline(y=observed, c='k')
    #clb_slope.set_ticks([0.025, 0.055, 0.085, 0.115])
    #clb_slope.set_ticklabels(['0.025', '0.055', '0.085', '0.115'])
    original_ticks = list(clb_slope.get_ticks())
    original_ticks = [round(k, 2) for k in original_ticks]

    if treatment_idx == 0:
        original_ticks.remove(0.06)


    clb_slope.set_ticks(original_ticks + [observed])
    clb_slope.set_ticklabels(original_ticks + ['Obs.'])

    ax_ks.set_title(utils.titles_str_no_inocula_dict[treatment], fontsize=14)
    ax_ks.text(-0.1, 1.04, plot_utils.sub_plot_labels[treatment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks.transAxes)


    
    # slope error
    pcm_slope_error = ax_ks_error.pcolor(x_axis_log10, y_axis, ks_rescaled_12_vs_18_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(ks_rescaled_12_vs_18_error_all), vcenter=np.median(np.ndarray.flatten(ks_rescaled_12_vs_18_error_all)), vmax=np.amax(ks_rescaled_12_vs_18_error_all)))
    clb_slope_error = plt.colorbar(pcm_slope_error, ax=ax_ks_error)
    #clb_slope_error.set_label(label='Relative error of ' + r'$D$'  + ' from simulated data', fontsize=9)
    clb_slope_error.set_label(label='Relative error of ' + r'$\mathrm{KS}$'  + ' from simulated data', fontsize=9)

    ax_ks_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 11)
    ax_ks_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 11)
    ax_ks_error.xaxis.set_major_formatter(plot_utils.fake_log)

    ax_ks_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[treatment_idx + 3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ks_error.transAxes)





fig.text(0.37, 0.95, "AFD simulations", va='center', fontsize=25)


fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_migration_heatmap.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
