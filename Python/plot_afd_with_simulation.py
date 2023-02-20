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

experiments = [('No_migration',4), ('Global_migration',4), ('Parent_migration', 4)  ]
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

            print(experiment, rescaled_status)

            ks_statistic, p_value = utils.run_permutation_ks_test(afd_dict[experiment][transfers[0]][rescaled_status], afd_dict[experiment][transfers[1]][rescaled_status], n=1000)

            distances_dict[experiment][rescaled_status] = {}
            distances_dict[experiment][rescaled_status]['D'] = ks_statistic
            distances_dict[experiment][rescaled_status]['pvalue'] = p_value


    with open(ks_dict_path, 'wb') as outfile:
        pickle.dump(distances_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_ks_dict():

    dict_ = pickle.load(open(ks_dict_path, "rb"))
    return dict_


#make_ks_dict()

ks_dict = load_ks_dict()

fig = plt.figure(figsize = (12, 10))
fig.subplots_adjust(bottom= 0.15)





rescaled_status = 'afd_rescaled'

x_label = 'Rescaled ' + r'$\mathrm{log}_{10}$' + ' relative abundance'

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((3, 3), (0, experiment_idx), colspan=1)

    for transfer in transfers:

        colors_experiment_transfer = utils.color_dict_range[experiment][transfer-1]
        afd = afd_dict[experiment][transfer][rescaled_status]
        #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)


    ks_statistic = ks_dict[experiment][rescaled_status]['D']
    p_value = ks_dict[experiment][rescaled_status]['pvalue']

    ax.text(0.70,0.7, '$D=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.68,0.62, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)




# plot simulation
simulation_dict = slm_simulation_utils.load_simulation_all_migration_dict()
tau_all = np.asarray(list(simulation_dict.keys()))
sigma_all = np.asarray(list(simulation_dict[tau_all[0]].keys()))

x_axis = sigma_all
y_axis = tau_all
x_axis_log10 = np.log10(x_axis)

for treatment_idx, treatment in enumerate(['no_migration', 'global_migration', 'parent_migration']):

    observed = ks_dict[experiment][rescaled_status]['D']

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


    ax_ks = plt.subplot2grid((3, 3), (1, treatment_idx), colspan=1)
    ax_ks_error = plt.subplot2grid((3, 3), (2, treatment_idx), colspan=1)

    delta_range = max([observed  - np.amin(ks_rescaled_12_vs_18_all),  np.amax(ks_rescaled_12_vs_18_all) - observed])
    pcm_slope = ax_ks.pcolor(x_axis_log10, y_axis, ks_rescaled_12_vs_18_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=observed-delta_range, vcenter=observed, vmax=observed+delta_range))
    #fmt = lambda x, pos: '{:.1%}'.format(x)
    clb_slope = plt.colorbar(pcm_slope, ax=ax_ks)
    clb_slope.set_label(label='Disance between AFDs, ' + r'$D$' , fontsize=9)
    ax_ks.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 10)
    ax_ks.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 10)
    ax_ks.xaxis.set_major_formatter(plot_utils.fake_log)
    # Set observed marking and label
    clb_slope.ax.axhline(y=observed, c='k')
    original_ticks = list(clb_slope.get_ticks())
    clb_slope.set_ticks(original_ticks + [observed])
    clb_slope.set_ticklabels(original_ticks + ['Obs.'])

    # Set observed marking and label
    clb_slope .ax.axhline(y=observed, c='k')
    clb_slope.set_ticks([0.025, 0.075, 0.15, 0.2])
    clb_slope.set_ticklabels(['0.025', '0.075', '0.15', '0.20'])
    original_ticks = list(clb_slope.get_ticks())
    clb_slope.set_ticks(original_ticks + [observed])
    clb_slope.set_ticklabels(original_ticks + ['Obs.'])

    print(observed+delta_range)
    


    # slope error
    pcm_slope_error = ax_ks_error.pcolor(x_axis_log10, y_axis, ks_rescaled_12_vs_18_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(ks_rescaled_12_vs_18_error_all), vcenter=np.median(np.ndarray.flatten(ks_rescaled_12_vs_18_error_all)), vmax=np.amax(ks_rescaled_12_vs_18_error_all)))
    clb_slope_error = plt.colorbar(pcm_slope_error, ax=ax_ks_error)
    clb_slope_error.set_label(label='Relative error of ' + r'$D$'  + ' from simulated data', fontsize=9)
    ax_ks_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 11)
    ax_ks_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 11)
    ax_ks_error.xaxis.set_major_formatter(plot_utils.fake_log)








fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_with_simulation.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
