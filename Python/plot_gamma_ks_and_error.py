from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

import scipy.stats as stats
from scipy.stats import gamma, kstest

#from macroecotools import obs_pred_rsquare
import utils

from itertools import combinations
import statsmodels.stats.multitest as multitest



def get_mae_dict():

    mae_dict = {}

    for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

        mae_dict[migration_innoculum] = {}

        for transfer in utils.transfers:

            mae_dict[migration_innoculum][transfer] = {}

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0], inocula=migration_innoculum[1])
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            ESVs = np.asarray(ESVs)
            occupancies, predicted_occupancies, mad, beta, species_occupancies  = utils.predict_occupancy(s_by_s, ESVs)

            error = np.absolute(occupancies - predicted_occupancies)/occupancies

            #for species_occupancies_idx, species_occupancies in enumerate(species_occupancies):
            ks_all = []
            mean_all = []
            for afd in rel_s_by_s:

                mean_afd = np.mean(afd)
                std_afd = np.std(afd)

                beta_afd = (mean_afd**2)/(std_afd**2)
                beta_div_mean_afd = beta_afd/mean_afd

                ks, p_value = kstest(afd, 'gamma', args=(beta_afd, 0, 1/beta_div_mean_afd))

                mean_all.append(mean_afd)
                ks_all.append(ks)

            #occupancies_no_nan = occupancies[np.logical_not(np.isnan(predicted_occupancies))]
            #predicted_occupancies_no_nan = predicted_occupancies[np.logical_not(np.isnan(predicted_occupancies))]
            #ESVs_no_nan = ESVs[np.logical_not(np.isnan(predicted_occupancies))]

            #mae = np.mean(np.absolute(occupancies_no_nan - predicted_occupancies_no_nan) / occupancies_no_nan)

            #rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            #mean_rel_abundances = np.mean(rel_s_by_s, axis=1)
            #mean_rel_abundances_no_nan = mean_rel_abundances[np.logical_not(np.isnan(predicted_occupancies))]



            mae_dict[migration_innoculum][transfer]['All'] = {}
            mae_dict[migration_innoculum][transfer]['All']['mean_abs_error'] = np.mean(error)
            mae_dict[migration_innoculum][transfer]['All']['mean_log_abs_error'] = np.mean(np.log10(error[error>0]))
            mae_dict[migration_innoculum][transfer]['All']['occupancies'] = occupancies
            mae_dict[migration_innoculum][transfer]['All']['predicted_occupancies'] = predicted_occupancies
            mae_dict[migration_innoculum][transfer]['All']['species'] = species_occupancies
            mae_dict[migration_innoculum][transfer]['All']['mean_relative_abundances'] = mad
            mae_dict[migration_innoculum][transfer]['All']['beta'] = beta
            mae_dict[migration_innoculum][transfer]['All']['error_all'] = error

            mae_dict[migration_innoculum][transfer]['All']['ks_all'] = ks_all
            mae_dict[migration_innoculum][transfer]['All']['mean_for_ks_all'] = mean_all


    return mae_dict





mae_dict = get_mae_dict()


fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)

ax_error = plt.subplot2grid((2,2), (0,0))
ax_ks = plt.subplot2grid((2,2), (0,1))

ax_mean_vs_error = plt.subplot2grid((2,2), (1,0))
ax_mean_vs_ks = plt.subplot2grid((2,2), (1,1))


x_tick_labels = []
x_tick_idxs = []

ax_error_count = 0
for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

    x = []
    maes = []
    ks_all_to_format = []

    for transfer in utils.transfers:

        mae = mae_dict[migration_innoculum][transfer]['All']['mean_log_abs_error']
        #print(mae)
        #mae = 10**mae

        ks_all = mae_dict[migration_innoculum][transfer]['All']['ks_all']
        mean_for_ks_all = mae_dict[migration_innoculum][transfer]['All']['mean_for_ks_all']
        mean_ks = np.mean(ks_all)

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]

        print(migration_innoculum, transfer, mae)

        ax_error.scatter(ax_error_count, mae, alpha=1, s=60, zorder=2, c=[color_], edgecolors='k')
        ax_ks.scatter(ax_error_count, mean_ks, alpha=1, s=60, zorder=2, c=[color_], edgecolors='k')

        x.append(ax_error_count)
        maes.append(mae)
        ks_all_to_format.append(mean_ks)

        x_tick_labels.append(transfer)
        x_tick_idxs.append(ax_error_count)


        # plot mean abundance vs error
        mean_relative_abundances = mae_dict[migration_innoculum][transfer]['All']['mean_relative_abundances']
        error_all = mae_dict[migration_innoculum][transfer]['All']['error_all']
        beta_all = mae_dict[migration_innoculum][transfer]['All']['beta']

        ax_mean_vs_error.scatter(mean_relative_abundances, error_all, alpha=0.3, c=color_.reshape(1,-1), s=14, zorder=2)#, linewidth=0.8, edgecolors='k')
        ax_mean_vs_ks.scatter(mean_for_ks_all, ks_all, alpha=0.3, c=color_.reshape(1,-1), s=14, zorder=2)#, linewidth=0.8, edgecolors='k')

        ax_error_count += 1

    ax_error.plot(x, maes, markersize = 8, c='k', linewidth=1.5, alpha=1, zorder=1)
    ax_ks.plot(x, ks_all_to_format, markersize = 8, c='k', linewidth=1.5, alpha=1, zorder=1)

    if migration_innoculum_idx < 3:
        ax_error.axvline(ax_error_count-0.5, lw=1.5, ls=':',color='k', zorder=1)
        ax_ks.axvline(ax_error_count-0.5, lw=1.5, ls=':',color='k', zorder=1)

    ax_error.text((migration_innoculum_idx/len(utils.migration_innocula))+0.12, -0.12, utils.titles_new_line_dict[migration_innoculum], fontsize=5.8, color='k', ha='center', va='center', transform=ax_error.transAxes )
    ax_ks.text((migration_innoculum_idx/len(utils.migration_innocula))+0.12, -0.12, utils.titles_new_line_dict[migration_innoculum], fontsize=5.8, color='k', ha='center', va='center', transform=ax_ks.transAxes )



#ax_error.set_xlim([-0.3, max(x_tick_idxs)+0.3])
ax_error.set_xticks(x_tick_idxs)
ax_error.set_xticklabels(x_tick_labels, fontsize=8)

ax_error.set_ylabel('Mean log relative error of the gamma', fontsize=12)
#ax_migration.text(0.5, -0.1, "Transfer", fontweight='bold', fontsize=14, color='k', ha='center', va='center', transform=ax.transAxes )
ax_error.text(0.5, -0.2, "Transfer", fontsize=14, color='k', ha='center', va='center', transform=ax_error.transAxes )


ax_ks.set_xlim([-0.3, max(x_tick_idxs)+0.3])
ax_ks.set_xticks(x_tick_idxs)
ax_ks.set_xticklabels(x_tick_labels, fontsize=8)


ax_ks.set_ylabel('Mean KS statistic', fontsize=12)
#ax_migration.text(0.5, -0.1, "Transfer", fontweight='bold', fontsize=14, color='k', ha='center', va='center', transform=ax.transAxes )
ax_ks.text(0.5, -0.2, "Transfer", fontsize=14, color='k', ha='center', va='center', transform=ax_error.transAxes )





ax_mean_vs_error.set_xscale('log', basex=10)
ax_mean_vs_error.set_yscale('log', basey=10)

ax_mean_vs_ks.set_xscale('log', basex=10)

ax_mean_vs_error.set_xlabel('Mean relative abundance', fontsize=12)
ax_mean_vs_error.set_ylabel('Relative error of the gamma', fontsize=12)


ax_mean_vs_ks.set_xlabel('Mean relative abundance', fontsize=12)
ax_mean_vs_ks.set_ylabel('KS statistic', fontsize=12)





fig.subplots_adjust(wspace=0.3, hspace=0.15)
fig.savefig(utils.directory + "/figs/gamma_ks_and_error.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
