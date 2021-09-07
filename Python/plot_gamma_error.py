from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

from itertools import combinations
import statsmodels.stats.multitest as multitest

transfers = [12, 18]

mae_dict = {}

for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

    mae_dict[migration_innoculum] = {}

    for transfer in transfers:

        mae_dict[migration_innoculum][transfer] = {}

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0], inocula=migration_innoculum[1])
        ESVs = np.asarray(ESVs)
        occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)

        occupancies_no_nan = occupancies[np.logical_not(np.isnan(predicted_occupancies))]
        predicted_occupancies_no_nan = predicted_occupancies[np.logical_not(np.isnan(predicted_occupancies))]
        ESVs_no_nan = ESVs[np.logical_not(np.isnan(predicted_occupancies))]

        mae = np.mean(np.absolute(occupancies_no_nan - predicted_occupancies_no_nan) / occupancies_no_nan)

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        mean_rel_abundances = np.mean(rel_s_by_s, axis=1)
        mean_rel_abundances_no_nan = mean_rel_abundances[np.logical_not(np.isnan(predicted_occupancies))]

        mae_dict[migration_innoculum][transfer]['All'] = {}
        mae_dict[migration_innoculum][transfer]['All']['MAE'] = mae
        mae_dict[migration_innoculum][transfer]['All']['occupancies'] = occupancies_no_nan
        mae_dict[migration_innoculum][transfer]['All']['predicted_occupancies'] = predicted_occupancies_no_nan
        mae_dict[migration_innoculum][transfer]['All']['species'] = ESVs_no_nan
        mae_dict[migration_innoculum][transfer]['All']['mean_relative_abundances'] = mean_rel_abundances_no_nan




migration_innocula = [('No_migration',4), ('Parent_migration',4)]


afd_dict = {}

afd_dict_merged_attractors = {}

for migration_innoculum in migration_innocula:

    attractor_dict = utils.get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])

    for attractor_idx, attractor in enumerate(attractor_dict.keys()):

        for transfer in transfers:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])

            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            s_by_s_attractor = s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            s_by_s_attractor = s_by_s_attractor[attractor_species_idx]


            occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s_attractor)

            occupancies_no_nan = occupancies[np.logical_not(np.isnan(predicted_occupancies))]
            predicted_occupancies_no_nan = predicted_occupancies[np.logical_not(np.isnan(predicted_occupancies))]
            attractor_species_no_nan = attractor_species[np.logical_not(np.isnan(predicted_occupancies))]

            mae = np.mean(np.absolute(occupancies_no_nan - predicted_occupancies_no_nan) / occupancies_no_nan)

            rel_s_by_s_attractor = (s_by_s_attractor/s_by_s_attractor.sum(axis=0))
            mean_rel_abundances = np.mean(rel_s_by_s_attractor, axis=1)
            mean_rel_abundances_no_nan = mean_rel_abundances[np.logical_not(np.isnan(predicted_occupancies))]

            mae_dict[migration_innoculum][transfer][attractor] = {}
            mae_dict[migration_innoculum][transfer][attractor]['MAE'] = mae
            mae_dict[migration_innoculum][transfer][attractor]['occupancies'] = occupancies_no_nan
            mae_dict[migration_innoculum][transfer][attractor]['predicted_occupancies'] = predicted_occupancies_no_nan
            mae_dict[migration_innoculum][transfer][attractor]['species'] = attractor_species_no_nan
            mae_dict[migration_innoculum][transfer][attractor]['mean_relative_abundances'] = mean_rel_abundances_no_nan




fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)
ax_migration = plt.subplot2grid((1, 2), (0,0))
ax_attractors = plt.subplot2grid((1, 2), (0,1))

x_tick_labels = []
x_tick_idxs = []

ax_migration_count = 0
for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

    mae_all = [mae_dict[migration_innoculum][transfer]['All']['MAE'] for transfer in transfers]

    x = []
    maes = []

    for transfer in transfers:

        mae = mae_dict[migration_innoculum][transfer]['All']['MAE']

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]

        ax_migration.scatter(ax_migration_count, mae, alpha=1, s=60, zorder=2, c=[color_], edgecolors='k')


        x.append(ax_migration_count)
        maes.append(mae)

        x_tick_labels.append(transfer)
        x_tick_idxs.append(ax_migration_count)

        ax_migration_count += 1

    ax_migration.plot(x, maes, markersize = 8, c='k', linewidth=1.5, alpha=1, zorder=1)

    if migration_innoculum_idx < 3:
        ax_migration.axvline(ax_migration_count-0.5, lw=1.5, ls=':',color='k', zorder=1)

    ax_migration.text((migration_innoculum_idx/len(utils.migration_innocula))+0.12, -0.12, utils.titles_new_line_dict[migration_innoculum], fontsize=5.8, color='k', ha='center', va='center', transform=ax_migration.transAxes )



ax_migration.set_xlim([-0.3, max(x_tick_idxs)+0.3])
ax_migration.set_xticks(x_tick_idxs)
ax_migration.set_xticklabels(x_tick_labels, fontsize=8)

ax_migration.set_ylabel('Mean relative error of the gamma', fontsize=12)
#ax_migration.text(0.5, -0.1, "Transfer", fontweight='bold', fontsize=14, color='k', ha='center', va='center', transform=ax.transAxes )
ax_migration.text(0.5, -0.2, "Transfer", fontsize=14, color='k', ha='center', va='center', transform=ax_migration.transAxes )



attractors = list(attractor_dict.keys())
attractors.insert(0, 'All')

ax_attractors_count = 0
x_tick_labels_attractor = []
x_tick_idxs_attractor = []
legend_offsets = [0.25, 0.28]
for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

    for transfer_idx, transfer in enumerate(transfers):

        x_tick_labels_attractor.append('Transfer %d' % transfer)
        x_tick_idxs_attractor.append(ax_attractors_count+1)

        for attractor in attractors:

            mae = mae_dict[migration_innoculum][transfer][attractor]['MAE']

            color = utils.get_color_attractor(attractor, transfer)

            ax_attractors.scatter(ax_attractors_count, mae, alpha=1, s=60, zorder=2, c=[color], edgecolors='k')

            ax_attractors_count += 1

        if transfer_idx < len(transfers)-1:

            ax_attractors.axvline(ax_attractors_count-0.5, lw=1.5, ls=':',color='k', zorder=1)


    if migration_innoculum_idx == 0:
        ax_attractors.axvline(ax_attractors_count-0.5, lw=1.5, ls='-',color='k', zorder=1)

    ax_attractors.text((migration_innoculum_idx/len(migration_innocula)) + legend_offsets[migration_innoculum_idx], -0.15, utils.titles_new_line_dict[migration_innoculum], fontsize=11, color='k', ha='center', va='center', transform=ax_attractors.transAxes)




ax_attractors.set_ylabel('Mean relative error of the gamma', fontsize=12)

ax_attractors.set_xlim([-0.5, ax_attractors_count-0.5])

ax_attractors.set_xticks(x_tick_idxs_attractor)
ax_attractors.set_xticklabels(x_tick_labels_attractor, fontsize=7)





legend_elements = [Line2D([0], [0], marker='o', color='none', markerfacecolor=utils.get_color_attractor('Alcaligenaceae', 12), label='Alcaligenaceae', markeredgecolor='k', markersize=6),
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=utils.get_color_attractor('Pseudomonadaceae', 12), label='Pseudomonadaceae', markeredgecolor='k', markersize=6),
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=utils.get_color_attractor('All', 12), label='Both attractors', markeredgecolor='k', markersize=6)]


ax_attractors.legend(handles=legend_elements, loc='upper left', prop={'size': 5})




#ax_attractors.set_xticks([])

fig.subplots_adjust(wspace=0.3, hspace=0.15)
fig.savefig(utils.directory + "/figs/gamma_error.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()




def plot_observed_vs_predicted_occupancy():

    fig = plt.figure(figsize = (6, 6)) #
    #fig.subplots_adjust()

    ax_1 = plt.subplot2grid((2, 2), (0,0))
    ax_2 = plt.subplot2grid((2, 2), (0,1))
    ax_3 = plt.subplot2grid((2, 2), (1,0))
    ax_4 = plt.subplot2grid((2, 2), (1,1))

    ax_all = [ax_1, ax_2, ax_3, ax_4]
    ax_count = 0
    for migration_innoculum in migration_innocula:

        attractor_dict = utils.get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])
        for transfer in transfers:

            ax = ax_all[ax_count]

            ax.plot([0.01,1],[0.01,1], lw=2,ls='--',c='k',zorder=1)
            ax.set_xscale('log', basex=10)
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('Observed occupancy', fontsize=9)
            ax.set_ylabel('Predicted occupancy', fontsize=9)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            ax.tick_params(axis='both', which='major', labelsize=6)

            ax.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=9)


            ax_count += 1
            for attractor in attractors:

                if attractor == 'All':
                    continue

                occupancies = mae_dict[migration_innoculum][transfer][attractor]['occupancies']
                predicted_occupancies = mae_dict[migration_innoculum][transfer][attractor]['predicted_occupancies']
                species = mae_dict[migration_innoculum][transfer][attractor]['species']

                color = utils.get_color_attractor(attractor, transfer)
                ax.scatter(occupancies, predicted_occupancies, alpha=0.8, s=15, zorder=2, c=[color], label=attractor, linewidth=0.8, edgecolors='k')


            ax.legend(loc="upper left", fontsize=6)

    fig.subplots_adjust(wspace=0.45, hspace=0.4)
    fig.savefig(utils.directory + "/figs/gamma_predicted_vs_observed_occupancies.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def plot_error_vs_mean_abundance():
    # plot error vs difference in relative abundances

    fig = plt.figure(figsize = (4, 4)) #
    fig.subplots_adjust(bottom= 0.15)

    ax_1 = plt.subplot2grid((2, 2), (0,0))
    ax_2 = plt.subplot2grid((2, 2), (0,1))
    ax_3 = plt.subplot2grid((2, 2), (1,0))
    ax_4 = plt.subplot2grid((2, 2), (1,1))

    ax_all = [ax_1, ax_2, ax_3, ax_4]
    ax_count = 0
    for migration_innoculum in migration_innocula:

        attractor_dict = utils.get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])
        for transfer in transfers:

            occupancies = mae_dict[migration_innoculum][transfer]['All']['occupancies']
            predicted_occupancies = mae_dict[migration_innoculum][transfer]['All']['predicted_occupancies']
            species = mae_dict[migration_innoculum][transfer]['All']['species']

            relative_errors = np.absolute(occupancies - predicted_occupancies) / occupancies


            mean_relative_abundance_pseudomonadaceae = mae_dict[migration_innoculum][transfer]['Pseudomonadaceae']['mean_relative_abundances']
            species_pseudomonadaceae = mae_dict[migration_innoculum][transfer]['Pseudomonadaceae']['species']

            mean_relative_abundance_alcaligenaceae = mae_dict[migration_innoculum][transfer]['Alcaligenaceae']['mean_relative_abundances']
            species_alcaligenaceae = mae_dict[migration_innoculum][transfer]['Alcaligenaceae']['species']

            delta_mean_rel_abund = []
            mre_to_plot = []

            for s_idx, s in enumerate(species):

                if (s in species_pseudomonadaceae) and (s in species_alcaligenaceae):

                    s_pseudo_rel_abund = mean_relative_abundance_pseudomonadaceae[np.where(species_pseudomonadaceae == s)[0][0]]
                    s_alcal_rel_abund = mean_relative_abundance_alcaligenaceae[np.where(species_alcaligenaceae == s)[0][0]]

                    delta_mean_rel_abund.append(np.absolute(s_pseudo_rel_abund-s_alcal_rel_abund))

                    mre_to_plot.append(relative_errors[s_idx])

            ax = ax_all[ax_count]

            ax.scatter(delta_mean_rel_abund, mre_to_plot, s=12, alpha=0.7)

            ax.set_xscale('log', basex=10)
            ax.set_yscale('log', basey=10)

            ax.tick_params(axis='both', which='minor', labelsize=6)
            ax.tick_params(axis='both', which='major', labelsize=6)

            ax.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=8)


            #ax.set_xlabel('Absolute difference in relative abundance between attractors', fontsize=9)
            #ax.set_ylabel('Relative error, gamma', fontsize=9)


            ax_count += 1

    fig.text(0, 0.05, "Absolute difference in mean relative abundance between attractors", va='center',  fontsize=8)
    fig.text(0, 0.55, "Relative error, gamma", va='center', rotation='vertical', fontsize=11)


    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(utils.directory + "/figs/gamma_error_vs_mean_abundance.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



#plot_observed_vs_predicted_occupancy()


plot_error_vs_mean_abundance()
