from __future__ import division
import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from scipy.stats import gamma
import scipy.spatial as spatial

#from macroecotools import obs_pred_rsquare
import utils

from itertools import combinations


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors




T_range = list(range(16))




experiments = [('No_migration', 4), ('Global_migration', 4)]


def calculate_dissimilarity(afd_temporal, T):

    diss_all = []
    # number of dissimilarities decreases with lag
    for t in range(len(afd_temporal) - T):

        n_t = afd_temporal[t]
        n_t_plus_T = afd_temporal[t + T]

        d_plus = n_t + n_t_plus_T
        d_minus = n_t - n_t_plus_T

        diss = ((d_minus**2) - d_plus)/(d_plus * (d_plus-1))
        diss_all.append(diss)

    mean_diss = np.mean(diss_all)

    return mean_diss




def calculate_dissimilarity_between_communities(afd_temporal_1, afd_temporal_2, T):

    diss_all = []
    # number of dissimilarities decreases with lag
    for t in range(len(afd_temporal_1) - T):

        n_1_t = afd_temporal_1[t]
        n_2_t_plus_T = afd_temporal_2[t + T]

        d_plus = n_1_t + n_2_t_plus_T
        d_minus = n_1_t - n_2_t_plus_T

        diss = ((d_minus**2) - d_plus)/(d_plus * (d_plus-1))
        diss_all.append(diss)

    #print(diss_all)
    mean_diss = np.mean(diss_all)

    return mean_diss




diss_dict = {}
# make each community into a time-by-species matrix
for experiment_idx, experiment in enumerate(experiments):

    communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
    communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]

    afd_dict = {}
    #species_all = []
    for transfer in range(1, 19):

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities= communities_keep)
        #species_all.append(species)

        for comm_rep_idx, comm_rep in enumerate(comm_rep_list):

            if comm_rep not in afd_dict:
                afd_dict[comm_rep] = {}

            for s_idx, s in enumerate(species):

                if s not in afd_dict[comm_rep]:
                    afd_dict[comm_rep][s] = {}
                    afd_dict[comm_rep][s]['transfer'] = []
                    afd_dict[comm_rep][s]['abundance'] = []

                afd_dict[comm_rep][s]['transfer'].append(transfer)
                afd_dict[comm_rep][s]['abundance'].append(s_by_s[s_idx,comm_rep_idx])

            #afd_dict[comm_rep].append(s_by_s[:,comm_rep_idx])


    #diss_dict[experiment[0]] = {}
    # go through each community to make matrix
    comm_merged = list(afd_dict.keys())
    for c in comm_merged:

        species_c = list(afd_dict[c].keys())

        for s in species_c:

            afd_temporal = afd_dict[c][s]['abundance']

            if len(afd_temporal) < 18:
                continue

            # remove if there's a zero
            afd_temporal = np.asarray(afd_temporal)
            if sum(afd_temporal==0) > 0:
                continue

            T_all = [calculate_dissimilarity(afd_temporal, t) for t in T_range]
            T_all = np.asarray(T_all)

            #if c not in diss_dict[experiment[0]]:
            #    diss_dict[experiment[0]][c] = {}

            #diss_dict[experiment[0]][c][s] = T_all

            if s not in diss_dict:
                diss_dict[s] = {}

            if experiment[0] not in diss_dict[s]:
                diss_dict[s][experiment[0]] = {}
                diss_dict[s][experiment[0]]['communities'] = {}
                diss_dict[s][experiment[0]]['communities_between'] = {}


            diss_dict[s][experiment[0]]['communities'][c] = {}
            diss_dict[s][experiment[0]]['communities'][c]['afd_temporal'] = afd_temporal
            diss_dict[s][experiment[0]]['communities'][c]['dissimilarity_within'] = T_all


    # go through species and get dissimilarity between replicate communities
    comm_pair_all = list(combinations(comm_merged, 2))
    for species in diss_dict.keys():

        if experiment[0] not in diss_dict[species]:
            continue

        communities_species = list(diss_dict[species][experiment[0]]['communities'].keys())

        if len(communities_species) == 1:
            continue

        communities_species_pair_all = list(combinations(communities_species, 2))
        for communities_species_pair in communities_species_pair_all:

            afd_temporal_1 = diss_dict[species][experiment[0]]['communities'][communities_species_pair[0]]['afd_temporal']
            afd_temporal_2 = diss_dict[species][experiment[0]]['communities'][communities_species_pair[1]]['afd_temporal']

            T_between_all = [calculate_dissimilarity_between_communities(afd_temporal_1, afd_temporal_2, t) for t in T_range]
            T_between_all = np.asarray(T_between_all)

            diss_dict[species][experiment[0]]['communities_between'][communities_species_pair] = T_between_all



# get species to plot
species_to_plot = []
for key, value in diss_dict.items():

    if len(value) < 2:
        continue

    #if (len(value['Global_migration']) > 5) and len(value['No_migration']) > 5:
    species_to_plot.append(key)

    #else:
    #    print(key)


print(species_to_plot)

#species_to_plot = species_to_plot[:4]
fig = plt.figure(figsize = (8, 16)) #
fig.subplots_adjust(bottom= 0.15)

color_global = utils.color_dict_range[('Global_migration', 4)][15]
color_no = utils.color_dict_range[('No_migration', 4)][15]

species_count = 0
species_chunk_all = [species_to_plot[x:x+2] for x in range(0, len(species_to_plot), 2)]
for species_chunk_idx, species_chunk in enumerate(species_chunk_all):
    for species_idx, species in enumerate(species_chunk):

        ax = plt.subplot2grid((4, 2), (species_chunk_idx, species_idx))

        dissimilarity_within_global_all = []
        for key, value in diss_dict[species]['Global_migration']['communities'].items():

            dissimilarity_within = value['dissimilarity_within']
            #ax.plot(T_range, dissimilarity_within, c=color_global, lw=1, alpha=0.4, zorder=1)
            dissimilarity_within_global_all.append(dissimilarity_within)


        # plot mean
        if len(dissimilarity_within_global_all) >= 3:
            ax.plot(T_range, np.mean(np.asarray(dissimilarity_within_global_all), axis=0), c=color_global, ls='--', lw=3, alpha=1, zorder=2, label='Global, within communities')


        dissimilarity_within_no_all = []
        for key, value in diss_dict[species]['No_migration']['communities'].items():

            dissimilarity_within = value['dissimilarity_within']
            #ax.plot(T_range, dissimilarity_within, c=color_no, lw=1, alpha=0.5, zorder=1)
            dissimilarity_within_no_all.append(dissimilarity_within)

        # plot mean
        if len(dissimilarity_within_no_all) >= 3:
            ax.plot(T_range, np.mean(np.asarray(dissimilarity_within_no_all), axis=0), c=color_no, ls='--', lw=3, alpha=1, zorder=2, label='No, within communities')


        # plot between communities
        dissimilarity_within_global_all = []
        for key, value in diss_dict[species]['Global_migration']['communities_between'].items():
            dissimilarity_within_global_all.append(value)


        # plot mean
        if len(dissimilarity_within_global_all) >= 3:
            ax.plot(T_range, np.mean(np.asarray(dissimilarity_within_global_all), axis=0), c=color_global, ls=':', lw=3, alpha=1, zorder=2, label='Global, across communities')


        dissimilarity_within_no_all = []
        for key, value in diss_dict[species]['No_migration']['communities_between'].items():
            dissimilarity_within_no_all.append(value)

        # plot mean
        if len(dissimilarity_within_no_all) >= 3:
            ax.plot(T_range, np.mean(np.asarray(dissimilarity_within_no_all), axis=0), c=color_no, ls=':', lw=3, alpha=1, zorder=2, label='No, across communities')


    
        print(species_count+1, species)


        ax.set_title('ASV %d' % (species_count+1), fontsize=12, fontweight='bold' )
        print('ASV %d' % (species_count+1), species)
        ax.set_xlabel("Obersvation lag, " + r'$T$', fontsize = 10)
        ax.set_ylabel("Mean dissimilarity, " + r'$\left< \Phi_{i}(T) \right>$', fontsize = 10)



        if species_count == 0:

            ax.legend(loc="upper left", fontsize=8)

            #from matplotlib.lines import Line2D
            #custom_lines = [Line2D([0], [0], color=color_global, lw=3, label='Global migration'),
            #                Line2D([0], [0], color=color_no, lw=3, label='No migration')]

            #ax.legend(handles=custom_lines, loc='upper left')

        species_count+=1

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig_name = utils.directory + '/figs/dissimilarity_temporal.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
