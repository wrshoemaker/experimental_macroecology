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
import slm_simulation_utils


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors



n=10000


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)

experiments = [('No_migration', 4), ('Global_migration', 4)]

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((1, 2), (0, experiment_idx))

    #communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
    #communities_keep = [str(key) for key, value in communities.items() if len(value) == 18]
    dist_array_all = []
    for transfer in [12, 18]:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        dist_array = spatial.distance.pdist(rel_s_by_s.T, metric='braycurtis')
        dist_array_all.append(dist_array)

        # plot
        color = utils.color_dict_range[experiment][transfer-1]
        ax.hist(dist_array, histtype='step', lw=2, alpha=0.9, bins=15, density=True, color=color, label='Transfer %d' % transfer)
        ax.set_xlabel("Bray-Curtis dissimilarity", fontsize = 12)
        ax.set_ylabel("Probability density", fontsize = 12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(utils.titles_no_inocula_dict[experiment], fontsize=14)

    cv_12 = np.std(dist_array_all[0])/np.mean(dist_array_all[0])
    cv_18 = np.std(dist_array_all[1])/np.mean(dist_array_all[1])

    delta_cv = cv_18 - cv_12

    def delta_cv_permutation():

        dist_array_merged = np.concatenate([dist_array_all[0], dist_array_all[1]])
        delta_cv_null_all = []

        for n_i in range(n):

            np.random.shuffle(dist_array_merged)

            array_12_null = dist_array_merged[:len(dist_array_all[0])]
            array_18_null = dist_array_merged[len(dist_array_all[0]):]

            cv_12_null = np.std(array_12_null)/np.mean(array_12_null)
            cv_18_null = np.std(array_18_null)/np.mean(array_18_null)

            delta_cv_null = cv_18_null - cv_12_null
            delta_cv_null_all.append(delta_cv_null)


        delta_cv_null_all = np.asarray(delta_cv_null_all)
        p_value = sum(delta_cv_null_all>delta_cv)/n

        print(experiment, delta_cv, p_value)



fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig_name = utils.directory + '/figs/dissimilarity.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


# make plot!!!



# distances between transfer 12 and 18
#for experiment_idx, experiment in enumerate(experiments):

#    communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
#    communities_keep = [str(key) for key, value in communities.items() if (12 in value) and (18 in value)]

#    species_nested = []
#    s_by_s_nested = []
#    comm_rep_list_nested = []
#    for transfer in [12, 18]:

#        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities=communities_keep)
#        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

#        species_nested.append(species)
#        s_by_s_nested.append(s_by_s)
#        comm_rep_list_nested.append(comm_rep_list)

#    species_intersect = np.intersect1d(species_nested[0], species_nested[1])
#    species_12_idx = np.asarray([species_nested[0].index(s) for s in species_intersect])
#    species_18_idx = np.asarray([species_nested[1].index(s) for s in species_intersect])

#    comm_rep_intersect = np.intersect1d(comm_rep_list_nested[0], comm_rep_list_nested[1])

    #comm_rep_list_12 = np.asarray([comm_rep_list_nested[0].index(s) for s in comm_rep_intersect])
    #comm_rep_list_18 = np.asarray([comm_rep_list_nested[1].index(s) for s in comm_rep_intersect])

#    sad_all = []
#    distances_all = []
#    for comm_rep in comm_rep_intersect:

#        comm_rep_12_idx = comm_rep_list_nested[0].index(comm_rep)
#        comm_rep_18_idx = comm_rep_list_nested[1].index(comm_rep)

#        sad_12 = s_by_s_nested[0][species_12_idx,comm_rep_12_idx]
#        sad_18 = s_by_s_nested[1][species_18_idx,comm_rep_18_idx]

#        distances_all.append(spatial.distance.braycurtis(sad_12, sad_18))

#        sad_all.append([sad_12, sad_18])

#    cv = np.std(distances_all)/np.mean(distances_all)
