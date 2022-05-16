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




experiments = [('No_migration',4), ('Parent_migration',4)]
transfers = [12, 18]

afd_dict = {}

afd_dict_merged_attractors = {}

for experiment in experiments:

    attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

    afd_dict_merged_attractors[experiment] = {}

    for transfer in transfers:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=experiment[0],inocula=experiment[1])
        relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        afd = relative_s_by_s.flatten()
        afd = afd[afd>0]
        afd = np.log10(afd)

        afd_dict_merged_attractors[experiment][transfer] = afd


    afd_dict[experiment] = {}
    for attractor_idx, attractor in enumerate(attractor_dict.keys()):

        afd_dict[experiment][attractor] = {}

        for transfer in transfers:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=experiment[0],inocula=experiment[1])
            relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))


            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(relative_s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            relative_s_by_s_attractor = relative_s_by_s_attractor[attractor_species_idx]


            afd = relative_s_by_s_attractor.flatten()
            afd = afd[afd>0]
            afd = np.log10(afd)

            # rescale
            #if len(afd) > 5:
            afd_dict[experiment][attractor][transfer] = (afd - np.mean(afd)) / np.std(afd)



fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(bottom= 0.15)




for experiment_idx, experiment in enumerate(experiments):

    for transfer_idx, transfer in enumerate(transfers):

        ax = plt.subplot2grid((2, 2), (experiment_idx, transfer_idx), colspan=1)

        attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

        for attractor_idx, attractor in enumerate(attractor_dict.keys()):

            afd = afd_dict[experiment][attractor][transfer]
            colors_experiment_transfer = utils.get_color_attractor(attractor, transfer)

            hist_f, bin_edges_f = np.histogram(afd, density=True, bins=20)
            bins_mean_f = [0.5 * (bin_edges_f[i] + bin_edges_f[i+1]) for i in range(0, len(bin_edges_f)-1 )]
            bins_mean_f = np.asarray(bins_mean_f)
            hist_f_to_plot = hist_f[hist_f>0]
            bins_mean_f_to_plot = bins_mean_f[hist_f>0]

            colors_experiment_transfer=np.array(colors_experiment_transfer).reshape(1,-1)

            ax.scatter(bins_mean_f_to_plot, hist_f_to_plot, s=30, alpha=0.9, c=colors_experiment_transfer)


        #ks_statistic_experiment, p_value_experiment = stats.ks_2samp(, afd_dict[experiment]['Pseudomonadaceae'][transfer])

        afd_1 = afd_dict[experiment]['Alcaligenaceae'][transfer]
        afd_2 = afd_dict[experiment]['Pseudomonadaceae'][transfer]

        D_null = []

        for b in range(1000):

            afd_null_1 = np.random.choice(afd_1, size=len(afd_1), replace=True)
            afd_null_2 = np.random.choice(afd_2, size=len(afd_2), replace=True)

            ks_statistic_experiment, p_value_experiment = stats.ks_2samp(afd_null_1, afd_null_2)
            D_null.append(ks_statistic_experiment)


        D_null = np.asarray(D_null)
        #p_value = sum(D_null==0)/10000

        #print(p_value)



        ax.text(0.70,0.8, '$D=%0.4f$' % np.mean(D_null), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        #ax.text(0.68,0.73, utils.get_p_value(p_value_experiment), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )


        ax.text(0.5, 1.06, '%s\ntransfer %d' % (utils.titles_dict[experiment], transfer), fontweight='bold', fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

        ax.set_xlabel('Rescaled log10 relative abundance', fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)

        ax.set_ylim([-0.02, 1])
            #s=30,



fig.subplots_adjust(wspace=0.5, hspace=0.4)
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.tight_layout()
#  bbox_inches = "tight",
fig.savefig(utils.directory + "/figs/afd_rescaled_temporal_attractor_both_treatments.png", format='png', pad_inches = 0.5, dpi = 600)
plt.close()
