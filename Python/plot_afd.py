from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm

from itertools import combinations
import statsmodels.stats.multitest as multitest


#afd_migration_transfer_12

afd_dict = {}
transfers = np.asarray([12, 18])

experiments = [('No_migration',4), ('Global_migration',4), ('Parent_migration', 4)  ]

for experiment in experiments:

    afd_dict[experiment] = {}

    for transfer in transfers:

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

        afd = relative_s_by_s.flatten()
        afd = afd[afd>0]
        afd = np.log10(afd)

        afd_dict[experiment][transfer] = afd




def old_fig():

    fig = plt.figure(figsize = (4*len(experiments), 4))
    fig.subplots_adjust(bottom= 0.15)

    for transfer in transfers:

        for combo in combinations(experiments,2):
            afd_experiment_1 = afd_dict[combo[0]][transfer]
            afd_experiment_2 = afd_dict[combo[1]][transfer]
            KS_statistic, p_value = stats.ks_2samp(afd_experiment_1, afd_experiment_2)

            sys.stdout.write("Transfer %d, %s vs. %s, D = %g, P= %g\n" % (transfer, combo[0][0], combo[1][0], KS_statistic, p_value))




    for experiment_idx, experiment in enumerate(experiments):

        ax = plt.subplot2grid((1, len(experiments)), (0, experiment_idx), colspan=1)


        for transfer in transfers:

            colors_experiment_transfer = utils.color_dict[experiment][transfer-1]
            afd = afd_dict[experiment][transfer]
            label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)

            ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

        KS_statistic, p_value = stats.ks_2samp(afd_dict[experiment][transfers[0]], afd_dict[experiment][transfers[1]])

        ax.text(0.20,0.8, '$D=%0.3f$' % KS_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        ax.text(0.18,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )


        ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )
        ax.legend(loc="upper left", fontsize=8)

        ax.set_xlabel('Log relative abundance', fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)

        ax.set_xlim(-5.5, 0.2)

    #ax.set_xscale('log', basex=10)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/afd_temporal.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()









fig = plt.figure(figsize = (4*(len(experiments) +1), 4))
fig.subplots_adjust(bottom= 0.15)


distances_dict = {}
combo_pairs = []
pvalues = []
treatment_combinations = list(combinations(experiments,2))
for combo in treatment_combinations:

    if combo not in distances_dict:
        distances_dict[combo] = {}

    for transfer in transfers:

        afd_experiment_1 = afd_dict[combo[0]][transfer]
        afd_experiment_2 = afd_dict[combo[1]][transfer]
        D, pvalue = stats.ks_2samp(afd_experiment_1, afd_experiment_2)

        distances_dict[combo][transfer] = {}
        distances_dict[combo][transfer]['D'] = D
        distances_dict[combo][transfer]['pvalue'] = pvalue

        combo_pairs.append((combo, transfer))
        pvalues.append(pvalue)

        #sys.stdout.write("Transfer %d, %s vs. %s, D = %g, P= %g\n" % (transfer, combo[0][0], combo[1][0], D, pvalue))

#key_pavalues = [(key, distances_dict[key]['pvalue']) for key in distances_dict.keys()]
#pvalues = [x[1] for x in key_pavalues]
reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(pvalues, alpha=0.05, method='fdr_bh')
for combo_pair_idx, combo_pair in enumerate(combo_pairs):
    distances_dict[combo_pair[0]][combo_pair[1]]['pvalue_bh'] = pvals_corrected[combo_pair_idx]



for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((1, len(experiments)+1), (0, experiment_idx), colspan=1)

    for transfer in transfers:

        colors_experiment_transfer = utils.color_dict[experiment][transfer-1]
        afd = afd_dict[experiment][transfer]
        label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)

        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

    KS_statistic, p_value = stats.ks_2samp(afd_dict[experiment][transfers[0]], afd_dict[experiment][transfers[1]])

    ax.text(0.20,0.8, '$D=%0.3f$' % KS_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.18,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel('Relative abundance, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_xlim(-5.5, 0.2)

#ax.set_xscale('log', basex=10)

ax_distances = plt.subplot2grid((1, len(experiments)+1), (0, 3), colspan=1)
marker_dict = {(('No_migration', 4), ('Global_migration', 4)): 'D',
                                    (('No_migration', 4), ('Parent_migration', 4)): 'X',
                                    (('Global_migration', 4), ('Parent_migration', 4)): 'o'}

label_dict = {(('No_migration', 4), ('Global_migration', 4)): 'No migration vs. Global migration',
            (('No_migration', 4), ('Parent_migration', 4)): 'No migration vs. Parent migration',
            (('Global_migration', 4), ('Parent_migration', 4)): 'Global migration vs. Parent migration'}

for combo in treatment_combinations:

    distances_combo = [distances_dict[combo][transfer]['D'] for transfer in transfers]
    pvalues_combo = [distances_dict[combo][transfer]['pvalue_bh'] for transfer in transfers]

    ax_distances.plot(transfers, distances_combo, color = 'k', zorder=1)
    ax_distances.scatter(transfers, distances_combo, alpha=1, edgecolors='k', marker=marker_dict[combo], s = 120, label=label_dict[combo], zorder=2)


#transfers

ax_distances.legend(loc="upper right", fontsize=6, markerscale=0.5)
ax_distances.set_xlabel('Transfers' , fontsize=12)
ax_distances.set_ylabel('Kolmogorov–Smirnov distance, '+ r'$D$', fontsize=12)

ax_distances.set_xlim([11, 19])
ax_distances.set_ylim([-0.05, 0.55 ])
ax_distances.axhline(0, lw=3, ls=':',color='k', zorder=1)

labels = [item.get_text() for item in ax_distances.get_xticklabels()]

ax_distances.set_xticks([12, 18])
ax_distances.set_xticklabels([12,18])


#print(list(ax_distances.get_xticklabels()))

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_temporal.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
