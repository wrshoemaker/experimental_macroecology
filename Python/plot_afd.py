from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm

from itertools import combinations
import statsmodels.stats.multitest as multitest


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

fig = plt.figure(figsize = (4*(len(experiments) +1), 8))
fig.subplots_adjust(bottom= 0.15)



#combo_pairs = []
#pvalues = []

        #sys.stdout.write("Transfer %d, %s vs. %s, D = %g, P= %g\n" % (transfer, combo[0][0], combo[1][0], D, pvalue))

#key_pavalues = [(key, distances_dict[key]['pvalue']) for key in distances_dict.keys()]
#pvalues = [x[1] for x in key_pavalues]
#reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(pvalues, alpha=0.05, method='fdr_bh')
#for combo_pair_idx, combo_pair in enumerate(combo_pairs):
#    distances_dict[combo_pair[0]][combo_pair[1]]['pvalue_bh'] = pvals_corrected[combo_pair_idx]



for rescaled_status_idx, rescaled_status in enumerate(rescaled_status_all):

    if rescaled_status == 'afd':
        x_label = 'Relative abundance, ' +  r'$\mathrm{log}_{10}$'

    else:
        x_label = 'Rescaled relative abundance, ' +  r'$\mathrm{log}_{10}$'

    for experiment_idx, experiment in enumerate(experiments):

        ax = plt.subplot2grid((2, len(experiments)+1), (rescaled_status_idx, experiment_idx), colspan=1)

        for transfer in transfers:

            colors_experiment_transfer = utils.color_dict_range[experiment][transfer-1]
            afd = afd_dict[experiment][transfer][rescaled_status]
            #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
            label = '%s, transfer %d' %(utils.titles_dict[experiment], transfer)

            ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)


        ks_statistic = ks_dict[experiment][rescaled_status]['D']
        p_value = ks_dict[experiment][rescaled_status]['pvalue']

        ax.text(0.70,0.8, '$D=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        ax.text(0.68,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

        ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)
        #ax.set_xlim(-5.5, 0.2)




marker_dict = {(('No_migration', 4), ('Global_migration', 4)): 'D',
                                    (('No_migration', 4), ('Parent_migration', 4)): 'X',
                                    (('Global_migration', 4), ('Parent_migration', 4)): 'o'}

label_dict = {(('No_migration', 4), ('Global_migration', 4)): 'No migration vs. Global migration',
            (('No_migration', 4), ('Parent_migration', 4)): 'No migration vs. Parent migration',
            (('Global_migration', 4), ('Parent_migration', 4)): 'Global migration vs. Parent migration'}


for rescaled_status_idx, rescaled_status in enumerate(rescaled_status_all):

    ax_distances = plt.subplot2grid((2, len(experiments)+1), (rescaled_status_idx, 3), colspan=1)

    for combo in treatment_combinations:

        distances_combo = [ks_dict[combo][transfer][rescaled_status]['D'] for transfer in transfers]
        pvalues_combo = [ks_dict[combo][transfer][rescaled_status]['pvalue'] for transfer in transfers]

        ax_distances.plot(transfers, distances_combo, color = 'k', zorder=1)


    for combo in treatment_combinations:

        for transfer in transfers:

            ks_statistic = ks_dict[combo][transfer][rescaled_status]['D']
            p_value = ks_dict[combo][transfer][rescaled_status]['pvalue']

            colors_experiment_transfer_1 = utils.color_dict_range[combo[0]][transfer-3]
            colors_experiment_transfer_2 = utils.color_dict_range[combo[1]][transfer-3]

            if p_value < 0.05:

                ax_distances.text(transfer, ks_statistic+0.025, '*', fontsize=12, color='k', ha='center', va='center')#, transform=ax.transAxes )


            marker_style = dict(color='k', marker='o',
                                markerfacecoloralt=colors_experiment_transfer_1,
                                markerfacecolor=colors_experiment_transfer_2)

            #ax_distances.plot(transfer, distances_combo_transfer, alpha=1, edgecolors='k', marker=marker_dict[combo], s = 120, label=label_dict[combo], zorder=2)

            ax_distances.plot(transfer, ks_statistic, markersize = 16, linewidth=2,  alpha=1, zorder=3, fillstyle='left', **marker_style)

    #transfers


    #legend_elements = [Line2D([0], [0], color='k', ls='--', lw=1.5, label='Mean ' + r'$\left | \beta_{1} -1 \right |$'),
    #                    Line2D([0], [0], color='k', ls=':', lw=1.5, label='Null')]
    #ax_distances.legend(handles=legend_elements, loc='upper left')
    #ax_distances.legend(loc="upper right", fontsize=6, markerscale=0.5)

    ax_distances.set_xlabel('Transfers' , fontsize=12)
    ax_distances.set_ylabel('Kolmogorov–Smirnov distance, '+ r'$D$', fontsize=12)

    ax_distances.set_xlim([11, 19])
    ax_distances.set_ylim([-0.02, 0.35 ])
    ax_distances.axhline(0, lw=3, ls=':',color='k', zorder=1)

    labels = [item.get_text() for item in ax_distances.get_xticklabels()]

    ax_distances.set_xticks([12, 18])
    ax_distances.set_xticklabels([12,18])


#print(list(ax_distances.get_xticklabels()))

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_temporal.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
