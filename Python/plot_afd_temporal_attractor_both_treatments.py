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





#experiments = [('No_migration',4)]#, ('Parent_migration', 4)  ]

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

            afd_dict[experiment][attractor][transfer] = afd






fig = plt.figure(figsize = (12, 8))
fig.subplots_adjust(bottom= 0.15)

for experiment_idx, experiment in enumerate(experiments):

    distances_dict = {}
    combo_pairs = []
    pvalues = []
    treatment_combinations = list(combinations(list(attractor_dict.keys()),2))
    for combo in treatment_combinations:

        if combo not in distances_dict:
            distances_dict[combo] = {}

        for transfer in transfers:

            afd_experiment_1 = afd_dict[experiment][combo[0]][transfer]
            afd_experiment_2 = afd_dict[experiment][combo[1]][transfer]

            D, pvalue = stats.ks_2samp(afd_experiment_1, afd_experiment_2)

            distances_dict[combo][transfer] = {}
            distances_dict[combo][transfer]['D'] = D
            distances_dict[combo][transfer]['pvalue'] = pvalue


            combo_pairs.append((combo, transfer))
            pvalues.append(pvalue)

            #sys.stdout.write("Transfer %d, %s vs. %s, D = %g, P= %g\n" % (transfer, combo[0][0], combo[1][0], D, pvalue))


    for attractor in attractor_dict.keys():

        distances_dict[(attractor, 'All')] = {}

        treatment_combinations.append((attractor, 'All'))

        for transfer in transfers:

            afd_merged_attractors = distances_dict[(attractor, 'All')][transfer] = {}


            afd_attractor = afd_dict[experiment][attractor][transfer]

            afd_merged_attractors = afd_dict_merged_attractors[experiment][transfer]

            D, pvalue = stats.ks_2samp(afd_attractor, afd_merged_attractors)


            afd_merged_attractors = distances_dict[(attractor, 'All')][transfer]['D'] = D
            afd_merged_attractors = distances_dict[(attractor, 'All')][transfer]['pvalue'] = pvalue

            combo_pairs.append( ((attractor, 'All'), transfer) )
            pvalues.append(pvalue)




    #key_pavalues = [(key, distances_dict[key]['pvalue']) for key in distances_dict.keys()]
    #pvalues = [x[1] for x in key_pavalues]
    reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(pvalues, alpha=0.05, method='fdr_bh')
    for combo_pair_idx, combo_pair in enumerate(combo_pairs):
        distances_dict[combo_pair[0]][combo_pair[1]]['pvalue_bh'] = pvals_corrected[combo_pair_idx]



    for attractor_idx, attractor in enumerate(list(attractor_dict.keys())):

        ax = plt.subplot2grid((2, len(list(attractor_dict.keys()))+1), (experiment_idx, attractor_idx), colspan=1)

        for transfer in transfers:

            colors_experiment_transfer = utils.get_color_attractor(attractor, transfer)

            afd = afd_dict[experiment][attractor][transfer]
            label = '%s, transfer %d' %(utils.attractor_latex_dict[attractor], transfer)

            ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

        KS_statistic, p_value = stats.ks_2samp(afd_dict[experiment][attractor][transfers[0]], afd_dict[experiment][attractor][transfers[1]])

        ax.text(0.70,0.8, '$D=%0.3f$' % KS_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
        ax.text(0.68,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

        ax.set_title(utils.attractor_latex_dict[attractor], fontsize=12, fontweight='bold' )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel('Relative abundance, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)
        ax.set_xlim(-5.5, 0.2)


        if attractor_idx == 1:

            ax.text(0.5, 1.14, utils.titles_dict[experiment], fontweight='bold', fontsize=14, color='k', ha='center', va='center', transform=ax.transAxes )





    ax_distances = plt.subplot2grid((2, len(attractor_dict.keys())+1), (experiment_idx, 2), colspan=1)


    for combo in treatment_combinations:

        distances_combo = [distances_dict[combo][transfer]['D'] for transfer in transfers]
        pvalues_combo = [distances_dict[combo][transfer]['pvalue_bh'] for transfer in transfers]

        ax_distances.plot(transfers, distances_combo, color = 'k', zorder=1)

    for combo in treatment_combinations:

        for transfer in transfers:

            distances_combo_transfer = distances_dict[combo][transfer]['D']
            pvalues_combo_transfer = distances_dict[combo][transfer]['pvalue_bh']

            colors_experiment_transfer_1 = utils.get_color_attractor(combo[0], transfer)

            colors_experiment_transfer_2 = utils.get_color_attractor(combo[1], transfer)

            if pvalues_combo_transfer < 0.05:

                ax_distances.text(transfer, distances_combo_transfer+0.025, '*', fontsize=12, color='k', ha='center', va='center')#, transform=ax.transAxes )


            marker_style = dict(color='k', marker='o',
                                markerfacecoloralt=colors_experiment_transfer_1,
                                markerfacecolor=colors_experiment_transfer_2)


            ax_distances.plot(transfer, distances_combo_transfer, markersize = 16,   \
                linewidth=2,  alpha=1, zorder=3, fillstyle='left', **marker_style)



    ax_distances.set_xlabel('Transfers' , fontsize=12)
    ax_distances.set_ylabel('Kolmogorov–Smirnov distance, '+ r'$D$', fontsize=12)

    ax_distances.set_xlim([11, 19])
    ax_distances.set_ylim([-0.02, 0.35 ])
    ax_distances.axhline(0, lw=3, ls=':',color='k', zorder=1)

    labels = [item.get_text() for item in ax_distances.get_xticklabels()]

    ax_distances.set_xticks([12, 18])
    ax_distances.set_xticklabels([12,18])

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Merged attractors',
                              markerfacecolor='k', markersize=12)]


    ax_distances.legend(handles=legend_elements, loc='upper right')


#plt.text(0.5,1.1,'No migration', fontsize=12, color='k', ha='center', va='center')
#fig.suptitle('No migration', x =0.52, fontsize=14, fontweight='bold')



fig.subplots_adjust(wspace=0.5, hspace=1.2)
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.tight_layout()
#  bbox_inches = "tight",
fig.savefig(utils.directory + "/figs/afd_temporal_attractor_both_treatments.pdf", format='pdf', pad_inches = 0.5, dpi = 600)
plt.close()




# test whether the KS distance b/w AFDs of different treatments with the same migration
# is greater than the KS distance of different attractors in the same treatment

def run_ks_difference_test(n_iter = 10000):

    attractors = list(attractor_dict.keys())
    for transfer in transfers:

        for experiment_idx, experiment in enumerate(experiments):

            for attractor in attractors:

                afd_experiment_1 = afd_dict[experiments[0]][attractor][transfer]
                afd_experiment_2 = afd_dict[experiments[1]][attractor][transfer]

                afd_attractor_1 = afd_dict[experiment][attractors[0]][transfer]
                afd_attractor_2 = afd_dict[experiment][attractors[1]][transfer]


                ks_statistic_experiment, p_value_experiment = stats.ks_2samp(afd_experiment_1, afd_experiment_2)
                ks_statistic_attractor, p_value_attractor = stats.ks_2samp(afd_attractor_1, afd_attractor_2)


                delta_ks = ks_statistic_experiment - ks_statistic_attractor
                delta_ks_abs = np.absolute(delta_ks)

                afd_merged = np.concatenate([afd_experiment_1, afd_experiment_2, afd_attractor_1, afd_attractor_2])
                delta_ks_abs_null = []

                for n_i in range(n_iter):

                    np.random.shuffle(afd_merged)

                    afd_experiment_1_null = afd_merged[:len(afd_experiment_1)]
                    afd_experiment_2_null = afd_merged[len(afd_experiment_1):len(afd_experiment_1)+len(afd_experiment_2)]

                    afd_atractor_1_null = afd_merged[len(afd_experiment_1)+len(afd_experiment_2):len(afd_experiment_1)+len(afd_experiment_2)+len(afd_attractor_1)]
                    afd_atractor_2_null = afd_merged[len(afd_experiment_1)+len(afd_experiment_2)+len(afd_attractor_1):]

                    ks_statistic_experiment_null, p_value_experiment_null = stats.ks_2samp(afd_experiment_1_null, afd_experiment_2_null)
                    ks_statistic_attractor_null, p_value_attractor_null = stats.ks_2samp(afd_atractor_1_null, afd_atractor_2_null)

                    delta_ks_abs_null_i = np.absolute(ks_statistic_experiment_null - ks_statistic_attractor_null)

                    delta_ks_abs_null.append(delta_ks_abs_null_i)


                delta_ks_abs_null = np.asarray(delta_ks_abs_null)

                p_ks_abs = (sum(delta_ks_abs_null>delta_ks_abs)+1) / (n_iter+1)

                print(experiment, attractor, transfer, delta_ks, p_ks_abs)


#run_ks_difference_test(n_iter = 10000)
