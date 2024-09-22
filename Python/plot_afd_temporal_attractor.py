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



#afd_dict = {}
#transfers = np.asarray([12, 18])

#experiments = [('No_migration',4), ('Global_migration',4), ('Parent_migration', 4)  ]

#for experiment in experiments:

#    afd_dict[experiment] = {}

#    for transfer in transfers:

#        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

#        afd = relative_s_by_s.flatten()
#        afd = afd[afd>0]
#        afd = np.log10(afd)

#        afd_dict[experiment][transfer] = afd




#experiments = [('No_migration',4)]#, ('Parent_migration', 4)  ]



#print(n_Alcaligenaceae, n_Pseudomonadaceae)

def make_attractor_dict():

    experiments = [('No_migration',4)]

    #afd_dict[experiment][transfer]['afd_rescaled_occupancy_one']
    attractor_afd_dict = {}
    attractor_afd_dict['merged'] = {}
    attractor_afd_dict['per_asv'] = {}

    attractor_dict = utils.get_attractor_status(migration='No_migration', inocula=4)
    n_Alcaligenaceae = len(attractor_dict['Alcaligenaceae'])
    n_Pseudomonadaceae = len(attractor_dict['Pseudomonadaceae'])

    afd_dict_merged_attractors = {}

    for transfer in [12, 18]:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration='No_migration',inocula=4)
        #relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        #afd = relative_s_by_s.flatten()
        #afd = afd[afd>0]
        #afd = np.log10(afd)

        afd, rescaled_afd = utils.get_flat_rescaled_afd(s_by_s)

        afd_dict_merged_attractors[transfer] = rescaled_afd


    for attractor_idx, attractor in enumerate(attractor_dict.keys()):

        attractor_afd_dict['merged'][attractor] = {}

        for transfer in [12, 18]:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration='No_migration',inocula=4)
            relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(relative_s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            relative_s_by_s_attractor = relative_s_by_s_attractor[attractor_species_idx]

            afd, rescaled_afd = utils.get_flat_rescaled_afd(relative_s_by_s_attractor)
            afd_occupancy_one, rescaled_afd_occupancy_one = utils.get_flat_rescaled_afd(relative_s_by_s_attractor, min_occupancy=1)


            attractor_afd_dict['merged'][attractor][transfer]['afd'] = afd
            attractor_afd_dict['merged'][attractor][transfer]['afd_rescaled'] = rescaled_afd

            attractor_afd_dict['merged'][attractor][transfer]['afd_occupancy_one'] = afd_occupancy_one
            attractor_afd_dict['merged'][attractor][transfer]['afd_rescaled_occupancy_one'] = rescaled_afd_occupancy_one




    for attractor_idx, attractor in enumerate(attractor_dict.keys()):

        attractor_afd_dict['per_asv'][attractor] = {}
        attractor_afd_dict['per_asv'][attractor]['asv'] = {}


        for transfer in [12, 18]:

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration='Parent_migration',inocula=4)
            relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))
            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]

            asv_to_keep_idx = (np.sum(relative_s_by_s_attractor>0, axis=1) == relative_s_by_s_attractor.shape[1])

            species = np.asarray(species)
            asv_to_keep = species[asv_to_keep_idx]
            relative_s_by_s_attractor_to_keep = relative_s_by_s_attractor[asv_to_keep_idx,:]

            for asv_idx, asv in enumerate(asv_to_keep):
                if asv not in attractor_afd_dict['per_asv'][attractor]['asv']:
                    attractor_afd_dict['per_asv'][attractor]['asv'][asv] = {}

                attractor_afd_dict['per_asv'][attractor]['asv'][asv][transfer] = relative_s_by_s_attractor_to_keep[asv_idx,:]

        asv_attractor = list(attractor_afd_dict['per_asv'][attractor]['asv'].keys())

        ks_stat_all = []
        for asv in asv_attractor:
            
            # occupancy of one at both timepoints
            if len(attractor_afd_dict['per_asv'][attractor]['asv'][asv]) != 2:
                continue

            afd_12 = attractor_afd_dict['per_asv'][attractor]['asv'][asv][12]
            afd_18 = attractor_afd_dict['per_asv'][attractor]['asv'][asv][18]

            afd_log10_12 = np.log10(afd_12)
            afd_log10_18 = np.log10(afd_18)

            rescaled_afd_log10_12 = (afd_log10_12 - np.mean(afd_log10_12))/np.std(afd_log10_12)
            rescaled_afd_log10_18 = (afd_log10_18 - np.mean(afd_log10_18))/np.std(afd_log10_18)


            D, pvalue = stats.ks_2samp(rescaled_afd_log10_12, rescaled_afd_log10_18)
            scaled_d = D*np.sqrt(len(rescaled_afd_log10_12)*len(rescaled_afd_log10_18)/ (len(rescaled_afd_log10_12) + len(rescaled_afd_log10_18)))

            ks_stat_all.append(scaled_d)

            # mean scaled ks 
            # Alcaligenaceae 0.7786122309834377
            # Pseudomonadaceae 0.554024732920621


        attractor_afd_dict['per_asv'][attractor]['ks_stats'] = {}
        attractor_afd_dict['per_asv'][attractor]['ks_stats']['mean_ks_over_rescald_afds'] = np.mean(ks_stat_all)








fig = plt.figure(figsize = (10, 4))
fig.subplots_adjust(bottom= 0.15)



distances_dict = {}
combo_pairs = []
pvalues = []
treatment_combinations = list(combinations(list(attractor_dict.keys()),2))
for combo in treatment_combinations:

    if combo not in distances_dict:
        distances_dict[combo] = {}

    for transfer in transfers:

        afd_experiment_1 = afd_dict[combo[0]][transfer]
        afd_experiment_2 = afd_dict[combo[1]][transfer]

        #print(afd_experiment_1)
        #print(afd_experiment_2)

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


        afd_attractor = afd_dict[attractor][transfer]

        afd_merged_attractors = afd_dict_merged_attractors[transfer]

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



for experiment_idx, experiment in enumerate(list(attractor_dict.keys())):

    ax = plt.subplot2grid((1, len(list(attractor_dict.keys()))+1), (0, experiment_idx), colspan=1)

    for transfer in transfers:

        colors_experiment_transfer = utils.get_color_attractor(experiment, transfer)


        afd = afd_dict[experiment][transfer]
        label = '%s, transfer %d' %(utils.attractor_latex_dict[experiment], transfer)

        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

    KS_statistic, p_value = stats.ks_2samp(afd_dict[experiment][transfers[0]], afd_dict[experiment][transfers[1]])

    ax.text(0.70,0.8, '$D=%0.3f$' % KS_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.68,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.set_title(utils.attractor_latex_dict[experiment], fontsize=12, fontweight='bold' )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel('Relative abundance, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    #ax.set_xlim(-5.5, 0.2)


#ax.set_xscale('log', basex=10)

ax_distances = plt.subplot2grid((1, len(attractor_dict.keys())+1), (0, 2), colspan=1)
#marker_dict = {(('No_migration', 4), ('Global_migration', 4)): 'D',
#                                    (('No_migration', 4), ('Parent_migration', 4)): 'X',
#                                    (('Global_migration', 4), ('Parent_migration', 4)): 'o'}

#label_dict = {(('No_migration', 4), ('Global_migration', 4)): 'No migration vs. Global migration',
#            (('No_migration', 4), ('Parent_migration', 4)): 'No migration vs. Parent migration',
#            (('Global_migration', 4), ('Parent_migration', 4)): 'Global migration vs. Parent migration'}



for combo in treatment_combinations:

    distances_combo = [distances_dict[combo][transfer]['D'] for transfer in transfers]
    pvalues_combo = [distances_dict[combo][transfer]['pvalue_bh'] for transfer in transfers]

    #colors_experiment_transfer = utils.color_dict[experiment][transfer-1]

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

        #ax_distances.plot(transfer, distances_combo_transfer, alpha=1, edgecolors='k', marker=marker_dict[combo], s = 120, label=label_dict[combo], zorder=2)

        ax_distances.plot(transfer, distances_combo_transfer, markersize = 16,   \
            linewidth=2,  alpha=1, zorder=3, fillstyle='left', **marker_style)

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

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Merged attractors',
                          markerfacecolor='k', markersize=12)]


ax_distances.legend(handles=legend_elements, loc='upper right')


#plt.text(0.5,1.1,'No migration', fontsize=12, color='k', ha='center', va='center')
fig.suptitle('No migration', x =0.52, fontsize=14, fontweight='bold')



fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#  bbox_inches = "tight",
fig.savefig(utils.directory + "/figs/afd_temporal_attractor.png", format='png',pad_inches = 0.5, dpi = 600)
#fig.savefig(utils.directory + "/figs/afd_temporal_attractor.pdf", format='pdf',pad_inches = 0.5, dpi = 600)

plt.close()
