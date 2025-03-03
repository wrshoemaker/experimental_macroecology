from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma
import scipy.special as special
#from macroecotools import obs_pred_rsquare
import utils
import tree_utils



tree = tree_utils.get_tree()


transfers = [12,18]
max_n_zeros = 3

distance_rho_dict = {}

# ensamble correlations...
for migration_innoculum in utils.migration_innocula:

    for transfer in transfers:

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        
        # sort esvs alphabetically so we know if we've entered one in the dictionary

        ESVs_sorted_idx = np.asarray([i[0] for i in sorted(enumerate(ESVs), key=lambda x:x[1])])
        ESVs = np.asarray(ESVs)
        ESVs_sorted = ESVs[ESVs_sorted_idx]
        s_by_s_sorted = s_by_s[ESVs_sorted_idx,:]

        # remove esvs with occupancies < 80%
        esv_to_keep_idx = np.sum((s_by_s_sorted > 0), axis=1) / len(comm_rep_list) > 0.8
        ESVs_sorted = ESVs_sorted[esv_to_keep_idx]
        s_by_s_sorted = s_by_s_sorted[esv_to_keep_idx,:]

        tree_subset = tree_utils.subset_tree(ESVs_sorted.tolist(), tree)

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        rho = np.corrcoef(rel_s_by_s)

        for i in range(len(ESVs_sorted)):
            for j in range(i):

                esv_pair = (ESVs_sorted[i], ESVs_sorted[j])
                if esv_pair not in distance_rho_dict:
                    distance_rho_dict[esv_pair] = {}
                    distance_rho_dict[esv_pair]['ensemble_rho'] = {}
                    distance_ij = tree_subset.get_distance(str(esv_pair[0]), str(esv_pair[1]))
                    distance_rho_dict[esv_pair]['distance'] = distance_ij
                    # calculate distance

                if migration_innoculum not in distance_rho_dict[esv_pair]['ensemble_rho']:
                    distance_rho_dict[esv_pair]['ensemble_rho'][migration_innoculum] = {}

                rho_ij = rho[i,j]
                distance_rho_dict[esv_pair]['ensemble_rho'][migration_innoculum][transfer] = rho_ij



# temporal correlations
count_dict = utils.get_otu_dict()
for treatment in ['No_migration.4.T%s', 'No_migration.40.T%s', 'Global_migration.4.T%s', 'Parent_migration.4.T%s']:

    #if ('No_migration.4.' in treatment) or ('Global_migration.4.' in treatment):
    #    transfers = utils.transfers_all

    if 'Parent_migration.NA.T0' in treatment:
        # empty string
        continue
    
    else:
        transfers = utils.transfers_all


    rel_abundance_dict = {}
    samples_transfers_dict = {}
    for transfer in transfers:

        treatment_transfer = treatment % str(transfer)
        count_dict_to_keep = {key: value for key, value in count_dict.items() if treatment_transfer in key}
        
        abundance_dict = {}
        n_samples = len(count_dict_to_keep.keys())
        for sample, asv_dict in count_dict_to_keep.items():

            replicate = sample.split('.')[-1]
            
            if replicate not in rel_abundance_dict:
                rel_abundance_dict[replicate] = {}

            if replicate not in samples_transfers_dict:
                samples_transfers_dict[replicate] = []
            samples_transfers_dict[replicate].append(transfer)

            N = sum(asv_dict.values())
            
            for asv, abundance in asv_dict.items():
                
                if asv not in rel_abundance_dict[replicate]:
                    rel_abundance_dict[replicate][asv] = {}

                rel_abundance_dict[replicate][asv][transfer] = abundance/N

    
    # identify samples to keep
    samples_to_keep = []
    for key, value in samples_transfers_dict.items():
        if len(set(value)) == 18:
            samples_to_keep.append(key)

    if len(samples_to_keep) == 0:
        continue

    treatment_split = treatment.split('.')
    treatment_tuple = (treatment_split[0], int(treatment_split[1]))

    for sample in samples_to_keep:

        rel_abundance_sample_dict = rel_abundance_dict[sample]
        esv = list(rel_abundance_sample_dict.keys())
        # alphabetical
        esv.sort()
        tree_subset = tree_utils.subset_tree(esv, tree)


        for i in range(len(esv)):
            
            esv_i = esv[i]
            afd_i = []

            for t in range(1, 19):
                    
                if t not in rel_abundance_sample_dict[esv_i]:
                    afd_i.append(0)
                else:
                    afd_i.append(rel_abundance_sample_dict[esv_i][t])
            
            if afd_i.count(0) <= max_n_zeros:

                for j in range(i):
                    
                    esv_j = esv[j]
                    afd_j = []
                    esv_pair = (esv_i, esv_j)

                    for t in range(1, 19):
    
                        if t not in rel_abundance_sample_dict[esv_j]:
                            afd_j.append(0)
                        else:
                            afd_j.append(rel_abundance_sample_dict[esv_j][t])

                    # skip AFDs with many zeros
                    if afd_j.count(0) <= max_n_zeros:

                        if esv_pair not in distance_rho_dict:
                            distance_rho_dict[esv_pair] = {}
                            distance_ij = tree_subset.get_distance(esv_pair[0], esv_pair[1])
                            distance_rho_dict[esv_pair]['distance'] = distance_ij

                        if 'temporal_rho' not in distance_rho_dict[esv_pair]:
                            distance_rho_dict[esv_pair]['temporal_rho'] = {}

                        if treatment_tuple not in distance_rho_dict[esv_pair]['temporal_rho']:
                            distance_rho_dict[esv_pair]['temporal_rho'][treatment_tuple] = {}

                        rho_ij = np.corrcoef(afd_i, afd_j)[0,1]

                        distance_rho_dict[esv_pair]['temporal_rho'][treatment_tuple][sample] = rho_ij



# add phylogenetic distance calculations

# plots 
# 

fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)     

ax_ensamble = plt.subplot2grid((1, 2), (0, 0))
ax_temporal = plt.subplot2grid((1, 2), (0, 1))

ensemble_distance = []
ensemble_rho = []
temporal_distance = []
temporal_rho = []

target_migration_innoculum = ('No_migration', 4)

for key, value in distance_rho_dict.items():

    distance = value['distance']

    if 'ensemble_rho' in value:
        
        migration_innoculum = value['ensemble_rho'].keys()

        for m in migration_innoculum:

            if m == target_migration_innoculum:

                ensemble_rho_m = list(value['ensemble_rho'][m].values())
                ensemble_distance.extend([distance] * len(ensemble_rho_m))
                ensemble_rho.extend(ensemble_rho_m)


    if 'temporal_rho' in value:

        migration_innoculum = value['temporal_rho'].keys()

        for m in migration_innoculum:

            if m == target_migration_innoculum:

                temporal_rho_m = list(value['temporal_rho'][m].values())

                temporal_distance.extend([distance] * len(temporal_rho_m))
                temporal_rho.extend(temporal_rho_m)



ensemble_distance = np.asarray(ensemble_distance)
ensemble_rho = np.asarray(ensemble_rho)
temporal_distance = np.asarray(temporal_distance)
temporal_rho = np.asarray(temporal_rho)


ax_ensamble.scatter(ensemble_distance, ensemble_rho, c='k', alpha=0.1, s=8)
ax_ensamble.set_xscale('log', basex=10)
ax_ensamble.set_xlabel('Phylogenetic distance', fontsize=9)
ax_ensamble.set_ylabel('Correlation', fontsize=9)
ax_ensamble.set_title('Ensemble correlation', fontsize=11)

ensemble_distance_log10 = np.log10(ensemble_distance)
temporal_distance_log10 = np.log10(temporal_distance)


hist_all, bin_edges_all = np.histogram(ensemble_distance_log10, density=True, bins=10)
bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
bins_mean_all_to_keep = []
bins_occupancies = []
for i in range(0, len(bin_edges_all)-1 ):
    predicted_occupancies_log10_i = ensemble_rho[(ensemble_distance_log10>=bin_edges_all[i]) & (ensemble_distance_log10<bin_edges_all[i+1])]
    bins_mean_all_to_keep.append(bin_edges_all[i])
    bins_occupancies.append(np.mean(predicted_occupancies_log10_i))


bins_mean_all_to_keep = np.asarray(bins_mean_all_to_keep)
bins_occupancies = np.asarray(bins_occupancies)

bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]
bins_occupancies_no_nan = bins_occupancies[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]

ax_ensamble.plot(10**bins_mean_all_to_keep_no_nan,   bins_occupancies_no_nan, lw=3, ls='-',c='k', zorder=2, label='Mean')




ax_temporal.scatter(temporal_distance, temporal_rho, c='k', alpha=0.1, s=8)
ax_temporal.set_xscale('log', basex=10)
ax_temporal.set_xlabel('Phylogenetic distance', fontsize=9)
ax_temporal.set_ylabel('Correlation', fontsize=9)
ax_temporal.set_title('Temporal correlation', fontsize=11)

   


hist_all, bin_edges_all = np.histogram(temporal_distance_log10, density=True, bins=10)
bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
bins_mean_all_to_keep = []
bins_occupancies = []
for i in range(0, len(bin_edges_all)-1 ):
    predicted_occupancies_log10_i = temporal_rho[(temporal_distance_log10>=bin_edges_all[i]) & (temporal_distance_log10<bin_edges_all[i+1])]
    bins_mean_all_to_keep.append(bin_edges_all[i])
    bins_occupancies.append(np.mean(predicted_occupancies_log10_i))


bins_mean_all_to_keep = np.asarray(bins_mean_all_to_keep)
bins_occupancies = np.asarray(bins_occupancies)

bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]
bins_occupancies_no_nan = bins_occupancies[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]

ax_temporal.plot(10**bins_mean_all_to_keep_no_nan,   bins_occupancies_no_nan, lw=3, ls='-',c='k', zorder=2, label='Mean')


        

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig_name = utils.directory + '/figs/distance_vs_rho.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
                




