from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections

color_dict = {'Parent_migration.4.T12': utils.color_dict_range[('Parent_migration', 4)][7],
                'No_migration.4.T12': utils.color_dict_range[('No_migration',4)][7],
                'No_migration.40.T12': utils.color_dict_range[('No_migration',40)][7],
                'Global_migration.4.T12': utils.color_dict_range[('Global_migration',4)][7],
                'Parent_migration.4.T18': utils.color_dict_range[('Parent_migration', 4)][13],
                'No_migration.4.T18': utils.color_dict_range[('No_migration',4)][13],
                'No_migration.40.T18': utils.color_dict_range[('No_migration',40)][13],
                'Global_migration.4.T18': utils.color_dict_range[('Global_migration',4)][13],
                'Parent_migration.NA.T0': 'k'}


count_dict = utils.get_otu_dict()

mean_rel_abund_all_treatments_dict = {}

for treatment in ['No_migration.4.T12', 'No_migration.40.T12', 'Global_migration.4.T12', 'Parent_migration.4.T12', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.4.T18', 'Parent_migration.NA.T0']:

    #mean_abundance_dict[treatment] = {}
    #samples_to_keep = [sample for sample in samples if treatment in sample]
    count_dict_to_keep = {key: value for key, value in count_dict.items() if treatment in key}
    abundance_dict = {}
    n_samples = len(count_dict_to_keep)
    for sample, asv_dict in count_dict_to_keep.items():
        N = sum(asv_dict.values())
        for asv, abundance in asv_dict.items():
            if asv not in abundance_dict:
                abundance_dict[asv] = []
            abundance_dict[asv].append(abundance/N)


    for asv, rel_abundance_list in abundance_dict.items():
        mean_rel_abundance = sum(rel_abundance_list)/n_samples
        if asv not in mean_rel_abund_all_treatments_dict:
            mean_rel_abund_all_treatments_dict[asv] = {}
        mean_rel_abund_all_treatments_dict[asv][treatment] = mean_rel_abundance


fig = plt.figure(figsize = (16, 8)) #
fig.subplots_adjust(bottom= 0.15,  wspace=0.25)

plot_idx = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

rho_dict = {}

for treatment_idx, treatment in enumerate(['No_migration.4.T12', 'No_migration.40.T12', 'Global_migration.4.T12', 'Parent_migration.4.T12', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.4.T18']):

    source = []
    final = []

    for asv, mean_rel_abundance_dict in mean_rel_abund_all_treatments_dict.items():

        if ('Parent_migration.NA.T0' in mean_rel_abundance_dict) and (treatment in mean_rel_abundance_dict):
            source.append(mean_rel_abundance_dict['Parent_migration.NA.T0'])
            final.append(mean_rel_abundance_dict[treatment])


    ax = plt.subplot2grid((2, 4), plot_idx[treatment_idx], colspan=1)


    ax.plot([0.9*(10**-7),1.01], [0.9*(10**-7),1.01], lw=3,ls='--',c='k',zorder=1, label='1:1')
    ax.scatter(source, final, alpha=0.8, c=color_dict[treatment].reshape(1,-1), zorder=2)#, c='#87CEEB')

    # regressions
    #slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(source), np.log10(final))

    #t_value = (slope - 1)/std_err
    #p_value = stats.t.sf(np.abs(t_value), len(source)-2)
    #p_value_to_plot = utils.get_p_value(p_value)

    #ax.text(0.15,0.92, r'$\beta=$' + str(round(slope,3)), fontsize=10, color='k', ha='center', va='center', transform=ax.transAxes )
    ##ax.text(0.15,0.84, r'$t=$' + str(round(t_value,3)), fontsize=10, color='k', ha='center', va='center', transform=ax.transAxes )
    #ax.text(0.15,0.76, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax.transAxes )
    #ax.text(0.18,0.68, r'$\rho^{2}=$' + str(round(r_value**2,4)), fontsize=10, color='k', ha='center', va='center', transform=ax.transAxes )

    rho = np.corrcoef(np.log10(source), np.log10(final))[0,1]
    rho_dict[treatment] = rho

    ax.text(0.26,0.93, r'$\rho_{\mathrm{Pearson}}^{2}=$' + str(round(rho**2,5)), fontsize=10, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.legend(loc="lower right", fontsize=8)

    #print(slope, p_value)

    #if p_value < 0.05:
    #    ax.text(0.15,0.885, r'$P < 0.05$', fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes)

    #else:
    #    ax.text(0.15,0.885, r'$P \nless 0.05$', fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes)

    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Relative abundance\n' + "in progenitor community", fontsize=11)
    ax.set_ylabel('Mean relative abundance\n' + utils.label_dict[treatment] + ', transfer ' + treatment.split('.')[-1][1:], fontsize=10)


fig.subplots_adjust(wspace=0.35, hspace=0.3)
#fig.savefig(utils.directory + "/figs/initial_vs_final_abundance.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + '/figs/initial_vs_final_abundance.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()




# plot correlation between timepoints
#fig, ax = plt.subplots(figsize=(4,4))
#for treatment_idx, treatment in enumerate(['No_migration.4', 'No_migration.40', 'Global_migration.4', 'Parent_migration.4']):

#    rho_transfer_12 = rho_dict[treatment + '.T12']
#    rho_transfer_18 = rho_dict[treatment + '.T18']

#    print(rho_transfer_12, rho_transfer_18)

#    ax.plot([12, 18], [rho_transfer_12, rho_transfer_18], alpha=1, c='k', zorder=2)
#    ax.scatter([12, 18], [rho_transfer_12, rho_transfer_18], alpha=1, s=50, c=color_dict[treatment+'.T18'].reshape(1,-1), label=utils.label_dict[treatment+'.T18'], linewidth=0.8, edgecolors='k', zorder=3)


#ax.axhline(0, lw=1.5, ls=':',color='k', zorder=1)
#ax.legend(loc="lower center", fontsize=8)

#ax.set_xlabel('Transfer', fontsize=12)
#ax.set_ylabel('Correlation in relative abundances\nbetween parent and descendent communities',  fontsize=12)

    #ax.scatter(occupancies, predicted_occupancies, alpha=0.8, s=15, zorder=2, c=[color], label='Merged', linewidth=0.8, edgecolors='k')
#ax.set_yscale('log', basey=10)

#fig.subplots_adjust(wspace=0.35, hspace=0.3)
#fig.savefig(utils.directory + "/figs/autocorrelation.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
#plt.close()
