from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma
import scipy.special as special
import utils
import pickle
import plot_utils

from scipy.optimize import fsolve
from scipy.special import erf
from sklearn.linear_model import LogisticRegression


remove_zeros = False


n_range = np.linspace(1, 24874, num=50, endpoint=True)
count_dict = utils.get_otu_dict()

iter_ = 100
intermediate_filename = utils.directory + "/data/subsample_richness.dat"
treatments = ['Parent_migration.4.T18', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.NA.T0']

color_dict = {'Parent_migration.4.T18': utils.color_dict_range[('Parent_migration', 4)][13],
                'No_migration.4.T18': utils.color_dict_range[('No_migration',4)][13],
                'No_migration.40.T18': utils.color_dict_range[('No_migration',40)][13],
                'Global_migration.4.T18': utils.color_dict_range[('Global_migration',4)][13],
                'Parent_migration.NA.T0': 'k'}



def make_subsample_richness_dict():

    rarefaction_dict = {}
    for sample, sad_dict in count_dict.items():
        sad = list(sad_dict.values())
        sad = np.asarray(sad)
        abundance = sum(sad)

        richness_subsample = []
        for n in n_range:
            n = int(n)
            if n > abundance:
                continue
            richness_subsample_n = [len(utils.subsample_sad(sad, replace=False, n_subsample = n)) for i in range(iter_)]
            richness_subsample.append(np.mean(richness_subsample_n))

        rarefaction_dict[sample] = richness_subsample


    sys.stderr.write("Saving allelic dict...\n")
    with open(intermediate_filename, 'wb') as handle:
        pickle.dump(rarefaction_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_subsample_richness_dict():

    with open(intermediate_filename, 'rb') as handle:
        b = pickle.load(handle)
    return b



# load rarefaction dictionart
rarefaction_dict = load_subsample_richness_dict()


# get parent vs. offspring data

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


################################
# logistic regression analysis #
################################
# get parent mean relative abundances
mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
#mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

species_in_descendants = []

# exclude parent migration
migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4)]
for migration_innoculum in migration_innocula: #utils.migration_innocula:

    s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
    means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=True)
    species_in_descendants.extend(ESVs)


species_in_descendants = list(set(species_in_descendants))

mean_rel_abundances_parent_present = [mean_rel_abundances_parent[s_idx] for (s_idx,s) in enumerate(species_parent) if s in species_in_descendants]
mean_rel_abundances_parent_absent = [mean_rel_abundances_parent[s_idx] for (s_idx,s) in enumerate(species_parent) if s not in species_in_descendants]
mean_rel_abundances_parent_present_log10 = np.log10(mean_rel_abundances_parent_present)
mean_rel_abundances_parent_absent_log10 = np.log10(mean_rel_abundances_parent_absent)


presnt_in_descendants = []
for i in species_parent:
    if i in species_in_descendants:
        presnt_in_descendants.append(1)
    else:
        presnt_in_descendants.append(0)
presnt_in_descendants = np.asarray(presnt_in_descendants)
mean_rel_abundances_parent_log10 = np.log10(mean_rel_abundances_parent)



##############
# make plots #
##############
fig = plt.figure(figsize = (12, 12.5)) #
fig.subplots_adjust(bottom= 0.15)

transfers = [12,18]

ax_rarefaction = plt.subplot2grid((2, 2), (0,0), colspan=1)
ax_parent_vs_offspring = plt.subplot2grid((2, 2), (0,1), colspan=1)
ax_parent_hist = plt.subplot2grid((2, 2), (1,0), colspan=1)
ax_parent_logisitc = plt.subplot2grid((2, 2), (1,1), colspan=1)


ax_rarefaction.text(-0.1, 1.04, plot_utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_rarefaction.transAxes)
ax_parent_vs_offspring.text(-0.1, 1.04, plot_utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_parent_vs_offspring.transAxes)
ax_parent_hist.text(-0.1, 1.04, plot_utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_parent_hist.transAxes)
ax_parent_logisitc.text(-0.1, 1.04, plot_utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_parent_logisitc.transAxes)


# plot rarefaction
for treatment in treatments:

    ax_rarefaction.plot([-100000, -1000], [-100000,-1000], lw=1.5, c=color_dict[treatment], label=utils.label_dict[treatment])

    for sample, rarefied_richness in rarefaction_dict.items():

        sample_split = sample.rsplit('.',1)[0]

        if sample_split == 'Parent_migration.NA.T0':
            alpha=0.7
        else:
            alpha=0.05

        richness = rarefaction_dict[sample]
        n_range_sample = n_range[:len(richness)]

        ax_rarefaction.plot(n_range_sample, richness, lw=1.5, alpha=alpha, c=color_dict[sample_split])


    #ax_rarefaction.set_title('Transfer 18', fontsize=13)
    ax_rarefaction.set_xlim(-500.5, 26000)
    ax_rarefaction.set_ylim(0.8, 1600)
    ax_rarefaction.set_yscale('log', basey=10)
    ax_rarefaction.set_xlabel('Number of reads', fontsize=12)
    ax_rarefaction.set_ylabel('Subsampled ASV richness without replacement', fontsize=12)
    ax_rarefaction.legend(loc="lower right", fontsize=8)




# plot parent vs. offspring example
parent_mad = []
final_mad = []
for asv, mean_rel_abundance_dict in mean_rel_abund_all_treatments_dict.items():

    if ('Parent_migration.NA.T0' in mean_rel_abundance_dict) and ('No_migration.4.T18' in mean_rel_abundance_dict):
        parent_mad.append(mean_rel_abundance_dict['Parent_migration.NA.T0'])
        final_mad.append(mean_rel_abundance_dict['No_migration.4.T18'])


parent_vs_offspring_color = utils.color_dict_range[('No_migration',4)][-2]
rho = np.corrcoef(np.log10(parent_mad), np.log10(final_mad))[0,1]

min_ = min(parent_mad + final_mad)*0.7
max_ = max(parent_mad + final_mad)*1.2

ax_parent_vs_offspring.set_xlim([min_, max_])
ax_parent_vs_offspring.set_ylim([min_, max_])
ax_parent_vs_offspring.plot([min_, max_], [min_, max_], lw=3,ls='--',c='k',zorder=1, label='1:1')
ax_parent_vs_offspring.scatter(parent_mad, final_mad, alpha=0.8, c=parent_vs_offspring_color.reshape(1,-1), zorder=2)#, c='#87CEEB')
ax_parent_vs_offspring.text(0.22,0.87, r'$\rho_{\mathrm{Pearson}}^{2}=$' + str(round(rho**2, 5)), fontsize=12, color='k', ha='center', va='center', transform=ax_parent_vs_offspring.transAxes )
ax_parent_vs_offspring.legend(loc="upper left", fontsize=10)
ax_parent_vs_offspring.set_xscale('log', basex=10)
ax_parent_vs_offspring.set_yscale('log', basey=10)
ax_parent_vs_offspring.set_xlabel('Relative abundance in progenitor', fontsize=12)
ax_parent_vs_offspring.set_ylabel('Mean relative abundance, no migration, transfer 18', fontsize=12)




# plot before/after histogram
ax_parent_hist.hist(mean_rel_abundances_parent_present_log10, lw=3, alpha=0.8, bins= 12, color='k', ls='-', histtype='step', density=True, label='Present in descendents')
ax_parent_hist.hist(mean_rel_abundances_parent_absent_log10, lw=3, alpha=0.8, bins= 12, color='k', ls=':', histtype='step', density=True, label='Absent in descendents')

ax_parent_hist.set_xlabel('Relative abundance in regional community, ' +  r'$\mathrm{log}_{10}$', fontsize=12)
ax_parent_hist.set_ylabel('Probability density', fontsize=12)
ax_parent_hist.legend(loc="upper right", fontsize=8)

#ks_statistic, p_value = utils.run_permutation_ks_test(mean_rel_abundances_parent_present_log10, mean_rel_abundances_parent_absent_log10)
ks_statistic = 0.3899878193141073
p_value = 0

ax_parent_hist.text(0.8,0.81, r'$D = {{{}}}$'.format(str(round(ks_statistic, 3))), fontsize=11, color='k', ha='center', va='center', transform=ax_parent_hist.transAxes)
ax_parent_hist.text(0.8,0.73, r'$P < 0.05$', fontsize=11, color='k', ha='center', va='center', transform=ax_parent_hist.transAxes)



# plot logistic regression
ax_parent_logisitc.scatter(mean_rel_abundances_parent_log10, presnt_in_descendants, color='k', alpha=0.3)
ax_parent_logisitc.set_xlabel('Relative abundance in regional community, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
ax_parent_logisitc.set_ylabel('Presence in descendant communities', fontsize=12)
ax_parent_logisitc.set_yticks([0,1])
ax_parent_logisitc.set_yticklabels(['0', '1'])

model = LogisticRegression(solver='liblinear', random_state=0).fit(mean_rel_abundances_parent_log10.reshape(-1, 1), presnt_in_descendants)
slope = model.coef_[0][0] # 0.9410534572734155
intercept = model.intercept_[0] # 0.7123794529201377

# p(x) = 1 / (1 + np.exp( -1 * (intercept + slope*x) ))
range_ = np.linspace(min(mean_rel_abundances_parent_log10), max(mean_rel_abundances_parent_log10), num=1000, endpoint=True)
prediction = [model.predict_proba([[value]])[0][1] for value in range_]
ax_parent_logisitc.plot(range_, prediction, c='k', ls='-', lw=2.5, label='Logistic regression')
ax_parent_logisitc.legend(loc="center left", fontsize=10)




fig_name = utils.directory + '/figs/experimental_details.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
