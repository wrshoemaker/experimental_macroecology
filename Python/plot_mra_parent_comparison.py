from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from sklearn.linear_model import LogisticRegression



# get parent mean relative abundances
mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
#mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

species_in_descendants = []

for migration_innoculum in utils.migration_innocula:

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
#presnt_in_descendants_binary = []
for i in species_parent:
    if i in species_in_descendants:
        presnt_in_descendants.append(1)
    else:
        presnt_in_descendants.append(0)
presnt_in_descendants = np.asarray(presnt_in_descendants)
mean_rel_abundances_parent_log10 = np.log10(mean_rel_abundances_parent)




#hist_, bin_edges_ = np.histogram(mean_rel_abundances_parent_log10, density=True, bins=10)
#bins_mean_f_mean = [0.5 * (bin_edges_[i] + bin_edges_[i+1]) for i in range(0, len(bin_edges_)-1 )]
#prob_present = [sum(presnt_in_descendants[((mean_rel_abundances_parent_log10 >= bin_edges_[i]) & (mean_rel_abundances_parent_log10 < bin_edges_[i+1]))])/len(presnt_in_descendants) for i in range(0, len(bin_edges_)-1 ) ]
#ax_f_mean.scatter(bins_mean_f_mean, hist_f_mean, alpha=0.9, s=10, c=species_color_map[species_name])
#prob_present = np.asarray(prob_present)







#D, p = stats.ks_2samp(mean_rel_abundances_parent_present_log10, mean_rel_abundances_parent_absent_log10)

#print(D, p)

#fig, ax = plt.subplots(figsize=(4,4))
#fig.subplots_adjust(bottom= 0.15)


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)
ax_hist = plt.subplot2grid((1, 2), (0,0))
ax_regression = plt.subplot2grid((1, 2), (0,1))


ax_hist.hist(mean_rel_abundances_parent_present_log10, lw=3, alpha=0.8, bins= 15, color='dodgerblue', histtype='step', density=True, label='Present in descendents')
ax_hist.hist(mean_rel_abundances_parent_absent_log10, lw=3, alpha=0.8, bins= 15, color='k', histtype='step', density=True, label='Absent in descendents')

ax_hist.set_xlabel('Mean relative abundance\nin parent community, ' +  r'$\mathrm{log}_{10}$', fontsize=12)
ax_hist.set_ylabel('Probability density', fontsize=12)
ax_hist.legend(loc="upper right", fontsize=8)


#ax.axvline(n_reads_all_log10_mean, lw=2, ls='--',color='k', zorder=1, label='Mean of ' + r'$\mathrm{log}_{10} $')

ax_regression.scatter(mean_rel_abundances_parent_log10, presnt_in_descendants, color='dodgerblue', alpha=0.5)

ax_regression.set_xlabel('Mean relative abundance\nin parent community, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
ax_regression.set_ylabel('Presence in descendant communities', fontsize=12)

#


model = LogisticRegression(solver='liblinear', random_state=0).fit(mean_rel_abundances_parent_log10.reshape(-1, 1), presnt_in_descendants)
#prediction = model.predict_proba(mean_rel_abundances_parent_log10)
#model = LogisticRegression(solver='liblinear', random_state=0)
slope = model.coef_[0][0] # 1.086147398369049
intercept = model.intercept_[0] # 1.6207715018444455
# p(x) = 1 / (1 + np.exp( -1 * (intercept + slope*x) ))
range = np.linspace(min(mean_rel_abundances_parent_log10), max(mean_rel_abundances_parent_log10), num=1000, endpoint=True)
prediction = [model.predict_proba([[value]])[0][1] for value in range]
ax_regression.plot(range, prediction, c='k', ls='-', label='Logistic regression')
ax_regression.legend(loc="center left", fontsize=8)


fig.subplots_adjust(wspace=0.3, hspace=0.5)
fig.savefig(utils.directory + "/figs/mean_rel_abund_parent_comparison.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()