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
import plot_utils

from scipy.optimize import fsolve
from scipy.special import erf


remove_zeros = True


transfers = [12,18]

fig, ax_taylors = plt.subplots(figsize=(4,4))



# get mean and std for rescaling
all_means = []
all_vars = []

slope_all = []


for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):
        
    for transfer in transfers:

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        #print(migration_innoculum, transfer , s_by_s.shape, len(comm_rep_list))

        afd = rel_s_by_s.flatten()
        afd_log10 = np.log(afd[afd>0])
        afd_log10_rescaled = (afd_log10 - np.mean(afd_log10)) / np.std(afd_log10)

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)

        # taylors law
        means = []
        variances = []
        for afd_i in rel_s_by_s:
            afd_i = afd_i[afd_i>0]
            if len(afd_i) < 3:
                continue

            means.append(np.mean(afd_i))
            variances.append(np.var(afd_i))


        label = utils.titles_abbreviated_dict[migration_innoculum] + ', transfer ' + str(transfer)

        ax_taylors.scatter(means, variances, alpha=0.3, c=color_, zorder=2, label=label)

        # get slopes
        means_array = np.asarray(means)
        variances_array = np.asarray(variances)
        means_for_taylors = means_array[means_array<=0.95]
        variances_for_taylors = variances_array[means_array<=0.95]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means_for_taylors), np.log10(variances_for_taylors))
        print(migration_innoculum, transfer, slope, intercept)
        beta_taylors = 10**intercept
        print((2*beta_taylors) / (1+beta_taylors))

        slope_all.append(slope)

        all_means.extend(means)
        all_vars.extend(variances)


se_slope = np.std(slope_all, ddof=1) / np.sqrt(np.size(slope_all))
print("Mean slope +/- S.E.")
print(np.mean(slope_all), se_slope)



#ax_taylors.set_xlim([3e-6 , 3])


ax_taylors.set_xscale('log', basex=10)
ax_taylors.set_yscale('log', basey=10)
ax_taylors.set_xlabel('Mean relative abundance', fontsize=12)
ax_taylors.set_ylabel('Variance of relative abundance', fontsize=12)
ax_taylors.xaxis.set_tick_params(labelsize=9)
ax_taylors.yaxis.set_tick_params(labelsize=9)

ax_taylors.minorticks_off()

all_means = np.asarray(all_means)
all_vars = np.asarray(all_vars)

intercept = np.mean(np.log10(all_vars) - 2*np.log10(all_means))
x_log10_range =  np.linspace(min(np.log10(all_means)) , max(np.log10(all_means)) , 10000)
y_log10_null_range = 10 ** (2*x_log10_range + intercept)
ax_taylors.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='-', zorder=2, label= r'$y \sim x^{2}$')

mean_range = np.linspace(min(all_means), max(all_means), num=1000)
variance_range = (1-mean_range) * mean_range

#ax_taylors.plot(mean_range, variance_range, lw=3, ls=':', c = 'k', label='Bhatia–Davis inequality')
#ax_taylors.plot(mean_range, variance_range, lw=2.5, ls=':', c='k', label='Max. ' + r'$\sigma^{2}_{x}$', zorder=1)
ax_taylors.legend(loc="lower right", fontsize=6)





fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/taylors_law.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/taylors_law.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)

plt.close()
