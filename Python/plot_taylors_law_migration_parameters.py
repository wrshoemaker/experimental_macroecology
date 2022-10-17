from __future__ import division
import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm


afd_dict = {}
transfers = [12, 18]

#migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]
migration_innocula = [('No_migration',4), ('Global_migration',4), ('Parent_migration',4)]


zeros=True


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)

ax_slope = plt.subplot2grid((1, 2), (0, 0))
ax_intercept = plt.subplot2grid((1, 2), (0, 1))

y_idx = 0

for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

    for trasfer_idx, transfer in enumerate(utils.transfers):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0], inocula=migration_innoculum[1])

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=zeros)

        idx_to_keep = means<0.1

        #means = means[idx_to_keep]
        #variances = variances[idx_to_keep]

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

        s_xx, s_yy, s_xy = utils.get_pair_stats(np.log10(means), np.log10(variances))
        t = stats.t.ppf(1-(utils.alpha/2), len(means)-2)
        #maximim likelihood estimator
        mse = sum((np.log10(variances) - (intercept + slope * np.log10(means)))**2) / (len(means)-2)
        slope_ci = t*np.sqrt(mse/s_xx)

        intercept_ci = t*np.sqrt(mse*((1/len(means)) + ((np.mean(np.log10(means))**2)/s_xx)))

        #slope_ci_lower = slope - slope_ci
        #slope_ci_upper = slope + slope_ci

        #intercept_ci_lower = intercept - intercept_ci
        #intercept_ci_upper = intercept + intercept_ci

        #print((slope-slope_ci_lower, slope_ci_upper-slope))

        ax_slope.errorbar(slope, y_idx, xerr=slope_ci, linestyle='-', marker='o', c='k', elinewidth=2.5, alpha=1, zorder=1)
        ax_slope.scatter(slope, y_idx, alpha=1,  s = 120, linewidth=2, c=utils.color_dict_range[migration_innoculum][transfer-4].reshape(1,-1), edgecolors='k', zorder=2)#, c='#87CEEB')


        ax_intercept.errorbar(intercept, y_idx, xerr=intercept_ci, linestyle='-', marker='o', c='k', elinewidth=2.5, alpha=1, zorder=1)
        ax_intercept.scatter(intercept, y_idx, alpha=1, s = 120, linewidth=2, c=utils.color_dict_range[migration_innoculum][transfer-4].reshape(1,-1), edgecolors='k', zorder=2)#, c='#87CEEB')


        y_idx += 1

        #ax.text(0.3,0.9, r'$\sigma^{{2}}_{{x}} = {{{}}} * \left \langle x \right \rangle^{{{}}}$'.format(str(round(10**intercept, 3)),  str(round(slope, 3)) ), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes  )



        #ax.legend(loc="lower right", fontsize=8)





y_axis_positions = (np.asarray(list(range(len(migration_innocula))))*2 )+ 1.15
y_axis_labels = [utils.titles_no_inocula_dict[x] for x in migration_innocula]

ax_slope.set_yticks(y_axis_positions)
ax_slope.set_yticklabels(y_axis_labels, rotation=90, fontsize=7, fontweight='bold', ha='center')
ax_slope.tick_params(axis=u'y',length=0)

ax_intercept.set_yticks(y_axis_positions)
ax_intercept.set_yticklabels(y_axis_labels, rotation=90, fontsize=7, fontweight='bold', ha='center')
ax_intercept.tick_params(axis=u'y',length=0)

#ax_intercept.set_yticks(y_axis_positions)
#ax_intercept.set_yticklabels(y_axis_labels, rotation=90, fontsize=5.5, fontweight='bold', ha='center')
#ax_intercept.tick_params(axis=u'y',length=0)
ax_slope.set_xlabel("Taylor's Law slope", fontsize=12)
ax_intercept.set_xlabel("Taylor's Law intercept", fontsize=12)






#ax_slope.set_yticks(y_axis_positions)
#ax_slope.set_yticklabels(y_axis_labels, rotation=90, fontsize=5.5, fontweight='bold', ha='center')
#ax_slope.tick_params(axis=u'y',length=0)


#ax_intercept.set_yticks(y_axis_positions)
#ax_intercept.set_yticklabels(y_axis_labels, rotation=90, fontsize=5.5, fontweight='bold', ha='center')
#ax_intercept.tick_params(axis=u'y',length=0)

#ax_slope.axvline(0, lw=3, ls=':',color='k', zorder=1)
#ax_intercept.axvline(0, lw=3, ls=':',color='k', zorder=1)

#ax_slope.set_xlabel('Standardized slope difference\nbetween individual and merged attractors', fontsize=12)
#ax_intercept.set_xlabel('Standardized intercept difference\nbetween individual and merged attractors', fontsize=12)

#ax_slope.text(-0.07,0.25, 'No migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )
#ax_slope.text(-0.07,0.75, 'Parent migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )





#wspace=0.3, hspace=0.3
fig_name = utils.directory + '/figs/taylors_law_migration_params.png'
fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
