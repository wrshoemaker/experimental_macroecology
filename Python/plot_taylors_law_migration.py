from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from matplotlib import cm


zeros = True

migration_innocula = [[('No_migration',4), ('No_migration',40)], [('Global_migration',4), ('Parent_migration',4)]]

fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)

for migration_innoculum_row_idx, migration_innoculum_row in enumerate(migration_innocula):

    for migration_innoculum_column_idx, migration_innoculum_column in enumerate(migration_innoculum_row):

        title = utils.titles_dict[migration_innoculum_column]

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum_column[0], inocula=migration_innoculum_column[1])

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))


        means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=zeros)

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))


        ax_plot = plt.subplot2grid((2, 2), (migration_innoculum_row_idx, migration_innoculum_column_idx), colspan=1)

        ax_plot.scatter(means, variances, alpha=0.8, c=utils.color_dict[migration_innoculum_column].reshape(1,-1), edgecolors='k')#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        y_log10_null_range = 10 ** (utils.slope_null*x_log10_range + intercept)

        ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
        ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )

        ax_plot.set_xlabel('Average relative\nabundance', fontsize=12)
        ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)

        ax_plot.set_title(title, fontsize=12, fontweight='bold' )


        t_value = (slope - (utils.slope_null))/std_err
        p_value = stats.t.sf(np.abs(t_value), len(means)-2)

        p_value_to_plot = utils.get_p_value(p_value)


        ax_plot.text(0.2,0.8, r'$t=$' + str(round(t_value,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_plot.transAxes )
        ax_plot.text(0.2,0.7, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_plot.transAxes )


        sys.stdout.write("Slope = %g, t = %g, P= %g\n" % (slope, t_value, p_value))



        ax_plot.legend(loc="lower right", fontsize=8)



fig.subplots_adjust(wspace=0.3, hspace=0.5)
fig.savefig(utils.directory + "/figs/taylors_law_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()