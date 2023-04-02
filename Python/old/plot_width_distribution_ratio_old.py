from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed(123456789)

number_transfers = 18






species_names_no_migration, species_transfers_no_migration, mean_relative_abundances_no_migration, mean_absolute_differences_no_migration, width_distribution_transfers_no_migration, width_distribution_ratios_no_migration, mean_relative_abundances_for_width_no_migration, variance_transfers_no_migration, variance_no_migration, mean_no_migration = utils.get_temporal_patterns(('No_migration',4))
species_names_global_migration, species_transfers_global_migration, mean_relative_abundances_global_migration, mean_absolute_differences_global_migration, width_distribution_transfers_global_migration, width_distribution_ratios_global_migration, mean_relative_abundances_for_width_global_migration, variance_transfers_global_migration, variance_global_migration, mean_global_migration = utils.get_temporal_patterns(('Global_migration',4))





width_colors_no_migration = [utils.rgb_blue[width_transfer_] for width_transfer_ in variance_transfers_no_migration]
width_colors_global_migration = [utils.rgb_red[width_transfer_] for width_transfer_ in variance_transfers_global_migration]


colors_no_migration = [utils.rgb_blue[species_transfer] for species_transfer in species_transfers_no_migration]
colors_global_migration = [utils.rgb_red[species_transfer] for species_transfer in species_transfers_global_migration]


width_all_colors_no_migration = [utils.rgb_blue[species_transfer] for species_transfer in width_distribution_transfers_no_migration]
width_all_colors_global_migration = [utils.rgb_red[species_transfer] for species_transfer in width_distribution_transfers_global_migration]








fig = plt.figure(figsize = (8, 12)) #
fig.subplots_adjust(bottom= 0.15)


ax_scatter_no_migration = plt.subplot2grid((3, 2), (0, 0))#, colspan=1)
ax_scatter_global_migration = plt.subplot2grid((3, 2), (0, 1))#, colspan=1)

ax_width_no_migration = plt.subplot2grid((3, 2), (1, 0))#, colspan=1)
ax_width_global_migration = plt.subplot2grid((3, 2), (1, 1))#, colspan=1)

#ax_lognormal = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_mean = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_cv = plt.subplot2grid((3, 2), (2, 1))#, colspan=1)

mean_relative_abundances_no_migration_log10 = np.log10(mean_relative_abundances_no_migration)

mean_absolute_differences_no_migration_log10 = np.log10(mean_absolute_differences_no_migration)


width_distribution_ratios_no_migration_log10 = np.log10(width_distribution_ratios_no_migration)
width_distribution_ratios_global_migration_log10 = np.log10(width_distribution_ratios_global_migration)



#mean_relative_abundances_no_migration_filter = mean_relative_abundances_no_migration[mean_relative_abundances_no_migration< -1]
#mean_absolute_differences_no_migration_filter = mean_absolute_differences_no_migration[mean_relative_abundances_no_migration< -1]



ax_scatter_no_migration.scatter(mean_relative_abundances_no_migration, mean_absolute_differences_no_migration, alpha=1, color=colors_no_migration, edgecolors='k')
ax_scatter_no_migration.set_xscale('log', basex=10)
ax_scatter_no_migration.set_yscale('log', basey=10)
ax_scatter_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_no_migration.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)
#slope_no_migration, intercept_no_migration, r_value_no_migration, p_value_no_migration, std_err_no_migration = stats.linregress(mean_relative_abundances_no_migration_filter, mean_absolute_differences_no_migration_filter)
#x_log10_range_no_migration =  np.linspace(min(mean_relative_abundances_no_migration) , max(mean_relative_abundances_no_migration) , 10000)
#y_log10_fit_range_no_migration = 10 ** (slope_no_migration*x_log10_range_no_migration + intercept_no_migration)
ax_scatter_no_migration.set_title(utils.titles_dict[('No_migration',4)], fontsize=12, fontweight='bold' )


max_difference_x, max_difference_y = utils.max_difference_between_timepoints(min(mean_relative_abundances_no_migration),  max(mean_relative_abundances_no_migration))


#ax_scatter_no_migration.axvline(x=0.5, ls = ':', c='k', lw=2, zorder=1)



#cutoffs_no_migration, slopes_no_migration = utils.get_slopes_cutoffs(mean_relative_abundances_no_migration, mean_absolute_differences_no_migration)
#ins_no_migration = inset_axes(ax_scatter_no_migration, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.68,0.11,0.3,0.3), bbox_transform=ax_scatter_no_migration.transAxes)
#ins_no_migration.set_xlabel('Max.' + r'$\left \langle x(t) \right \rangle$', fontsize=8)
#ins_no_migration.set_ylabel("Slope", fontsize=8)

#ins_no_migration.plot(10**cutoffs_no_migration, slopes_no_migration, ls='-', c='k')

#ins_no_migration.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
#ins_no_migration.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

#ins_no_migration.set_xscale('log', basex=10)
#ins_no_migration.tick_params(labelsize=5)
#ins_no_migration.tick_params(axis='both', which='major', pad=1)
#ins_no_migration.set_ylim(0.85, 1.02)

#ins_no_migration.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )
#ins_no_migration.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )




#mean_relative_abundances_global_migration_filter = mean_relative_abundances_global_migration[mean_relative_abundances_global_migration< -1]
#mean_absolute_differences_global_migration_filter = mean_absolute_differences_global_migration[mean_relative_abundances_global_migration< -1]

ax_scatter_global_migration.scatter(mean_relative_abundances_global_migration, mean_absolute_differences_global_migration, alpha=0.9, color=colors_global_migration, edgecolors='k')
ax_scatter_global_migration.set_xscale('log', basex=10)
ax_scatter_global_migration.set_yscale('log', basey=10)
ax_scatter_global_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_global_migration.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)

slope_global_migration, intercept_global_migration, r_value_global_migration, p_value_global_migration, std_err_global_migration = stats.linregress(np.log10(mean_relative_abundances_global_migration), np.log10(mean_absolute_differences_global_migration))
x_log10_range_global_migration =  np.linspace(min(np.log10(mean_relative_abundances_global_migration)) , max(np.log10(mean_relative_abundances_global_migration)) , 10000)
y_log10_fit_range_global_migration = 10 ** (slope_global_migration*x_log10_range_global_migration + intercept_global_migration)
ax_scatter_global_migration.plot(10**x_log10_range_global_migration, y_log10_fit_range_global_migration, ls='--', lw=4, c='k')



ax_scatter_global_migration.set_title(utils.titles_dict[('Global_migration',4)], fontsize=12, fontweight='bold' )


ax_scatter_global_migration.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_global_migration, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_scatter_global_migration.transAxes  )


#ax_scatter_global_migration.axvline(x=0.5, ls = ':', c='k', lw=2, zorder=1)



#cutoffs_global_migration, slopes_global_migration = utils.get_slopes_cutoffs(mean_relative_abundances_global_migration, mean_absolute_differences_global_migration)
#ins_global_migration = inset_axes(ax_scatter_global_migration, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.68,0.11,0.3,0.3), bbox_transform=ax_scatter_global_migration.transAxes)
#ins_global_migration.set_xlabel('Max.' + r'$\left \langle x(t) \right \rangle$', fontsize=8)
#ins_global_migration.set_ylabel("Slope", fontsize=8)

#ins_global_migration.plot(10**cutoffs_global_migration, slopes_global_migration, ls='-', c='k')
#ins_global_migration.set_xscale('log', basex=10)
#ins_global_migration.tick_params(labelsize=5)
#ins_global_migration.tick_params(axis='both', which='major', pad=1)

#ins_global_migration.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
#ins_global_migration.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

#ins_global_migration.set_ylim(0.85, 1.02)

#ins_global_migration.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_global_migration.transAxes )
#ins_global_migration.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_global_migration.transAxes )





ax_width_no_migration.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_no_migration.scatter(mean_relative_abundances_for_width_no_migration, width_distribution_ratios_no_migration, alpha=1, color=width_all_colors_no_migration, zorder=2, edgecolors='k')
ax_width_no_migration.set_xscale('log', basex=10)
ax_width_no_migration.set_yscale('log', basey=10)
ax_width_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_no_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\frac{x(t + \delta t) }{x(t ) }$', fontsize=12)

ax_width_no_migration.set_xlim([1e-4, 1])
ax_width_no_migration.set_ylim([0.5e-2, 3e2])


ax_width_global_migration.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_global_migration.scatter(mean_relative_abundances_for_width_global_migration, width_distribution_ratios_global_migration, alpha=1, color=width_all_colors_global_migration, zorder=2, edgecolors='k')



ax_width_global_migration.set_xscale('log', basex=10)
ax_width_global_migration.set_yscale('log', basey=10)
ax_width_global_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_global_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$ \frac{x(t + \delta t) }{x(t ) } $', fontsize=12)

ax_width_global_migration.set_xlim([1e-4, 1])
ax_width_global_migration.set_ylim([0.5e-2, 3e2])









ins_width_no_migration = inset_axes(ax_width_no_migration, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_no_migration.transAxes)
ins_width_global_migration = inset_axes(ax_width_global_migration, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_global_migration.transAxes)


shape_no_migration, loc_no_migration, scale_no_migration = stats.lognorm.fit(width_distribution_ratios_no_migration_log10)
shape_global_migration, loc_global_migration, scale_global_migration = stats.lognorm.fit(width_distribution_ratios_global_migration_log10)


ks_no_migration, p_no_migration = stats.kstest(width_distribution_ratios_no_migration_log10, 'lognorm', args=(shape_no_migration, loc_no_migration, scale_no_migration), alternative='two-sided')

ks_global_migration, p_global_migration = stats.kstest(width_distribution_ratios_global_migration_log10, 'lognorm', args=(shape_global_migration, loc_global_migration, scale_global_migration), alternative='two-sided')


sys.stdout.write("KS test against lognormal, no migration: D = %g,  P = %g\n" % (ks_no_migration, p_no_migration))
sys.stdout.write("KS test against lognormal, global migration: D = %g,  P = %g\n" % ( ks_global_migration, p_global_migration  ))


# try with laplace, just for kicks

loc_laplace, scale_laplace = stats.laplace.fit(width_distribution_ratios_no_migration)
ks_laplace, p_laplace = stats.kstest(width_distribution_ratios_no_migration, 'laplace', args=(loc_laplace, scale_laplace), alternative='two-sided')
sys.stdout.write("KS test against laplace, no migration: D = %g,  P = %g\n" % (ks_laplace, p_laplace))



x_range_no_migration = np.linspace(min(width_distribution_ratios_no_migration_log10)  , max(width_distribution_ratios_no_migration_log10) , 10000)
x_range_global_migration = np.linspace(min(width_distribution_ratios_global_migration) , max(width_distribution_ratios_global_migration) , 10000)

samples_fit_log_no_migration = stats.lognorm.pdf(x_range_no_migration, shape_no_migration, loc_no_migration, scale_no_migration)
samples_fit_log_global_migration = stats.lognorm.pdf(x_range_global_migration, shape_global_migration, loc_global_migration, scale_global_migration)

ins_width_no_migration.hist(width_distribution_ratios_no_migration_log10, histtype='step', color='k', lw=3, alpha=0.9, bins= 12, density=True, zorder=2)
ins_width_global_migration.hist(width_distribution_ratios_global_migration_log10, histtype='step', color=colors_global_migration[8], lw=3, alpha=0.8, bins= 12, density=True, zorder=2)



ins_width_no_migration.axvline(np.mean(x_range_no_migration), lw=2.5, ls=':', color=width_colors_no_migration[-3],zorder=3)
ins_width_global_migration.axvline(np.mean(x_range_global_migration), lw=2.5, ls=':', color=width_colors_global_migration[-3],zorder=3)

ins_width_no_migration.plot(x_range_no_migration, samples_fit_log_no_migration, color=width_colors_no_migration[-3], label='Lognormal fit', lw=3, zorder=3)
#ins_width_global_migration.plot(x_range_global_migration, samples_fit_log_global_migration, color=width_colors_global_migration[-3], label='Lognormal fit', lw=3, zorder=3)

ins_width_no_migration.set_yscale('log', basey=10)
ins_width_global_migration.set_yscale('log', basey=10)

ins_width_no_migration.set_xlabel('Width dist., ' + r'$\mathrm{log}_{10}$', fontsize=7)
ins_width_global_migration.set_xlabel('Width dist., ' + r'$\mathrm{log}_{10}$', fontsize=7)

ins_width_no_migration.set_ylabel('Prob. density', fontsize=7)
ins_width_global_migration.set_ylabel('Prob. density', fontsize=7)

ins_width_no_migration.tick_params(labelsize=5)
ins_width_no_migration.tick_params(axis='both', which='major', pad=1)

ins_width_global_migration.tick_params(labelsize=5)
ins_width_global_migration.tick_params(axis='both', which='major', pad=1)


ins_width_no_migration.set_xlim([-2.4, 2.4])
ins_width_global_migration.set_xlim([-2.4, 2.4])



KS_statistic, p_value = stats.ks_2samp(width_distribution_ratios_no_migration, width_distribution_ratios_global_migration)
sys.stdout.write("Difference between width dists: KS = %g, P= %g\n" % (KS_statistic, p_value))





ax_mean.axhline(0, lw=3, ls=':',color='grey', zorder=1)
ax_mean.plot(variance_transfers_no_migration, mean_no_migration, color = 'k', zorder=2)
ax_mean.plot(variance_transfers_global_migration, mean_global_migration, color = 'k', zorder=2)

ax_mean.scatter(variance_transfers_no_migration, mean_no_migration, color = width_colors_no_migration, edgecolors='k', zorder=3)
ax_mean.scatter(variance_transfers_global_migration, mean_global_migration, color = width_colors_global_migration, edgecolors='k', zorder=3)
ax_mean.set_xlabel('Transfer', fontsize=12)
ax_mean.set_ylabel('Mean of log-transformed\nrelative abundance ratios, ' + r'$\mu$', fontsize=12)

ax_mean.set_ylim(-0.65, 0.65)



ax_cv.axhline(1, lw=3, ls=':',color='grey', zorder=1)
ax_cv.plot(variance_transfers_no_migration, variance_no_migration, color = 'k', zorder=2)
#ax_cv.plot(variance_transfers_global_migration, variance_global_migration, color = 'k', zorder=2)



ax_cv.scatter(variance_transfers_no_migration, variance_no_migration, color = width_colors_no_migration, edgecolors='k', zorder=3)
ax_cv.scatter(variance_transfers_global_migration, variance_global_migration, color = width_colors_global_migration, edgecolors='k', zorder=3)

ax_cv.set_xlabel('Transfer', fontsize=12)
#ax_cv.set_ylabel('CV of log-transformed relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)
ax_cv.set_ylabel('CV of log-transformed\nrelative abundance ratios, ' + r'$\frac{\sigma}{\left | \mu \right |}$', fontsize=12)

#ax_cv.text(0.8,0.9, '$P< 0.05$', fontsize=12, color='k', ha='center', va='center', transform=ax_cv.transAxes )


ax_cv.set_yscale('log', basey=10)




#transfers_intersect = np.intersect1d(variance_transfers_no_migration, variance_transfers_global_migration)

#sorter_no_migration = np.argsort(variance_transfers_no_migration)
#no_migration_idx = sorter_no_migration[np.searchsorted(variance_transfers_no_migration, transfers_intersect, sorter=sorter_no_migration)]
#variances_no_migration_intersect = variance_no_migration[no_migration_idx]

#sorter_global_migration = np.argsort(variance_transfers_global_migration)
#global_migration_idx = sorter_global_migration[np.searchsorted(variance_transfers_global_migration, transfers_intersect, sorter=sorter_global_migration)]
#variances_global_migration_intersect = variance_global_migration[global_migration_idx]


transfers_intersect_mean, means_no_migration_intersect, means_global_migration_intersect  = utils.get_intersecting_timepoints(variance_transfers_no_migration, mean_no_migration, variance_transfers_global_migration, mean_global_migration)
transfers_intersect_cv, cv_no_migration_intersect, cv_global_migration_intersect  = utils.get_intersecting_timepoints(variance_transfers_no_migration, variance_no_migration, variance_transfers_global_migration, variance_global_migration)

mean_ratio_observed_mean, p_value_mean = utils.test_difference_two_time_series(means_no_migration_intersect, means_global_migration_intersect)
mean_ratio_observed_cv, p_value_cv = utils.test_difference_two_time_series(cv_no_migration_intersect, cv_global_migration_intersect)


#_matrix = np.array([variances_no_migration_intersect, variances_global_migration_intersect])



sys.stdout.write("Mean difference test, D = %g,  P = %g\n" % (mean_ratio_observed_mean, p_value_mean))
sys.stdout.write("CV difference test, D = %g,  P = %g\n" % (mean_ratio_observed_cv, p_value_cv))

ax_mean.text(0.82,0.9, utils.get_p_value(p_value_mean), fontsize=12, color='k', ha='center', va='center', transform=ax_mean.transAxes )
ax_cv.text(0.82,0.9, utils.get_p_value(p_value_cv), fontsize=12, color='k', ha='center', va='center', transform=ax_cv.transAxes )




ax_mean.axvline(x=12, c='k', ls=':', lw=2, label='End of migration')
ax_cv.axvline(x=12, c='k', ls=':', lw=2, label='End of migration')

ax_mean.legend(loc="upper left", fontsize=8)
ax_cv.legend(loc="upper left", fontsize=8)


fig.subplots_adjust(wspace=0.5, hspace=0.3)
#fig.savefig(utils.directory + "/figs/temporal_width_distribution_ratios.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/temporal_width_distribution_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
