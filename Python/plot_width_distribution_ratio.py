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





species_names_no_migration, species_transfers_no_migration, mean_relative_abundances_no_migration, mean_absolute_differences_no_migration, width_distribution_species_no_migration, width_distribution_transfers_no_migration, width_distribution_ratios_no_migration, mean_relative_abundances_for_width_no_migration, variance_transfers_no_migration, variance_no_migration, mean_no_migration = utils.get_temporal_patterns(('No_migration',4))
species_names_global_migration, species_transfers_global_migration, mean_relative_abundances_global_migration, mean_absolute_differences_global_migration, width_distribution_species_global_migration, width_distribution_transfers_global_migration, width_distribution_ratios_global_migration, mean_relative_abundances_for_width_global_migration, variance_transfers_global_migration, variance_global_migration, mean_global_migration = utils.get_temporal_patterns(('Global_migration',4))





width_colors_no_migration = [utils.rgb_blue[width_transfer_] for width_transfer_ in variance_transfers_no_migration]
width_colors_global_migration = [utils.rgb_red[width_transfer_] for width_transfer_ in variance_transfers_global_migration]


colors_no_migration = [utils.rgb_blue[species_transfer] for species_transfer in species_transfers_no_migration]
colors_global_migration = [utils.rgb_red[species_transfer] for species_transfer in species_transfers_global_migration]


width_all_colors_no_migration = [utils.rgb_blue[species_transfer] for species_transfer in width_distribution_transfers_no_migration]
width_all_colors_global_migration = [utils.rgb_red[species_transfer] for species_transfer in width_distribution_transfers_global_migration]






fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)


ax_scatter_no_migration = plt.subplot2grid((3, 2), (0, 0))#, colspan=1)
ax_scatter_global_migration = plt.subplot2grid((3, 2), (0, 1))#, colspan=1)

ax_scatter_no_migration_slope = plt.subplot2grid((3, 2), (1, 0))#, colspan=1)
ax_scatter_global_migration_slope = plt.subplot2grid((3, 2), (1, 1))#, colspan=1)

ax_width_no_migration = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_width_global_migration = plt.subplot2grid((3, 2), (2, 1))#, colspan=1)

#ax_lognormal = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)

mean_relative_abundances_no_migration_log10 = np.log10(mean_relative_abundances_no_migration)
mean_absolute_differences_no_migration_log10 = np.log10(mean_absolute_differences_no_migration)

width_distribution_ratios_no_migration_log10 = np.log10(width_distribution_ratios_no_migration)
width_distribution_ratios_global_migration_log10 = np.log10(width_distribution_ratios_global_migration)



ax_scatter_no_migration.scatter(mean_relative_abundances_no_migration, mean_absolute_differences_no_migration, alpha=0.9, color=colors_no_migration, edgecolors='k')
ax_scatter_no_migration.set_xscale('log', basex=10)
ax_scatter_no_migration.set_yscale('log', basey=10)
ax_scatter_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_no_migration.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)

slope_no_migration, intercept_no_migration, r_value_no_migration, p_value_no_migration, std_err_no_migration = stats.linregress(np.log10(mean_relative_abundances_no_migration), np.log10(mean_absolute_differences_no_migration))
x_log10_range_no_migration =  np.linspace(min(np.log10(mean_relative_abundances_no_migration)) , max(np.log10(mean_relative_abundances_no_migration)) , 10000)
y_log10_fit_range_no_migration = 10 ** (slope_no_migration*x_log10_range_no_migration + intercept_no_migration)

ax_scatter_no_migration.plot(10**x_log10_range_no_migration, y_log10_fit_range_no_migration, ls='--', lw=4, c='k', zorder=3)

ax_scatter_no_migration.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_no_migration, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_scatter_no_migration.transAxes  )
ax_scatter_no_migration.set_title(utils.titles_dict[('No_migration',4)], fontsize=12, fontweight='bold' )






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





ax_width_no_migration.axhline(1, lw=3, ls=':',color='k', zorder=3)
ax_width_no_migration.scatter(mean_relative_abundances_for_width_no_migration, width_distribution_ratios_no_migration, alpha=0.5, s=10, color=width_all_colors_no_migration, zorder=2, edgecolors='none')
ax_width_no_migration.set_xscale('log', basex=10)
ax_width_no_migration.set_yscale('log', basey=10)
ax_width_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_no_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\frac{x(t + \delta t) }{x(t ) }$', fontsize=12)

ax_width_no_migration.set_xlim([1e-4, 1])
ax_width_no_migration.set_ylim([0.5e-2, 3e2])


ax_width_global_migration.axhline(1, lw=3, ls=':',color='k', zorder=3)
ax_width_global_migration.scatter(mean_relative_abundances_for_width_global_migration, width_distribution_ratios_global_migration, alpha=0.5, s=10, color=width_all_colors_global_migration, zorder=2, edgecolors='none')

ax_width_global_migration.set_xscale('log', basex=10)
ax_width_global_migration.set_yscale('log', basey=10)
ax_width_global_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_global_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$ \frac{x(t + \delta t) }{x(t ) } $', fontsize=12)

ax_width_global_migration.set_xlim([1e-4, 1])
ax_width_global_migration.set_ylim([0.5e-2, 3e2])





# plot slope
for t in range(1,18):

    no_migration_idx = species_transfers_no_migration==t
    migration_idx = species_transfers_global_migration==t

    mean_relative_abundances_no_migration_t = mean_relative_abundances_no_migration[no_migration_idx]
    mean_absolute_differences_no_migration_t = mean_absolute_differences_no_migration[no_migration_idx]

    mean_relative_abundances_global_migration_t = mean_relative_abundances_global_migration[migration_idx]
    mean_absolute_differences_global_migration_t = mean_absolute_differences_global_migration[migration_idx]

    slope_no_migration, intercept_no_migration, r_value_no_migration, p_value_no_migration, std_err_no_migration = stats.linregress(np.log10(mean_relative_abundances_no_migration_t), np.log10(mean_absolute_differences_no_migration_t))
    slope_global_migration, intercept_global_migration, r_value_global_migration, p_value_global_migration, std_err_global_migration = stats.linregress(np.log10(mean_relative_abundances_global_migration_t), np.log10(mean_absolute_differences_global_migration_t))

    print(t, slope_no_migration, slope_global_migration)

    ax_scatter_no_migration_slope.scatter(t, slope_no_migration, alpha=0.9, color=utils.rgb_blue[t], edgecolors='k')

    ax_scatter_global_migration_slope.scatter(t, slope_global_migration, alpha=0.9, color=utils.rgb_red[t], edgecolors='k')



ax_scatter_no_migration_slope.set_ylim([0.7, 1.05])
ax_scatter_global_migration_slope.set_ylim([0.7, 1.05])

ax_scatter_no_migration_slope.set_xlabel('Transfer', fontsize=12)
ax_scatter_global_migration_slope.set_xlabel('Transfer', fontsize=12)

ax_scatter_no_migration_slope.set_ylabel('Slope', fontsize=12)
ax_scatter_global_migration_slope.set_ylabel('Slope', fontsize=12)




ins_width_no_migration = inset_axes(ax_width_no_migration, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_no_migration.transAxes)
ins_width_global_migration = inset_axes(ax_width_global_migration, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_global_migration.transAxes)


shape_no_migration, loc_no_migration, scale_no_migration = stats.lognorm.fit(width_distribution_ratios_no_migration_log10)
shape_global_migration, loc_global_migration, scale_global_migration = stats.lognorm.fit(width_distribution_ratios_global_migration_log10)



x_range_no_migration = np.linspace(min(width_distribution_ratios_no_migration_log10)  , max(width_distribution_ratios_no_migration_log10) , 10000)
x_range_global_migration = np.linspace(min(width_distribution_ratios_global_migration_log10) , max(width_distribution_ratios_global_migration_log10) , 10000)

samples_fit_log_no_migration = stats.lognorm.pdf(x_range_no_migration, shape_no_migration, loc_no_migration, scale_no_migration)
samples_fit_log_global_migration = stats.lognorm.pdf(x_range_global_migration, shape_global_migration, loc_global_migration, scale_global_migration)

ins_width_no_migration.hist(width_distribution_ratios_no_migration_log10, histtype='step', color='k', lw=3, alpha=0.9, bins= 50, density=True, zorder=2)
ins_width_global_migration.hist(width_distribution_ratios_global_migration_log10, histtype='step', color='k', lw=3, alpha=0.8, bins= 50, density=True, zorder=2)




ins_width_no_migration.axvline(np.mean(x_range_no_migration), lw=2.5, ls=':', color=width_colors_no_migration[-3],zorder=3)
ins_width_global_migration.axvline(np.mean(x_range_global_migration), lw=2.5, ls=':', color=width_colors_global_migration[-3],zorder=3)


#ins_width_no_migration.plot(x_range_no_migration, samples_fit_log_no_migration, color=width_colors_no_migration[-3], label='Lognormal fit', lw=3, zorder=3)

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


#ins_width_no_migration.set_xlim([-2.4, 2.4])
#ins_width_global_migration.set_xlim([-2.4, 2.4])


#ins_width_global_migration.set_ylim([10**-4, 10**0])



#KS_statistic, p_value = stats.ks_2samp(width_distribution_ratios_no_migration, width_distribution_ratios_global_migration)
#sys.stdout.write("Difference between width dists: KS = %g, P= %g\n" % (KS_statistic, p_value))


# KS test of width distribution after migration treatment ends

def ks_test_permute_transfer_labels_per_asv(width_array, transfer_array, asv_array):

    log_width_array = np.log10(width_array)

    set_asv_array = list(set(asv_array))

    # before/after count dict
    n_obs_dict = {}
    for a in set_asv_array:
        n_obs_dict[a] = sum((transfer_array[asv_array==a] < 12))

    log_width_array_before_null = []
    log_width_array_after_null = []

    for a in set_asv_array:
        log_width_array_a = log_width_array[asv_array==a]
        np.random.permutation(log_width_array_a)

        log_width_array_a_before = log_width_array_a[:n_obs_dict[a]]
        log_width_array_a_after = log_width_array_a[n_obs_dict[a]:]

        log_width_array_before_null.append(log_width_array_a_before)
        log_width_array_after_null.append(log_width_array_a_after)

    log_width_array_before_null = np.concatenate(log_width_array_before_null).ravel()
    log_width_array_after_null = np.concatenate(log_width_array_after_null).ravel()


    ks_statistic_null, p_value_null = stats.ks_2samp(log_width_array_before_null, log_width_array_after_null)

    print(ks_statistic_null)






ks_test_permute_transfer_labels_per_asv(width_distribution_ratios_global_migration, width_distribution_transfers_global_migration, width_distribution_species_global_migration)

#log_width_distribution_ratios_no_migration = np.log10(width_distribution_ratios_no_migration)
#log_width_distribution_ratios_no_migration_before = log_width_distribution_ratios_no_migration[width_distribution_transfers_no_migration < 12]
#log_width_distribution_ratios_no_migration_after = log_width_distribution_ratios_no_migration[width_distribution_transfers_no_migration >= 12]


#log_width_distribution_ratios_global_migration = np.log10(width_distribution_ratios_global_migration)
#log_width_distribution_ratios_global_migration_before = log_width_distribution_ratios_global_migration[width_distribution_transfers_global_migration < 12]
#log_width_distribution_ratios_global_migration_after = log_width_distribution_ratios_global_migration[width_distribution_transfers_global_migration >= 12]

#print(log_width_distribution_ratios_no_migration_after)

#KS_statistic_no_migration, p_value_no_migration = stats.ks_2samp(log_width_distribution_ratios_no_migration_before, log_width_distribution_ratios_no_migration_after)
#KS_statistic_global_migration, p_value_global_migration = stats.ks_2samp(log_width_distribution_ratios_global_migration_before, log_width_distribution_ratios_global_migration_after)




fig.subplots_adjust(wspace=0.5, hspace=0.3)
#fig.savefig(utils.directory + "/figs/temporal_width_distribution_ratios.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/temporal_width_distribution_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
