from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from matplotlib import cm

number_transfers = 18




def get_temporal_patterns(migration_innoculum):

    relative_abundance_dict = utils.get_relative_abundance_dictionary_temporal_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])

    species_mean_relative_abundances = []
    species_mean_absolute_differences = []
    species_mean_width_distribution_ratios = []
    species_transfers = []

    species_mean_width_distribution_ratios_dict = {}

    for species, species_dict in relative_abundance_dict.items():

        species_abundance_difference_dict = {}

        for replicate, species_replicate_dict in species_dict.items():

            if len(species_replicate_dict['transfers']) < 2:
                continue

            transfers = np.asarray(species_replicate_dict['transfers'])

            relative_abundances = np.asarray(species_replicate_dict['relative_abundances'])

            transfer_differences = transfers[:-1]
            absolute_differences = np.abs(relative_abundances[1:] - relative_abundances[:-1])

            #width_distribution_ratios = np.abs(relative_abundances[1:] / relative_abundances[:-1])
            width_distribution_ratios = relative_abundances[1:] / relative_abundances[:-1]

            for transfer_difference_idx, transfer_difference in enumerate(transfer_differences):

                if str(transfer_difference) not in species_abundance_difference_dict:
                    species_abundance_difference_dict[str(transfer_difference)] = {}
                    species_abundance_difference_dict[str(transfer_difference)]['absolute_differences'] = []
                    species_abundance_difference_dict[str(transfer_difference)]['relative_abundances'] = []
                    species_abundance_difference_dict[str(transfer_difference)]['width_distribution_ratios'] = []

                species_abundance_difference_dict[str(transfer_difference)]['absolute_differences'].append(absolute_differences[transfer_difference_idx])
                species_abundance_difference_dict[str(transfer_difference)]['relative_abundances'].append(relative_abundances[transfer_difference_idx])
                species_abundance_difference_dict[str(transfer_difference)]['width_distribution_ratios'].append(width_distribution_ratios[transfer_difference_idx])


        for transfer, transfer_dict in species_abundance_difference_dict.items():

            if len(transfer_dict['relative_abundances']) < 3:
                continue

            species_mean_relative_abundances.append(np.mean(np.log10(transfer_dict['relative_abundances'])))
            species_mean_absolute_differences.append(np.mean(np.log10(transfer_dict['absolute_differences'])))
            species_mean_width_distribution_ratios.append(np.mean(np.log10(transfer_dict['width_distribution_ratios'])))
            species_transfers.append(int(transfer))


            if int(transfer) not in species_mean_width_distribution_ratios_dict:
                species_mean_width_distribution_ratios_dict[int(transfer)] = []

            species_mean_width_distribution_ratios_dict[int(transfer)].append(np.mean(np.log10(transfer_dict['width_distribution_ratios'])))


    # calcualte variance in width distribution
    variance_width_distribution_transfers = []
    variance_width_distribution = []

    for transfer_, widths_ in species_mean_width_distribution_ratios_dict.items():

        widths_ = np.asarray(widths_)

        if transfer_ == 6:
            widths_ = widths_[widths_<-1.8]

        if len(widths_) < 4:
            continue

        variance_width_distribution_transfers.append(transfer_)
        variance_width_distribution.append(np.var(widths_))

    variance_width_distribution_transfers = np.asarray(variance_width_distribution_transfers)
    variance_width_distribution = np.asarray(variance_width_distribution)

    species_mean_relative_abundances = np.asarray(species_mean_relative_abundances)
    species_mean_absolute_differences = np.asarray(species_mean_absolute_differences)
    species_mean_width_distribution_ratios = np.asarray(species_mean_width_distribution_ratios)

    species_transfers = np.asarray(species_transfers)

    return species_transfers, species_mean_relative_abundances, species_mean_absolute_differences, species_mean_width_distribution_ratios, variance_width_distribution_transfers, variance_width_distribution



species_transfers_no_migration, mean_relative_abundances_no_migration, mean_absolute_differences_no_migration, mean_width_distribution_ratios_no_migration, variance_transfers_no_migration, variance_no_migration = get_temporal_patterns(('No_migration',4))
species_transfers_global_migration, mean_relative_abundances_global_migration, mean_absolute_differences_global_migration, mean_width_distribution_ratios_global_migration, variance_transfers_global_migration, variance_global_migration = get_temporal_patterns(('Global_migration',4))

width_colors_no_migration = [utils.rgb[width_transfer_] for width_transfer_ in variance_transfers_no_migration]
width_colors_global_migration = [utils.rgb_red[width_transfer_] for width_transfer_ in variance_transfers_global_migration]

colors_no_migration = [utils.rgb[species_transfer] for species_transfer in species_transfers_no_migration]
colors_global_migration = [utils.rgb_red[species_transfer] for species_transfer in species_transfers_global_migration]


fig = plt.figure(figsize = (8, 12)) #
fig.subplots_adjust(bottom= 0.15)


ax_scatter_no_migration = plt.subplot2grid((3, 2), (0, 0))#, colspan=1)
ax_scatter_global_migration = plt.subplot2grid((3, 2), (0, 1))#, colspan=1)

ax_width_no_migration = plt.subplot2grid((3, 2), (1, 0))#, colspan=1)
ax_width_global_migration = plt.subplot2grid((3, 2), (1, 1))#, colspan=1)

ax_lognormal = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_variance = plt.subplot2grid((3, 2), (2, 1))#, colspan=1)

mean_relative_abundances_no_migration_filter = mean_relative_abundances_no_migration[mean_relative_abundances_no_migration< -1]
mean_absolute_differences_no_migration_filter = mean_absolute_differences_no_migration[mean_relative_abundances_no_migration< -1]

ax_scatter_no_migration.scatter(10**mean_relative_abundances_no_migration, 10**mean_absolute_differences_no_migration, alpha=1, color=colors_no_migration, edgecolors='k')
ax_scatter_no_migration.set_xscale('log', basex=10)
ax_scatter_no_migration.set_yscale('log', basey=10)
ax_scatter_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_no_migration.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)
slope_no_migration, intercept_no_migration, r_value_no_migration, p_value_no_migration, std_err_no_migration = stats.linregress(mean_relative_abundances_no_migration_filter, mean_absolute_differences_no_migration_filter)
x_log10_range_no_migration =  np.linspace(min(mean_relative_abundances_no_migration) , max(mean_relative_abundances_no_migration) , 10000)
y_log10_fit_range_no_migration = 10 ** (slope_no_migration*x_log10_range_no_migration + intercept_no_migration)
ax_scatter_no_migration.plot(10**x_log10_range_no_migration, y_log10_fit_range_no_migration, c='k', lw=3, linestyle='--', zorder=2)
ax_scatter_no_migration.set_title(utils.titles_dict[('No_migration',4)], fontsize=12, fontweight='bold' )
ax_scatter_no_migration.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_no_migration, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_scatter_no_migration.transAxes  )

ax_scatter_no_migration.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)


mean_relative_abundances_global_migration_filter = mean_relative_abundances_global_migration[mean_relative_abundances_global_migration< -1]
mean_absolute_differences_global_migration_filter = mean_absolute_differences_global_migration[mean_relative_abundances_global_migration< -1]

ax_scatter_global_migration.scatter(10**mean_relative_abundances_global_migration, 10**mean_absolute_differences_global_migration, alpha=1, color=colors_global_migration, edgecolors='k')
ax_scatter_global_migration.set_xscale('log', basex=10)
ax_scatter_global_migration.set_yscale('log', basey=10)
ax_scatter_global_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_global_migration.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)
slope_global_migration, intercept_global_migration, r_value_global_migration, p_value_global_migration, std_err_global_migration = stats.linregress(mean_relative_abundances_global_migration_filter, mean_absolute_differences_global_migration_filter)
x_log10_range_global_migration =  np.linspace(min(mean_relative_abundances_global_migration) , max(mean_relative_abundances_global_migration) , 10000)
y_log10_fit_range_global_migration = 10 ** (slope_global_migration*x_log10_range_global_migration + intercept_global_migration)
ax_scatter_global_migration.plot(10**x_log10_range_global_migration, y_log10_fit_range_global_migration, c='k', lw=3, linestyle='--', zorder=2)
ax_scatter_global_migration.set_title(utils.titles_dict[('Global_migration',4)], fontsize=12, fontweight='bold' )
ax_scatter_global_migration.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_global_migration, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_scatter_global_migration.transAxes  )

ax_scatter_global_migration.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)



ax_width_no_migration.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_no_migration.scatter(10**mean_relative_abundances_no_migration, 10**mean_width_distribution_ratios_no_migration, alpha=1, color=colors_no_migration, zorder=2, edgecolors='k')
ax_width_no_migration.set_xscale('log', basex=10)
ax_width_no_migration.set_yscale('log', basey=10)
ax_width_no_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_no_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)



ax_width_global_migration.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_global_migration.scatter(10**mean_relative_abundances_global_migration, 10**mean_width_distribution_ratios_global_migration, alpha=1, color=colors_global_migration, zorder=2, edgecolors='k')
ax_width_global_migration.set_xscale('log', basex=10)
ax_width_global_migration.set_yscale('log', basey=10)
ax_width_global_migration.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_global_migration.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)




#ax_1.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )


#numpy.concatenate([a,b])


# stats


KS_statistic, p_value = stats.ks_2samp(mean_width_distribution_ratios_no_migration, mean_width_distribution_ratios_global_migration)
sys.stdout.write("KS = %g, P= %g\n" % (KS_statistic, p_value))



shape_no_migration, loc_no_migration, scale_no_migration = stats.lognorm.fit(mean_width_distribution_ratios_no_migration)
shape_global_migration, loc_global_migration, scale_global_migration = stats.lognorm.fit(mean_width_distribution_ratios_global_migration)

x_range_no_migration = np.linspace(min(mean_width_distribution_ratios_no_migration) , max(mean_width_distribution_ratios_no_migration) , 10000)
x_range_global_migration = np.linspace(min(mean_width_distribution_ratios_global_migration) , max(mean_width_distribution_ratios_global_migration) , 10000)

ax_lognormal.hist(mean_width_distribution_ratios_no_migration, histtype='step', color=colors_no_migration[8], lw=3, alpha=0.8, bins= 12, density=True, zorder=2)
ax_lognormal.hist(mean_width_distribution_ratios_global_migration, histtype='step', color=colors_global_migration[8], lw=3, alpha=0.8, bins= 12, density=True, zorder=2)

samples_fit_log_no_migration = stats.lognorm.pdf(x_range_no_migration, shape_no_migration, loc_no_migration, scale_no_migration)
samples_fit_log_global_migration = stats.lognorm.pdf(x_range_global_migration, shape_global_migration, loc_global_migration, scale_global_migration)

ax_lognormal.axvline(0, lw=3, ls=':',color='darkgrey', zorder=3)


ax_lognormal.plot(x_range_no_migration, samples_fit_log_no_migration, color=width_colors_no_migration[-3], label='Lognormal fit', lw=3, zorder=3)
ax_lognormal.plot(x_range_global_migration, samples_fit_log_global_migration, color=width_colors_global_migration[-3], label='Lognormal fit', lw=3, zorder=3)

ax_lognormal.set_yscale('log', basey=10)

ax_lognormal.set_xlabel('Width distribution of relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$' + ', log10', fontsize=12)
ax_lognormal.set_ylabel('Probability density', fontsize=12)

ax_lognormal.legend(loc="lower center", fontsize=8)



ax_variance.plot(variance_transfers_no_migration, variance_no_migration, color = 'k', zorder=1)
ax_variance.plot(variance_transfers_global_migration, variance_global_migration, color = 'k', zorder=1)


ax_variance.scatter(variance_transfers_no_migration, variance_no_migration, color = width_colors_no_migration, edgecolors='k', zorder=2)
ax_variance.scatter(variance_transfers_global_migration, variance_global_migration, color = width_colors_global_migration, edgecolors='k', zorder=2)

ax_variance.set_xlabel('Transfer', fontsize=12)
ax_variance.set_ylabel('Variance of log-transformed\nrelative abundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)

## Variance difference test

transfers_intersect = np.intersect1d(variance_transfers_no_migration, variance_transfers_global_migration)

sorter_no_migration = np.argsort(variance_transfers_no_migration)
no_migration_idx = sorter_no_migration[np.searchsorted(variance_transfers_no_migration, transfers_intersect, sorter=sorter_no_migration)]
variances_no_migration_intersect = variance_no_migration[no_migration_idx]

sorter_global_migration = np.argsort(variance_transfers_global_migration)
global_migration_idx = sorter_global_migration[np.searchsorted(variance_transfers_global_migration, transfers_intersect, sorter=sorter_global_migration)]
variances_global_migration_intersect = variance_global_migration[global_migration_idx]

variances_matrix = np.array([variances_no_migration_intersect, variances_global_migration_intersect])
mean_ratio_observed = sum(variances_matrix[0,:] - variances_matrix[1,:])

null_values = []
for i in range(10000):
    variances_matrix_copy = np.copy(variances_matrix)
    for i in range(variances_matrix_copy.shape[1]):
        np.random.shuffle(variances_matrix_copy[:,i])

    null_values.append(sum(variances_matrix_copy[0,:] - variances_matrix_copy[1,:]))


null_values = np.asarray(null_values)
p_value_variance = len(null_values[null_values < mean_ratio_observed]) / 10000

sys.stdout.write("Variance difference test P = %g\n" % ( p_value_variance))

ax_variance.text(0.75,0.9, '$P< 0.05$', fontsize=12, color='k', ha='center', va='center', transform=ax_variance.transAxes )


#variance_no_migration_exp = 10**variance_no_migration
#variance_global_migration_exp = 10**variance_global_migration

#print(np.mean(variance_no_migration_exp/variance_global_migration_exp) )



fig.subplots_adjust(wspace=0.5, hspace=0.3)
fig.savefig(utils.directory + "/figs/temporal_width_distribution_ratios.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
