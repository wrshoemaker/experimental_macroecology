from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


np.random.seed(123456789)


attractors = utils.attractors


species_names_a, species_transfers_a, mean_relative_abundances_a, mean_absolute_differences_a, mean_width_distribution_ratios_a, variance_transfers_a, variance_a, mean_a = utils.get_temporal_patterns(('No_migration',4), attractor=attractors[0])
species_names_p, species_transfers_p, mean_relative_abundances_p, mean_absolute_differences_p, mean_width_distribution_ratios_p, variance_transfers_p, variance_p, mean_p = utils.get_temporal_patterns(('No_migration',4), attractor=attractors[1])

# get the ESVs that are in that weird cluster for Pseudo
species_names_p_outliers = species_names_p[ (mean_relative_abundances_p > -0.3)  & (mean_absolute_differences_p  > -4)]
outlying_species = set(species_names_p_outliers)
### BLAST shows that this ESV belongs to Enterobacteriaceae

width_colors_a = [utils.rgb_alcaligenaceae[width_transfer_] for width_transfer_ in variance_transfers_a]
width_colors_p = [utils.rgb_pseudomonadaceae[width_transfer_] for width_transfer_ in variance_transfers_p]

colors_a = [utils.rgb_alcaligenaceae[species_transfer] for species_transfer in species_transfers_a]
colors_p = [utils.rgb_pseudomonadaceae[species_transfer] for species_transfer in species_transfers_p]


# get slope for each timepoint

def calculate_slopes_each_timepoint(species_transfers, mean_relative_abundances, mean_absolute_differences):
    species_transfers_unique = list(set(species_transfers))
    species_transfers_unique.sort()

    slopes = []
    for species_transfer in species_transfers_unique:

        mean_relative_abundances_i = mean_relative_abundances[species_transfers == species_transfer]
        mean_absolute_differences_i = mean_absolute_differences[species_transfers == species_transfer]
        slope_i, intercept_i, r_value_i, p_value_i, std_err_i = stats.linregress(mean_relative_abundances_i, mean_absolute_differences_i)
        slopes.append(slope_i)

    transfers = np.asarray(species_transfers_unique)
    slopes = np.asarray(slopes)

    return transfers, slopes





fig = plt.figure(figsize = (8, 12)) #
fig.subplots_adjust(bottom= 0.15)



ax_scatter_a = plt.subplot2grid((3, 2), (0, 0))#, colspan=1)
ax_scatter_p = plt.subplot2grid((3, 2), (0, 1))#, colspan=1)


ax_width_a = plt.subplot2grid((3, 2), (1, 0))#, colspan=1)
ax_width_p = plt.subplot2grid((3, 2), (1, 1))#, colspan=1)

#ax_lognormal = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_mean = plt.subplot2grid((3, 2), (2, 0))#, colspan=1)
ax_cv = plt.subplot2grid((3, 2), (2, 1))#, colspan=1)

mean_relative_abundances_filter_a = mean_relative_abundances_a[mean_relative_abundances_a< -1]
mean_absolute_differences_filter_a = mean_absolute_differences_a[mean_relative_abundances_a< -1]

ax_scatter_a.scatter(10**mean_relative_abundances_a, 10**mean_absolute_differences_a, alpha=1, color=colors_a, edgecolors='k')
ax_scatter_a.set_xscale('log', basex=10)
ax_scatter_a.set_yscale('log', basey=10)
ax_scatter_a.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_a.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)
slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(mean_relative_abundances_filter_a, mean_absolute_differences_filter_a)
x_log10_range_a =  np.linspace(min(mean_relative_abundances_filter_a) , max(mean_relative_abundances_filter_a) , 10000)
y_log10_fit_range_a = 10 ** (slope_a*x_log10_range_a + intercept_a)
ax_scatter_a.set_title(utils.attractor_latex_dict[attractors[0]], fontsize=12, fontweight='bold' )


#transfers_a, slopes_a = calculate_slopes_each_timepoint(species_transfers_a, mean_relative_abundances_a, mean_absolute_differences_a)
cutoffs_a, slopes_cutoff_a = utils.get_slopes_cutoffs(mean_relative_abundances_a, mean_absolute_differences_a)

#slopes_colors_a = [utils.rgb_alcaligenaceae[transfer_] for transfer_ in transfers_a]
#width_colors_p = [utils.rgb_pseudomonadaceae[width_transfer_] for width_transfer_ in variance_transfers_p]

ins_a = inset_axes(ax_scatter_a, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.68,0.11,0.3,0.3), bbox_transform=ax_scatter_a.transAxes)
#ins_a.set_xlabel("Transfer", fontsize=8)
#ins_a.set_ylabel("Slope", fontsize=8)

#ins_no_migration.plot(transfers_a, slopes_a, ls='-', c='k')
#ins_a.plot(transfers_a, slopes_a, color = 'k', zorder=2)
#ins_a.scatter(transfers_a, slopes_a, color = slopes_colors_a, edgecolors='k', zorder=3)
#ins_no_migration.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)
#ins_a.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
#ins_a.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

#ins_no_migration.set_xscale('log', basex=10)
#ins_a.tick_params(labelsize=5)
#ins_a.tick_params(axis='both', which='major', pad=1)
#ins_a.set_ylim(0.3, 1.2)
#ins_no_migration.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )
#ins_no_migration.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )

ins_a.set_xlabel('Max.' + r'$\left \langle x(t) \right \rangle$', fontsize=8)
ins_a.set_ylabel("Slope", fontsize=8)

ins_a.plot(10**cutoffs_a, slopes_cutoff_a, ls='-', c='k')
ins_a.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)

ins_a.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
ins_a.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

ins_a.set_xscale('log', basex=10)
ins_a.tick_params(labelsize=5)
ins_a.tick_params(axis='both', which='major', pad=1)
ins_a.set_ylim(0.65, 1.02)

ins_a.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_a.transAxes )
ins_a.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_a.transAxes )






#
mean_relative_abundances_filter_p = mean_relative_abundances_p[mean_relative_abundances_p< -1]
mean_absolute_differences_filter_p = mean_absolute_differences_p[mean_relative_abundances_p< -1]

mean_relative_abundances_p_no_outlier = mean_relative_abundances_p[ (mean_relative_abundances_p <= -0.3)  ]
mean_absolute_differences_p_no_outlier = mean_absolute_differences_p[ (mean_relative_abundances_p <= -0.3)  ]
species_transfers_p_no_outlier = species_transfers_p[(mean_relative_abundances_p <= -0.3)]

#colors_p_no_outlier = [utils.rgb_pseudomonadaceae[species_transfer] for species_transfer in species_transfers_p_no_outlier]


ax_scatter_p.scatter(10**mean_relative_abundances_p, 10**mean_absolute_differences_p, alpha=1, color=colors_p, edgecolors='k')
ax_scatter_p.set_xscale('log', basex=10)
ax_scatter_p.set_yscale('log', basey=10)
ax_scatter_p.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_scatter_p.set_ylabel('Average difference between\ntime points,' + r'$\left \langle \left |  x(t + \delta t)  - x(t )\right | \right \rangle$', fontsize=12)
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(mean_relative_abundances_filter_p, mean_absolute_differences_filter_p)
x_log10_range_p =  np.linspace(min(mean_relative_abundances_filter_p) , max(mean_relative_abundances_filter_p) , 10000)
y_log10_fit_range_p = 10 ** (slope_p*x_log10_range_p + intercept_p)
ax_scatter_p.set_title(utils.attractor_latex_dict[attractors[1]], fontsize=12, fontweight='bold' )


#transfers_p, slopes_p = calculate_slopes_each_timepoint(species_transfers_p_no_outlier, mean_relative_abundances_p_no_outlier, mean_absolute_differences_p_no_outlier)
cutoffs_p, slopes_cutoff_p = utils.get_slopes_cutoffs(mean_relative_abundances_p_no_outlier, mean_absolute_differences_p_no_outlier)

#slopes_colors_p = [utils.rgb_pseudomonadaceae[transfer_] for transfer_ in transfers_p]
#width_colors_p = [utils.rgb_pseudomonadaceae[width_transfer_] for width_transfer_ in variance_transfers_p]

ins_p = inset_axes(ax_scatter_p, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.68,0.11,0.3,0.3), bbox_transform=ax_scatter_p.transAxes)
#ins_p.set_xlabel("Transfer", fontsize=8)
#ins_p.set_ylabel("Slope", fontsize=8)

#ins_no_migration.plot(transfers_a, slopes_a, ls='-', c='k')
#ins_p.plot(transfers_p, slopes_p, color = 'k', zorder=2)
#ins_p.scatter(transfers_p, slopes_p, color = slopes_colors_p, edgecolors='k', zorder=3)
#ins_no_migration.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)
#ins_p.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
#ins_p.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

#ins_no_migration.set_xscale('log', basex=10)
#ins_p.tick_params(labelsize=5)
#ins_p.tick_params(axis='both', which='major', pad=1)
#ins_p.set_ylim(0.3, 1.2)
#ins_no_migration.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )
#ins_no_migration.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_no_migration.transAxes )

ins_p.set_xlabel('Max.' + r'$\left \langle x(t) \right \rangle$', fontsize=8)
ins_p.set_ylabel("Slope", fontsize=8)

ins_p.plot(10**cutoffs_p, slopes_cutoff_p, ls='-', c='k')
ins_p.axvline(0.1, lw=1.5, ls=':',color='k', zorder=1)

ins_p.axhline(2/3, lw=1.5, ls='--',color='k', zorder=1)
ins_p.axhline(1, lw=1.5, ls='--',color='k', zorder=1)

ins_p.set_xscale('log', basex=10)
ins_p.tick_params(labelsize=5)
ins_p.tick_params(axis='both', which='major', pad=1)
ins_p.set_ylim(0.65, 1.02)

ins_p.text(0.21,0.88, 'Linear', fontsize=5, color='k', ha='center', va='center', transform=ins_p.transAxes )
ins_p.text(0.26,0.18, 'Square\nroot', fontsize=5, color='k', ha='center', va='center', transform=ins_p.transAxes )







ax_width_a.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_a.scatter(10**mean_relative_abundances_a, 10**mean_width_distribution_ratios_a, alpha=1, color=colors_a, zorder=2, edgecolors='k')
ax_width_a.set_xscale('log', basex=10)
ax_width_a.set_yscale('log', basey=10)
ax_width_a.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_a.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)
ax_width_a.set_xlim([0.5e-4, 1.3])
ax_width_a.set_ylim([0.5e-2, 3e2])


shape_a, loc_a, scale_a = stats.lognorm.fit(mean_width_distribution_ratios_a)
x_range_a = np.linspace(min(mean_width_distribution_ratios_a)  , max(mean_width_distribution_ratios_a) , 10000)
samples_fit_log_a = stats.lognorm.pdf(x_range_a, shape_a, loc_a, scale_a)
ks_a, p_a = stats.kstest(mean_width_distribution_ratios_a, 'lognorm', args=(shape_a, loc_a, scale_a), alternative='two-sided')
sys.stdout.write("KS test against lognormal, Alcaligenaceae: D = %g,  P = %g\n" % (ks_a, p_a))


ins_width_a = inset_axes(ax_width_a, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_a.transAxes)
ins_width_a.hist(mean_width_distribution_ratios_a, histtype='step', color=colors_a[8], lw=3, alpha=0.8, bins= 12, density=True, zorder=2)
ins_width_a.axvline(np.mean(x_range_a), lw=2.5, ls=':', color=width_colors_a[-3],zorder=3)
ins_width_a.plot(x_range_a, samples_fit_log_a, color=width_colors_a[-3], label='Lognormal fit', lw=3, zorder=3)
ins_width_a.set_yscale('log', basey=10)
ins_width_a.set_xlabel('Width dist., ' + r'$\mathrm{log}_{10}$', fontsize=7)
ins_width_a.set_ylabel('Prob. density', fontsize=7)
ins_width_a.tick_params(labelsize=5)
ins_width_a.tick_params(axis='both', which='major', pad=1)
ins_width_a.set_xlim([-2.4, 2.4])





ax_width_p.axhline(1, lw=3, ls=':',color='k', zorder=1)
ax_width_p.scatter(10**mean_relative_abundances_p, 10**mean_width_distribution_ratios_p, alpha=1, color=colors_p, zorder=2, edgecolors='k')
ax_width_p.set_xscale('log', basex=10)
ax_width_p.set_yscale('log', basey=10)
ax_width_p.set_xlabel('Average relative abundance\nat time $t$, ' + r'$\left \langle x(t) \right \rangle$', fontsize=12)
ax_width_p.set_ylabel('Width distribution of relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)
#ax_width_p.set_xlim([0.8e-4, 1])
ax_width_p.set_xlim([0.5e-4, 1.3])
ax_width_p.set_ylim([0.5e-2, 3e2])



shape_p, loc_p, scale_p = stats.lognorm.fit(mean_width_distribution_ratios_p)
x_range_p = np.linspace(min(mean_width_distribution_ratios_p)  , max(mean_width_distribution_ratios_p) , 10000)
samples_fit_log_p = stats.lognorm.pdf(x_range_p, shape_p, loc_p, scale_p)
ks_p, p_p = stats.kstest(mean_width_distribution_ratios_p, 'lognorm', args=(shape_p, loc_p, scale_p), alternative='two-sided')
sys.stdout.write("KS test against lognormal, Pseudomonadaceae: D = %g,  P = %g\n" % (ks_p, p_p))


ins_width_p = inset_axes(ax_width_p, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.73,0.75,0.25,0.25), bbox_transform=ax_width_p.transAxes)
ins_width_p.hist(mean_width_distribution_ratios_p, histtype='step', color=colors_p[8], lw=3, alpha=0.8, bins= 12, density=True, zorder=2)
ins_width_p.axvline(np.mean(x_range_p), lw=2.5, ls=':', color=width_colors_p[-3],zorder=3)
ins_width_p.plot(x_range_p, samples_fit_log_p, color=width_colors_p[-3], label='Lognormal fit', lw=3, zorder=3)
ins_width_p.set_yscale('log', basey=10)
ins_width_p.set_xlabel('Width dist., ' + r'$\mathrm{log}_{10}$', fontsize=7)
ins_width_p.set_ylabel('Prob. density', fontsize=7)
ins_width_p.tick_params(labelsize=5)
ins_width_p.tick_params(axis='both', which='major', pad=1)
ins_width_p.set_xlim([-2.4, 2.4])





# test whether there's a significant difference in pooled width distributions
KS_statistic, p_value = stats.ks_2samp(mean_width_distribution_ratios_a, mean_width_distribution_ratios_p)
sys.stdout.write("Difference between attractor width dists: KS = %g, P= %g\n" % (KS_statistic, p_value))






transfers_intersect_mean, means_a_intersect, means_p_intersect  = utils.get_intersecting_timepoints(variance_transfers_a, mean_a, variance_transfers_p, mean_p)
transfers_intersect_cv, cv_a_intersect, cv_p_intersect  = utils.get_intersecting_timepoints(variance_transfers_a, variance_a, variance_transfers_p, variance_p)

mean_ratio_observed_mean, p_value_mean = utils.test_difference_two_time_series(means_a_intersect, means_p_intersect)
mean_ratio_observed_cv, p_value_cv = utils.test_difference_two_time_series(cv_a_intersect, cv_p_intersect)


sys.stdout.write("Mean difference test, D = %g,  P = %g\n" % (mean_ratio_observed_mean, p_value_mean))
sys.stdout.write("CV difference test, D = %g,  P = %g\n" % (mean_ratio_observed_cv, p_value_cv))



ax_mean.axhline(0, lw=3, ls=':',color='grey', zorder=1)
ax_mean.plot(variance_transfers_a, mean_a, color = 'k', zorder=2)
ax_mean.plot(variance_transfers_p, mean_p, color = 'k', zorder=2)
ax_mean.scatter(variance_transfers_a, mean_a, color = width_colors_a, edgecolors='k', zorder=3)
ax_mean.scatter(variance_transfers_p, mean_p, color = width_colors_p, edgecolors='k', zorder=3)
ax_mean.set_xlabel('Transfer', fontsize=12)
ax_mean.set_ylabel('Mean of log-transformed\nrelative abundance ratios, ' + r'$\mu$', fontsize=12)
ax_mean.set_ylim(-0.65, 0.65)
ax_mean.text(0.82,0.9, utils.get_p_value(p_value_mean), fontsize=12, color='k', ha='center', va='center', transform=ax_mean.transAxes )




ax_cv.axhline(1, lw=3, ls=':',color='grey', zorder=1)
ax_cv.plot(variance_transfers_a, variance_a, color = 'k', zorder=2)
ax_cv.plot(variance_transfers_p, variance_p, color = 'k', zorder=2)
ax_cv.scatter(variance_transfers_a, variance_a, color = width_colors_a, edgecolors='k', zorder=3)
ax_cv.scatter(variance_transfers_p, variance_p, color = width_colors_p, edgecolors='k', zorder=3)
ax_cv.set_xlabel('Transfer', fontsize=12)
#ax_cv.set_ylabel('CV of log-transformed relative\nabundance ratios, ' + r'$\left \langle \frac{x(t + \delta t) }{x(t ) } \right \rangle$', fontsize=12)
ax_cv.set_ylabel('CV of log-transformed\nrelative abundance ratios, ' + r'$\frac{\sigma}{\left | \mu \right |}$', fontsize=12)
ax_cv.set_yscale('log', basey=10)

#ax_cv.text(0.8,0.9, '$P< 0.05$', fontsize=12, color='k', ha='center', va='center', transform=ax_cv.transAxes )
ax_cv.text(0.65,0.9, utils.get_p_value(p_value_cv), fontsize=12, color='k', ha='center', va='center', transform=ax_cv.transAxes )









fig.subplots_adjust(wspace=0.5, hspace=0.3)
#fig.savefig(utils.directory + "/figs/temporal_width_distribution_ratios.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/temporal_width_distribution_attractor.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
