from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

import utils
import plot_utils
import slm_simulation_utils

from matplotlib import cm


zeros = True

np.random.seed(123456789)

migration_innocula = [('No_migration',4), ('Parent_migration',4), ('Global_migration',4)]

fig = plt.figure(figsize = (12, 12)) #
#fig = plt.figure(figsize = (16, 8)) #
fig.subplots_adjust(bottom= 0.15)



t_slope_all = []
plot_count = 0
for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

    sys.stdout.write("Running analyses for %s treatment\n" % (migration_innoculum[0]))

    moments_dict = {}

    for trasfer_idx, transfer in enumerate(utils.transfers):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0], inocula=migration_innoculum[1])

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=zeros)


        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))


        ax_plot = plt.subplot2grid((3, 3), (trasfer_idx, migration_innoculum_idx), colspan=1)

        ax_plot.scatter(means, variances, alpha=0.8, c=utils.color_dict_range[migration_innoculum][transfer-2].reshape(1,-1), edgecolors='k')#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        y_log10_null_range = 10 ** (utils.slope_null*x_log10_range + intercept)

        ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
        #ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Slope = 2")
        ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label=r'$y \sim x^{2}$')

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )

        ax_plot.set_xlabel('Mean relative\nabundance', fontsize=12)
        ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)


        if trasfer_idx == 0:

            title = utils.titles_no_inocula_dict[migration_innoculum]
            ax_plot.set_title(title, fontsize=12, fontweight='bold' )


        if plot_count == 0:
            ax_plot.legend(loc="lower right", fontsize=8)



        #t_value = (slope - (utils.slope_null))/std_err
        #p_value = stats.t.sf(np.abs(t_value), len(means)-2)

        #p_value_to_plot = utils.get_p_value(p_value)


        #ax_plot.text(0.2,0.8, r'$t=$' + str(round(t_value,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_plot.transAxes )
        #ax_plot.text(0.2,0.7, p_value_to_plot, fontsize=10, color='k', ha='center', va='center', transform=ax_plot.transAxes )

        #print(migration_innoculum[0], transfer)

        sys.stdout.write("Transfer %d, Slope = %g\n" % (transfer, slope))
        sys.stdout.write("Transfer %d, Intercept = %g\n" % (transfer, intercept))
        #print(slope, intercept)

        

        if migration_innoculum_idx == 0:

            ax_plot.text(-0.3,0.5, 'Transfer %d' % transfer, fontsize=12, fontweight='bold', color='k', ha='center', rotation=90, va='center', transform=ax_plot.transAxes )

        
        #ax_plot.text(-0.1, 1.04, plot_utils.sub_plot_labels[3*migration_innoculum_idx + trasfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_plot.transAxes)
        ax_plot.text(-0.1, 1.04, plot_utils.sub_plot_labels[migration_innoculum_idx + 3*trasfer_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_plot.transAxes)

        
        plot_count+=1

        # save for paired t-test
        idx_to_keep = means<0.95

        means = means[idx_to_keep]
        variances = variances[idx_to_keep]

        moments_dict[transfer] = {}
        moments_dict[transfer]['mean_log10'] = np.log10(means)
        moments_dict[transfer]['variance_log10'] = np.log10(variances)



    # run t-test
    mean_log10_18 = np.asarray(moments_dict[18]['mean_log10'])
    variance_log10_18 = np.asarray(moments_dict[18]['variance_log10'])
    mean_log10_12 = np.asarray(moments_dict[12]['mean_log10'])
    variance_log10_12 = np.asarray(moments_dict[12]['variance_log10'])
    slope_18, slope_12, t_slope, intercept_18, intercept_12, t_intercept, r_value_18, r_value_12 = utils.t_statistic_two_slopes(mean_log10_18, variance_log10_18, mean_log10_12, variance_log10_12)

    # permute temporal labels while prerving (mean, var) pairs
    n_12 = len(mean_log10_12)
    n_18 = len(mean_log10_18)

    merged_mean_log10 = np.concatenate((mean_log10_18, mean_log10_12))
    merged_var_log10 = np.concatenate((variance_log10_18, variance_log10_12))

    idx_slope = np.arange(n_12 + n_18)

    slope_18, slope_12, t_slope, intercept_18, intercept_12, t_intercept, r_value_18, r_value_12 = utils.t_statistic_two_slopes(mean_log10_18, variance_log10_18, mean_log10_12, variance_log10_12)
    t_slope_null_all = []
    t_intercept_null_all = []
    iter_ = 10000
    for i in range(iter_):

        np.random.shuffle(idx_slope)

        merged_mean_log10_12_i = merged_mean_log10[idx_slope[:n_12]]
        merged_mean_log10_18_i = merged_mean_log10[idx_slope[n_12:]]

        merged_var_log10_12_i = merged_var_log10[idx_slope[:n_12]]
        merged_var_log10_18_i = merged_var_log10[idx_slope[n_12:]]

        slope_18_null, slope_12_null, t_slope_null, intercept_18_null, intercept_12_null, t_intercept_null, r_value_18_null, r_value_12_null = utils.t_statistic_two_slopes(merged_mean_log10_18_i, merged_var_log10_18_i, merged_mean_log10_12_i, merged_var_log10_12_i)
        t_slope_null_all.append(t_slope_null)
        t_intercept_null_all.append(t_intercept_null)


    t_slope_null_all = np.asarray(t_slope_null_all)
    t_intercept_null_all = np.asarray(t_intercept_null_all)

    p_value_slope = sum(np.absolute(t_slope_null_all) > np.absolute(t_slope))/iter_
    p_value_intercept = sum(np.absolute(t_intercept_null_all) > np.absolute(t_intercept))/iter_
    

    sys.stdout.write("Slope difference test, t = %g, P = %g\n" % (t_slope, p_value_slope))
    sys.stdout.write("Intercept difference test, t = %g, P = %g\n" % (t_intercept, p_value_intercept))

    t_slope_all.append(t_slope)




treatments_no_innoculum = ['no_migration', 'parent_migration', 'global_migration']

def run_best_parameter_simulations():

    simulation_all_migration_abc_dict = slm_simulation_utils.load_simulation_all_migration_abc_dict()

    tau_all = np.asarray(simulation_all_migration_abc_dict['tau_all'])
    sigma_all = np.asarray(simulation_all_migration_abc_dict['sigma_all'])

    for treatment_idx, treatment in enumerate(treatments_no_innoculum):
        
        slope_t_test_simulation = np.asarray(simulation_all_migration_abc_dict['slope_12_vs_18'][treatment]['slope_t_test'])

        euc_dist = np.sqrt((t_slope_all[treatment_idx] - slope_t_test_simulation)**2)
        min_parameter_idx = np.argmin(euc_dist)

        tau_best = tau_all[min_parameter_idx]
        sigma_best = sigma_all[min_parameter_idx]
        
        label = '%s_taylors' % treatment
        slm_simulation_utils.run_simulation_all_migration_fixed_parameters(tau_best, sigma_best, label, n_iter=1000)



for treatment_idx, treatment in enumerate(treatments_no_innoculum):

    label = '%s_taylors' % treatment
    simulation_all_migration_fixed_parameters_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_dict(label)
    
    slope_t_test = np.asarray(simulation_all_migration_fixed_parameters_dict['slope_12_vs_18'][treatment]['slope_t_test'])

    tau_best = simulation_all_migration_fixed_parameters_dict['tau_all']
    sigma_best = simulation_all_migration_fixed_parameters_dict['sigma_all']

    ax = plt.subplot2grid((3, 3), (2, treatment_idx), colspan=1)

    p_value = sum(slope_t_test > t_slope_all[treatment_idx])/len(slope_t_test)
    #print(p_value)

    ax.hist(slope_t_test, lw=3, alpha=0.8, bins=10, color=utils.color_dict[migration_innocula[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
    ax.axvline(x=0, ls=':', lw=3, c='k', label='Null')
    ax.axvline(x=t_slope_all[treatment_idx], ls='--', lw=3, c='k', label='Observed ' +  r'$t_{\mathrm{slope}}$')
    ax.set_xlabel('Simulated ' + r'$t_{\mathrm{slope}}$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=11)
    ax.set_ylabel('Probability density',  fontsize=11)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[6 + treatment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


    if treatment_idx == 0:
        ax.legend(loc="upper right", fontsize=8)




#slope_t_test_simulation_no_migration = np.asarray(simulation_all_migration_abc_dict['slope_12_vs_18']['no_migration']['slope_t_test'])
#slope_t_test_simulation_parent_migration = np.asarray(simulation_all_migration_abc_dict['slope_12_vs_18']['parent_migration']['slope_t_test'])
#slope_t_test_simulation_global_migration = np.asarray(simulation_all_migration_abc_dict['slope_12_vs_18']['global_migration']['slope_t_test'])

#t_slope_all = np.asarray(t_slope_all)
#slope_t_test_simulation = np.asarray([slope_t_test_simulation_no_migration, slope_t_test_simulation_parent_migration, slope_t_test_simulation_global_migration])

#t_slope_all = np.asarray([t_slope_all[1]])
#slope_t_test_simulation = np.asarray([slope_t_test_simulation_parent_migration])


#best_tau, best_sigma = utils.weighted_euclidean_distance(tau_all, sigma_all, t_slope_all, slope_t_test_simulation)




fig.subplots_adjust(wspace=0.32, hspace=0.4)
#fig.savefig(utils.directory + "/figs/taylors_law_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/taylors_law_migration.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)

plt.close()
