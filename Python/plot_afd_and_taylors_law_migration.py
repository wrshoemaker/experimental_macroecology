from __future__ import division
import os, sys
import numpy as np
import pickle

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from itertools import combinations
#import statsmodels.stats.multitest as multitest
import plot_utils
import slm_simulation_utils


#afd_migration_transfer_12
zeros = True
migration_innocula = [('No_migration',4), ('Parent_migration',4), ('Global_migration',4)]

ks_dict_path = "%s/data/afd_ks_dict.pickle" %  utils.directory

afd_dict = {}
transfers = np.asarray([12, 18])

experiments = [('No_migration',4), ('Parent_migration', 4) , ('Global_migration',4) ]
treatment_combinations = list(combinations(experiments,2))

rescaled_status_all = ['afd', 'afd_rescaled']

for experiment in experiments:

    afd_dict[experiment] = {}

    for transfer in transfers:

        #relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])
        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=experiment[0],inocula=experiment[1])
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        afd = rel_s_by_s.flatten()
        afd = afd[afd>0]
        afd = np.log10(afd)

        afd_rescaled = (afd - np.mean(afd))/np.std(afd)

        afd_dict[experiment][transfer] = {}
        afd_dict[experiment][transfer]['afd'] = afd
        afd_dict[experiment][transfer]['afd_rescaled'] = afd_rescaled



#
afd_paired_dict = {}
for experiment in experiments:

    afd_paired_dict[experiment] = {}

    s_by_s_dict = {}

    for transfer in transfers:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        s_by_s_dict[transfer] = {}
        s_by_s_dict[transfer]['rel_s_by_s'] = rel_s_by_s
        s_by_s_dict[transfer]['species'] = np.asarray(species)
        s_by_s_dict[transfer]['comm_rep_list'] = np.asarray(comm_rep_list)

    #communities_to_keep = list( set(s_by_s_dict[12]['comm_rep_list'].tolist() ) &  set(s_by_s_dict[18]['comm_rep_list'].tolist() ))
    communities_to_keep = np.intersect1d( s_by_s_dict[12]['comm_rep_list'], s_by_s_dict[18]['comm_rep_list'] )
    
    paired_rel_abundances_all = []
    for community in communities_to_keep:

        community_12_idx = np.where( s_by_s_dict[12]['comm_rep_list'] == community)[0][0]
        community_18_idx = np.where( s_by_s_dict[18]['comm_rep_list'] == community)[0][0]

        community_asv_union = np.union1d(s_by_s_dict[12]['species'], s_by_s_dict[18]['species'])

        for community_asv_j in community_asv_union:

            if community_asv_j in s_by_s_dict[12]['species']:
                
                community_asv_j_12_idx = np.where(s_by_s_dict[12]['species'] == community_asv_j)[0][0]
                rel_abundance_12 = s_by_s_dict[12]['rel_s_by_s'][community_asv_j_12_idx, community_12_idx]

            else:
                rel_abundance_12 = float(0)  

            
            if community_asv_j in s_by_s_dict[18]['species']:
                
                community_asv_j_18_idx = np.where(s_by_s_dict[18]['species'] == community_asv_j)[0][0]
                rel_abundance_18 = s_by_s_dict[18]['rel_s_by_s'][community_asv_j_18_idx, community_18_idx]

            else:
                rel_abundance_18 = float(0)


            if (rel_abundance_12 == float(0)) and (rel_abundance_18 == float(0)):
                continue

            paired_rel_abundances_all.append([rel_abundance_12, rel_abundance_18])

    afd_paired_dict[experiment] = paired_rel_abundances_all








def make_ks_dict():

    distances_dict = {}

    for combo in treatment_combinations:

        if combo not in distances_dict:
            distances_dict[combo] = {}

        for transfer in transfers:

            distances_dict[combo][transfer] = {}

            for rescaled_status in rescaled_status_all:

                print(combo, transfer, rescaled_status)

                afd_experiment_1 = afd_dict[combo[0]][transfer][rescaled_status]
                afd_experiment_2 = afd_dict[combo[1]][transfer][rescaled_status]

                ks_statistic, p_value = utils.run_permutation_ks_test(afd_experiment_1, afd_experiment_2, n=1000)

                distances_dict[combo][transfer][rescaled_status] = {}
                distances_dict[combo][transfer][rescaled_status]['D'] = ks_statistic
                distances_dict[combo][transfer][rescaled_status]['pvalue'] = p_value



    for experiment_idx, experiment in enumerate(experiments):

        distances_dict[experiment] = {}

        for rescaled_status in rescaled_status_all:

            #print(experiment, rescaled_status)

            ks_statistic, p_value = utils.run_permutation_ks_test(afd_dict[experiment][transfers[0]][rescaled_status], afd_dict[experiment][transfers[1]][rescaled_status], n=1000)

            distances_dict[experiment][rescaled_status] = {}
            distances_dict[experiment][rescaled_status]['D'] = ks_statistic
            distances_dict[experiment][rescaled_status]['pvalue'] = p_value



        ks_statistic, p_value = utils.run_permutation_ks_test_control(afd_paired_dict[experiment])

        distances_dict[experiment]['afd_rescaled_and_paired'] = {}
        distances_dict[experiment]['afd_rescaled_and_paired']['D'] = ks_statistic
        distances_dict[experiment]['afd_rescaled_and_paired']['pvalue'] = p_value


    with open(ks_dict_path, 'wb') as outfile:
        pickle.dump(distances_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_ks_dict():

    dict_ = pickle.load(open(ks_dict_path, "rb"))
    return dict_




#make_ks_dict()

ks_dict = load_ks_dict()

fig = plt.figure(figsize = (18, 30))
fig.subplots_adjust(bottom= 0.15)


rescaled_status = 'afd_rescaled'

x_label = 'Rescaled ' + r'$\mathrm{log}_{10}$' + ' relative abundance'

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((5, 3), (0, experiment_idx), colspan=1)

    for transfer in transfers:

        colors_experiment_transfer = utils.color_dict_range[experiment][transfer-1]
        afd = afd_dict[experiment][transfer][rescaled_status]
        #label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)
        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)


    ks_statistic = ks_dict[experiment][rescaled_status]['D']
    p_value = ks_dict[experiment][rescaled_status]['pvalue']

    #ax.text(0.70,0.7, '$D=%0.3f$' % ks_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.70,0.65, '$\mathrm{KS}=%0.3f$' % ks_statistic, fontsize=18, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.68,0.58, utils.get_p_value(p_value), fontsize=18, color='k', ha='center', va='center', transform=ax.transAxes )

    ax.set_title(utils.titles_no_inocula_dict[experiment], fontsize=25, fontweight='bold' )
    ax.legend(loc="upper right", fontsize=18)
    ax.set_xlabel(x_label, fontsize=19)
    ax.set_ylabel('Probability density', fontsize=20)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[experiment_idx], fontsize=18, fontweight='bold', ha='center', va='center', transform=ax.transAxes)



treatments_no_innoculum = ['no_migration', 'parent_migration', 'global_migration']

def run_best_parameter_simulations():

    # plot simulation
    simulation_dict = slm_simulation_utils.load_simulation_all_migration_abc_dict()

    tau_all = np.asarray(simulation_dict['tau_all'])
    sigma_all = np.asarray(simulation_dict['sigma_all'])

    for treatment_idx, treatment in enumerate(treatments_no_innoculum):

        ks_rescaled_12_vs_18_simulation = np.asarray(simulation_dict['ks_rescaled_12_vs_18'][treatment])

        ks_rescaled_12_vs_18 =  ks_dict[experiments[treatment_idx]][rescaled_status]['D']  

        euc_dist = np.sqrt((ks_rescaled_12_vs_18 - ks_rescaled_12_vs_18_simulation)**2)
        min_parameter_idx = np.argmin(euc_dist)

        tau_best = tau_all[min_parameter_idx]
        sigma_best = sigma_all[min_parameter_idx]

        label = '%s_afd' % treatment
        slm_simulation_utils.run_simulation_all_migration_fixed_parameters(tau_best, sigma_best, label, n_iter=1000)





for treatment_idx, treatment in enumerate(treatments_no_innoculum):

    label = '%s_afd' % treatment
    simulation_all_migration_fixed_parameters_dict = slm_simulation_utils.load_simulation_all_migration_fixed_parameters_dict(label)

    ks_simulated = np.asarray(simulation_all_migration_fixed_parameters_dict['ks_12_vs_18'][treatment])
    ks_observed = ks_dict[experiments[treatment_idx]]['afd_rescaled']['D']

    tau_best = simulation_all_migration_fixed_parameters_dict['tau_all']
    sigma_best = simulation_all_migration_fixed_parameters_dict['sigma_all']

    ax = plt.subplot2grid((5, 3), (1, treatment_idx), colspan=1)

    p_value = sum(ks_simulated > ks_observed)/len(ks_simulated)

    print(np.median(ks_simulated), p_value)


    ax.hist(ks_simulated, lw=3, alpha=0.8, bins=10, color=utils.color_dict[experiments[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
    #ax.axvline(x=0, ls=':', lw=3, c='k', label='Null')
    #ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$D$')
    ax.axvline(x=ks_observed, ls='--', lw=3, c='k', label='Observed ' +  r'$\mathrm{KS}$')

    #ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$D$')
    ax.axvline(x=np.median(ks_simulated), ls=':', lw=3, c='k', label='Median simulated ' +  r'$\mathrm{KS}$')

    ax.set_xlabel('Simulated ' + r'$\mathrm{KS}$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=18)
    ax.set_ylabel('Probability density',  fontsize=20)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[3 + treatment_idx], fontsize=18, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


    if treatment_idx == 0:
        ax.legend(loc="upper right", fontsize=16)





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

        ax_plot = plt.subplot2grid((5, 3), (2+trasfer_idx, migration_innoculum_idx), colspan=1)

        ax_plot.scatter(means, variances, alpha=0.8, c=utils.color_dict_range[migration_innoculum][transfer-2].reshape(1,-1), edgecolors='k')#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        y_log10_null_range = 10 ** (utils.slope_null*x_log10_range + intercept)

        ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
        #ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Slope = 2")
        ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label=r'$y \sim x^{2}$')

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=18, color='k', ha='center', va='center', transform=ax_plot.transAxes  )

        ax_plot.set_xlabel('Mean relative abundance', fontsize=20)
        ax_plot.set_ylabel('Variance of relative abundance', fontsize=18)


        #if trasfer_idx == 0:

            #title = utils.titles_no_inocula_dict[migration_innoculum]
            #ax_plot.set_title(title, fontsize=12, fontweight='bold' )


        if plot_count == 0:
            ax_plot.legend(loc="lower right", fontsize=16)



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
            ax_plot.text(-0.25,0.5, 'Transfer %d' % transfer, fontsize=18, fontweight='bold', color='k', ha='center', rotation=90, va='center', transform=ax_plot.transAxes )

        
        ax_plot.text(-0.1, 1.04, plot_utils.sub_plot_labels[6 + migration_innoculum_idx + 3*trasfer_idx], fontsize=18, fontweight='bold', ha='center', va='center', transform=ax_plot.transAxes)

        
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

    ax = plt.subplot2grid((5, 3), (4, treatment_idx), colspan=1)

    p_value = sum(slope_t_test > t_slope_all[treatment_idx])/len(slope_t_test)
    #print(p_value)

    ax.hist(slope_t_test, lw=3, alpha=0.8, bins=10, color=utils.color_dict[migration_innocula[treatment_idx]], histtype='stepfilled', density=True, zorder=2)
    ax.axvline(x=0, ls=':', lw=3, c='k', label='Null')
    ax.axvline(x=t_slope_all[treatment_idx], ls='--', lw=3, c='k', label='Observed ' +  r'$t_{\mathrm{slope}}$')
    ax.set_xlabel('Simulated ' + r'$t_{\mathrm{slope}}$' + ' from optimal\n' + r'$\tau = $' + str(round(tau_best, 2)) + ' and ' + r'$\sigma = $' + str(round(sigma_best, 3)), fontsize=18)
    ax.set_ylabel('Probability density',  fontsize=20)

    ax.text(-0.1, 1.04, plot_utils.sub_plot_labels[12 + treatment_idx], fontsize=18, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

    if treatment_idx == 0:
        ax.legend(loc="upper right", fontsize=16)







fig.subplots_adjust(wspace=0.3, hspace=0.25)
#fig.savefig(utils.directory + "/figs/afd_and_taylors_law_migration.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/afd_and_taylors_law_migration.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
