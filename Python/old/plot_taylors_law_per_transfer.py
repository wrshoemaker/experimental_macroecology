from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from matplotlib import cm

#color_range =  np.linspace(0.0, 1.0, 18)
#rgb = cm.get_cmap('Blues')( color_range )
alpha = 0.05
zeros = True

#experiments = [('No_migration',4), ('Global_migration',4), ('Glucose',  np.nan) ]
experiments = [('No_migration', 4), ('Global_migration', 4)]

transfer_max_dict = {('No_migration', 4): 18, ('Global_migration', 4): 18, ('Glucose', np.nan): 12}

#s_by_s_1, species_1, comm_rep_list_1 = utils.get_s_by_s("Glucose", transfer=1)


def get_bootstrapped_ci_regression_params(x, y, iter=1000, loglog=True):

    if loglog == True:
        x = np.log10(x)
        y = np.log10(y)

    idx_ = np.arange(len(x))

    slope_resample = []
    intercept_resample = []
    for i in range(iter):
        idx_resample = np.random.choice(idx_, size=len(idx_), replace=True)

        x_resample = x[idx_resample]
        y_resample = y[idx_resample]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        slope_resample.append(slope)
        intercept_resample.append(intercept)

    slope_resample = np.asarray(slope_resample)
    slope_resample = np.sort(slope_resample)

    intercept_resample = np.asarray(intercept_resample)
    intercept_resample = np.sort(intercept_resample)

    slope_ci_lower = slope_resample[int(iter*0.025)]
    slope_ci_upper = slope_resample[int(iter*0.975)]

    intercept_ci_lower = intercept_resample[int(iter*0.025)]
    intercept_ci_upper = intercept_resample[int(iter*0.975)]


    return slope_ci_lower, slope_ci_upper, intercept_ci_lower, intercept_ci_upper





experiment_dict = {}
for experiment_idx, experiment in enumerate(experiments):

    transfer_max = transfer_max_dict[experiment]

    transfers = []
    means = []
    variances = []
    colors = []

    intercept_all = []
    slope_all = []

    slope_ci_lower_all = []
    slope_ci_upper_all = []

    intercept_ci_lower_all = []
    intercept_ci_upper_all = []

    species_relative_abundances_dict = {}
    for transfer in range(1, transfer_max+1):

        if experiment[0] == 'Glucose':

            s_by_s, species, comm_rep_list = utils.get_s_by_s("Glucose", transfer=transfer)
            #if transfer==1:
            #    communities_keep = comm_rep_list
            #    s_by_s, species, comm_rep_list = utils.get_s_by_s("Glucose", transfer=transfer)

        else:

            communities = utils.get_migration_time_series_community_names(migration=experiment[0], inocula=experiment[1])
            communities_keep = [str(key) for key, value in communities.items() if len(value) == transfer_max]

            s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1], communities=communities_keep)

        comm_rep_array = np.asarray(comm_rep_list)

        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        print(experiment, transfer, s_by_s.shape[1])

        #for afd_idx, afd in enumerate(rel_s_by_s):

        #species_i = species[afd_idx]

        #if species_i not in species_relative_abundances_dict:
        #    species_relative_abundances_dict[species_i] = {}

        #afd_no_zeros = afd[afd>0]
        #comm_rep_array_no_zeros = comm_rep_array[afd>0]

        #for comm_rep_array_no_zeros_i, afd_no_zeros_i in zip(comm_rep_array_no_zeros, afd_no_zeros):

        #    if comm_rep_array_no_zeros_i not in species_relative_abundances_dict[species_i]:
        #        species_relative_abundances_dict[species_i][comm_rep_array_no_zeros_i] = {}
        #        species_relative_abundances_dict[species_i][comm_rep_array_no_zeros_i]['transfers'] = []
        #        species_relative_abundances_dict[species_i][comm_rep_array_no_zeros_i]['relative_abundances'] = []

        #    species_relative_abundances_dict[species_i][comm_rep_array_no_zeros_i]['transfers'].append(transfer, afd_no_zeros_i)

        means_transfer, variances_transfer, variances_species = utils.get_species_means_and_variances(rel_s_by_s, species, zeros=True)

        if len(means_transfer) < 5:
            continue

        #colors_transfer = [color_range[transfer-1] for i in range(len(means_transfer))]
        colors_transfer = [utils.color_dict_range[experiment]
                           [transfer-1] for i in range(len(means_transfer))]

        transfers.append(transfer)
        means.extend(list(means_transfer))
        variances.extend(list(variances_transfer))
        colors.extend(colors_transfer)

        idx_to_keep = means_transfer<0.9
        means_transfer = means_transfer[idx_to_keep]
        variances_transfer = variances_transfer[idx_to_keep]

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means_transfer), np.log10(variances_transfer))

        s_xx, s_yy, s_xy = utils.get_pair_stats(np.log10(means_transfer), np.log10(variances_transfer))
        t = stats.t.ppf(1-(alpha/2), len(means_transfer)-2)
        #maximim likelihood estimator
        mse = sum((np.log10(variances) - (intercept + slope * np.log10(means)))**2) / (len(means)-2)
        #sigma_hat = np.sqrt((1/len(means_transfer))*(s_yy-slope*s_xy)
        slope_ci = t*np.sqrt(mse/s_xx)

        intercept_ci = t*np.sqrt(mse*((1/len(means_transfer)) + ((np.mean(np.log10(means))**2)/s_xx)))

        slope_ci_lower = slope - slope_ci
        slope_ci_upper = slope + slope_ci

        intercept_ci_lower = intercept - intercept_ci
        intercept_ci_upper = intercept + intercept_ci


        #slope_ci_lower, slope_ci_upper, intercept_ci_lower, intercept_ci_upper = get_bootstrapped_ci_regression_params(means_transfer, variances_transfer)

        slope_all.append(slope)
        intercept_all.append(intercept)

        slope_ci_lower_all.append(slope_ci_lower)
        slope_ci_upper_all.append(slope_ci_upper)

        intercept_ci_lower_all.append(intercept_ci_lower)
        intercept_ci_upper_all.append(intercept_ci_upper)

    transfers = np.asarray(transfers)
    means = np.asarray(means)
    variances = np.asarray(variances)
    colors = np.asarray(colors)

    slope_all = np.asarray(slope_all)
    intercept_all = np.asarray(intercept_all)

    slope_ci_lower_all = np.asarray(slope_ci_lower_all)
    slope_ci_upper_all = np.asarray(slope_ci_upper_all)
    intercept_ci_lower_all = np.asarray(intercept_ci_lower_all)
    intercept_ci_upper_all = np.asarray(intercept_ci_upper_all)


    experiment_dict[experiment] = {}
    experiment_dict[experiment]['transfers'] = transfers
    experiment_dict[experiment]['means'] = means
    experiment_dict[experiment]['variances'] = variances
    experiment_dict[experiment]['colors'] = colors

    experiment_dict[experiment]['slope_all'] = slope_all
    experiment_dict[experiment]['intercept_all'] = intercept_all

    experiment_dict[experiment]['slope_ci_lower_all'] = slope_ci_lower_all
    experiment_dict[experiment]['slope_ci_upper_all'] = slope_ci_upper_all

    experiment_dict[experiment]['intercept_ci_lower_all'] = intercept_ci_lower_all
    experiment_dict[experiment]['intercept_ci_upper_all'] = intercept_ci_upper_all


#fig = plt.figure(figsize = (12, 8)) #
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(bottom=0.15)

for experiment_idx, experiment in enumerate(experiments):

    ax_scatter = plt.subplot2grid((3, 3), (0, experiment_idx))  # , colspan=1)
    ax_slopes = plt.subplot2grid((3, 3), (1, experiment_idx))  # , colspan=1)
    ax_intercepts = plt.subplot2grid((3, 3), (2, experiment_idx))  # , colspan=1)

    #ax_scatter = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    #ax_slopes = plt.subplot2grid((1, 2), (0, 1), colspan=1)

    transfers = experiment_dict[experiment]['transfers']
    means = experiment_dict[experiment]['means']
    variances = experiment_dict[experiment]['variances']
    colors = experiment_dict[experiment]['colors']

    slope_all = experiment_dict[experiment]['slope_all']
    intercept_all = experiment_dict[experiment]['intercept_all']

    slope_ci_lower_all = experiment_dict[experiment]['slope_ci_lower_all']
    slope_ci_upper_all = experiment_dict[experiment]['slope_ci_upper_all']

    intercept_ci_lower_all = experiment_dict[experiment]['intercept_ci_lower_all']
    intercept_ci_upper_all = experiment_dict[experiment]['intercept_ci_upper_all']

    transfer_max = transfer_max_dict[experiment]
    # run slope test
    #t, p = stats.ttest_ind(dnds_treatment[0], dnds_treatment[1], equal_var=False)
    #t_value = (slope - (slope_null))/std_err
    #p_value = stats.t.sf(np.abs(t_value), len(means)-2)

    #sys.stdout.write("Slope = %g, t = %g, P= %g\n" % (slope, t_value, p_value))

    ax_scatter.set_ylabel('Variance of relative abundance', fontsize=10)

    # Bhatia–Davis inequality
    mean_range = np.linspace(min(means), max(means), num=1000)
    variance_range = (1-mean_range) * mean_range

    #ax_scatter.plot(mean_range, variance_range, lw=3, ls=':', c = 'k', label='Bhatia–Davis inequality')
    ax_scatter.plot(mean_range, variance_range, lw=3, ls=':', c='k', label='Max. ' + r'$\sigma^{2}_{x}$')

    #colors_scatter = [utils.color_dict_range[t] for t in transfers]

    ax_scatter.scatter(means, variances, c=colors, cmap=utils.color_dict_range[experiment], alpha=0.8, edgecolors='k', zorder=2)  # , c='#87CEEB')

    ax_scatter.set_xscale('log', basex=10)
    ax_scatter.set_yscale('log', basey=10)
    ax_scatter.set_xlabel('Mean relative abundance', fontsize=11)
    ax_scatter.set_ylabel('Variance of relative abundance', fontsize=10)
    ax_scatter.legend(loc="lower right", fontsize=8)
    ax_scatter.set_title(
        utils.titles_dict[experiment], fontsize=12, fontweight='bold')

    ax_slopes.axhline(y=2, color='darkgrey', linestyle=':', lw=3, zorder=1)

    #slope_colors = [color_range[t-1] for t in transfers]
    slope_colors = [utils.color_dict_range[experiment][t-1] for t in transfers]

    ax_slopes.errorbar(transfers, slope_all, yerr=(slope_all-slope_ci_lower_all, slope_ci_upper_all-slope_all), linestyle='-',
                       marker='o', c='k', elinewidth=1.5, alpha=1, zorder=2)
    ax_slopes.scatter(transfers, slope_all, c=slope_colors, cmap='Blues',
                      edgecolors='k', alpha=1, zorder=3)  # , c='#87CEEB')
    ax_slopes.set_xlabel('Transfer', fontsize=12)
    ax_slopes.set_ylabel('Slope', fontsize=10)



    # intercepts
    ax_intercepts.errorbar(transfers, intercept_all, yerr=(intercept_all-intercept_ci_lower_all, intercept_ci_upper_all-intercept_all), linestyle='-', marker='o', c='k', elinewidth=1.5, alpha=1, zorder=2)
    ax_intercepts.scatter(transfers, intercept_all, c=slope_colors, cmap='Blues', edgecolors='k', alpha=1, zorder=3)  # , c='#87CEEB')
    ax_intercepts.set_xlabel('Transfer', fontsize=12)
    ax_intercepts.set_ylabel('Intercept', fontsize=10)

    #ax_slopes.set_ylim(-1, 3)
    ax_slopes.set_ylim([1.3, 2.1])
    ax_intercepts.set_ylim([-3.5, 1])


fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/taylors_law_time_series_all.png",
            format='png', bbox_inches="tight", pad_inches=0.5, dpi=600)
plt.close()
