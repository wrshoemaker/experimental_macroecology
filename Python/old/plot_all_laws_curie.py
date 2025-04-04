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

from scipy.optimize import fsolve
from scipy.special import erf


remove_zeros = True

migration_innocula = [('No_migration',4), ('Parent_migration',4), ('Global_migration',4)]



prevalence_range = np.logspace(-4, 1, num=1000)

fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)

transfers = [18]

#ax_occupancy = plt.subplot2grid((1, 3), (0,0), colspan=1)
ax_afd = plt.subplot2grid((2, 2), (0,0), colspan=1)
ax_occupancy = plt.subplot2grid((2, 2), (0,1), colspan=1)
ax_mad_vs_occupancy = plt.subplot2grid((2, 2), (1,0), colspan=1)
ax_error = plt.subplot2grid((2, 2), (1,1), colspan=1)

ax_afd.text(-0.1, 1.08, 'a', fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_afd.transAxes)
ax_occupancy.text(-0.1, 1.08, 'b', fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_occupancy.transAxes)
ax_mad_vs_occupancy.text(-0.1, 1.08, 'c', fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_mad_vs_occupancy.transAxes)
ax_error.text(-0.1, 1.08, 'd', fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_error.transAxes)




def gamma_dist(x_range, x_bar, beta=1):
    return (1/special.gamma(beta)) * ((beta/x_bar)**beta) * (x_range**(beta-1)) * np.exp(-1*beta *x_range / x_bar)


# get mean and std for rescaling
afd_log10_rescaled_all = []
all_means = []
all_vars = []
all_mads = []
all_mads_occupancies = []
all_predicted_occupancies = []
all_mu = []
all_sigma = []
for transfer in transfers:

    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        #print(migration_innoculum, transfer , s_by_s.shape, len(comm_rep_list))

        afd = rel_s_by_s.flatten()
        afd_log10 = np.log(afd[afd>0])
        afd_log10_rescaled = (afd_log10 - np.mean(afd_log10)) / np.std(afd_log10)

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)

        hist, bin_edges = np.histogram(afd_log10_rescaled, density=True, bins=10)
        #bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]
        bins_mean = [bin_edges[i+1] for i in range(0, len(bin_edges)-1 )]

        if transfer == 18:
            label_ = utils.titles_no_inocula_dict[migration_innoculum]
            ax_afd.scatter(bins_mean, hist, alpha=0.8, c=color_, label=label_)
        else:
            ax_afd.scatter(bins_mean, hist, alpha=0.8, c=color_)


        # taylors law
        means = []
        variances = []
        for afd_i in rel_s_by_s:
            afd_i = afd_i[afd_i>0]
            if len(afd_i) < 3:
                continue

            means.append(np.mean(afd_i))
            variances.append(np.var(afd_i))

        # mad
        mad = np.mean(rel_s_by_s, axis=1)
        mad_log10 = np.log(mad)
        mad_log10_rescaled = (mad_log10 - np.mean(mad_log10)) / np.std(mad_log10)

        hist_mad, bin_edges_mad = np.histogram(mad_log10_rescaled, density=True, bins=10)
        bins_mean_mad = [0.5 * (bin_edges_mad[i] + bin_edges_mad[i+1]) for i in range(0, len(bin_edges_mad)-1 )]
        prob_to_plot = [sum( (mad_log10_rescaled>=bin_edges_mad[i]) & (mad_log10_rescaled<bin_edges_mad[i+1])  ) / len(mad_log10_rescaled) for i in range(0, len(bin_edges_mad)-1 )]
        #prob_to_plot = [sum( (mad_log10>=bin_edges_mad[i]) & (mad_log10<=bin_edges_mad[i+1])  ) / len(mad_log10) for i in range(0, len(bin_edges_mad)-1 )]



        mu, sigma = utils.Klogn(mad, utils.c)
        all_mu.append(mu)
        all_sigma.append(sigma)
        #mu_2, sigma_2 = utils.Klogn(mad, 0.00001)

        # occupancy
        occupancies, predicted_occupancies, mad_occupancies, beta_occupancies, species_occupances = utils.predict_occupancy(s_by_s, ESVs)
        ax_occupancy.scatter(occupancies, predicted_occupancies, alpha=0.5, c=color_, s=18, zorder=2)#, linewidth=0.8, edgecolors='k')

        # errors
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies
        errors = errors[errors>0]
        mean_error = 10**np.mean(np.log10(errors))
        std_error = np.std(np.log10(errors))

        ax_error.errorbar(migration_innoculum_idx, mean_error, 0.05, linestyle='-', marker='o', c='k', elinewidth=1.5, alpha=1, zorder=1)

        #ax_survival.plot(prevalence_range, survival_array, ls='-', lw=2, c=utils.color_dict_range[migration_innoculum][transfer-3], alpha=0.6, zorder=1)

        print(migration_innoculum_idx, mean_error)

        ax_error.scatter(migration_innoculum_idx, mean_error, c=utils.color_dict_range[migration_innoculum][transfer-3], s=45, zorder=2)

        # abundance vs occupancy
        ax_mad_vs_occupancy.scatter(mad_occupancies, occupancies, alpha=0.5, c=color_, s=18, zorder=2)#, linewidth=0.8, edgecolors='k')

        all_predicted_occupancies.extend(predicted_occupancies.tolist())
        all_means.extend(means)
        all_vars.extend(variances)
        afd_log10_rescaled_all.extend(afd_log10_rescaled)
        all_mads.extend(mad.tolist())
        all_mads_occupancies.extend(mad_occupancies.tolist())




afd_log10_rescaled_all = np.asarray(afd_log10_rescaled_all)
afd_log10_rescaled_all_cutoff = afd_log10_rescaled_all[afd_log10_rescaled_all<1.8]
all_mads = np.asarray(all_mads)


ax_afd.set_xlabel('Rescaled log relative abundance', fontsize=12)
ax_afd.set_ylabel('Probability', fontsize=12)


x_range = np.linspace(min(afd_log10_rescaled_all_cutoff) , max(afd_log10_rescaled_all_cutoff) , 10000)
#x_range_log_rescaled = (np.log10(x_range) - afd_log_all_mean) / afd_log_all_std
#gammalog  <- function(x, k) { (1.13*x - 0.9 * exp(x)) + 0.5 }
#gammalog  <- function(x, k = 1.7) { ( k*trigamma(k)*x - exp( sqrt(trigamma(k))*x+ digamma(k)) ) - log(gamma(k)) + k*digamma(k) + log10(exp(1)) }
#gammalog = (1.13*x_range - 0.9 * np.exp(x_range)) + 0.5
k = 2.3
k_digamma = special.digamma(k)
k_trigamma = special.polygamma(1,k)

gammalog = k*k_trigamma*x_range - np.exp(np.sqrt(k_trigamma)*x_range + k_digamma) - np.log(special.gamma(k)) + k*k_digamma + np.log10(np.exp(1))

ax_afd.plot(x_range, 10**gammalog, 'k', label='SLM prediction', lw=2)
ax_afd.legend(loc="upper right", fontsize=8)
#ax_afd.set_yscale('log', basey=10)
# trigamma = second derivatives of the logarithm of the gamma function
# digamma = first derivatives of the logarithm of the gamma function







# occupancy
ax_occupancy.plot([0.01,1],[0.01,1], lw=2,ls='--',c='k',zorder=2, label='1:1')
#ax_occupancy.set_xlim([0.008,1.03])

ax_occupancy.set_xlim([0.008, 1.08])
ax_occupancy.set_ylim([0.008, 1.08])

#ax_occupancy.set_ylim([0.008,1.03])
ax_occupancy.set_xscale('log', basex=10)
ax_occupancy.set_yscale('log', basey=10)
ax_occupancy.set_xlabel('Observed occupancy', fontsize=12)
ax_occupancy.set_ylabel('Predicted occupancy, SLM', fontsize=12)
ax_occupancy.tick_params(axis='both', which='minor', labelsize=9)
ax_occupancy.tick_params(axis='both', which='major', labelsize=9)
#ax_occupancy.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=9)
ax_occupancy.legend(loc="upper left", fontsize=9)


# survival
#ax_survival.set_xscale('log', basex=10)
#ax_error.set_yscale('log', basey=10)
#ax_error.set_xlabel('Relative error, ' + r'$\epsilon$', fontsize=12)
ax_error.set_ylabel('SLM occupancy error', fontsize=12)
#ax_survival.tick_params(axis='both', which='minor', labelsize=9)
#ax_survival.tick_params(axis='both', which='major', labelsize=9)

labels = ['No\nmigration', 'Parent\nmigration', 'Global\nmigration']


ax_error.set_xticks([0,1,2])
ax_error.set_xticklabels(labels, fontsize=10)
ax_error.set_xlim([-0.5, 2.5])


# mad vs occupancy
all_mads_occupancies_log10 = np.log10(all_mads_occupancies)
all_predicted_occupancies_log10 = np.log10(all_predicted_occupancies)
hist_all, bin_edges_all = np.histogram(all_mads_occupancies_log10, density=True, bins=13)
bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
bins_mean_all_to_keep = []
bins_occupancies = []
for i in range(0, len(bin_edges_all)-1 ):
    all_predicted_occupancies_log10_i = all_predicted_occupancies_log10[ (all_mads_occupancies_log10>=bin_edges_all[i]) & (all_mads_occupancies_log10<bin_edges_all[i+1])]
    bins_mean_all_to_keep.append(bins_mean_all[i])
    bins_occupancies.append(np.mean(all_predicted_occupancies_log10_i))


bins_mean_all_to_keep = np.asarray(bins_mean_all_to_keep)
bins_occupancies = np.asarray(bins_occupancies)

bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]
bins_occupancies_no_nan = bins_occupancies[(~np.isnan(bins_mean_all_to_keep)) & (~np.isnan(bins_occupancies))]

ax_mad_vs_occupancy.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_occupancies_no_nan, lw=2,ls='-',c='k',zorder=2, label='SLM prediction')
ax_mad_vs_occupancy.set_xscale('log', basex=10)
ax_mad_vs_occupancy.set_yscale('log', basey=10)
ax_mad_vs_occupancy.set_xlabel('Mean relative abundance', fontsize=12)
ax_mad_vs_occupancy.set_ylabel('Occupancy', fontsize=12)




ax_mad_vs_occupancy.tick_params(axis='both', which='minor', labelsize=9)
ax_mad_vs_occupancy.tick_params(axis='both', which='major', labelsize=9)
ax_mad_vs_occupancy.legend(loc="lower right", fontsize=10)




fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/all_laws_curie.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
