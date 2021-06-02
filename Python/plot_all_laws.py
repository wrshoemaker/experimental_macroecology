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

remove_zeros = True


fig = plt.figure(figsize = (12, 12)) #
fig.subplots_adjust(bottom= 0.15)

ax_occupancy = plt.subplot2grid((2, 2), (0,0), colspan=1)
ax_afd = plt.subplot2grid((2, 2), (0,1), colspan=1)
ax_taylors = plt.subplot2grid((2, 2), (1,0), colspan=1)
ax_mad = plt.subplot2grid((2, 2), (1,1), colspan=1)


def gamma_dist(x_range, x_bar, beta=1):
    return (1/special.gamma(beta)) * ((beta/x_bar)**beta) * (x_range**(beta-1)) * np.exp(-1*beta *x_range / x_bar)


# get mean and std for rescaling

afd_log_all_list = []
mad_all_list = []
for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

    s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

    afd = rel_s_by_s.flatten()
    afd = np.log10(afd[afd>0])

    afd_log_all_list.extend(afd.tolist())

    for afd_i in rel_s_by_s:

        afd_log_i = afd_i[afd_i>0]

        if len(afd_log_i) < 3:
            continue

        mad_all_list.append(np.mean(afd_log_i))


afd_log_all_mean = np.mean(afd_log_all_list)
afd_log_all_std = np.std(afd_log_all_list)

mad_all_list = np.log10(mad_all_list)
mad_all_mean = np.mean(mad_all_list)
mad_all_std = np.std(mad_all_list)


all_means = []
all_vars = []
for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

    s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

    occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)

    color_ = utils.color_dict[migration_innoculum]
    label_ = utils.titles_dict[migration_innoculum]


    ax_occupancy.scatter(occupancies, predicted_occupancies, alpha=0.8, c=color_, label=label_, zorder=2)#, c='#87CEEB')


    afd = rel_s_by_s.flatten()
    afd = np.log10(afd[afd>0])

    afd_rescaled  = (afd - afd_log_all_mean) / afd_log_all_std

    #bins = np.arange(0, max(afd_rescaled) + 1, 0.5)

    hist, bin_edges = np.histogram(afd_rescaled, density=True, bins=20)

    bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]

    ax_afd.scatter(bins_mean, hist, alpha=0.8, c=color_)

    means = []
    variances = []
    for afd_i in rel_s_by_s:

        afd_i = afd_i[afd_i>0]

        if len(afd_i) < 3:
            continue

        means.append(np.mean(afd_i))
        variances.append(np.var(afd_i))

    ax_taylors.scatter(means, variances, alpha=0.8, c=color_, zorder=2)

    all_means.extend(means)
    all_vars.extend(variances)


    means_log_rescaled = (np.log10(means) - mad_all_mean) / mad_all_std
    hist_mad, bin_edges_mad = np.histogram(means_log_rescaled, density=True, bins=20)
    bins_mean_mad = [0.5 * (bin_edges_mad[i] + bin_edges_mad[i+1]) for i in range(0, len(bin_edges_mad)-1 )]
    ax_mad.scatter(bins_mean_mad, hist_mad, alpha=0.8, c=color_)






ax_occupancy.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
ax_occupancy.set_xscale('log', basex=10)
ax_occupancy.set_yscale('log', basey=10)
ax_occupancy.set_xlabel('Observed occupancy', fontsize=14)
ax_occupancy.set_ylabel('Predicted occupancy', fontsize=14)
ax_occupancy.legend(loc="upper left", fontsize=8)


ax_afd.set_xlabel('Rescaled log relative abundance', fontsize=14)
ax_afd.set_ylabel('Probability density', fontsize=14)



afd_log_all = np.asarray(afd_log_all_list)
afd_log_all_rescaled = (afd_log_all - afd_log_all_mean) / afd_log_all_std

afd_log_all_rescaled_cutoff = afd_log_all_rescaled[afd_log_all_rescaled<1.8]

#afd_all_rescaled = 10**np.asarray(afd_log_all_rescaled)

#x_gamma_range = np.linspace(min(afd_all), max(afd_all), num=50)

#y_gamma = gamma_dist(x_gamma_range, np.mean(afd_all), beta=1)

#x_gamma_range_log_rescaled = (np.log10(x_gamma_range) - afd_log_all_mean) / afd_log_all_std


ag,bg,cg = gamma.fit(afd_log_all_rescaled_cutoff)
x_range = np.linspace(min(afd_log_all_rescaled_cutoff) , max(afd_log_all_rescaled_cutoff) , 10000)


#x_range_log_rescaled = (np.log10(x_range) - afd_log_all_mean) / afd_log_all_std

#ax_afd.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', label='Gamma', lw=2)
ax_afd.legend(loc="upper right", fontsize=8)


#gammalog  <- function(x, k) { (1.13*x - 0.9 * exp(x)) + 0.5 }
#gammalog  <- function(x, k = 1.7) { ( k*trigamma(k)*x - exp( sqrt(trigamma(k))*x+ digamma(k)) ) - log(gamma(k)) + k*digamma(k) + log10(exp(1)) }


gammalog = (1.13*x_range - 0.9 * np.exp(x_range)) + 0.5
k = 2.4

k_digamma = special.digamma(k)
k_trigamma = special.polygamma(1,k)

gammalog = k*k_trigamma*x_range - np.exp(np.sqrt(k_trigamma)*x_range + k_digamma) - np.log(special.gamma(k)) + k*k_digamma + np.log10(np.exp(1))

ax_afd.plot(x_range, 10**gammalog, 'k', label='Gamma', lw=2)

# trigamma = second derivatives of the logarithm of the gamma function
# digamma = first derivatives of the logarithm of the gamma function



#ax_afd.plot(x_gamma_range_log_rescaled, y_gamma)


ax_taylors.set_xscale('log', basex=10)
ax_taylors.set_yscale('log', basey=10)
ax_taylors.set_xlabel('Average relative abundance', fontsize=14)
ax_taylors.set_ylabel('Variance of relative abundance', fontsize=14)


all_means = np.asarray(all_means)
all_vars = np.asarray(all_vars)

#slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(all_means), np.log10(all_vars))

intercept = np.mean(np.log10(all_vars) - 2*np.log10(all_means))


x_log10_range =  np.linspace(min(np.log10(all_means)) , max(np.log10(all_means)) , 10000)
#y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
y_log10_null_range = 10 ** (2*x_log10_range + intercept)

ax_taylors.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='-', zorder=1, label= r'$y \sim x^{2}$')
ax_taylors.legend(loc="upper left", fontsize=8)




#ax_mad.set_yscale('log', basey=10)
ax_mad.set_xlabel('Rescaled log average relative abundance', fontsize=14)
ax_mad.set_ylabel('Probability density', fontsize=14)


#ax_taylors.set_xlim([])


fig.text(0.5, 0.97, "Transfer 18", va='center', ha='center', fontweight='bold',fontsize=16)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/all_laws.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
