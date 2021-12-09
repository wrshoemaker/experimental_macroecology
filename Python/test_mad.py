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

prevalence_range = np.logspace(-4, 1, num=1000)

fig = plt.figure(figsize = (12.5, 8)) #
fig.subplots_adjust(bottom= 0.15)

transfers = [12,18]

#ax_occupancy = plt.subplot2grid((1, 3), (0,0), colspan=1)
ax_afd = plt.subplot2grid((2, 3), (0,0), colspan=1)
ax_taylors = plt.subplot2grid((2, 3), (0,1), colspan=1)
ax_mad = plt.subplot2grid((2, 3), (0,2), colspan=1)
#ax_mad_vs_occupancy = plt.subplot2grid((2, 2), (1,0), colspan=1)

ax_occupancy = plt.subplot2grid((2, 3), (1,0), colspan=1)
ax_survival = plt.subplot2grid((2, 3), (1,1), colspan=1)
ax_mad_vs_occupancy = plt.subplot2grid((2, 3), (1,2), colspan=1)


def gamma_dist(x_range, x_bar, beta=1):
    return (1/special.gamma(beta)) * ((beta/x_bar)**beta) * (x_range**(beta-1)) * np.exp(-1*beta *x_range / x_bar)


# get mean and std for rescaling
afd_log10_rescaled_all = []
all_means = []
all_vars = []
all_mads = []
all_predicted_occupancies = []
for transfer in transfers:

    for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        afd = rel_s_by_s.flatten()
        afd_log10 = np.log(afd[afd>0])
        afd_log10_rescaled = (afd_log10 - np.mean(afd_log10)) / np.std(afd_log10)

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)
        label_ = utils.titles_dict[migration_innoculum] + ', transfer ' + str(transfer)
        hist, bin_edges = np.histogram(afd_log10_rescaled, density=True, bins=10)
        #bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]
        bins_mean = [bin_edges[i+1] for i in range(0, len(bin_edges)-1 )]


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
        #mad_log10_rescaled = (mad_log10 - np.mean(mad_log10)) / np.std(mad_log10)
        hist_mad, bin_edges_mad = np.histogram(mad_log10, density=True, bins=10)
        bins_mean_mad = [0.5 * (bin_edges_mad[i] + bin_edges_mad[i+1]) for i in range(0, len(bin_edges_mad)-1 )]

        prob_to_plot = [sum( (mad_log10>=bin_edges_mad[i]) & (mad_log10<=bin_edges_mad[i+1])  ) / len(mad_log10) for i in range(0, len(bin_edges_mad)-1 )]


        ax_mad.scatter(bins_mean_mad, prob_to_plot, alpha=0.4, c=color_, size=10)




fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/test_mad.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
