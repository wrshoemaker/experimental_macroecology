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

c = 10**-6
transfers = [12,18]


#fig = plt.figure(figsize = (8, 8)) #
#fig.subplots_adjust(bottom= 0.15)


# get mean and std for rescaling

fig, ax = plt.subplots(figsize=(4,4))

migration_innocula_nested_list = [utils.migration_innocula[:2], utils.migration_innocula[2:]]

for row_idx, row_list in enumerate(migration_innocula_nested_list):

    for column_idx, migration_innoculum in enumerate(row_list):

        mu_all = []
        sigma_all = []
        color_all = []

        for transfer in transfers:

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            mad = np.mean(rel_s_by_s, axis=1)

            #afd_log10 = np.log(afd[afd>0])
            #afd_log10_rescaled = (afd_log10 - np.mean(afd_log10)) / np.std(afd_log10)


            mu, sigma = utils.Klogn(mad, 0.00001)
            color_ = utils.color_dict_range[migration_innoculum][transfer-3]
            #color_ = color_.reshape(1,-1)

            mu_all.append(mu)
            sigma_all.append(sigma)
            color_all.append(color_)


        mu_all = np.asarray(mu_all)
        sigma_all = np.asarray(sigma_all)
        ax.scatter(mu_all, sigma_all, alpha=1, c=color_all, zorder=2, label=utils.titles_dict[migration_innoculum])


        #print(10**(np.log10(mu_all[1]) - np.log10(mu_all[0])))


        #ax.arrow(mu_all[0], sigma_all[0], (mu_all[1] - mu_all[0]), (sigma_all[1]-sigma_all[0]), fc="k", ec="k", zorder=3, head_width=0.1, head_length=0.1)
        #ax.quiver(mu_all[0], sigma_all[0], (mu_all[1] - mu_all[0]), (sigma_all[1]))
        ax.annotate("", xy=(mu_all[1], sigma_all[1]), xycoords='data', xytext=(mu_all[0], sigma_all[0]), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), zorder=1)
        #ax.set_xscale('log', basex=10)

        ax.set_title('Lognormal parameters', fontsize=14)

        ax.set_xlabel('Location parameter, ' + r'$\mu$', fontsize=12)
        ax.set_ylabel('Shape parameter, ' + r'$\sigma$', fontsize=12)


        #label_ = utils.titles_dict[migration_innoculum] + ', transfer ' + str(transfer)
        #hist, bin_edges = np.histogram(afd_log10_rescaled, density=True, bins=10)
        #bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]
        #bins_mean = [bin_edges[i+1] for i in range(0, len(bin_edges)-1 )]
        #ax_afd.scatter(bins_mean, hist, alpha=0.8, c=color_)#, label=label_)

ax.legend(loc="lower left", fontsize=8)


fig_name = utils.directory + '/figs/mad_compare.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
