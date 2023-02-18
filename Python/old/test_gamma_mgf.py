from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

import utils

carbons = utils.carbons


fig, ax = plt.subplots(figsize=(4,4))

#for carbon_idx, carbon in enumerate(carbons):

carbon = carbons[1]

s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)

relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))

means, variances, species_to_keep = utils.get_species_means_and_variances(relative_s_by_s, species, zeros=False)

slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

print(slope)



q_range = np.linspace(0, 5, num=50)

N_array = s_by_s.sum(axis=0)

for afd_idx, afd in enumerate(s_by_s):

    if sum(afd>0) < 10:
        continue


    afd_rel = relative_s_by_s[afd_idx,:]

    afd_rel_no_zeros = afd_rel #afd_rel[afd_rel>0]
    afd_no_zeros = afd #afd[afd_rel>0]
    N_array_no_zeros = N_array #N_array[afd_rel>0]


    beta_i = (np.mean(afd_rel_no_zeros) ** 2) / np.var(afd_rel_no_zeros)

    mgf_q = [np.mean((1 + (q/N_array_no_zeros)) ** afd_no_zeros) ** (-1/beta_i) for q in q_range ]

    q_range_i = q_range #* np.mean(afd_rel) / np.var(afd_rel)

    ax.plot(q_range_i, mgf_q, lw=1, alpha=0.7, ls='--',c='k',zorder=1)

        #print(mgf_q)

    #print(s_by_s[80,:])

    #print(s_by_s>0)


fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/mgf.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
