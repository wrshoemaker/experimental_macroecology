from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma
from scipy.special import kv

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm

from itertools import combinations
import statsmodels.stats.multitest as multitest



s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=18, migration='No_migration', inocula=4)
rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=True)

mu, s = utils.Klogn(means, 10**-4)

k_lognorm = np.random.lognormal(mean=mu, sigma=s, size=1000)

#print(k_lognorm)



sigma = 1.4

m_tilde = 0.2
#k = np.logspace(-10, 0, num=10000)
k = k_lognorm


mean_slm = k*(1-sigma/2)
var_slm = (k**2)*(sigma/2)*(1-sigma/2)

#mean_slm_m = (np.sqrt(m_tilde*k))*((2/sigma)-1) / ((2/sigma) * np.sqrt(m_tilde/k) )
#var_slm_m = m_tilde*k * (2/sigma)*((2/sigma)-1) / (((2/sigma) * np.sqrt(m_tilde/k) )**2)

#mean_slm_m = np.sqrt(m_tilde*k)*kv(2/sigma, (4/sigma)*np.sqrt(m_tilde/k))/kv((2/sigma)-1, (4/sigma)*np.sqrt(m_tilde/k))
#var_slm_m = m_tilde*k * kv((2/sigma)+1, (4/sigma)*np.sqrt(m_tilde/k))/kv((2/sigma)-1, (4/sigma)*np.sqrt(m_tilde/k))




def get_mean_var_slm_m(m_tilde, k, sigma):
    mean_slm_m = np.sqrt(m_tilde*k) * ((32/sigma)*np.sqrt(m_tilde/k)+( ( 4*(2/sigma))**2)-1)/((32/sigma)*np.sqrt(m_tilde/k)+((4*((2/sigma)-1))**2)-1)
    var_slm_m = m_tilde*k*(((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma)+1)**2)))-1)/((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma)-1)**2)))-1)) - ((((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma))**2)))-1)/((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma)-1)**2)))-1)/((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma)+1)**2)))-1)/((32/sigma)*np.sqrt(m_tilde/k)+((4*(((2/sigma)-1)**2)))-1))**2)

    return mean_slm_m, var_slm_m


#idx_to_keep = mean_slm_m > 10**-5

#mean_slm = mean_slm[idx_to_keep]
#var_slm = var_slm[idx_to_keep]
#mean_slm_m = mean_slm_m[idx_to_keep]
#var_slm_m = var_slm_m[idx_to_keep]


mean_slm_m_low, var_slm_m_low = get_mean_var_slm_m(0.01, k, sigma)
mean_slm_m_med, var_slm_m_med = get_mean_var_slm_m(1, k, sigma)
mean_slm_m_high, var_slm_m_high = get_mean_var_slm_m(100, k, sigma)


fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(mean_slm, var_slm, ls='-', c='dodgerblue', s=4, label=r'$\mathrm{SLM}$')


ax.scatter(mean_slm_m_low, var_slm_m_low, ls='-', c='coral', s=4, label=r'$\mathrm{SLMm},\, \tilde{m}_{i}=0.01$')
ax.scatter(mean_slm_m_med, var_slm_m_med, ls='-', c='orangered', s=4, label=r'$\mathrm{SLMm},\,\tilde{m}_{i}=1$')
ax.scatter(mean_slm_m_high, var_slm_m_high, ls='-', c='firebrick', s=4, label=r'$\mathrm{SLMm},\,\tilde{m}_{i}=100$')


ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

ax.set_xlim(10**-5, 10**-1)
ax.set_ylim(10**-11, 10**0)



ax.set_xlabel('Mean relative abundance', fontsize=12)
ax.set_ylabel('Variance of relative abundance', fontsize=12)

ax.legend(loc="upper left", fontsize=8)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/taylors_law_theory.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
