from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
#from scipy.stats import gamma
from scipy.special import kv, gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm

from itertools import combinations

sigma = 1
k = 0.0001

#x = 0.1
x = np.logspace(-6, 0, base=10, num=1000)
step_size = (np.log10(x[1:]) - np.log10(x[:-1]))[0]


#x, step = np.linspace(lower_bound, upper_bound, N, retstep=True)

#prob_slm = (gamma((2/sigma)-1)**-1)  * ((2/(k*sigma))**((2/sigma)-1)) * np.exp((-2/(k*sigma))*x)*(x**((2/sigma)-2))

#prob_slm = gamma.pdf(x, ((2/sigma)-1), scale=(2/(k*sigma)))
#prob_slm = prob_slm/np.sum(prob_slm * step_size)
#prob_slm = prob_slm/sum(prob_slm)
#idx_slm_to_plot = prob_slm>0
#x_slm_to_plot = x[idx_slm_to_plot]
#prob_slm_to_plot = prob_slm[idx_slm_to_plot]

m_tilde_all = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

color_range =  np.linspace(0.3, 0.9, 6)
rgb_ = cm.get_cmap('Reds')( color_range )


fig, ax = plt.subplots(figsize=(4,4))

for m_tilde_idx, m_tilde in enumerate(m_tilde_all):

    prob = ((2*((m_tilde*k)**((1/2)*((2/sigma)-1))) * kv((2/sigma)-1, (4/sigma)*np.sqrt(m_tilde/k)))**-1) * np.exp((-2/(sigma*x)) * (m_tilde + ((x**2)/k))) * (x**((2/sigma) -2))
    # approximate integral over PDF using a Riemann sum
    #prob = prob/np.sum(prob * step_size)
    prob = prob/np.sum(prob)
    print(sum(prob))

    idx_to_plot = prob>0
    x_to_plot = x[idx_to_plot]
    prob_to_plot = prob[idx_to_plot]

    ax.plot(x_to_plot, prob_to_plot, ls='-', c=rgb_[m_tilde_idx], label=r'$\tilde{{m}}_{{i}}= 10^{{{}}}$'.format(int(np.log10(m_tilde))))


    #r'$test \frac{{1}}{{{}}}$')


ax.set_xscale('log', basex=10)
#ax.set_yscale('log', basey=10)

ax.set_ylim(0, 0.12)

ax.set_xlabel('Relative abundance', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)

ax.axvline(k, lw=1.5, ls=':',color='k', zorder=1, label=r'$K_{i}$')


ax.legend(loc="upper left", fontsize=8)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_theory.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
