from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.colors as clr

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

import slm_simulation_utils



fig, ax = plt.subplots(figsize=(4,4))

#means_all = []
#vars_all = []

mean_var_dict = {}

for i in range(10):

    x, t = slm_simulation_utils.run_simulation()
    #x_final = x[-1,:,:]
    for t_i_idx, t_i in enumerate(t):
        x_t_i = x[t_i_idx,:,:]
        x_t_i_rel = (x_t_i.T/x_t_i.sum(axis=1)).T

        sad_sample_all = []
        for s in x_t_i_rel:
            sad_sample_all.append(np.random.multinomial(utils.n_reads, s))

        x_t_i_sample = np.concatenate(sad_sample_all).reshape(x_t_i_rel.shape)
        x_t_i_sample = x_t_i_sample[:,~np.all(x_t_i_sample == 0, axis=0)]
        x_t_i_sample_rel = (x_t_i_sample.T/x_t_i_sample.sum(axis=1)).T

        species = list(range(x_t_i_sample_rel.shape[1]))
        mean_rel_abundances, var_rel_abundances, species_to_keep = utils.get_species_means_and_variances(x_t_i_sample_rel.T, species, min_observations=3, zeros=False)

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))
        if t_i not in mean_var_dict:
            mean_var_dict[t_i] = []
        mean_var_dict[t_i].append(slope)

        #if t_i == 0:
        #    ax.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.5)



#x_log10_range =  np.linspace(min(np.log10(means_all)) , max(np.log10(means_all)) , 10000)
#y_log10_fit_range = slope*x_log10_range + intercept
#ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

for key, value in mean_var_dict.items():

    print(key, np.mean(value))


ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

fig_name = utils.directory + '/figs/slm_taylors_law.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
