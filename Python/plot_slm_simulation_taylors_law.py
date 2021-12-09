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

import test_sde_simulation



fig, ax = plt.subplots(figsize=(4,4))

for i in range(5):

    x, t = test_sde_simulation.run_simulation()

    x_final = x[-1,:,:]
    x_final_rel = (x_final.T/x_final.sum(axis=1)).T

    sad_sample_all = []
    for s in x_final_rel:
        sad_sample_all.append(np.random.multinomial(utils.n_reads, s))

    x_final_sample = np.concatenate(sad_sample_all).reshape(x_final_rel.shape)
    x_final_sample = x_final_sample[:,~np.all(x_final_sample == 0, axis=0)]
    x_final_sample_rel = (x_final_sample.T/x_final_sample.sum(axis=1)).T


    species = list(range(x_final_sample_rel.shape[1]))

    mean_rel_abundances, var_rel_abundances, species_to_keep = utils.get_species_means_and_variances(x_final_sample_rel.T, species, min_observations=3, zeros=False)

    #print(len(mean_rel_abundances))

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))

    print(slope)



    ax.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.5)


ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

fig_name = utils.directory + '/figs/slm_taylors_law.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
