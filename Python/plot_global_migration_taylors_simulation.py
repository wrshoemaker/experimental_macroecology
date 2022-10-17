from __future__ import division
import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
import slm_simulation_utils


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors


with open(slm_simulation_utils.simulation_migration_all_path, 'rb') as handle:
    simulation_dict = pickle.load(handle)


tau_all = list(simulation_dict.keys())
tau_all.sort()

sigma_all = list(simulation_dict[tau_all[0]].keys())
sigma_all.sort()

t_all = np.asarray(range(18))
t_all_to_plot = t_all + 1

fig, ax = plt.subplots(figsize=(4,4))

n_itere = 100
for tau_i in tau_all:
    for sigma_i in sigma_all:

        mean_delta_intercept_all = []
        for t in t_all:
            mean_delta_intercept_all.append(np.mean(np.asarray(simulation_dict[tau_i][sigma_i][t]['global_migration']['taylors_intercept']) - np.asarray(simulation_dict[tau_i][sigma_i][t]['no_migration']['taylors_intercept'])))

        mean_delta_intercept_all = np.asarray(mean_delta_intercept_all)

        #print(sum(mean_delta_intercept_all<0)/len(mean_delta_intercept_all))

        ax.plot(t_all_to_plot, mean_delta_intercept_all, c='k', alpha=0.1, lw=1)


ax.set_xlabel('Transfer', fontsize=12)
ax.set_ylabel(r'$\Delta \mathrm{Intercept}$', fontsize=12)


fig_name = utils.directory + '/figs/global_migration_taylors_simulation.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
