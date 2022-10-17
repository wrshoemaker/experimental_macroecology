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




def load_simulation_global_rho_dict():

    with open(slm_simulation_utils.simulation_global_rho_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



#slm_simulation_utils.run_simulation_global_rho()

simulation_global_rho_dict = load_simulation_global_rho_dict()

tau_all = np.asarray(list(simulation_global_rho_dict.keys()))
sigma_all = np.asarray(list(simulation_global_rho_dict[tau_all[0]].keys()))

np.sort(tau_all)
np.sort(sigma_all)

mean_delta_rho_cv_all_nested = []
for tau_i in tau_all:

    mean_delta_rho_cv_all = []

    for sigma_i in sigma_all:

        rho_cv_12 = simulation_global_rho_dict[tau_i][sigma_i]['per_transfer_stats'][11]['cv_rho']
        rho_cv_18 = simulation_global_rho_dict[tau_i][sigma_i]['per_transfer_stats'][17]['cv_rho']

        rho_cv_12 = np.asarray(rho_cv_12)
        rho_cv_18 = np.asarray(rho_cv_18)

        mean_delta_rho_cv = np.mean(rho_cv_18 - rho_cv_12)
        mean_delta_rho_cv_all.append(mean_delta_rho_cv)

    mean_delta_rho_cv_all_nested.append(mean_delta_rho_cv_all)



# rows = tau
# columns = sigma
mean_delta_rho_cv_all_nested = np.asarray(mean_delta_rho_cv_all_nested)


fig, ax = plt.subplots(figsize=(4,4))

#fig = plt.figure(figsize = (9.5, 4))
#gs = gridspec.GridSpec(nrows=1, ncols=2)

#ax_rho = fig.add_subplot(gs[0, 0])
#ax_slope_rho = fig.add_subplot(gs[0, 1])

x_axis = sigma_all
y_axis = tau_all


pcm_rho = ax.pcolor(x_axis, y_axis, mean_delta_rho_cv_all_nested, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),

clb_rho = plt.colorbar(pcm_rho, ax=ax)
clb_rho.set_label(label='Change in CV correlation b/w transfers 12 and 18, ' +  r'$\Delta \rho_{CV}$')

ax.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)




fig.subplots_adjust(hspace=0.25,wspace=0.25)
fig.savefig(utils.directory + "/figs/simulation_global_rho.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
