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




def load_simulation_parent_rho_dict():

    with open(slm_simulation_utils.simulation_parent_rho_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



# 18 - 12
observed_delta_rho = 0.6646895356148901 - 0.14137037547948292
observed_delta_rho_slope = 0.0023231389940615776 - 0.25774404196024386


#slm_simulation_utils.run_simulation_parent_rho()



simulation_parent_rho_dict = load_simulation_parent_rho_dict()

tau_all = np.asarray(list(simulation_parent_rho_dict.keys()))
sigma_all = np.asarray(list(simulation_parent_rho_dict[tau_all[0]].keys()))

np.sort(tau_all)
np.sort(sigma_all)

delta_rho_all = []
delta_slope_rho_all = []
delta_rho_error_all = []
delta_slope_rho_error_all = []
for tau in tau_all:

    tau_delta_rho = []
    tau_delta_slope_rho = []

    tau_delta_rho_error = []
    tau_delta_rho_slope_error = []

    for sigma in sigma_all:

        migration_12 = np.asarray(simulation_parent_rho_dict[tau][sigma][11]['migration'])
        migration_18 = np.asarray(simulation_parent_rho_dict[tau][sigma][17]['migration'])

        migration_12 = migration_12**2
        migration_18 = migration_18**2

        slope_rho_12 = np.asarray(simulation_parent_rho_dict[tau][sigma][11]['mad_ratio_vs_parent'])
        slope_rho_18 = np.asarray(simulation_parent_rho_dict[tau][sigma][17]['mad_ratio_vs_parent'])
        slope_rho_12 = slope_rho_12**2
        slope_rho_18 = slope_rho_18**2

        delta_rho = migration_18-migration_12
        delta_slope_rho = slope_rho_18-slope_rho_12

        error_delta_rho = np.absolute((delta_rho - observed_delta_rho) / observed_delta_rho)
        error_delta_rho_slope = np.absolute((delta_slope_rho - observed_delta_rho_slope) / observed_delta_rho_slope)

        tau_delta_rho.append(np.mean(delta_rho))
        tau_delta_slope_rho.append(np.mean(delta_slope_rho))

        tau_delta_rho_error.append(np.mean(error_delta_rho))
        tau_delta_rho_slope_error.append(np.mean(error_delta_rho_slope))


    delta_rho_all.append(tau_delta_rho)
    delta_slope_rho_all.append(tau_delta_slope_rho)

    delta_rho_error_all.append(tau_delta_rho_error)
    delta_slope_rho_error_all.append(tau_delta_rho_slope_error)


# rows = tau
# columns = sigma
delta_rho_all = np.asarray(delta_rho_all)
delta_slope_rho_all = np.asarray(delta_slope_rho_all)

delta_rho_error_all = np.asarray(delta_rho_error_all)
delta_slope_rho_error_all = np.asarray(delta_slope_rho_error_all)


#fig, ax = plt.subplots(figsize=(4,4))

fig = plt.figure(figsize = (9.5, 9.5))
gs = gridspec.GridSpec(nrows=2, ncols=2)

ax_rho = fig.add_subplot(gs[0, 0])
ax_slope_rho = fig.add_subplot(gs[0, 1])

ax_rho_error = fig.add_subplot(gs[1, 0])
ax_slope_rho_error = fig.add_subplot(gs[1, 1])


x_axis = sigma_all
y_axis = tau_all


pcm_rho = ax_rho.pcolor(x_axis, y_axis, delta_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),

clb_rho = plt.colorbar(pcm_rho, ax=ax_rho)
clb_rho.set_label(label='Change in MAD correlation b/w transfers 12 and 18, ' +  r'$\Delta \rho^{2}$')

ax_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)



pcm_slope_rho = ax_slope_rho.pcolor(x_axis, y_axis, delta_slope_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),

clb_slope_rho = plt.colorbar(pcm_slope_rho, ax=ax_slope_rho)
clb_slope_rho.set_label(label='Change in MAD vs. parent abundance\ncorrelation b/w transfers 12 and 18, ' +  r'$\Delta \rho^{2}$')

ax_slope_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_slope_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)




pcm_rho_error = ax_rho_error.pcolor(x_axis, y_axis, delta_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.min(delta_rho_error_all), vcenter=np.median(delta_rho_error_all), vmax=np.max(delta_rho_error_all)))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),
clb_rho_error = plt.colorbar(pcm_rho_error, ax=ax_rho_error)
clb_rho_error.set_label(label='Relative error of change in MAD correlation')
ax_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)



pcm_slope_rho_error = ax_slope_rho_error.pcolor(x_axis, y_axis, delta_slope_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.min(delta_slope_rho_error_all), vcenter=np.median(delta_slope_rho_error_all), vmax=np.max(delta_slope_rho_error_all)))
clb_slope_rho_error = plt.colorbar(pcm_slope_rho_error, ax=ax_slope_rho_error)
clb_slope_rho_error.set_label(label='Relative error of change in MAD vs. parent abundance correlation')
ax_slope_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_slope_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)



fig.subplots_adjust(hspace=0.25,wspace=0.4)
fig.savefig(utils.directory + "/figs/simulation_parent_rho.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
