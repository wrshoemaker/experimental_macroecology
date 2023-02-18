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
import plot_utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors




def load_simulation_parent_rho_dict():

    with open(slm_simulation_utils.simulation_parent_rho_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



# 18 - 12
observed_z_rho = 2.672481133312821
observed_mad_slope_t_test = -2.6431339333972756


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

        z_rho = simulation_parent_rho_dict[tau][sigma]['rho_12_vs_18']['Z']
        mad_slope_t_test = simulation_parent_rho_dict[tau][sigma]['slope_12_vs_18']['mad_slope_t_test']

        z_rho = np.asarray(z_rho)
        mad_slope_t_test = np.asarray(mad_slope_t_test)

        error_z_rho = np.absolute((z_rho - observed_z_rho) / observed_z_rho)
        error_mad_slope_t_test = np.absolute((mad_slope_t_test - observed_mad_slope_t_test) / observed_mad_slope_t_test)

        tau_delta_rho.append(np.mean(z_rho))
        tau_delta_slope_rho.append(np.mean(mad_slope_t_test))

        tau_delta_rho_error.append(np.mean(error_z_rho))
        tau_delta_rho_slope_error.append(np.mean(error_mad_slope_t_test))


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

fig = plt.figure(figsize = (10, 9.5))
gs = gridspec.GridSpec(nrows=2, ncols=2)

ax_rho = fig.add_subplot(gs[0, 0])
ax_slope_rho = fig.add_subplot(gs[0, 1])

ax_rho_error = fig.add_subplot(gs[1, 0])
ax_slope_rho_error = fig.add_subplot(gs[1, 1])


x_axis = sigma_all
y_axis = tau_all

x_axis_log10 = np.log10(x_axis)

pcm_rho = ax_rho.pcolor(x_axis_log10, y_axis, delta_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=np.amin(delta_rho_all), vcenter=observed_z_rho, vmax=np.amax(delta_rho_all)))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),

clb_rho = plt.colorbar(pcm_rho, ax=ax_rho)
clb_rho.set_label(label='Change in MAD corr. after cessation of migration, ' +  r'$Z_{\rho}$')

ax_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)
ax_rho.xaxis.set_major_formatter(plot_utils.fake_log)


# Set observed marking and label
clb_rho.ax.axhline(y=observed_z_rho, c='k')
original_ticks = list(clb_rho.get_ticks())
clb_rho.set_ticks(original_ticks + [observed_z_rho])
clb_rho.set_ticklabels(original_ticks + ['Obs.'])






pcm_slope_rho = ax_slope_rho.pcolor(x_axis_log10, y_axis, delta_slope_rho_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=np.amin(delta_slope_rho_all), vcenter=observed_mad_slope_t_test, vmax=np.amax(delta_slope_rho_all)))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),

clb_slope_rho = plt.colorbar(pcm_slope_rho, ax=ax_slope_rho)
clb_slope_rho.set_label(label='Change in MAD ratio vs. progenitor slope\nafter cessation of migration, ' +  r'$t_{b}$')

ax_slope_rho.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_slope_rho.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)

# Set observed marking and label
clb_slope_rho.ax.axhline(y=observed_mad_slope_t_test, c='k')
original_ticks = list(clb_slope_rho.get_ticks())
clb_slope_rho.set_ticks(original_ticks + [observed_mad_slope_t_test])
clb_slope_rho.set_ticklabels(original_ticks + ['Obs.'])





pcm_rho_error = ax_rho_error.pcolor(x_axis_log10, y_axis, delta_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.min(delta_rho_error_all), vcenter=np.median(delta_rho_error_all), vmax=np.max(delta_rho_error_all)))
# norm=colors.LogNorm(vmin=min(rel_s_by_s_np_dendo[rel_s_by_s_np_dendo>0]), vmax=1),
clb_rho_error = plt.colorbar(pcm_rho_error, ax=ax_rho_error)
clb_rho_error.set_label(label='Relative error of ' + r'$Z_{\rho}$'  + ' from simulated data')
ax_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)



pcm_slope_rho_error = ax_slope_rho_error.pcolor(x_axis_log10, y_axis, delta_slope_rho_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.min(delta_slope_rho_error_all), vcenter=np.median(delta_slope_rho_error_all), vmax=np.max(delta_slope_rho_error_all)))
clb_slope_rho_error = plt.colorbar(pcm_slope_rho_error, ax=ax_slope_rho_error)
clb_slope_rho_error.set_label(label='Relative error of ' + r'$t_{b}$' + ' from simulated data')
ax_slope_rho_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
ax_slope_rho_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)



fig.subplots_adjust(hspace=0.25,wspace=0.3)
fig.savefig(utils.directory + "/figs/simulation_parent_rho.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
