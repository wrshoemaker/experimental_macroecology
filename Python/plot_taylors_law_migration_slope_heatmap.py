from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import pickle

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import slm_simulation_utils
import plot_utils




simulation_dict = slm_simulation_utils.load_simulation_all_migration_dict()

tau_all = np.asarray(list(simulation_dict.keys()))
sigma_all = np.asarray(list(simulation_dict[tau_all[0]].keys()))

x_axis = sigma_all
y_axis = tau_all

x_axis_log10 = np.log10(x_axis)


########################
# plot the simulations #
########################

fig = plt.figure(figsize = (18, 12)) #


t_test_dict = {}
t_test_dict['no_migration'] = {}
t_test_dict['no_migration']['slope'] = -1.1159
t_test_dict['no_migration']['intercept'] = -0.604559

t_test_dict['global_migration'] = {}
t_test_dict['global_migration']['slope'] = -0.390222
t_test_dict['global_migration']['intercept'] = -0.380296

t_test_dict['parent_migration'] = {}
t_test_dict['parent_migration']['slope'] = 2.74544
t_test_dict['parent_migration']['intercept'] = 3.34061

for treatment_idx, treatment in enumerate(['no_migration', 'parent_migration', 'global_migration']):

    tau_t_test_slope_all = []
    tau_t_test_slope_error_all = []

    tau_t_test_intercept_all = []
    tau_t_test_intercept_error_all = []

    for tau in tau_all:

        tau_t_test_slope = []
        tau_t_test_slope_error = []

        tau_t_test_intercept = []
        tau_t_test_intercept_error = []

        for sigma in sigma_all:
            
            t_test_slope = np.asarray(simulation_dict[tau][sigma]['slope_12_vs_18'][treatment]['slope_t_test'])
            t_test_intercept = np.asarray(simulation_dict[tau][sigma]['slope_12_vs_18'][treatment]['intercept_t_test'])

            mean_t_test_slope = np.mean(t_test_slope)
            mean_t_test_intercept = np.mean(t_test_intercept)

            error_t_test_slope = np.mean(np.absolute((t_test_slope - t_test_dict[treatment]['slope'] ) / t_test_dict[treatment]['slope'] ))
            error_t_test_intercept = np.mean(np.absolute((t_test_slope - t_test_dict[treatment]['intercept'] ) / t_test_dict[treatment]['intercept'] ))

            tau_t_test_slope.append(mean_t_test_slope)
            tau_t_test_intercept.append(mean_t_test_intercept)

            tau_t_test_slope_error.append(error_t_test_slope)
            tau_t_test_intercept_error.append(error_t_test_intercept)


        tau_t_test_slope_all.append(tau_t_test_slope)
        tau_t_test_slope_error_all.append(tau_t_test_slope_error)

        tau_t_test_intercept_all.append(tau_t_test_intercept)
        tau_t_test_intercept_error_all.append(tau_t_test_intercept_error)

        
    tau_t_test_slope_all = np.asarray(tau_t_test_slope_all)
    tau_t_test_slope_error_all = np.asarray(tau_t_test_slope_error_all)

    tau_t_test_intercept_all = np.asarray(tau_t_test_intercept_all)
    tau_t_test_intercept_error_all = np.asarray(tau_t_test_intercept_error_all)
    

    ax_slope = plt.subplot2grid((2, 3), (0, treatment_idx), colspan=1)
    ax_slope_error = plt.subplot2grid((2, 3), (1, treatment_idx), colspan=1)

    # slope
    delta_range = max([t_test_dict[treatment]['slope']  - np.amin(tau_t_test_slope_all),  np.amax(tau_t_test_slope_all) - t_test_dict[treatment]['slope'] ])
    pcm_slope = ax_slope.pcolor(x_axis_log10, y_axis, tau_t_test_slope_all, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=t_test_dict[treatment]['slope']-delta_range, vcenter=t_test_dict[treatment]['slope'], vmax=t_test_dict[treatment]['slope']+delta_range))
    clb_slope = plt.colorbar(pcm_slope, ax=ax_slope)
    clb_slope.set_label(label='Change in exponent after cessation of migration, ' + r'$t_{\mathrm{exponent}}$' , fontsize=10)
    ax_slope.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
    ax_slope.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)
    ax_slope.xaxis.set_major_formatter(plot_utils.fake_log)
    # Set observed marking and label
    clb_slope.ax.axhline(y=t_test_dict[treatment]['slope'], c='k')
    original_ticks = list(clb_slope.get_ticks())
    original_ticks = [round(k, 3) for k in original_ticks]
    
    if treatment_idx == 2:
        original_ticks.remove(-0.4)

    clb_slope.set_ticks(original_ticks + [t_test_dict[treatment]['slope']])
    clb_slope.set_ticklabels(original_ticks + ['Obs.'])
    ax_slope.text(-0.1, 1.04, plot_utils.sub_plot_labels[treatment_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes)

    #ax_slope.set
    ax_slope.set_title(utils.titles_str_no_inocula_dict[treatment], fontsize=16)

    

    # slope error
    pcm_slope_error = ax_slope_error.pcolor(x_axis_log10, y_axis, tau_t_test_slope_error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=np.amin(tau_t_test_slope_error_all), vcenter=np.median(np.ndarray.flatten(tau_t_test_slope_error_all)), vmax=np.amax(tau_t_test_slope_error_all)))
    clb_slope_error = plt.colorbar(pcm_slope_error, ax=ax_slope_error)
    clb_slope_error.set_label(label='Relative error of ' + r'$t_{\mathrm{exponent}}$'  + ' from simulated data', fontsize=10)
    ax_slope_error.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 12)
    ax_slope_error.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 12)
    ax_slope_error.xaxis.set_major_formatter(plot_utils.fake_log)
    ax_slope_error.text(-0.1, 1.04, plot_utils.sub_plot_labels[treatment_idx + 3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_slope_error.transAxes)



fig.text(0.36, 0.95, "Taylor's Law exponent simulations", va='center', fontsize=25)



fig.subplots_adjust(wspace=0.3, hspace=0.25)
fig.savefig(utils.directory + "/figs/taylors_law_migration_slope_heatmap.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/taylors_law_migration_slope_heatmap.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
