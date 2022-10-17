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




def load_simulation_all_dict():

    with open(slm_simulation_utils.simulation_migration_all_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


dict_ = load_simulation_all_dict()

print(dict_.keys())

tau_all = np.asarray(list(dict_.keys()))
sigma_all = np.asarray(list(dict_[tau_all[0]].keys()))


fig = plt.figure(figsize = (11, 9.5))
gs = gridspec.GridSpec(nrows=3, ncols=3)



result_dict = {'no_migration': {'taylors_intercept': -0.23910487202635533, 'taylors_slope': 1.83181735277136, 'mean_log_error': -0.7198265811341613},
                'global_migration': {'taylors_intercept': -1.5177411136045995, 'taylors_slope': 1.544625265838575, 'mean_log_error': -1.4098623149648624},
                'parent_migration': {'taylors_intercept': -1.371192519883035, 'taylors_slope': 1.5339236365354767, 'mean_log_error': -1.0126866124216345}}



migration_treatments = ['no_migration', 'global_migration', 'parent_migration']

for m_idx, m in enumerate(migration_treatments):

    ax_taylor_slope = fig.add_subplot(gs[0, m_idx])
    ax_taylor_inter = fig.add_subplot(gs[1, m_idx])
    ax_gamma = fig.add_subplot(gs[2, m_idx])

    slope_all = []
    intercept_all = []
    error_all = []

    for tau in tau_all:

        slope_sigma_all = []
        intercept_sigma_all = []
        error_sigma_all = []

        for sigma in sigma_all:

            slope_tau_sigma = np.mean(dict_[tau][sigma][11][m]['taylors_slope'])
            slope_sigma_all.append(slope_tau_sigma)

            intercept_tau_sigma = np.mean(dict_[tau][sigma][11][m]['taylors_intercept'])
            intercept_sigma_all.append(intercept_tau_sigma)

            error_tau_sigma = np.mean(dict_[tau][sigma][11][m]['mean_log_error'])
            error_sigma_all.append(error_tau_sigma)

        slope_all.append(slope_sigma_all)
        intercept_all.append(intercept_sigma_all)
        error_all.append(error_sigma_all)

    slope_all = np.asarray(slope_all)
    intercept_all = np.asarray(intercept_all)
    error_all = np.asarray(error_all)


    print(np.max(error_all), np.min(error_all))

    # plot slope
    pcm_taylor_slope = ax_taylor_slope.pcolor(sigma_all, tau_all, slope_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=1.5, vcenter=2.5, vmax=3.5))
    clb_taylor_slope = plt.colorbar(pcm_taylor_slope, ax=ax_taylor_slope)
    observed_slope = result_dict[m]['taylors_slope']
    original_ticks = list(clb_taylor_slope.get_ticks())
    clb_taylor_slope.set_ticks(original_ticks + [observed_slope])
    clb_taylor_slope.set_ticklabels(original_ticks + ['Obs.'])
    clb_taylor_slope.ax.tick_params(labelsize=8)


    # plot intercept
    pcm_taylor_inter = ax_taylor_inter.pcolor(sigma_all, tau_all, intercept_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2))
    clb_taylor_inter = plt.colorbar(pcm_taylor_inter, ax=ax_taylor_inter)
    observed_inter = result_dict[m]['taylors_intercept']
    original_ticks = list(clb_taylor_inter.get_ticks())
    clb_taylor_inter.set_ticks(original_ticks + [observed_inter])
    clb_taylor_inter.set_ticklabels(original_ticks + ['Obs.'])
    clb_taylor_inter.ax.tick_params(labelsize=8)



    # plot gamma
    pcm_gamma = ax_gamma.pcolor(sigma_all, tau_all, error_all, cmap='YlOrRd', norm=colors.TwoSlopeNorm(vmin=-11, vcenter=-5.5, vmax=0))
    clb_gamma = plt.colorbar(pcm_gamma, ax=ax_gamma)
    observed_inter = result_dict[m]['mean_log_error']
    original_ticks = list(clb_gamma.get_ticks())
    clb_gamma.set_ticks(original_ticks + [observed_inter])
    clb_gamma.set_ticklabels(original_ticks + ['Obs.'])
    clb_gamma.ax.tick_params(labelsize=8)


    #ax_taylor_inter.set_xlabel("Strength of growth rate fluctuations, " + r'$\sigma$', fontsize = 11)
    #ax_taylor_inter.set_ylabel("Timescale of growth, " + r'$\tau$', fontsize = 11)


    if  m_idx == 2:
        clb_taylor_slope.set_label(label="Taylor's Law slope")
        clb_taylor_inter.set_label(label="Taylor's Law intercept")
        clb_gamma.set_label(label="Occupancy error, gamma")


    if m_idx == 0:
        title = 'No migration'

    elif m_idx == 1:
        title = 'Global migration'

    else:
        title = 'Parent migration'


    ax_taylor_slope.set_title(title, fontsize=12, fontweight='bold' )


fig.text(0.5, 0.06, "Strength of growth rate fluctuations, " + r'$\sigma$', va='center', ha='center', fontsize=16)

fig.text(0.06, 0.5,"Timescale of growth, " + r'$\tau$', va='center',rotation='vertical', fontsize=16)



fig.subplots_adjust(hspace=0.2,wspace=0.2)
fig.savefig(utils.directory + "/figs/simulation_all.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
