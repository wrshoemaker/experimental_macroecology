from __future__ import division
import os, sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

from itertools import combinations
import statsmodels.stats.multitest as multitest


transfers = [12, 18]




fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))


row_count = 0

migration_innocula = [('No_migration',4), ('Parent_migration',4)]
transfers = [12, 18]

for migration_innoculum in migration_innocula:

    attractor_dict = utils.get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    for transfer in transfers:

        ax_shape = axes[0, row_count] #fig.add_subplot(gs[0, row_count])
        ax_shape_vs_error = axes[1, row_count] #fig.add_subplot(gs[1, row_count])


        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])
        species = np.asarray(species)
        occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies

        theta_dict = {}

        # plot attractors for all
        color = utils.get_color_attractor('All', transfer)

        relative_errors = np.absolute(occupancies - predicted_occupancies) / occupancies


        for attractor_idx, attractor in enumerate(attractor_dict.keys()):

            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            s_by_s_attractor = s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            s_by_s_attractor = s_by_s_attractor[attractor_species_idx]


            # plot attractors
            occupancies_attractor, predicted_occupancies_attractor = utils.predict_occupancy(s_by_s_attractor)
            color = utils.get_color_attractor(attractor, transfer)

            relative_errors_attractor = np.absolute(occupancies_attractor - predicted_occupancies_attractor) / occupancies_attractor

            rel_s_by_s_attractor = (s_by_s_attractor/s_by_s_attractor.sum(axis=0))

            means = np.mean(rel_s_by_s_attractor, axis=1)
            vars = np.var(s_by_s_attractor, axis=1)

            #thetas = means / vars
            thetas = means

            for s_idx, s in enumerate(attractor_species):

                if s not in theta_dict:
                    theta_dict[s] = {}

                theta_dict[s][attractor] = thetas[s_idx]


        thetas_a = []
        thetas_p = []
        errors_all = []
        for k, d in theta_dict.items():

            if len(d) != 2:
                continue

            error_k = errors[species==k][0]

            if np.isnan(error_k) == True:
                continue

            if error_k < 10**-5:
                continue


            # delta shape param vs. error

            thetas_a.append(d['Alcaligenaceae'])
            thetas_p.append(d['Pseudomonadaceae'])
            errors_all.append(error_k)


        # plot shape params
        #if len(thetas_p) == 0:
        #    continue

        thetas_a = np.asarray(thetas_a)
        thetas_p = np.asarray(thetas_p)
        errors_all = np.asarray(errors_all)

        thetas_a_no_nan = thetas_a[(np.isfinite(thetas_a)) & np.isfinite(thetas_p)]
        thetas_p_no_nan = thetas_p[(np.isfinite(thetas_a)) & np.isfinite(thetas_p)]
        errors_all_no_nan = errors_all[(np.isfinite(thetas_a)) & np.isfinite(thetas_p)]

        delta_theta = np.absolute(thetas_a_no_nan - thetas_p_no_nan)


        ax_shape.scatter(thetas_p_no_nan, thetas_a_no_nan, alpha=0.7, s=30, zorder=2, c='k')
        max_plot = max( [max(thetas_a_no_nan), max(thetas_p_no_nan)] )
        min_plot = min( [min(thetas_a_no_nan), min(thetas_p_no_nan)] )
        ax_shape.plot([min_plot, max_plot],[min_plot, max_plot], lw=3,ls='--',c='k',zorder=1, label='1:1')
        ax_shape.set_xscale('log', basex=10)
        ax_shape.set_yscale('log', basey=10)
        ax_shape.set_xlabel(r'$\beta_{i}/\bar{x}_{i}$' + ', Pseudomonadaceae', fontsize=12)
        ax_shape.set_ylabel(r'$\beta_{i}/\bar{x}_{i}$' + ', Alcaligenaceae', fontsize=12)
        ax_shape.tick_params(axis='both', which='minor', labelsize=5)
        ax_shape.tick_params(axis='both', which='major', labelsize=5)

        #print(slope, p_value)


        #print(thetas_a_no_nan)

        ax_shape_vs_error.scatter(delta_theta, errors_all_no_nan, alpha=0.7, s=30, zorder=2, c='k')
        ax_shape_vs_error.set_xscale('log', basex=10)
        ax_shape_vs_error.set_yscale('log', basey=10)
        ax_shape_vs_error.tick_params(axis='both', which='minor', labelsize=5)
        ax_shape_vs_error.tick_params(axis='both', which='major', labelsize=5)
        ax_shape_vs_error.set_xlabel('Absolute difference of rate parameters, ' + r'$\left | \Delta \beta_{i}/\bar{x}_{i} \right |$', fontsize=11)
        ax_shape_vs_error.set_ylabel('Relative error, merged attractors', fontsize=12)

        ax_shape_vs_error.set_xlim(0.5*min(delta_theta), 1.6*max(delta_theta))
        ax_shape_vs_error.set_ylim(0.2*min(errors_all_no_nan), 1.6*max(errors_all_no_nan))


        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(delta_theta), np.log10(errors_all_no_nan))

        ax_shape_vs_error.text(0.77,0.17, r'$\beta = {{{}}}$'.format(str( round(slope, 2) )), fontsize=12, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)

        if p_value < 0.05:
            x_log10_range =  np.linspace(min(np.log10(delta_theta)) , max(np.log10(delta_theta)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            ax_shape_vs_error.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=3, linestyle='--', zorder=2)#, label="OLS regression")
            ax_shape_vs_error.text(0.77,0.1, r'$P < 0.05$', fontsize=12, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)

        else:
            ax_shape_vs_error.text(0.77,0.09, r'$P \nless 0.05$', fontsize=12, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)


        ax_shape.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=12)

        if row_count == 0:
            ax_shape.legend(loc="upper left", fontsize=11)



        row_count += 1


fig.subplots_adjust(wspace=0.24, hspace=0.2)
fig.savefig(utils.directory + "/figs/gamma_attractors_mean.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
