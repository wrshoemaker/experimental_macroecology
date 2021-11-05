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


prevalence_range = np.logspace(-4, 1, num=1000)


#gs = gridspec.GridSpec(nrows=4, ncols=4)
#fig = plt.figure(figsize = (10, 10))


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))


row_count = 0

migration_innocula = [('No_migration',4), ('Parent_migration',4)]
transfers = [12, 18]

for migration_innoculum in migration_innocula:

    attractor_dict = utils.get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])
    for transfer in transfers:

        ax_occupancy = axes[0, row_count] #fig.add_subplot(gs[0, row_count])
        ax_survival = axes[1, row_count] #fig.add_subplot(gs[1, row_count])
        ax_shape = axes[2, row_count] #fig.add_subplot(gs[2, row_count])
        ax_shape_vs_error = axes[3, row_count]  #fig.add_subplot(gs[3, row_count])


        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])
        species = np.asarray(species)
        occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies

        theta_dict = {}

        # plot attractors for all
        color = utils.get_color_attractor('All', transfer)
        ax_occupancy.scatter(occupancies, predicted_occupancies, alpha=0.8, s=15, zorder=2, c=[color], label='Merged', linewidth=0.8, edgecolors='k')

        relative_errors = np.absolute(occupancies - predicted_occupancies) / occupancies

        # survival all
        survival_array = [sum(relative_errors>=i)/len(relative_errors) for i in prevalence_range]
        survival_array = [sum(relative_errors[np.isfinite(relative_errors)]>=i)/len(relative_errors[np.isfinite(relative_errors)]) for i in prevalence_range]

        survival_array = np.asarray(survival_array)
        #survival_array_no_nan = survival_array[np.isfinite(survival_array)]
        ax_survival.plot(prevalence_range, survival_array, ls='-', lw=3, c=utils.get_color_attractor('All', transfer), alpha=0.8, zorder=2, label='Merged')




        for attractor_idx, attractor in enumerate(attractor_dict.keys()):

            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            s_by_s_attractor = s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            s_by_s_attractor = s_by_s_attractor[attractor_species_idx]

            # plot attractors
            occupancies_attractor, predicted_occupancies_attractor = utils.predict_occupancy(s_by_s_attractor)
            color = utils.get_color_attractor(attractor, transfer)
            ax_occupancy.scatter(occupancies_attractor, predicted_occupancies_attractor, alpha=0.8, s=15, zorder=2, c=[color], label=attractor, linewidth=0.8, edgecolors='k')

            relative_errors_attractor = np.absolute(occupancies_attractor - predicted_occupancies_attractor) / occupancies_attractor

            # plot survival
            survival_array_attractor = [sum(relative_errors_attractor[np.isfinite(relative_errors_attractor)]>=i)/len(relative_errors_attractor[np.isfinite(relative_errors_attractor)]) for i in prevalence_range]
            survival_array_attractor = np.asarray(survival_array_attractor)
            ax_survival.plot(prevalence_range, survival_array_attractor, ls='-', lw=3, c=utils.get_color_attractor(attractor, transfer), alpha=0.7, zorder=2, label=attractor)





            means = np.mean(s_by_s_attractor, axis=1)
            vars = np.var(s_by_s_attractor, axis=1)

            thetas = means / vars



            #thetas = utils.calculate_theta(s_by_s_attractor)

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



        ax_occupancy.plot([0.01,1],[0.01,1], lw=2,ls='--',c='k',zorder=1, label='1:1')
        ax_occupancy.set_xlim([0.008,1.03])
        ax_occupancy.set_ylim([0.008,1.03])
        ax_occupancy.set_xscale('log', basex=10)
        ax_occupancy.set_yscale('log', basey=10)
        ax_occupancy.set_xlabel('Observed occupancy', fontsize=9)
        ax_occupancy.set_ylabel('Predicted occupancy', fontsize=9)
        ax_occupancy.tick_params(axis='both', which='minor', labelsize=5)
        ax_occupancy.tick_params(axis='both', which='major', labelsize=5)
        ax_occupancy.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=9)



        ax_survival.set_xscale('log', basex=10)
        ax_survival.set_yscale('log', basey=10)
        ax_survival.set_xlabel('Relative error, ' + r'$\epsilon$', fontsize=9)
        ax_survival.set_ylabel('Fraction of ASVs ' + r'$\geq \epsilon$', fontsize=9)
        ax_survival.tick_params(axis='both', which='minor', labelsize=5)
        ax_survival.tick_params(axis='both', which='major', labelsize=5)




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


        ax_shape.scatter(thetas_p_no_nan, thetas_a_no_nan, alpha=0.5, s=15, zorder=2, c='k')
        max_plot = max( [max(thetas_a_no_nan), max(thetas_p_no_nan)] )
        min_plot = min( [min(thetas_a_no_nan), min(thetas_p_no_nan)] )
        ax_shape.plot([min_plot, max_plot],[min_plot, max_plot], lw=2,ls='--',c='k',zorder=1, label='1:1')
        ax_shape.set_xscale('log', basex=10)
        ax_shape.set_yscale('log', basey=10)
        ax_shape.set_xlabel(r'$\beta_{i}/\bar{x}_{i}$' + ', Pseudomonadaceae', fontsize=9)
        ax_shape.set_ylabel(r'$\beta_{i}/\bar{x}_{i}$' + ', Alcaligenaceae', fontsize=9)
        ax_shape.tick_params(axis='both', which='minor', labelsize=5)
        ax_shape.tick_params(axis='both', which='major', labelsize=5)

        #print(slope, p_value)




        ax_shape_vs_error.scatter(delta_theta, errors_all_no_nan, alpha=0.4, s=15, zorder=2, c='k')
        ax_shape_vs_error.set_xscale('log', basex=10)
        ax_shape_vs_error.set_yscale('log', basey=10)
        ax_shape_vs_error.tick_params(axis='both', which='minor', labelsize=5)
        ax_shape_vs_error.tick_params(axis='both', which='major', labelsize=5)
        ax_shape_vs_error.set_xlabel('Absolute difference of\nrate parameters, ' + r'$\left | \Delta \beta_{i}/\bar{x}_{i} \right |$', fontsize=9)
        ax_shape_vs_error.set_ylabel('Relative error, merged attractors', fontsize=8.5)

        ax_shape_vs_error.set_xlim(0.5*min(delta_theta), 1.6*max(delta_theta))
        ax_shape_vs_error.set_ylim(0.2*min(errors_all_no_nan), 1.6*max(errors_all_no_nan))

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(delta_theta), np.log10(errors_all_no_nan))

        ax_shape_vs_error.text(0.77,0.22, r'$\beta = {{{}}}$'.format(str( round(slope, 2) )), fontsize=10, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)

        if p_value < 0.05:
            x_log10_range =  np.linspace(min(np.log10(delta_theta)) , max(np.log10(delta_theta)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            ax_shape_vs_error.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2)#, label="OLS regression")
            #ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")
            ax_shape_vs_error.text(0.77,0.1, r'$P < 0.05$', fontsize=10, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)

        else:
            ax_shape_vs_error.text(0.77,0.1, r'$P \nless 0.05$', fontsize=10, color='k', ha='center', va='center', transform=ax_shape_vs_error.transAxes)



        if row_count == 0:
            ax_occupancy.legend(loc="lower left", fontsize=6)
            ax_survival.legend(loc="lower left", fontsize=6)
            ax_shape.legend(loc="lower left", fontsize=6)



        row_count += 1


fig.subplots_adjust(wspace=0.34, hspace=0.4)
fig.savefig(utils.directory + "/figs/gamma_attractors.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
