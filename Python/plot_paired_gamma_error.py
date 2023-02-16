from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
import plot_utils


# keep this order
experiments = [('Parent_migration', 4), ('No_migration', 4), ('Global_migration', 4)]






# make error dict
error_dict = {}
for experiment_idx, experiment in enumerate(experiments):

    #ax = plt.subplot2grid((2,1), (experiment_idx,0), colspan=1)

    for transfer in [12,18]:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
        occupancies, predicted_occupancies, mad, beta, species = utils.predict_occupancy(s_by_s, species)
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies

        for s_idx, s in enumerate(species):

            if s not in error_dict:
                error_dict[s] = {}

            if transfer not in error_dict[s]:
                error_dict[s][transfer] = {}

            error_dict[s][transfer][experiment] = errors[s_idx]





species_all = list(error_dict.keys())

transfer = 12
n_iter = 10000

transfer_pairs = [[('Parent_migration', 4), ('No_migration', 4)],  [('Global_migration', 4), ('No_migration', 4)]]
for transfer_pair_idx, transfer_pair in enumerate(transfer_pairs):
    treatment_1 = []
    treatment_2 = []
    for s in species_all:

        if 12 in error_dict[s]:

            if (transfer_pair[0] in error_dict[s][12]) and (transfer_pair[1] in error_dict[s][12]):

                treatment_1.append(error_dict[s][12][transfer_pair[0]])
                treatment_2.append(error_dict[s][12][transfer_pair[1]])

    treatment_merged = np.asarray(treatment_1 + treatment_2)

    treatment_1 = np.asarray(treatment_1)
    treatment_2 = np.asarray(treatment_2)

    delta_treatment_observed = np.mean(treatment_1 - treatment_2)
    delta_treatment_observed_null = []
    for i in range(n_iter):
        np.random.shuffle(treatment_merged)

        delta_treatment_observed_null.append(np.mean(treatment_merged[:len(treatment_1)] - treatment_merged[len(treatment_1):]))

    delta_treatment_observed_null = np.asarray(delta_treatment_observed_null)

    # ask wheether mean is lower than null
    p = sum(delta_treatment_observed_null < delta_treatment_observed) / n_iter
    print(transfer_pair, delta_treatment_observed, p)






c_all = [utils.color_dict_range[e][transfer-3] for e in experiments ]


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)

ax_survival = plt.subplot2grid((1, 2), (0,0), colspan=1)
ax_comparison = plt.subplot2grid((1, 2), (0,1), colspan=1)

ax_survival.text(-0.1, 1.04, plot_utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_survival.transAxes)
ax_comparison.text(-0.1, 1.04, plot_utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_comparison.transAxes)




# plot survival
prevalence_range = np.logspace(-4, 1, num=1000)
for transfer in [12,18]:

    for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)



        # occupancy
        occupancies, predicted_occupancies, mad_occupancies, beta_occupancies, species_occupances = utils.predict_occupancy(s_by_s, ESVs)

        # errors
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies
        survival_array = [sum(errors>=i)/len(errors) for i in prevalence_range]
        survival_array = [sum(errors[np.isfinite(errors)]>=i)/len(errors[np.isfinite(errors)]) for i in prevalence_range]
        survival_array = np.asarray(survival_array)
        ax_survival.plot(prevalence_range, survival_array, ls='-', lw=2, c=utils.color_dict_range[migration_innoculum][transfer-3], alpha=0.6, zorder=1)




ax_survival.set_xscale('log', basex=10)
ax_survival.set_yscale('log', basey=10)
ax_survival.set_xlabel('Occupancy relative error, ' + r'$\epsilon$', fontsize=12)
ax_survival.set_ylabel('Fraction of ASVs ' + r'$\geq \epsilon$', fontsize=12)
ax_survival.tick_params(axis='both', which='minor', labelsize=9)
ax_survival.tick_params(axis='both', which='major', labelsize=9)
ax_survival.xaxis.set_tick_params(labelsize=7) 
ax_survival.yaxis.set_tick_params(labelsize=7)





### plot comparison




errors_to_plot_all = []
x_idx = np.asarray([0,1,2])
for s in species_all:

    error_all = []

    if transfer not in error_dict[s]:
        continue

    for experiment in experiments:

        # add dummy nan for now

        if experiment not in error_dict[s][transfer]:
            error_all.append(float('nan'))

        else:
            error_all.append(error_dict[s][transfer][experiment])


    error_all = np.asarray(error_all)

    x_idx_to_plot = x_idx[(~np.isnan(error_all)) & (error_all>0)]
    error_all_to_plot = error_all[(~np.isnan(error_all)) & (error_all>0)]

    # ignore ASVs that dont have an observation in the no migration treatment
    if (len(error_all_to_plot) < 2) and (1 not in x_idx_to_plot):
        continue


    ax_comparison.plot(x_idx_to_plot, error_all_to_plot, c='k', ls='-', lw=0.8, alpha=0.2, zorder=1)

    for error_all_to_plot_i_idx, error_all_to_plot_i in enumerate(error_all_to_plot):

        ax_comparison.scatter(x_idx_to_plot[error_all_to_plot_i_idx], error_all_to_plot[error_all_to_plot_i_idx], s=10, alpha=0.8, c=c_all[x_idx_to_plot[error_all_to_plot_i_idx]].reshape(1,-1) , zorder=2)

        errors_to_plot_all.append(error_all_to_plot[error_all_to_plot_i_idx])


ax_comparison.set_ylim(min(errors_to_plot_all)*0.5, 10)

ax_comparison.set_yscale('log', basey=10)
#ax_comparison.xaxis.set_tick_params(labelsize=7) 
ax_comparison.yaxis.set_tick_params(labelsize=6)


ax_comparison.set_xticks([0,1,2])
ax_comparison.set_xlim([-0.25, 2.25])
ax_comparison.set_xticklabels(['Regional\nmigration', 'No\nmigration', 'Global\nmigration'], fontsize=10)

ax_comparison.set_title('Transfer 12', fontsize=13)
ax_comparison.set_ylabel('Occupancy relative error', fontsize=12)


#experiments = [('Parent_migration', 4), ('No_migration', 4), ('Global_migration', 4)]

fig.subplots_adjust(wspace=0.35, hspace=0.3)

fig_name = utils.directory + '/figs/paried_gamma_error.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()





#a = np.asarray([1,3,4,5,6])
#b = np.asarray([7,8,9,10,11])

#utils.run_permutation_paired_t_test(a,b)
