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

transfer = 18

c_all = [utils.color_dict_range[e][transfer-3] for e in experiments ]

fig, ax = plt.subplots(figsize=(4,4))


errors_to_plot_all = []
x_idx = np.asarray([0,1,2])
for s in species:

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


    ax.plot(x_idx_to_plot, error_all_to_plot, c='k', ls='-', lw=1, alpha=0.5, zorder=1)

    print(x_idx_to_plot, error_all_to_plot)
    for error_all_to_plot_i_idx, error_all_to_plot_i in enumerate(error_all_to_plot):

        #print(c_all[x_idx_to_plot[error_all_to_plot_i_idx]])

        ax.scatter(x_idx_to_plot[error_all_to_plot_i_idx], error_all_to_plot[error_all_to_plot_i_idx], alpha=0.8, c=c_all[x_idx_to_plot[error_all_to_plot_i_idx]].reshape(1,-1) , zorder=2)

        errors_to_plot_all.append(error_all_to_plot[error_all_to_plot_i_idx])


ax.set_ylim(min(errors_to_plot_all)*0.5, 10)

ax.set_yscale('log', basey=10)


fig_name = utils.directory + '/figs/paried_gamma_error.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()





#a = np.asarray([1,3,4,5,6])
#b = np.asarray([7,8,9,10,11])

#utils.run_permutation_paired_t_test(a,b)