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


from matplotlib import cm, colors

#color_range =  np.linspace(0.0, 1.0, 18)
#rgb = cm.get_cmap('Blues')( color_range )
alpha = 0.05
zeros = True

#experiments = [('No_migration',4), ('Global_migration',4), ('Glucose',  np.nan) ]
experiments = [('No_migration', 4), ('Global_migration', 4)]

transfer_max_dict = {('No_migration', 4): 18,
                     ('Global_migration', 4): 18}

#s_by_s_1, species_1, comm_rep_list_1 = utils.get_s_by_s("Glucose", transfer=1)



fig = plt.figure(figsize = (7, 8)) #
fig.subplots_adjust(bottom= 0.15)


#ax_occupancy = plt.subplot2grid((1, 3), (0,0), colspan=1)

#ax_global = plt.subplot2grid((2,1), (1,0), colspan=1)



experiment_dict = {}

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((2,1), (experiment_idx,0), colspan=1)

    transfer_max = transfer_max_dict[experiment]

    species_relative_abundances_dict = {}

    error_per_transfer_dict = {}

    transfers_all = list(range(1, transfer_max+1))
    mean_error_all = []
    color_all = []
    for transfer in transfers_all:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=experiment[0], inocula=experiment[1])
        occupancies, predicted_occupancies, species_occupances = utils.predict_occupancy(s_by_s, species)
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies
        for species_i, errors_i in zip(species, errors):

            if species_i not in error_per_transfer_dict:
                error_per_transfer_dict[species_i] = {}
                error_per_transfer_dict[species_i]['transfers'] = []
                error_per_transfer_dict[species_i]['errors'] = []

            error_per_transfer_dict[species_i]['transfers'].append(transfer)
            error_per_transfer_dict[species_i]['errors'].append(errors_i)


        mean_error = np.mean(errors)
        mean_error_all.append(mean_error)

        color_ = utils.color_dict_range[experiment][transfer-1]
        color_ = color_.reshape(1,-1)
        color_all.append(color_)

        ax.scatter(transfer, mean_error, c=color_, s=40, edgecolor='k', zorder=2)

    #x = [4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18]
    #y = [0.06317090573997738, 2.699020673624375e-08, 4.250755636370229e-08, 0.00015494924588943526, 1.376442280931478e-05, 0.04998575304525481, 0.0005483043701858259, 0.06101576855562106, 4.16063924442156e-05, 5.1356317433803866e-05, 0.0011921814168054379]

    #ax.plot(x, y, alpha=0.4, c=utils.color_dict_range[experiment][7], zorder=1)

    y_min = 10**-7
    for key, value in error_per_transfer_dict.items():

        if len(value['errors']) < 4:
            continue

        transfers_i = np.asarray(value['transfers'])
        errors_i = np.asarray(value['errors'])

        transfers_i_to_plot = transfers_i[errors_i > y_min]
        errors_i_to_plot = errors_i[errors_i > y_min]
        ax.plot(transfers_i_to_plot, errors_i_to_plot, alpha=0.3, c=utils.color_dict_range[experiment][7], zorder=1)

    ax.plot([-4, -3], [100, 1000], alpha=0.8, c=utils.color_dict_range[experiment][7], zorder=1, label='One ASV')
    ax.scatter([-4, -3], [100, 1000], c='k', s=20 , edgecolor='k', zorder=2, label ='Mean')
    ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel('Transfer', fontsize=14)
    ax.set_ylabel('Relative error, ' + r'$\epsilon$', fontsize=14)
    ax.set_title(utils.titles_dict[experiment], fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylim([y_min, 10])
    ax.set_xlim([0.5, 18.5])
    ax.set_yscale('log', basey=10)



    #ax.plot(mean_range, variance_range, lw=3, ls=':', c='k', label='Max. ' + r'$\sigma^{2}_{x}$')



fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/gamma_error_per_transfer.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
