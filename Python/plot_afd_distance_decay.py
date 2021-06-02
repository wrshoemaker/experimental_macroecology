from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm
from matplotlib.lines import Line2D

from itertools import combinations
import statsmodels.stats.multitest as multitest


color_dict = {'No_migration': utils.rgb_blue, 'Global_migration': utils.rgb_red}


migration_innocula = [('No_migration',4), ('Global_migration',4)]

transfers = list(range(1, 19))

fig, ax = plt.subplots(figsize=(4,4))

distances_dict = {}

for migration_innoculum in migration_innocula:

    s_by_s_0, species_0, comm_rep_list_0 = utils.get_s_by_s_migration_test_singleton(transfer=1, migration=migration_innoculum[0],inocula=migration_innoculum[1])
    relative_s_by_s_0 = (s_by_s_0/s_by_s_0.sum(axis=0))


    afd_0 = relative_s_by_s_0.flatten()
    afd_0 = afd_0[afd_0 > 0]
    afd_0 = np.log10(afd_0)

    distances = [0]

    for transfer in transfers[1:]:

        s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0],inocula=migration_innoculum[1])

        relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        afd = relative_s_by_s.flatten()
        afd = afd[afd > 0]
        afd = np.log10(afd)

        KS_statistic, p_value = stats.ks_2samp(afd_0, afd)

        distances.append(KS_statistic)


    ax.plot(transfers, distances, c = 'k', zorder=2)

    colors = [color_dict[migration_innoculum[0]][transfer-1] for transfer in transfers]

    ax.scatter(transfers, distances, color = colors, edgecolors='k', alpha=1, zorder=3)


    ax.scatter(-10, -1, color = colors[-8], edgecolors='k', alpha=1, zorder=3, label=utils.titles_no_inocula_dict[migration_innoculum])


    distances_dict[migration_innoculum] = np.asarray(distances)

    #ax.scatter(variance_transfers_no_migration, mean_no_migration, color = width_colors_no_migration, edgecolors='k', zorder=3)



# add t test

ax.set_xlabel('Transfer, ' + r'$t$', fontsize=12)
ax.set_ylabel('KS distance of AFD\nfrom first transfer, ' + r'$d(1,t)$', fontsize=12)


ax.set_xlim([0, 19])
ax.set_ylim([-0.02, 0.35])

#legend_elements = [Line2D([0], [0], marker='o', markerfacecolor=utils.color_dict_range[('No_migration',4)][12],
#                    label='No migration', markeredgecolor='k', markersize=9)]


#ax.legend(handles=legend_elements, loc='lower right')

ax.legend(loc='lower right')


mean_difference, p_value = utils.test_difference_two_time_series(distances_dict[migration_innocula[0]], distances_dict[migration_innocula[1]])




ax.text(0.14,0.83, r'$\left \langle \Delta d \right \rangle = %0.3f$' % mean_difference, fontsize=9, color='k', ha='center', va='center', transform=ax.transAxes )
ax.text(0.1,0.76, utils.get_p_value(p_value), fontsize=9, color='k', ha='center', va='center', transform=ax.transAxes )


#('No_migration',4):rgb_blue, ('Global_migration',4):rgb_red


fig.savefig(utils.directory + "/figs/afd_distance_decay.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
