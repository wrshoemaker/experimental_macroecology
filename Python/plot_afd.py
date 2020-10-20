from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm


#afd_migration_transfer_12

afd_dict = {}
transfers = [12, 18]

experiments = [('No_migration',4), ('Global_migration',4), ('Parent_migration', 4)  ]

for experiment in experiments:

    afd_dict[experiment] = {}

    for transfer in transfers:

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

        afd = relative_s_by_s.flatten()
        afd = afd[afd>0]
        afd = np.log10(afd)

        afd_dict[experiment][transfer] = afd


fig = plt.figure(figsize = (4*len(experiments), 4))
fig.subplots_adjust(bottom= 0.15)

for experiment_idx, experiment in enumerate(experiments):

    ax = plt.subplot2grid((1, len(experiments)), (0, experiment_idx), colspan=1)


    for transfer in transfers:

        colors_experiment_transfer = utils.color_dict[experiment][transfer-1]
        afd = afd_dict[experiment][transfer]
        label = '%s, transfer %d' %(utils.titles_no_inocula_dict[experiment], transfer)

        ax.hist(afd, lw=3, alpha=0.8, bins= 15, color=colors_experiment_transfer, histtype='step', label='Transfer %d'%transfer,  density=True)

    KS_statistic, p_value = stats.ks_2samp(afd_dict[experiment][transfers[0]], afd_dict[experiment][transfers[1]])

    ax.text(0.20,0.8, '$D=%0.3f$' % KS_statistic, fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )
    ax.text(0.18,0.73, utils.get_p_value(p_value), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )


    ax.set_title(utils.titles_dict[experiment], fontsize=12, fontweight='bold' )
    ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel('Log relative abundance', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    ax.set_xlim(-5.5, 0.2)

#ax.set_xscale('log', basex=10)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/afd_temporal.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
