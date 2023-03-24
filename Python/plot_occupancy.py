from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma
import scipy.special as special
#from macroecotools import obs_pred_rsquare
import utils
import plot_utils

from scipy.optimize import fsolve
from scipy.special import erf


remove_zeros = True



prevalence_range = np.logspace(-4, 1, num=1000)

transfers = [12,18]
fig, ax_occupancy = plt.subplots(figsize=(4,4))


# get mean and std for rescaling
all_predicted_occupancies = []
all_observed_occupancies = []


for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):
    for transfer in transfers:

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)

        # occupancy
        occupancies, predicted_occupancies, mad_occupancies, beta_occupancies, species_occupances = utils.predict_occupancy(s_by_s, ESVs)
        label = utils.titles_abbreviated_dict[migration_innoculum] + ', transfer ' + str(transfer)
        ax_occupancy.scatter(occupancies, predicted_occupancies, alpha=0.5, c=color_, s=18, zorder=2, label=label)#, linewidth=0.8, edgecolors='k')

        # errors
        errors = np.absolute(occupancies - predicted_occupancies)/occupancies
        survival_array = [sum(errors>=i)/len(errors) for i in prevalence_range]
        survival_array = [sum(errors[np.isfinite(errors)]>=i)/len(errors[np.isfinite(errors)]) for i in prevalence_range]
        survival_array = np.asarray(survival_array)
        #survival_array_no_nan = survival_array[np.isfinite(survival_array)]
        #ax_survival.plot(prevalence_range, survival_array, ls='-', lw=2, c=utils.color_dict_range[migration_innoculum][transfer-3], alpha=0.6, zorder=1)

        all_observed_occupancies.extend(occupancies.tolist())
        all_predicted_occupancies.extend(predicted_occupancies.tolist())




# occupancyp
occupancy_min = min(all_observed_occupancies + all_predicted_occupancies)
occupancy_max = max(all_observed_occupancies + all_predicted_occupancies)

ax_occupancy.plot([occupancy_min*0.8, occupancy_max*1],[occupancy_min*0.8, occupancy_max*1], lw=2, ls='--',c='k',zorder=2, label='1:1')
ax_occupancy.set_xlim([occupancy_min*0.8, 1.1])
ax_occupancy.set_ylim([occupancy_min*0.8, 1.1])

ax_occupancy.set_xscale('log', basex=10)
ax_occupancy.set_yscale('log', basey=10)
ax_occupancy.set_xlabel('Observed occupancy', fontsize=12)
ax_occupancy.set_ylabel('Predicted occupancy', fontsize=12)
ax_occupancy.tick_params(axis='both', which='minor', labelsize=9)
ax_occupancy.tick_params(axis='both', which='major', labelsize=9)
#ax_occupancy.set_title('%s\nTransfer %d' % (utils.titles_dict[migration_innoculum], transfer), fontsize=9)
ax_occupancy.legend(loc="lower right", fontsize=5.5)









fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/occupancy.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
