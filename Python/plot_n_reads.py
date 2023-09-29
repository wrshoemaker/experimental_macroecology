from __future__ import division
import os, sys
import numpy as np

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

from matplotlib import cm
import matplotlib.pyplot as plt


transfer=18


#s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])


#migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]


n_reads_all = []
for migration_innoculum in utils.migration_innocula:

    s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])
    n_reads = s_by_s.sum(axis=0)

    n_reads_all.extend(n_reads.tolist())



n_reads_all_log10 = np.log10(n_reads_all)

n_reads_all_log10_mean = np.mean(n_reads_all_log10)


fig, ax = plt.subplots(figsize=(4,4))


print('Mean log10 = ' + str(round(n_reads_all_log10_mean, 3)))


ax.hist(n_reads_all_log10, lw=3, alpha=0.8, bins= 15, color='k', histtype='step', density=True)
ax.axvline(n_reads_all_log10_mean, lw=2, ls='--',color='k', zorder=1, label='Mean of ' + r'$\mathrm{log}_{10} $')


ax.set_xlabel('Total number of reads in a sample, ' +  r'$\mathrm{log}_{10}$' , fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)

ax.legend(loc="upper left", fontsize=8)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
#fig.savefig(utils.directory + "/figs/n_reads_hist.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + '/figs/n_reads_hist.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()



#for migration_innoculum_row_idx, migration_innoculum_row in enumerate(migration_innocula):

#    for migration_innoculum_column_idx, migration_innoculum_column in enumerate(migration_innoculum_row):

#        title = utils.titles_dict[migration_innoculum_column]

#        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum_column[0], inocula=migration_innoculum_column[1])


#migration_innocula = [('No_migration',4), ('Parent_migration',4)]
#transfers = [12, 18]
