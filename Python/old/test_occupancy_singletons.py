from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed(123456789)

s_by_s_singletons, species_singletons, comm_rep_list_singletons = utils.get_s_by_s_migration_test_singleton(transfer=18,migration='No_migration',inocula=4)

occupancies_singletons, predicted_occupancies_singletons = utils.predict_occupancy(s_by_s_singletons)


# ('No_migration',4)

#migration_innocula = [('No_migration',4), ('Parent_migration',4)]

s_by_s, species, comm_rep_list = utils.get_s_by_s_migration(transfer=18,migration='No_migration',inocula=4)

occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)


print(len(occupancies), len(occupancies_singletons))




fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15)
print(occupancies, predicted_occupancies)


ax_plot = plt.subplot2grid((1, 2), (0, 0), colspan=1)

ax_plot_singletons = plt.subplot2grid((1, 2), (0, 1), colspan=1)




ax_plot.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
ax_plot_singletons.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)


ax_plot.scatter(occupancies, predicted_occupancies, alpha=0.8,zorder=2)#, c='#87CEEB')

ax_plot_singletons.scatter(occupancies_singletons, predicted_occupancies_singletons, alpha=0.8,zorder=2)#, c='#87CEEB')

ax_plot.set_xscale('log', basex=10)
ax_plot.set_yscale('log', basey=10)
ax_plot.set_xlabel('Observed occupancy', fontsize=12)
ax_plot.set_ylabel('Predicted occupancy', fontsize=12)


ax_plot_singletons.set_xscale('log', basex=10)
ax_plot_singletons.set_yscale('log', basey=10)
ax_plot_singletons.set_xlabel('Observed occupancy', fontsize=12)
ax_plot_singletons.set_ylabel('Predicted occupancy', fontsize=12)


ax_plot.set_title("DADA2 without pooling\n(no singletons)", fontsize=12, fontweight='bold' )


ax_plot_singletons.set_title("DADA2 with pooling\n(singletons)", fontsize=12, fontweight='bold' )


fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/occupancy_singletons.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
