from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils
from matplotlib import cm


afd_dict = {}
transfers = [12, 18]

experiments = [('No_migration',4), ('Parent_migration', 4)  ]

#experiments_transfers = [[('No_migration',4), 12 ], [('No_migration',4), 18 ], [('Parent_migration',4), 12 ], [('Parent_migration',4), 18 ], [('No_migration',40), 18], [('Global_migration',4), ] ]

fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)

for experiment_idx, experiment in enumerate(experiments):

    afd_dict[experiment] = {}

    for transfer_idx, transfer in enumerate(transfers):

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

        means, variances = utils.get_species_means_and_variances(relative_s_by_s)
        attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))


        ax = plt.subplot2grid((2, 2), (experiment_idx, transfer_idx), colspan=1)

        ax.set_title("%s, transfer %d" % (utils.titles_no_inocula_dict[experiment], transfer ), fontsize=12, fontweight='bold' )
        ax.scatter(means, variances, c='k', alpha=0.8, label='Merged attractors')

        x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        #y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

        ax.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2)#, label="OLS regression")
        #ax.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)

        ax.set_xlabel('Average relative\nabundance', fontsize=12)
        ax.set_ylabel('Variance of relative abundance', fontsize=10)

        #ax.text(0.3,0.9, r'$\sigma^{2}_{x} = {{{}}} * x^{{{}}}$'.format(str(round(10**intercept, 3)),  str(round(slope, 3)) ), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes  )
        ax.text(0.3,0.9, r'$\sigma^{{2}}_{{x}} = {{{}}} * \left \langle x \right \rangle^{{{}}}$'.format(str(round(10**intercept, 3)),  str(round(slope, 3)) ), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes  )



        for attractor_idx, attractor in enumerate(attractor_dict.keys()):

            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]

            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            relative_s_by_s_attractor = relative_s_by_s_attractor[~np.all(relative_s_by_s_attractor == 0, axis=1)]

            means_attractor, variances_attractor = utils.get_species_means_and_variances(relative_s_by_s_attractor)
            slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))

            x_log10_range_attractor =  np.linspace(min(np.log10(means_attractor)) , max(np.log10(means_attractor)) , 10000)
            y_log10_fit_range_attractor = 10 ** (slope_attractor*x_log10_range_attractor + intercept_attractor)

            ax.scatter(means_attractor, variances_attractor, alpha=1, edgecolors='k', color=utils.family_colors[attractor], label=attractor)#, c='#87CEEB')
            ax.plot(10**x_log10_range_attractor, y_log10_fit_range_attractor, c=utils.family_colors[attractor], lw=2.5, linestyle='--', zorder=2)

            #ax.text(0.2,0.9-(0.1*(attractor_idx+1)), r'$y \sim x^{{{}}}$'.format(str( round(slope_attractor, 3) )), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
            #ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$y \sim {{{}}} * x^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
            ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$\sigma^{{2}}_{{x}} = {{{}}} * \left \langle x \right \rangle^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )


        ax.legend(loc="lower right", fontsize=8)



#wspace=0.3, hspace=0.3
fig_name = utils.directory + '/figs/taylors_law_attractor.pdf'
fig.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
