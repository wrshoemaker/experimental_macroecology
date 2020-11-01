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

def old_figure():

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)


    for experiment_idx, experiment in enumerate(experiments):

        afd_dict[experiment] = {}

        for transfer_idx, transfer in enumerate(transfers):

            relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

            means, variances = utils.get_species_means_and_variances(relative_s_by_s)
            attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

            mse = sum((np.log10(variances) - (intercept + slope*np.log10(means)))**2)
            s_xx, s_yy, s_xy = utils.get_pair_stats(np.log10(means), np.log10(variances))


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

                mse_attractor = sum((np.log10(variances_attractor) - (intercept_attractor + slope_attractor*np.log10(means_attractor)))**2)
                s_xx_attractor, s_yy_attractor, s_xy_attractor = utils.get_pair_stats(np.log10(means_attractor), np.log10(variances_attractor))

                df = (len(means) + len(means_attractor) - 4)

                # compare attractor to merged
                pooled_s2_y_x = (mse + mse_attractor) / df
                s2_intercept = pooled_s2_y_x * ( (1/len(means)) + (1/len(means_attractor))  +  (np.mean(np.log10(means_attractor))**2) /s_xx_attractor  +   (np.mean(np.log10(means))**2) /s_xx  )
                t_intercept = (intercept - intercept_attractor) / s2_intercept
                p_value_intercept = stats.t.sf(np.abs(t_intercept), df)*2


                s2_y_x_slope = (mse + mse_attractor)/df
                std_error_slope_difference = np.sqrt((s2_y_x_slope/s_xx_attractor) + (s2_y_x_slope/s_xx))
                t_slope =  (slope - slope_attractor) / std_error_slope_difference
                p_value_slope = stats.t.sf(np.abs(t_slope), df)*2

                x_log10_range_attractor =  np.linspace(min(np.log10(means_attractor)) , max(np.log10(means_attractor)) , 10000)
                y_log10_fit_range_attractor = 10 ** (slope_attractor*x_log10_range_attractor + intercept_attractor)

                ax.scatter(means_attractor, variances_attractor, alpha=1, edgecolors='k', color=utils.family_colors[attractor], label=attractor)#, c='#87CEEB')
                ax.plot(10**x_log10_range_attractor, y_log10_fit_range_attractor, c=utils.family_colors[attractor], lw=2.5, linestyle='--', zorder=2)

                #ax.text(0.2,0.9-(0.1*(attractor_idx+1)), r'$y \sim x^{{{}}}$'.format(str( round(slope_attractor, 3) )), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
                #ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$y \sim {{{}}} * x^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
                ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$\sigma^{{2}}_{{x}} = {{{}}} * \left \langle x \right \rangle^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )


            ax.legend(loc="lower right", fontsize=8)





fig = plt.figure(figsize = (12, 8)) #
fig.subplots_adjust(bottom= 0.15)

t_test_dict = {}

for experiment_idx, experiment in enumerate(experiments):

    t_test_dict[experiment] = {}

    for transfer_idx, transfer in enumerate(transfers):

        t_test_dict[experiment][transfer] = {}

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

        means, variances = utils.get_species_means_and_variances(relative_s_by_s)
        attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

        mse = sum((np.log10(variances) - (intercept + slope*np.log10(means)))**2)
        s_xx, s_yy, s_xy = utils.get_pair_stats(np.log10(means), np.log10(variances))


        ax = plt.subplot2grid((2, 3), (experiment_idx, transfer_idx), colspan=1)

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


            t_test_dict[experiment][transfer][attractor] = {}

            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]

            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            relative_s_by_s_attractor = relative_s_by_s_attractor[~np.all(relative_s_by_s_attractor == 0, axis=1)]

            means_attractor, variances_attractor = utils.get_species_means_and_variances(relative_s_by_s_attractor)
            slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))

            mse_attractor = sum((np.log10(variances_attractor) - (intercept_attractor + slope_attractor*np.log10(means_attractor)))**2)
            s_xx_attractor, s_yy_attractor, s_xy_attractor = utils.get_pair_stats(np.log10(means_attractor), np.log10(variances_attractor))

            df = (len(means) + len(means_attractor) - 4)
            critical_value = stats.t.ppf(0.95, df)

            pooled_s2_y_x = (mse + mse_attractor) / df

            std_error_intercept_difference =np.sqrt( pooled_s2_y_x * ( (1/len(means)) + (1/len(means_attractor))  + ( (np.mean(np.log10(means_attractor))**2) /s_xx_attractor)  +   (np.mean(np.log10(means))**2) /s_xx  ) )

            # compare attractor to merged

            t_intercept = (intercept - intercept_attractor) / std_error_intercept_difference
            p_value_intercept = stats.t.sf(np.abs(t_intercept), df)*2

            s2_y_x_slope = (mse + mse_attractor)/df

            std_error_slope_difference = np.sqrt((s2_y_x_slope/s_xx_attractor) + (s2_y_x_slope/s_xx))
            t_slope =  (slope - slope_attractor) / std_error_slope_difference

            p_value_slope = stats.t.sf(np.abs(t_slope), df)*2

            x_log10_range_attractor =  np.linspace(min(np.log10(means_attractor)) , max(np.log10(means_attractor)) , 10000)
            y_log10_fit_range_attractor = 10 ** (slope_attractor*x_log10_range_attractor + intercept_attractor)

            ax.scatter(means_attractor, variances_attractor, alpha=1, edgecolors='k', color=utils.family_colors[attractor], label=attractor)#, c='#87CEEB')
            ax.plot(10**x_log10_range_attractor, y_log10_fit_range_attractor, c=utils.family_colors[attractor], lw=2.5, linestyle='--', zorder=2)

            #ax.text(0.2,0.9-(0.1*(attractor_idx+1)), r'$y \sim x^{{{}}}$'.format(str( round(slope_attractor, 3) )), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
            #ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$y \sim {{{}}} * x^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )
            ax.text(0.3,0.9-(0.1*(attractor_idx+1)), r'$\sigma^{{2}}_{{x}} = {{{}}} * \left \langle x \right \rangle^{{{}}}$'.format(str(round(10**intercept_attractor, 3)),  str(round(slope_attractor, 3)) ), fontsize=11, color=utils.family_colors[attractor], ha='center', va='center', transform=ax.transAxes  )

            # margin of error = critical value * standard error
            # 95% CI for this sample is 0.55

            t_test_dict[experiment][transfer][attractor]['t_intercept'] = t_intercept
            t_test_dict[experiment][transfer][attractor]['p_value_intercept'] = p_value_intercept
            t_test_dict[experiment][transfer][attractor]['CI_intercept'] = std_error_intercept_difference * critical_value

            t_test_dict[experiment][transfer][attractor]['t_slope'] = t_slope
            t_test_dict[experiment][transfer][attractor]['p_value_slope'] = p_value_slope
            t_test_dict[experiment][transfer][attractor]['CI_slope'] = std_error_slope_difference * critical_value


        ax.legend(loc="lower right", fontsize=8)






#ax_slope = plt.subplot2grid((2, 3), (0, 2), colspan=1)
#ax_slope.set_xlabel('Difference in slope between\nattractor and merged attractors', fontsize=14)



#ax_intercept = plt.subplot2grid((2, 3), (1, 2), colspan=1)
# for major ticks
#ax_intercept.set_yticks([])
# for minor ticks
#ax_intercept.set_yticks([], minor=True)
#ax_intercept.set_xlabel('Difference in intercept between\nattractor and merged attractors', fontsize=14)








#wspace=0.3, hspace=0.3
fig_name = utils.directory + '/figs/taylors_law_attractor.pdf'
fig.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
