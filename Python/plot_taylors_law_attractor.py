from __future__ import division
import os, sys
import random
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



fig = plt.figure(figsize = (12, 8)) #
fig.subplots_adjust(bottom= 0.15)

t_test_dict = {}


for experiment_idx, experiment in enumerate(experiments):

    t_test_dict[experiment] = {}

    for transfer_idx, transfer in enumerate(transfers):

        t_test_dict[experiment][transfer] = {}

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])

        means, variances, species_to_keep = utils.get_species_means_and_variances(relative_s_by_s, species)
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
            attractor_species_idx = [~np.all(relative_s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            relative_s_by_s_attractor = relative_s_by_s_attractor[attractor_species_idx]


            means_attractor, variances_attractor, attractor_species_to_keep = utils.get_species_means_and_variances(relative_s_by_s_attractor, attractor_species)
            slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))

            mse_attractor = sum((np.log10(variances_attractor) - (intercept_attractor + slope_attractor*np.log10(means_attractor)))**2)
            s_xx_attractor, s_yy_attractor, s_xy_attractor = utils.get_pair_stats(np.log10(means_attractor), np.log10(variances_attractor))

            df = (len(means) + len(means_attractor) - 4)
            critical_value = stats.t.ppf(0.95, df)

            pooled_s2_y_x = (mse + mse_attractor) / df

            std_error_intercept_difference =np.sqrt( pooled_s2_y_x * ( (1/len(means)) + (1/len(means_attractor))  + ( (np.mean(np.log10(means_attractor))**2) /s_xx_attractor)  +   (np.mean(np.log10(means))**2) /s_xx  ) )

            # compare attractor to merged

            t_intercept = (intercept_attractor - intercept ) / std_error_intercept_difference
            p_value_intercept = stats.t.sf(np.abs(t_intercept), df)*2

            s2_y_x_slope = (mse + mse_attractor)/df

            std_error_slope_difference = np.sqrt((s2_y_x_slope/s_xx_attractor) + (s2_y_x_slope/s_xx))
            t_slope =  (slope_attractor - slope) / std_error_slope_difference


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




# perform null test





null_test_dict = {}
for experiment_idx, experiment in enumerate(experiments):

    null_test_dict[experiment] = {}

    for transfer_idx, transfer in enumerate(transfers):

        null_test_dict[experiment][transfer] = {}

        relative_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=experiment[0],inocula=experiment[1])


        means, variances, species_to_keep = utils.get_species_means_and_variances(relative_s_by_s, species)
        attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])


        for attractor_idx, attractor in enumerate(attractor_dict.keys()):

            null_test_dict[experiment][transfer][attractor] = {}


            attractor_idxs = [comm_rep_list.index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]
            relative_s_by_s_attractor = relative_s_by_s[:, attractor_idxs]
            attractor_species_idx = [~np.all(relative_s_by_s_attractor == 0, axis=1)][0]
            attractor_species = np.asarray(species)[attractor_species_idx]
            #attractor_species = attractor_species.tolist()
            relative_s_by_s_attractor = relative_s_by_s_attractor[attractor_species_idx]


            means_attractor, variances_attractor, attractor_species_to_keep = utils.get_species_means_and_variances(relative_s_by_s_attractor, attractor_species)
            #slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))

            species_idx = [np.where(species_to_keep == attractor_species_to_keep_i)[0][0] for attractor_species_to_keep_i in attractor_species_to_keep]
            species_idx = np.asarray(species_idx)
            means_subset = means[species_idx]
            variances_subset = variances[species_idx]

            df = (len(means_subset) + len(means_attractor) - 4)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means_subset), np.log10(variances_subset))
            slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))

            mse = sum((np.log10(variances) - (intercept + slope*np.log10(means)))**2)
            s_xx, s_yy, s_xy = utils.get_pair_stats(np.log10(means_subset), np.log10(variances_subset))

            mse_attractor = sum((np.log10(variances_attractor) - (intercept_attractor + slope_attractor*np.log10(means_attractor)))**2)
            s_xx_attractor, s_yy_attractor, s_xy_attractor = utils.get_pair_stats(np.log10(means_attractor), np.log10(variances_attractor))

            pooled_s2_y_x = (mse + mse_attractor) / df

            std_error_intercept_difference =np.sqrt( pooled_s2_y_x * ( (1/len(means)) + (1/len(means_attractor))  + ( (np.mean(np.log10(means_attractor))**2) /s_xx_attractor)  +   (np.mean(np.log10(means))**2) /s_xx  ) )

            # compare attractor to merged
            t_intercept = (intercept_attractor - intercept ) / std_error_intercept_difference
            s2_y_x_slope = (mse + mse_attractor)/df

            std_error_slope_difference = np.sqrt((s2_y_x_slope/s_xx_attractor) + (s2_y_x_slope/s_xx))
            t_slope =  (slope_attractor - slope) / std_error_slope_difference


            t_slope_null = []
            t_intercept_null = []
            n_permutations = 10000
            for j in range(n_permutations):
                means_permuted = []
                variances_permuted = []
                means_attractor_permuted = []
                variances_attractor_permuted = []
                # go through each species
                for s in range(len(means_subset)):

                    random_number = random.randrange(2)

                    if random_number == 0:
                        means_permuted.append(means_subset[s])
                        variances_permuted.append(variances_subset[s])
                        means_attractor_permuted.append(means_attractor[s])
                        variances_attractor_permuted.append(variances_attractor[s])

                    else:
                        means_permuted.append(means_attractor[s])
                        variances_permuted.append(variances_attractor[s])
                        means_attractor_permuted.append(means_subset[s])
                        variances_attractor_permuted.append(variances_subset[s])

                means_permuted = np.asarray(means_permuted)
                variances_permuted = np.asarray(variances_permuted)
                means_attractor_permuted = np.asarray(means_attractor_permuted)
                variances_attractor_permuted = np.asarray(variances_attractor_permuted)

                slope_permuted, intercept_permuted, r_value_permuted, p_value_permuted, std_err_permuted = stats.linregress(np.log10(means_permuted), np.log10(variances_permuted))
                slope_attractor_permuted, intercept_attractor_permuted, r_value_attractor_permuted, p_value_attractor_permuted, std_err_attractor_permuted = stats.linregress(np.log10(means_attractor_permuted), np.log10(variances_attractor_permuted))

                mse_permuted = sum((np.log10(variances_permuted) - (intercept_permuted + slope_permuted*np.log10(means_permuted)))**2)
                s_xx_permuted, s_yy_permuted, s_xy_permuted = utils.get_pair_stats(np.log10(means_permuted), np.log10(variances_permuted))

                mse_attractor_permuted = sum((np.log10(variances_attractor_permuted) - (intercept_attractor_permuted + slope_attractor_permuted*np.log10(means_attractor_permuted)))**2)
                s_xx_attractor_permuted, s_yy_attractor_permuted, s_xy_attractor_permuted = utils.get_pair_stats(np.log10(means_attractor_permuted), np.log10(variances_attractor_permuted))


                pooled_s2_y_x_permuted = (mse_permuted + mse_attractor_permuted) / df

                std_error_intercept_difference_permuted =np.sqrt( pooled_s2_y_x_permuted * ( (1/len(means_permuted)) + (1/len(means_attractor_permuted))  + ( (np.mean(np.log10(means_attractor_permuted))**2) /s_xx_attractor_permuted)  +   (np.mean(np.log10(means_permuted))**2) /s_xx_permuted  ) )

                # compare attractor to merged
                t_intercept_permuted = (intercept_attractor_permuted - intercept_permuted ) / std_error_intercept_difference_permuted
                s2_y_x_slope_permuted = (mse_permuted + mse_attractor_permuted)/df

                std_error_slope_difference_permuted = np.sqrt((s2_y_x_slope_permuted/s_xx_attractor_permuted) + (s2_y_x_slope_permuted/s_xx_permuted))
                t_slope_permuted =  (slope_attractor_permuted - slope_permuted) / std_error_slope_difference_permuted

                t_slope_null.append(t_slope_permuted)
                t_intercept_null.append(t_intercept_permuted)

            t_slope_null = np.asarray(t_slope_null)
            t_intercept_null = np.asarray(t_intercept_null)



            p_value_slope = len(t_slope_null[t_slope_null < t_slope]) / n_permutations
            p_value_intercept = len(t_intercept_null[t_intercept_null< t_intercept]) / n_permutations

            null_test_dict[experiment][transfer][attractor]['p_value_slope'] = p_value_slope
            null_test_dict[experiment][transfer][attractor]['p_value_intercept'] = p_value_intercept








ax_slope = plt.subplot2grid((2, 3), (0, 2), colspan=1)
ax_intercept = plt.subplot2grid((2, 3), (1, 2), colspan=1)



count = 0

y_axis_labels = []
y_axis_positions = []

for key in t_test_dict:

    for transfer in t_test_dict[key]:

        y_axis_labels.append('Transfer %d'%transfer)
        y_axis_positions.append(count+1.2)

        #for attractor in  t_test_dict[key][transfer]:
        for attractor in  [ 'Alcaligenaceae', 'Pseudomonadaceae']:

            dict_i = t_test_dict[key][transfer][attractor]

            ax_slope.scatter(dict_i['t_slope'], count, s = 120, alpha=1, linewidth=2, edgecolors='k', color=utils.family_colors[attractor], label=attractor,  zorder=3)#, c='#87CEEB')
            ax_intercept.scatter(dict_i['t_intercept'], count, s = 120, alpha=1, linewidth=2, edgecolors='k', color=utils.family_colors[attractor], label=attractor,  zorder=3)#, c='#87CEEB')

            null_test_dict[experiment][transfer][attractor]['p_value_slope']


            ax_slope.errorbar(dict_i['t_slope'], count, xerr=dict_i['CI_slope'],linestyle='-', marker='o', c='k', elinewidth=2, alpha=1, zorder=2)
            ax_intercept.errorbar(dict_i['t_intercept'], count, xerr=dict_i['CI_intercept'],linestyle='-', marker='o', c='k', elinewidth=2, alpha=1, zorder=2)

            p_value_slope = null_test_dict[experiment][transfer][attractor]['p_value_slope']
            p_value_intercept = null_test_dict[experiment][transfer][attractor]['p_value_intercept']

            #if p_value_slope < 0.05:

            #    ax_slope.text(dict_i['t_slope'], count+0.2, '*', fontsize=12, color='k', fontweight='bold', ha='center', va='center')


            #if p_value_intercept < 0.05:


            #    ax_intercept.text(dict_i['t_intercept'], count+0.2, '*', fontsize=12, color='k', fontweight='bold', ha='center', va='center' )


            count += 1





ax_slope.set_yticks(y_axis_positions)
ax_slope.set_yticklabels(y_axis_labels, rotation=90, fontsize=5.5, fontweight='bold', ha='center')
ax_slope.tick_params(axis=u'y',length=0)


ax_intercept.set_yticks(y_axis_positions)
ax_intercept.set_yticklabels(y_axis_labels, rotation=90, fontsize=5.5, fontweight='bold', ha='center')
ax_intercept.tick_params(axis=u'y',length=0)



ax_slope.set_xlim([-2.1, 0.1 ])
ax_intercept.set_xlim([-3.1, 0.15  ])

ax_slope.axvline(0, lw=3, ls=':',color='k', zorder=1)
ax_intercept.axvline(0, lw=3, ls=':',color='k', zorder=1)


ax_slope.set_xlabel('Standardized slope difference\nbetween individual and merged attractors', fontsize=12)
ax_intercept.set_xlabel('Standardized intercept difference\nbetween individual and merged attractors', fontsize=12)



#ax_slope.text(-0.12,0.25, 'No migration\nlow inoculum', fontsize=10, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )
#ax_slope.text(-0.12,0.25, 'No migration\nlow inoculum', fontsize=10, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )

ax_slope.text(-0.07,0.25, 'No migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )
ax_slope.text(-0.07,0.75, 'Parent migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_slope.transAxes )


ax_intercept.text(-0.07,0.25, 'No migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_intercept.transAxes )
ax_intercept.text(-0.07,0.75, 'Parent migration', fontsize=9, rotation=90, color='k', fontweight='bold', ha='center', va='center', transform=ax_intercept.transAxes )









#wspace=0.3, hspace=0.3
fig_name = utils.directory + '/figs/taylors_law_attractor.pdf'
fig.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
