from __future__ import division
import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.colors as clr

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

import slm_simulation_utils

iter = 200
dt = 7

simulation_dict_path = "%s/data/simulation_dict.pickle" %  utils.directory

def make_simulation_dict():

    simulation_dict = {}
    #for treatment in ['parent']:
    for treatment in ['none', 'global', 'parent']:

        simulation_dict[treatment] = {}

        #count = 0

        #for i in range(iter):
        #while count < iter:
        for i in range(iter):

            if i%100 == 0:
                print(treatment, i)

            x, k, t = slm_simulation_utils.run_simulation(migration_treatment=treatment, dt = dt)
            t = t/dt
            #x_final = x[-1,:,:]

            for t_i_idx, t_i in enumerate(t):
                x_t_i = x[t_i_idx,:,:]
                x_t_i_rel = (x_t_i.T/x_t_i.sum(axis=1)).T

                sad_sample_all = []
                for s in x_t_i_rel:
                    sad_sample_all.append(np.random.multinomial(utils.n_reads, s))

                x_t_i_sample = np.concatenate(sad_sample_all).reshape(x_t_i_rel.shape)
                # remove absent species
                x_t_i_sample = x_t_i_sample[:,~np.all(x_t_i_sample == 0, axis=0)]
                x_t_i_sample_rel = (x_t_i_sample.T/x_t_i_sample.sum(axis=1)).T

                #print(x_t_i_sample_rel[0,:])


                species = list(range(x_t_i_sample_rel.shape[1]))
                mean_rel_abundances, var_rel_abundances, species_to_keep = utils.get_species_means_and_variances(x_t_i_sample_rel.T, species, min_observations=3, zeros=False)
                #print(mean_rel_abundances, var_rel_abundances)

                if len(mean_rel_abundances) < 5:
                    continue

                #print(mean_rel_abundances, var_rel_abundances)

                #print(mean_rel_abundances, var_rel_abundances)

                slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))

                if slope < 0:
                    continue

                if t_i not in simulation_dict[treatment]:
                    simulation_dict[treatment][t_i] = {}
                    simulation_dict[treatment][t_i]['slope'] = []
                    simulation_dict[treatment][t_i]['gamma_mre'] = []

                # gamma slm
                occupancies_no_zeros, predicted_occupancies_no_zeros, mad_no_zeros, beta_no_zeros, species_no_zeros = utils.predict_occupancy(x_t_i_sample.T, range(x_t_i_sample.shape[1]))
                mean_error = np.mean(np.absolute(occupancies_no_zeros - predicted_occupancies_no_zeros) / occupancies_no_zeros)

                simulation_dict[treatment][t_i]['slope'].append(slope)
                simulation_dict[treatment][t_i]['gamma_mre'].append(mean_error)

                #print(slope)

                #count += 1


    with open(simulation_dict_path, 'wb') as outfile:
        pickle.dump(simulation_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_simulation_dict():

    dict_ = pickle.load(open(simulation_dict_path, "rb"))
    return dict_



#make_simulation_dict()

simulation_dict = load_simulation_dict()

#fig, ax = plt.subplots(figsize=(4,4))

color_range =  np.linspace(0.0, 1.0, 18)
rgb_blue = cm.get_cmap('Blues')( color_range )
rgb_red = cm.get_cmap('Reds')( color_range )
rgb_green = cm.get_cmap('Greens')( color_range )
rgb_orange = cm.get_cmap('Oranges')( color_range )
titles_no_inocula_dict = {'none': 'No migration', 'global': 'Global migration', 'parent': 'Parent migration'}
color_dict = {'none':rgb_blue[12], 'global':rgb_red[12], 'parent':rgb_green[12]}


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.15,  wspace=0.25)

ax_gamma = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_taylors = plt.subplot2grid((1, 2), (0, 1), colspan=1)

ax_all = [ax_gamma, ax_taylors]

measures  = ['gamma_mre', 'slope']

for m_idx, m in enumerate(measures):

    ax_i = ax_all[m_idx]

    for treatment in ['none', 'global', 'parent']:

        t = list(simulation_dict[treatment].keys())
        t.sort()
        mean_slope = []
        ci_upper = []
        ci_lower = []
        for t_i in t:
            slope_t_i = np.asarray(simulation_dict[treatment][t_i][m])
            slope_t_i = np.sort(slope_t_i)
            ci_upper.append(slope_t_i[int(len(slope_t_i)*0.975)])
            ci_lower.append(slope_t_i[int(len(slope_t_i)*0.025)])
            mean_slope.append(np.mean(slope_t_i))

        ax_i.plot(t, mean_slope, ls='-', label=titles_no_inocula_dict[treatment], c=color_dict[treatment])





#x_log10_range =  np.linspace(min(np.log10(means_all)) , max(np.log10(means_all)) , 10000)
#y_log10_fit_range = slope*x_log10_range + intercept
#ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")


#ax_taylors.set_yscale('log', basey=10)
#ax_taylors.set_yscale('log', basey=10)

ax_taylors.set_xlabel('Transfer', fontsize=12)
ax_taylors.set_ylabel("Taylor's Law slope", fontsize=12)

ax_gamma.set_xlabel('Transfer', fontsize=12)
ax_gamma.set_ylabel("Mean relative error of gamma occupacny", fontsize=12)


ax_gamma.axhline(0.35, lw=1.5, ls=':',color=color_dict['none'], zorder=1)
ax_gamma.axhline(0.19, lw=1.5, ls=':',color=color_dict['global'], zorder=1)
ax_gamma.axhline(0.44, lw=1.5, ls=':',color=color_dict['parent'], zorder=1)
ax_gamma.axvline(12, lw=1.5, ls='--',color='k', label='End of migration', zorder=1)


ax_taylors.axhline(1.735, lw=1.5, ls=':',color=color_dict['none'], zorder=1)
ax_taylors.axhline(1.427, lw=1.5, ls=':',color=color_dict['global'], zorder=1)
ax_taylors.axhline(1.545, lw=1.5, ls=':',color=color_dict['parent'], zorder=1)


ax_gamma.legend(loc="upper left", fontsize=8)



fig_name = utils.directory + '/figs/slm_taylors_law.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
