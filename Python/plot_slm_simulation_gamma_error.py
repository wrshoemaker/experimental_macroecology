from __future__ import division
import os, sys
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

iter=100

# for SLM, sigma = 2*(beta**-1) / (1+(beta**-1))

#sigma_range = np.linspace()

test_dict = {}

migration_treatments = ['global', 'parent', 'none']
for migration_treatment in migration_treatments:

    print(migration_treatment)

    mean_error_all = []

    for i in range(iter):

        n, k_to_keep, t = slm_simulation_utils.run_simulation(migration_treatment = migration_treatment)
        #n_final = n[-1,0,:]
        #print(n_final)
        n_final = n[-1,:,:]
        x_final = (n_final.T / np.sum(n_final, axis=1)).T

        # sample reletive abundances
        n_reads = np.zeros(n_final.shape)
        for x_final_i_idx in range(x_final.shape[0]):
            n_reads[x_final_i_idx,:] = np.random.multinomial(int(10**4.5), x_final[x_final_i_idx,:])

        occupancies_no_zeros, predicted_occupancies_no_zeros, mad_no_zeros, beta_no_zeros, species_no_zeros = utils.predict_occupancy(n_reads.T, range(len(k_to_keep)))
        mean_error = np.mean(np.absolute(occupancies_no_zeros - predicted_occupancies_no_zeros) / occupancies_no_zeros)

        mean_error_all.append(mean_error)

    #test_dict[]
    mean_error_all = np.asarray(mean_error_all)
    mean_mean_error = np.mean(mean_error_all)
    mean_error_all = np.sort(mean_error_all)

    lower_ci_mean_error = mean_error_all[int((iter*0.025))]
    upper_ci_mean_error = mean_error_all[int((iter*0.975))]

    print(mean_mean_error, lower_ci_mean_error, upper_ci_mean_error)



#x_final_flat = x_final.flatten()
#x_final_flat = x_final_flat[x_final_flat>0]
#x_final_log10 = np.log10(x_final_flat)
#x_final_log10_mean = np.mean(x_final_log10)
#x_final_log10_std = np.std(x_final_log10)



#fig, ax = plt.subplots(figsize=(4,4))


#for x_final_i in x_final.T:
#    x_final_i_log10 = np.log10(x_final_i[x_final_i>0])
#    if len(x_final_i_log10) < 30:
#        continue
#    x_final_i_log10_rescaled = (x_final_i_log10 - x_final_log10_mean)/x_final_log10_std
#    print(x_final_i_log10_rescaled)
#    hist, bin_edges = np.histogram(x_final_i_log10_rescaled, density=True, bins=10)
#    #bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]
#    bins_mean = [bin_edges[i+1] for i in range(0, len(bin_edges)-1 )]
#    ax.plot(bins_mean, hist, alpha=0.5, ls='-', c='k')#, label=label_)


#ax.scatter(t, x, alpha=0.7, s=4)

#fig.subplots_adjust(hspace=0.35,wspace=0.3)
#fig_name = "%s/figs/test_gamma_simulation.png" % utils.directory
#fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
#plt.close()
