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

#iter = 200
dt = 7
treatment = 'global'
tau = 0.3
sigma = 0.7

n, k, t = slm_simulation_utils.run_simulation_initial_condition_migration(migration_treatment=treatment, tau=tau, sigma=sigma, dt=dt)


mean_error_all = []
mean_cv_all = []
for i in range(10):

    t = t/dt
    #x_final = x[-1,:,:]
    #for t_i_idx, t_i in enumerate(t):
    n_t_i = n[int(t[-1]),:,:]
    x_t_i_rel = (n_t_i.T/n_t_i.sum(axis=1)).T

    sad_sample_all = []
    for s in x_t_i_rel:
        sad_sample_all.append(np.random.multinomial(utils.n_reads, s))

    x_t_i_sample = np.concatenate(sad_sample_all).reshape(x_t_i_rel.shape)
    # remove absent species
    x_t_i_sample = x_t_i_sample[:,~np.all(x_t_i_sample == 0, axis=0)]
    x_t_i_sample_rel = (x_t_i_sample.T/x_t_i_sample.sum(axis=1)).T


    species = list(range(x_t_i_sample_rel.shape[1]))
    mean_rel_abundances, var_rel_abundances, species_to_keep = utils.get_species_means_and_variances(x_t_i_sample_rel.T, species, min_observations=3, zeros=False)

    if len(mean_rel_abundances) < 5:
        continue

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))

    occupancies_no_zeros, predicted_occupancies_no_zeros, mad_no_zeros, beta_no_zeros, species_no_zeros = utils.predict_occupancy(x_t_i_sample.T, range(x_t_i_sample.shape[1]))
    mean_error = np.mean(np.absolute(occupancies_no_zeros - predicted_occupancies_no_zeros) / occupancies_no_zeros)
    mean_log_error = np.mean(np.log10(np.absolute(occupancies_no_zeros - predicted_occupancies_no_zeros) / occupancies_no_zeros))

    mean_error_all.append(mean_log_error)


    cv_abundances = np.sqrt(var_rel_abundances)/mean_rel_abundances

    #print(np.mean(cv_abundances))


#print(np.mean(mean_log_error))
