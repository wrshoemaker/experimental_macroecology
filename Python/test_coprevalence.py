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




transfers = [12,18]




def predict_cooccupancy(s_by_s, species, totreads=np.asarray([])):

    # get squared inverse cv
    # assume that entries are read counts.
    rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

    beta_all = []
    mean_all = []

    for s in rel_s_by_s_np:

        var = np.var(s)
        mean = np.mean(s)

        beta = (mean**2)/var

        mean_all.append(mean)
        beta_all.append(beta)

    beta_all = np.asarray(beta_all)
    mean_all = np.asarray(mean_all)


    s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]
    covariance_observed = np.cov(s_by_s_presence_absence)
    #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
    # calcualte total reads if no argument is passed
    # sloppy quick fix
    if len(totreads) == 0:
        totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(np.mean(afd_no_zeros/ totreads[afd>0]))
        #tf.append(np.mean(afd_no_zeros/s_by_s.sum(axis=0)[afd>0]))

    #tf = np.mean(rel_abundances)
    tf = np.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        N_reads = s_by_s.sum(axis=0)[np.nonzero(afd)[0]]
        #N_reads = s_by_s.sum(axis=0)[afd>0]

        tvpf_list.append(np.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    #tvpf = np.mean(tvpf_list)
    tvpf = np.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    predicted_occupancies = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    for beta_i, theta_i in zip(beta,theta):
        predicted_occupancies.append(1 -  ((1+theta_i*totreads)**(-1*beta_i )) )

    predicted_occupancies = np.asarray(predicted_occupancies)
    mean_predicted_occupancies = np.mean(predicted_occupancies, axis=1)

    n_species = len(species)
    observed_covariance_coprevalence = []
    predicted_covariance_coprevalence = []
    for i in range(n_species):

        for j in range(i):

            prevalence_i = predicted_occupancies[i,:]
            prevalence_j = predicted_occupancies[j,:]

            coprevalence_ij = np.mean(prevalence_i*prevalence_j)

            if np.isnan(coprevalence_ij) == True:
                continue

            covariance_coprevalence_ij = coprevalence_ij - mean_predicted_occupancies[i]*mean_predicted_occupancies[j]
            predicted_covariance_coprevalence.append(covariance_coprevalence_ij)


            observed_covariance_coprevalence.append(covariance_observed[i,j])


    observed_covariance_coprevalence = np.asarray(observed_covariance_coprevalence)
    predicted_covariance_coprevalence = np.asarray(predicted_covariance_coprevalence)

    return observed_covariance_coprevalence, predicted_covariance_coprevalence





fig, ax_occupancy = plt.subplots(figsize=(4,4))

for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):
    for transfer in transfers:

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        color_ = utils.color_dict_range[migration_innoculum][transfer-3]
        color_ = color_.reshape(1,-1)

        # occupancy
        observed_covariance_coprevalence, predicted_covariance_coprevalence = predict_cooccupancy(s_by_s, ESVs)
        print(np.corrcoef(observed_covariance_coprevalence, predicted_covariance_coprevalence)[0,1] )

        observed_covariance_coprevalence = observed_covariance_coprevalence**2
        predicted_covariance_coprevalence = predicted_covariance_coprevalence**2

        to_keep_idx = (predicted_covariance_coprevalence > 1e-6) & (observed_covariance_coprevalence>1e-6)
        observed_covariance_coprevalence = observed_covariance_coprevalence[to_keep_idx]
        predicted_covariance_coprevalence = predicted_covariance_coprevalence[to_keep_idx]

        ax_occupancy.scatter(observed_covariance_coprevalence, predicted_covariance_coprevalence, alpha=0.5, c=color_, s=18, zorder=2)#, linewidth=0.8, edgecolors='k')

ax_occupancy.set_xscale('log', basex=10)
ax_occupancy.set_yscale('log', basey=10) 


fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/cooccupancy.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()

        