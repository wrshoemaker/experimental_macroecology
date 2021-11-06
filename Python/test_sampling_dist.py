from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils



def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #return array[idx]
    return idx



def sample_from_prob_non_zero_k(x):

    # get parent mean relative abundances
    mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
    #mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

    species_in_descendants = []

    for migration_innoculum in utils.migration_innocula:

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1])
        rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
        means, variances, species_to_keep = utils.get_species_means_and_variances(rel_s_by_s, ESVs, zeros=True)
        species_in_descendants.extend(ESVs)


    species_in_descendants = list(set(species_in_descendants))

    mean_rel_abundances_parent_present = [mean_rel_abundances_parent[s_idx] for (s_idx,s) in enumerate(species_parent) if s in species_in_descendants]

    min_obs = min(mean_rel_abundances_parent_present)
    max_obs = max(mean_rel_abundances_parent_present)

    range = np.linspace(min_obs, max_obs, num=1000, endpoint=True)
    cdf = np.asarray([sum(mean_rel_abundances_parent_present<=i)/len(mean_rel_abundances_parent_present) for i in range])


    idx = find_nearest_idx(range, x)

    return cdf[idx]



print(sample_from_prob_non_zero_k(0.03))

print(sample_from_prob_non_zero_k(0.1))
