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

from scipy.optimize import fsolve
from scipy.special import erf


remove_zeros = True

c = 10**-6
transfers = [12,18]


fig = plt.figure(figsize = (8, 8)) #
fig.subplots_adjust(bottom= 0.15)


# get mean and std for rescaling

migration_innocula_nested_list = [utils.migration_innocula[:2], utils.migration_innocula[2:]]

for row_idx, row_list in enumerate(migration_innocula_nested_list):

    for column_idx, migration_innoculum in enumerate(row_list):

        ax = plt.subplot2grid((2, 2), (row_idx, column_idx), colspan=1)

        for transfer in transfers:

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            afd = rel_s_by_s.flatten()
            afd_log10 = np.log(afd[afd>0])
            afd_log10_rescaled = (afd_log10 - np.mean(afd_log10)) / np.std(afd_log10)

            color_ = utils.color_dict_range[migration_innoculum][transfer-3]
            color_ = color_.reshape(1,-1)
            label_ = utils.titles_dict[migration_innoculum] + ', transfer ' + str(transfer)
            hist, bin_edges = np.histogram(afd_log10_rescaled, density=True, bins=10)
            #bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )]
            bins_mean = [bin_edges[i+1] for i in range(0, len(bin_edges)-1 )]
            ax_afd.scatter(bins_mean, hist, alpha=0.8, c=color_)#, label=label_)
