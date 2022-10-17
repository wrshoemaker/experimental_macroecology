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

simulation_dict_path = "%s/data/simulation_global_rho.pickle" %  utils.directory


def load_simulation_dict():

    dict_ = pickle.load(open(simulation_dict_path, "rb"))
    return dict_



simulation_dict = load_simulation_dict()

tau_all = list(simulation_dict.keys())
tau_all.sort()

sigma_all = list(simulation_dict[tau_all[0]].keys())
sigma_all.sort()

for tau_i in tau_all:
    for sigma_i in sigma_all:


        #global_migration_delta = simulation_dict[tau][sigma]['ratio_stats']['global_migration']['mean_delta_cv_log_ratio']
        #no_migration_delta = simulation_dict[tau][sigma]['ratio_stats']['no_migration']['mean_delta_cv_log_ratio']

        global_migration_mean_cv_log_ratio = simulation_dict[tau_i][sigma_i]['ratio_stats']['global_migration']['mean_cv_log_ratio']
        global_migration_transfer = simulation_dict[tau_i][sigma_i]['ratio_stats']['global_migration']['transfer']

        no_migration_mean_cv_log_ratio = simulation_dict[tau_i][sigma_i]['ratio_stats']['no_migration']['mean_cv_log_ratio']
        no_migration_transfer = simulation_dict[tau_i][sigma_i]['ratio_stats']['no_migration']['transfer']

        global_migration_mean_cv_log_ratio = np.asarray(global_migration_mean_cv_log_ratio)
        global_migration_transfer = np.asarray(global_migration_transfer)

        no_migration_mean_cv_log_ratio = np.asarray(no_migration_mean_cv_log_ratio)
        no_migration_transfer = np.asarray(no_migration_transfer)

        # get mean over iterations
        transfer_set = np.asarray(list(set(global_migration_transfer)))

        global_migration_mean_cv_log_ratio_mean = np.asarray([np.mean(global_migration_mean_cv_log_ratio[global_migration_transfer==t]) for t in transfer_set])
        no_migration_mean_cv_log_ratio_mean = np.asarray([np.mean(no_migration_mean_cv_log_ratio[no_migration_transfer==t]) for t in transfer_set])

        print(tau_i, sigma_i)

        delta_mean_cv_log_ratio_mean = global_migration_mean_cv_log_ratio_mean/no_migration_mean_cv_log_ratio_mean

        print(np.mean(np.log10(delta_mean_cv_log_ratio_mean[(transfer_set>5) & (transfer_set<12)])))
