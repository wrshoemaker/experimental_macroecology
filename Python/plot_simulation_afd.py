from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import pickle

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import slm_simulation_utils
import plot_utils




simulation_dict = slm_simulation_utils.load_simulation_all_migration_dict()

tau_all = np.asarray(list(simulation_dict.keys()))
sigma_all = np.asarray(list(simulation_dict[tau_all[0]].keys()))


for treatment_idx, treatment in enumerate(['no_migration', 'global_migration', 'parent_migration']):

    for tau in tau_all:

        tau_t_test_slope = []
        tau_t_test_slope_error = []

        tau_t_test_intercept = []
        tau_t_test_intercept_error = []

        for sigma in sigma_all:

            ks_12_vs_18 = simulation_dict[tau][sigma]['ks_12_vs_18'][treatment]
            ks_rescaled_12_vs_18 = simulation_dict[tau][sigma]['ks_rescaled_12_vs_18'][treatment]

            print(treatment, tau, sigma, np.mean(ks_12_vs_18), np.mean(ks_rescaled_12_vs_18))


    # treatment label 
    #(treatment.capitalize(), 4)

