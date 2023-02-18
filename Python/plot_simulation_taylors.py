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

print(simulation_dict.keys())

for tau in tau_all:

    for sigma in sigma_all:

        t_slope_global_12 = np.asarray(simulation_dict[tau][sigma][11]['global_migration']['t_slope'])
        t_slope_global_18 = np.asarray(simulation_dict[tau][sigma][17]['global_migration']['t_slope'])

        t_slope_parent_12 = np.asarray(simulation_dict[tau][sigma][11]['parent_migration']['t_slope'])
        t_slope_parent_18 = np.asarray(simulation_dict[tau][sigma][17]['parent_migration']['t_slope'])

        print(np.mean(t_slope_global_18 - t_slope_global_12), np.mean(t_slope_parent_18 - t_slope_parent_12))