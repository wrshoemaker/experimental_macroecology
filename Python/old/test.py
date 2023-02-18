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




#print(np.mean(mean_log_error))
