from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import scipy.special as special

from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils


#m = 0.01
#sigma =

sigma = 0.3
m = 0.01
tau = 0.3
K = 0.5

print((2/sigma)-1)
print((-2/sigma)+1)

print(special.kv((2/sigma)-1, (4/sigma)*np.sqrt(tau*m/K)))
print(special.kv((-2/sigma)+1, (4/sigma)*np.sqrt(tau*m/K)))




def stationary_prob(x):

    c = 2*((tau*m*K)**(0.5 * ((2/sigma)-1))) * special.kv((2/sigma)-1, (4/sigma)*np.sqrt(tau*m/K))

    return (c**-1) * np.exp(-2/sigma*x * (tau*m + ((x**2)/K))) * (x**((2/sigma)-2))



print(stationary_prob(0.03))
