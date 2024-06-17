from __future__ import division
import os
import sys
import itertools
import random


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import numpy
#from macroecotools import obs_pred_rsquare
import utils




#migration_innocula = [('No_migration',4), ('Parent_migration',4), ('Global_migration',4)]
#for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

#    for trasfer_idx, transfer in enumerate(utils.transfers):

#        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration_innoculum[0], inocula=migration_innoculum[1])


count_dict = utils.get_otu_dict()



inocula_asvs = set(count_dict['Parent_migration.NA.T0.R1'].keys()) | set(count_dict['Parent_migration.NA.T0.R2'].keys()) | set(count_dict['Parent_migration.NA.T0.R3'].keys())


fraction_asv_absent_inocula_all = []
for key, value in count_dict.items():

    if 'Parent_migration.NA.T0' in key:
        continue

    asv = set(value.keys())
    n_asv_absent_inocula = len(asv - inocula_asvs)
    fraction_asv_absent_inocula = n_asv_absent_inocula/len(asv)

    fraction_asv_absent_inocula_all.append(fraction_asv_absent_inocula)
    

print(numpy.mean(fraction_asv_absent_inocula_all))


#print(len(inocula_asvs), len(set(count_dict['Parent_migration.NA.T0.R1'].keys())))