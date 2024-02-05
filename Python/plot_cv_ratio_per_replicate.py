from __future__ import division
import os, sys, re
import numpy as np
import pickle
import random

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections
import plot_utils


np.random.seed(123456789)
random.seed(123456789)

n_iter = 1000

count_dict = utils.get_otu_dict()


samples = list(count_dict.keys())

samples_dict = {}

treatments = ['no_migration', 'global_migration']

for treatment_idx, treatment in enumerate(['No_migration.4.T%s', 'Global_migration.4.T%s']):

    treatment_transfer = treatment % str(9)

    samples_dict[treatments[treatment_idx]] = [x.split('.')[-1] for x in samples if treatment_transfer in x]



# relative abundances of all species
abundnace_dict = {}
for treatment in treatments:

    replicates = samples_dict[treatment]

    for r in replicates:

        for t in range(8, 19):

            treatment_replicate_t = 'No_migration.4.T%s.%s' % (str(t), r)

            if treatment_replicate_t not in count_dict:
                continue

            treatment_replicate_t_count_dict = count_dict[treatment_replicate_t]

            n_reads = sum(treatment_replicate_t_count_dict.values())

            for asv, count in treatment_replicate_t_count_dict.items():

                if asv not in abundnace_dict:
                    abundnace_dict[asv] = {}

                if treatment not in abundnace_dict[asv]:
                    abundnace_dict[asv][treatment] = {}
                    
                if r not in abundnace_dict[asv][treatment]:
                    abundnace_dict[asv][treatment][r] = {}

                abundnace_dict[asv][treatment][r][t] = count/n_reads
                #print(asv, counts/n_reads)

            

def calculate_f_statistic_cv(x_1, x_2):

    '''Forkman (2009), "Estimator and Tests for Common Coefficients of Variation in Normal Distributions",
    Communications in Statistics - Theory and Methods, Vol. 38, pp. 233-251.'''

    cv_x_1 = np.std(x_1) / np.absolute(np.mean(x_1))
    cv_x_2 = np.std(x_2) / np.absolute(np.mean(x_2))

    n_1 = len(x_1)
    n_2 = len(x_2)

    F = ((cv_x_1**2)/(1 + (cv_x_1**2)*(n_1-1)/n_1))/((cv_x_2**2)/(1 + (cv_x_2**2)*(n_2-1)/n_2))

    return cv_x_1, cv_x_2, F



def unpaired_t_test(x_1, x_2):

    # ddof = 1, unbiased estimator
    var_1 = np.var(x_1, ddof=1)
    var_2 = np.var(x_2, ddof=1)

    mean_1 = np.mean(x_1)
    mean_2 = np.mean(x_2)

    s = np.sqrt((var_1/len(x_1)) +(var_2/len(x_2)))

    t_observed = (mean_1 - mean_2)/s

    return t_observed



def run_permutation_unpaired_t_test(x_1, x_2, n_iter=10000):

    t_observed = unpaired_t_test(x_1, x_2)

    x_merged = np.concatenate((x_1, x_2))
    
    t_null_all = []
    for i in range(n_iter):

        np.random.shuffle(x_merged)

        x_1_null = x_merged[:len(x_1)]
        x_2_null = x_merged[len(x_1):]
        t_null = unpaired_t_test(x_1_null, x_2_null)
        t_null_all.append(t_null)

    t_null_all = np.asarray(t_null_all)
    p_value = sum(t_null_all>t_observed)/n_iter

    return t_observed, p_value




# calculate CV of log-ratio
asv_all = abundnace_dict.keys()
cv_dict = {}
for asv in asv_all:

    for treatment in abundnace_dict[asv].keys():

        for replicate in abundnace_dict[asv][treatment].keys():

            if len(abundnace_dict[asv][treatment][replicate]) < 11:
                continue

            x_trajectory = np.asarray([abundnace_dict[asv][treatment][replicate][t] for t in range(8, 19)])
            log_ratio_x_trajectory = np.log10(x_trajectory[1:]/x_trajectory[:-1])
            log_ratio_x_trajectory_before = log_ratio_x_trajectory[:5]
            log_ratio_x_trajectory_after = log_ratio_x_trajectory[5:]

            cv_after, cv_before, F = calculate_f_statistic_cv(log_ratio_x_trajectory_after, log_ratio_x_trajectory_before)

            if asv not in cv_dict:
                cv_dict[asv] = {}

            if treatment not in cv_dict[asv]:
                cv_dict[asv][treatment] = {}


            # get null F
            F_null_all = []
            for i in range(n_iter):
                x_trajectory_null = np.random.permutation(x_trajectory)
                log_ratio_x_trajectory_null = np.log10(x_trajectory_null[1:]/x_trajectory_null[:-1])
                log_ratio_x_trajectory_before_null = log_ratio_x_trajectory_null[:5]
                log_ratio_x_trajectory_after_null = log_ratio_x_trajectory_null[5:]

                cv_after_null, cv_before_null, F_null = calculate_f_statistic_cv(log_ratio_x_trajectory_after_null, log_ratio_x_trajectory_before_null)
                F_null_all.append(F_null)

            F_null_all = np.asarray(F_null_all)
            p_value = sum(F_null_all>F)/n_iter

            cv_dict[asv][treatment][replicate] = {}
            cv_dict[asv][treatment][replicate]['cv_log_ratio_before'] = cv_before
            cv_dict[asv][treatment][replicate]['cv_log_ratio_after'] = cv_after
            cv_dict[asv][treatment][replicate]['F_cv'] = F
            cv_dict[asv][treatment][replicate]['F_cv_null_all'] = F_null_all.tolist()
            cv_dict[asv][treatment][replicate]['p_value'] = p_value

            




taxonomy_dict = utils.make_taxonomy_dict()


print('All replicates')


for asv in cv_dict.keys():

    if len(cv_dict[asv]) < 2:
        continue

    if (len(cv_dict[asv]['no_migration']) < 3) or len(cv_dict[asv]['global_migration']) < 3:
        continue
    
    f_cv_global_migration = np.asarray([cv_dict[asv]['global_migration'][r]['F_cv'] for r in cv_dict[asv]['global_migration'].keys()])
    f_cv_no_migration = np.asarray([cv_dict[asv]['no_migration'][r]['F_cv'] for r in cv_dict[asv]['no_migration'].keys()])

    t_observed, p_value = run_permutation_unpaired_t_test(f_cv_global_migration, f_cv_no_migration)

    # generate null distribution of t-statistics
    asv_cv_dict = cv_dict[asv]
    reps_no_migration = asv_cv_dict['no_migration'].keys()
    reps_global_migration = asv_cv_dict['global_migration'].keys()
    t_null_all = []
    for i in range(n_iter):

        

        f_cv_no_migration_null = np.asarray([asv_cv_dict['no_migration'][r]['F_cv_null_all'][i] for r in reps_no_migration])
        f_cv_global_migration_null = np.asarray([asv_cv_dict['global_migration'][r]['F_cv_null_all'][i] for r in reps_global_migration])

        t_null = unpaired_t_test(f_cv_global_migration_null, f_cv_no_migration_null)
        t_null_all.append(t_null)

    t_null_all = np.asarray(t_null_all)
    p_value_t = sum(t_null_all>t_observed)/n_iter

    print(asv, t_observed, p_value_t)




# repeat test, but constrain on attractor status
# only look at Alcaligenaceae, since all global migration populations are in this attractor state
attractor_status_no_migration = utils.get_attractor_status('No_migration', inocula=4)
attractor_status_global_migration = utils.get_attractor_status('Global_migration', inocula=4)

print('Constrain analysis on attractor status')

for asv in cv_dict.keys():

    if len(cv_dict[asv]) < 2:
        continue

    if (len(cv_dict[asv]['no_migration']) < 3) or len(cv_dict[asv]['global_migration']) < 3:
        continue


    f_cv_global_migration = np.asarray([cv_dict[asv]['global_migration'][r]['F_cv'] for r in cv_dict[asv]['global_migration'].keys()])
    no_migration_reps = [r[1:] for r in cv_dict[asv]['no_migration'].keys()]
    
    reps_to_keep = list(set(no_migration_reps) & set(attractor_status_no_migration['Alcaligenaceae']))
    reps_to_keep = ['R'+r for r in reps_to_keep]
    f_cv_no_migration = np.asarray([cv_dict[asv]['no_migration'][r]['F_cv'] for r in reps_to_keep])
    
    t_observed, p_value = run_permutation_unpaired_t_test(f_cv_global_migration, f_cv_no_migration)

    # generate null distribution of t-statistics
    asv_cv_dict = cv_dict[asv]
    reps_global_migration = asv_cv_dict['global_migration'].keys()
    t_null_all = []
    for i in range(n_iter):

        f_cv_no_migration_null = np.asarray([asv_cv_dict['no_migration'][r]['F_cv_null_all'][i] for r in reps_to_keep])
        f_cv_global_migration_null = np.asarray([asv_cv_dict['global_migration'][r]['F_cv_null_all'][i] for r in reps_global_migration])

        t_null = unpaired_t_test(f_cv_global_migration_null, f_cv_no_migration_null)
        t_null_all.append(t_null)

    t_null_all = np.asarray(t_null_all)
    p_value_t = sum(t_null_all>t_observed)/n_iter
    

    print(asv, t_observed, p_value_t)







# get observed and null F to plot

f_no_migration_all = []
f_no_migration_null_all = []
f_global_migration_all = []
f_global_migration_null_all = []

for k1,v1 in cv_dict.items(): # the basic way
    for k2,v2 in v1.items():

        for k3,v3 in v2.items():

            if k2 == 'no_migration':
                f_no_migration_all.append(v3['F_cv'])
                f_no_migration_null_all.extend(v3['F_cv_null_all'])
        
            else:
                f_global_migration_all.append(v3['F_cv'])
                f_global_migration_null_all.extend(v3['F_cv_null_all'])



#cv_dict[asv][treatment][replicate]['F_cv_null_all'] = F_null_all.tolist()


# plot

fig, ax = plt.subplots(figsize=(5,4))

fig = plt.figure(figsize = (8.5, 4))
fig.subplots_adjust(bottom= 0.15)


ax_no = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_global = plt.subplot2grid((1, 2), (0, 1), colspan=1)


ax_no.text(-0.1, 1.04, plot_utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_no.transAxes)
ax_global.text(-0.1, 1.04, plot_utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_global.transAxes)

ax_no.hist(f_no_migration_all, lw=3, alpha=0.8, bins=10, color=utils.color_dict[('No_migration',4)], histtype='stepfilled', label="Observed",  density=True, zorder=2)
ax_no.hist(f_no_migration_null_all, lw=3, alpha=0.8, bins=200, color='grey', histtype='stepfilled', label='Permutation-based null', density=True, zorder=1)
#ax_no.hist(f_no_migration_null_all, lw=3, alpha=0.8, bins=200, color='grey', histtype='stepfilled', label='Predicted', density=True, zorder=1)

# survival distribution
#x_range = np.logspace(0, max(), num=100, endpoint=True, base=10.0)

#survival_error = [sum(error_no_nan >= i)/len(error_no_nan) for i in x_range]
#survival_slm_error = [sum(error_slm_no_nan >= i)/len(error_slm_no_nan) for i in x_range]



ax_no.set_xlabel('Change in CV between\ntransfers 12 and 18, ' + r'$F_{\mathrm{CV}}$', fontsize=12)
ax_no.set_ylabel('Probability density', fontsize=12)
ax_no.legend(loc="upper right", fontsize=8)
ax_no.set_title(utils.titles_no_inocula_dict[('No_migration',4)], fontsize=14)
ax_no.set_xlim([0, max(f_no_migration_all)+1])


ax_global.hist(f_global_migration_all, lw=3, alpha=0.8, bins= 10, color=utils.color_dict[('Global_migration',4)], histtype='stepfilled', label="Observed",  density=True, zorder=2)
ax_global.hist(f_global_migration_null_all, lw=3, alpha=0.8, bins=200, color='grey', histtype='stepfilled', label='Null', density=True, zorder=1)
ax_global.set_xlabel('Change in CV between\ntransfers 12 and 18, ' + r'$F_{\mathrm{CV}}$', fontsize=12)
ax_global.set_ylabel('Probability density', fontsize=12)
ax_global.legend(loc="upper right", fontsize=8)
ax_global.set_title(utils.titles_no_inocula_dict[('Global_migration',4)], fontsize=14)
ax_global.set_xlim([0, max(f_global_migration_all)+1])



fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/cv_f_hist.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + '/figs/cv_f_hist.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)

plt.close()


