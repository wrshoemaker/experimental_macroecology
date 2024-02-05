from __future__ import division
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from scipy.stats import gamma

import utils
import collections

# number is max number of reads
#n_range = np.logspace(1, np.log10(218869), num=50, endpoint=True, base=10.0)
#median coverage = 24874
n_range = np.linspace(1, 24874, num=50, endpoint=True)

iter_ = 100
intermediate_filename = utils.directory + "/data/subsample_richness.dat"
treatments = ['Parent_migration.4.T18', 'No_migration.4.T18', 'No_migration.40.T18', 'Global_migration.4.T18', 'Parent_migration.NA.T0']

color_dict = {'Parent_migration.4.T18': utils.color_dict_range[('Parent_migration', 4)][13],
                'No_migration.4.T18': utils.color_dict_range[('No_migration',4)][13],
                'No_migration.40.T18': utils.color_dict_range[('No_migration',40)][13],
                'Global_migration.4.T18': utils.color_dict_range[('Global_migration',4)][13],
                'Parent_migration.NA.T0': 'k'}


#def chao_richness(subsampled_sad, n):
#    f_1 = subsampled_sad.count(1)
#    f_2 = subsampled_sad.count(2)

#    (f_1/n) * ( ((n-1)*f_1) /  )

#    (n-1)*f_1 + 2*f_2



def make_subsample_richness_dict():

    count_dict = utils.get_otu_dict()

    rarefaction_dict = {}
    for sample, sad_dict in count_dict.items():
        sad = list(sad_dict.values())
        sad = np.asarray(sad)
        abundance = sum(sad)

        richness_subsample = []
        for n in n_range:
            n = int(n)
            if n > abundance:
                continue
            richness_subsample_n = [len(utils.subsample_sad(sad, replace=False, n_subsample = n)) for i in range(iter_)]
            richness_subsample.append(np.mean(richness_subsample_n))

        rarefaction_dict[sample] = richness_subsample


    sys.stderr.write("Saving allelic dict...\n")
    with open(intermediate_filename, 'wb') as handle:
        pickle.dump(rarefaction_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_subsample_richness_dict():

    with open(intermediate_filename, 'rb') as handle:
        b = pickle.load(handle)
    return b




def plot_rarefaction():

    rarefaction_dict = load_subsample_richness_dict()

    fig, ax = plt.subplots(figsize=(4,4))

    for treatment in treatments:

        ax.plot([-100000, -1000], [-100000,-1000], lw=1.5, c=color_dict[treatment], label=utils.label_dict[treatment])


    for sample, rarefied_richness in rarefaction_dict.items():

        sample_split = sample.rsplit('.',1)[0]

        if sample_split == 'Parent_migration.NA.T0':
            alpha=0.7
        else:
            alpha=0.05

        richness = rarefaction_dict[sample]
        n_range_sample = n_range[:len(richness)]

        ax.plot(n_range_sample, richness, lw=1.5, alpha=alpha, c=color_dict[sample_split])


    ax.set_title('Transfer 18', fontsize=13)

    ax.set_xlim(-500.5, 26000)
    ax.set_ylim(0.8, 1600)
    #ax.axhline(1533, lw=1.5, ls=':',color='k', zorder=1)
    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Number of reads', fontsize=12)
    ax.set_ylabel('Subsampled ASV richness w/out replacement', fontsize=10)

    ax.legend(loc="lower right", fontsize=8)

    fig.savefig(utils.directory + '/figs/rarefied_richness.png', format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    fig.savefig(utils.directory + '/figs/rarefied_richness.eps', format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)

    plt.close()




plot_rarefaction()

#estimate_mean_abundances_parent()
