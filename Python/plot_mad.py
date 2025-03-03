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
import plot_utils

from scipy.optimize import fsolve
from scipy.special import erf

transfers = [12,18]

def plot_mad():

    fig = plt.figure(figsize = (16.5, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

        for transfer_idx, transfer in enumerate(transfers):

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            mad = np.mean(rel_s_by_s, axis=1)
            mad_log10 = np.log(mad)

            print(migration_innoculum, transfer, min(mad_log10))

            hist_mad, bin_edges_mad = np.histogram(mad_log10, density=True, bins=10)
            bins_mean_mad = [0.5 * (bin_edges_mad[i] + bin_edges_mad[i+1]) for i in range(0, len(bin_edges_mad)-1 )]
            prob_mad = [sum( (mad_log10>=bin_edges_mad[i]) & (mad_log10<bin_edges_mad[i+1])  ) / len(mad_log10) for i in range(0, len(bin_edges_mad)-1 )]
            
            bins_mean_mad = np.asarray(bins_mean_mad)
            prob_mad = np.asarray(prob_mad)
            to_keep_idx = prob_mad > 0
            
            bins_mean_mad = bins_mean_mad[to_keep_idx]
            prob_mad = prob_mad[to_keep_idx]

            ax = plt.subplot2grid((2, 4), (transfer_idx, migration_innoculum_idx), colspan=1)

            color_ = utils.color_dict_range[migration_innoculum][transfer-3]
            color_ = color_.reshape(1,-1)

            ax.scatter(bins_mean_mad, prob_mad, alpha=0.8, c=color_)#, label=label)

            #print()
            #print()
            #c=min(mad)
            #c = (1/max(s_by_s.sum(axis=0)))
            #print(c, min(mad))
            c = min(mad)*10
            mu_to_plot, sigma_to_plot = utils.Klogn(mad, c)
            x_mean_range = np.logspace(np.log10(min(mad))-5, 0, num=100)
            x_mean_range_log = np.log(x_mean_range)
            lognorm_pdf___ = utils.get_lognorma_mad_prediction(x_mean_range_log, mu_to_plot, sigma_to_plot, c)
            ax.plot(x_mean_range_log, lognorm_pdf___, c='k', lw=2.5, linestyle='-', zorder=2, label= 'Lognormal')

            if migration_innoculum_idx + transfer_idx == 0:
                ax.legend(loc='lower left')

            ax.set_xlim([-27, 0])

            ax.set_yscale('log', basey=10)
            label = utils.titles_abbreviated_dict[migration_innoculum] + ', transfer ' + str(transfer)
            label = label + '\n# ASVs = ' + str(len(mad)) 
            ax.set_title(label, fontsize=12)

            ax.set_xlabel('Mean relative abundance, ' + r'$\mathrm{log}_{10}$', fontsize=11)
            ax.set_ylabel('Probability density', fontsize=11)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/mad.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def make_cdf(data, range_):


    data = data[np.isfinite(data)]
    cdf_array = [sum(data<=i)/len(data) for i in range_]
    cdf_array = np.asarray(cdf_array)

    return cdf_array




def plot_cdf():

    fig, ax = plt.subplots(figsize=(4,4))

    for migration_innoculum_idx, migration_innoculum in enumerate(utils.migration_innocula):

        for transfer_idx, transfer in enumerate(transfers):

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            mad = np.mean(rel_s_by_s, axis=1)
            mad_log10 = np.log10(mad)
            rescaled_mad_log10 = (mad_log10 - np.mean(mad_log10))/np.std(mad_log10)

            range_ = np.linspace(min(rescaled_mad_log10), max(rescaled_mad_log10), num=100)

            cdf = make_cdf(rescaled_mad_log10, range_)
            #mad_log10 = np.log(mad)

            color_ = utils.color_dict_range[migration_innoculum][transfer-3]
            color_ = color_.reshape(1,-1)[0]
            ax.plot(range_, cdf, c=color_, lw=2, ls='-')


    #ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Resaled log10 mean abundance', fontsize=11)
    ax.set_ylabel('Cumulative probability density', fontsize=11)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/mad_cdf.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_example_mad():

    migration_innoculum = ('Parent_migration', 4)
    transfer = 12
    
    s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration=migration_innoculum[0], inocula=migration_innoculum[1], transfer=transfer)
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

    mad = np.mean(rel_s_by_s, axis=1)
    mad_log10 = np.log(mad)

    hist_mad, bin_edges_mad = np.histogram(mad_log10, density=True, bins=10)
    bins_mean_mad = [0.5 * (bin_edges_mad[i] + bin_edges_mad[i+1]) for i in range(0, len(bin_edges_mad)-1 )]
    prob_mad = [sum( (mad_log10>=bin_edges_mad[i]) & (mad_log10<bin_edges_mad[i+1])  ) / len(mad_log10) for i in range(0, len(bin_edges_mad)-1 )]
    
    bins_mean_mad = np.asarray(bins_mean_mad)
    prob_mad = np.asarray(prob_mad)
    to_keep_idx = prob_mad > 0
    
    bins_mean_mad = bins_mean_mad[to_keep_idx]
    prob_mad = prob_mad[to_keep_idx]

    fig, ax = plt.subplots(figsize=(4,4))

    color_ = utils.color_dict_range[('Parent_migration', 4)][12-3]
    color_ = color_.reshape(1,-1)

    ax.scatter(bins_mean_mad, prob_mad, alpha=0.8, c=color_)#, label=label)

    c = min(mad)*10
    mu_to_plot, sigma_to_plot = utils.Klogn(mad, c)
    x_mean_range = np.logspace(np.log(min(mad))-5, 0, num=100)
    x_mean_range_log = np.log(x_mean_range)
    lognorm_pdf___ = utils.get_lognorma_mad_prediction(x_mean_range_log, mu_to_plot, sigma_to_plot, c)
    ax.plot(x_mean_range_log, lognorm_pdf___, c='k', lw=2.5, linestyle='-', zorder=2, label= 'Lognormal')

    ax.legend(loc='lower left')

    ax.set_xlim([-27, 0])
    ax.set_ylim([0.0001, 1])
    ax.set_yscale('log', basey=10)
    label =  'Regional migration, transfer ' + str(12)
    label = label + '\n# ASVs = ' + str(len(mad)) 
    ax.set_title(label, fontsize=12)

    ax.set_xlabel('Log-transformed mean relative abundance', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)


    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/mad_example.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





#plot_cdf()

#plot_mad()

plot_example_mad()
