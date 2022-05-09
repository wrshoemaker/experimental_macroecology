from __future__ import division
import os, sys, re, math

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import sympy as sp

import bisect
import random
from collections import Counter
from scipy import stats, signal, optimize, special



np.seterr(divide='ignore', invalid='ignore')


# parameters for simulations
n_reads = 10**4.4


family_colors = {'Alcaligenaceae':'darkorange', 'Comamonadaceae': 'darkred',
                'Enterobacteriaceae':'dodgerblue', 'Enterococcaceae':'limegreen',
                'Lachnospiraceae':'deepskyblue', 'Pseudomonadaceae':'darkviolet'}


color_range =  np.linspace(0.0, 1.0, 18)
rgb_blue = cm.get_cmap('Blues')( color_range )
rgb_red = cm.get_cmap('Reds')( color_range )
rgb_green = cm.get_cmap('Greens')( color_range )
rgb_orange = cm.get_cmap('Oranges')( color_range )


rgb_alcaligenaceae = cm.get_cmap('Oranges')( color_range )
rgb_pseudomonadaceae = cm.get_cmap('Purples')( color_range )


slope_null=2

def get_color_attractor(attractor, transfer):

    if attractor == 'Alcaligenaceae':
        return rgb_alcaligenaceae[transfer-4]

    elif attractor == 'Pseudomonadaceae':
        return rgb_pseudomonadaceae[transfer-4]

    elif attractor == 'All':
        return 'k'

    else:
        print("Make a color map for this attractor!")




color_dict_range = {('No_migration',4):rgb_blue, ('Global_migration',4):rgb_red, ('Parent_migration', 4):rgb_green, ('No_migration',40): rgb_orange}


color_dict = {('No_migration',4):rgb_blue[12], ('Global_migration',4):rgb_red[12], ('Parent_migration', 4):rgb_green[12], ('No_migration',40):rgb_orange[12] }


attractor_latex_dict = {'Alcaligenaceae': r'$Alcaligenaceae$', 'Pseudomonadaceae': r'$Pseudomonadaceae$'}


label_dict = {'Parent_migration.4.T18': 'Parent migration, low inoculum',
                'No_migration.4.T18': 'No migration, low inoculum',
                'No_migration.40.T18': 'No migration, high inoculum',
                'Global_migration.4.T18': 'Global migration, low inoculum',
                'Parent_migration.NA.T0': 'Parent community'}





#from macroeco_distributions import pln, pln_solver, pln_ll
#from macroecotools import obs_pred_rsquare

directory = os.path.expanduser("~/GitHub/experimental_macroecology")
metadata_path = directory + '/data/metadata.csv'
otu_path = directory + '/data/otu_table.csv'


carbons = ['Glucose', 'Citrate', 'Leucine']
carbons_colors = ['royalblue', 'forestgreen', 'darkred']
carbons_shapes = ['o', 'D', 'X']
attractors = [ 'Alcaligenaceae', 'Pseudomonadaceae']

titles = ['No migration, low inoculum', 'No migration, high inoculum', 'Global migration, low inoculum', 'Parent migration, low inoculum' ]


titles_new_line = ['No migration\nlow inoculum', 'No migration\nhigh inoculum', 'Global migration\nlow inoculum', 'Parent migration\nlow inoculum' ]


titles_new_line_dict = {('No_migration',4):'No migration\nlow inoculum', ('No_migration',40):'No migration\nhigh inoculum', ('Global_migration',4):'Global migration\nlow inoculum', ('Parent_migration',4):'Parent migration\nlow inoculum'}


migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]

titles_dict = {('No_migration',4): 'No migration, low inoculum',
                ('No_migration',40): 'No migration, high inoculum',
                ('Global_migration',4): 'Global migration, low inoculum',
                ('Parent_migration',4): 'Parent migration, low inoculum',
                ('Glucose', np.nan): 'Glucose' }


titles_no_inocula_dict = {('No_migration',4): 'No migration',
                ('Global_migration',4): 'Global migration',
                ('Parent_migration',4): 'Parent migration',
                ('Glucose', np.nan): 'Glucose' }



def subsample_sad(sad, replace=True, n_subsample = 10):

    s = len(sad)

    relative_sad = sad / sum(sad)

    if replace == True:

        # sampling WITH replacement
        # write function to sample SAD without replacement
        individuals_subsample = np.random.choice(range(s), size=n_subsample, replace=True, p=relative_sad)

        species_subsample, sad_subsample = np.unique(individuals_subsample, return_counts=True)


    else:

        all_individuals = []
        for idx, i in enumerate(range(len(sad))):
            all_individuals.extend( [i]* sad[idx] )

        subsample_individuals = random.sample(all_individuals, n_subsample)

        species_counts = Counter(subsample_individuals)

        sad_subsample = list(species_counts.values())

        sad_subsample.sort(reverse=True)

        sad_subsample = np.asarray(sad_subsample)

    return sad_subsample






def bootstrap_estimate_ks(array_1, array_2, size=50, n=10000):

    ks_statistic_bs = []

    ks_statistic, p_value = stats.ks_2samp(array_1, array_2)


    for i in range(n):

        array_1_bs = random.choice(array_1, size=size, replace=True)
        array_2_bs = random.choice(array_2, size=size, replace=True)

        ks_statistic_bs_i, p_value_bs_i = stats.ks_2samp(array_1_bs, array_2_bs)

        ks_statistic_bs.append(ks_statistic_bs_i)

    ks_statistic_bs = np.asarray(ks_statistic_bs)
    ks_statistic_bs = np.sort(ks_statistic_bs)

    ks_ci_025 = ks_statistic_bs[int(n*0.025)]
    ks_ci_975 = ks_statistic_bs[int(n*0.975)]

    return ks_statistic, ks_ci_025, ks_ci_975







def run_permutational_ks_test(array_1, array_2, n=10000):

    ks_statistic, p_value = stats.ks_2samp(array_1, array_2)

    array_merged = np.concatenate([array_1, array_2])

    ks_null = []

    for n_i in range(n):

        np.random.shuffle(array_merged)

        array_1_null = array_merged[:len(array_1)]
        array_2_null = array_merged[len(array_1):]

        ks_statistic_null, p_value_null = stats.ks_2samp(array_1_null, array_2_null)

        ks_null.append(ks_statistic_null)

    ks_null = np.asarray(ks_null)

    p_value = sum(ks_null>ks_statistic) / n

    return ks_statistic, p_value






def weakly_damped_oscillator_(ts, x0_, gamma, omega_0, phi):

    return x0_ * np.exp(-1*ts*gamma) * np.cos(omega_0*ts + phi)



def fit_weakly_damped_oscillator(ts,xs):
    ts = np.asarray(ts)
    # shift to zero
    ts = ts - ts[0]
    x0 = xs[0]

    gamma_init = 0.8
    omega_0_init = 10
    phi_init = 0.3


    def weakly_damped_oscillator(ts, gamma, omega_0, phi):

        return x0 * np.exp(-1*ts*gamma) * np.cos(omega_0*ts + phi)


    xmin = optimize.fmin(lambda x: np.square(xs-weakly_damped_oscillator(ts, x[0],x[1], x[2])).sum(), np.array([gamma_init, omega_0_init, phi_init]))
    gamma = xmin[0]
    omega_0 = xmin[1]
    phi = xmin[2]

    return gamma, omega_0, phi





def max_difference_between_timepoints(x_min, x_max):
    #assume log range
    # returns log-spaced linear values
    x_range = np.logspace(x_min, x_max, num=1000, base=10.0)

    y_range = [max([1-x, x]) for x in x_range]
    y_range = np.asarray(y_range)

    return x_range, y_range


def max_and_min_width_between_timepoints(x_min, x_max):

    x_range = np.logspace(x_min, x_max, num=1000, base=10.0)

    #y_range = [1/x for x in x_range]
    y_max_range = 1/ x_range
    y_min_range = 1/ x_range
    #y_range = np.asarray(y_range)

    return x_range, y_min_range , y_max_range



def calculate_noise_color():

    # Fourier transform
    frq, f = signal.periodogram(data[c])



def get_null_correlations(s_by_s, species, comm_rep_list, n_iter=10000):

    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

    corr_dict = {}

    rel_s_by_s_permuted = rel_s_by_s[np.arange(len(rel_s_by_s))[:,None], np.random.randn(*rel_s_by_s.shape).argsort(axis=1)]

    for species_i_idx, species_i in enumerate(species):

        for species_j_idx, species_j in enumerate(species):

            if species_j_idx >= species_i_idx:
                continue

            afd_i = rel_s_by_s[species_i_idx,:]
            afd_j = rel_s_by_s[species_j_idx,:]

            c_ij = np.corrcoef(afd_i, afd_j)[0][1]

            corr_dict[(species_i, species_j)] = {}
            corr_dict[(species_i, species_j)]['correlation_obs'] = c_ij
            corr_dict[(species_i, species_j)]['correlation_null'] = []



    for k in range(n_iter):

        if k %1000 == 0:
            print(k)

        rel_s_by_s_permuted = rel_s_by_s[np.arange(len(rel_s_by_s))[:,None], np.random.randn(*rel_s_by_s.shape).argsort(axis=1)]

        for species_i_idx, species_i in enumerate(species):

            for species_j_idx, species_j in enumerate(species):

                if species_j_idx >= species_i_idx:
                    continue

                afd_i = rel_s_by_s_permuted[species_i_idx,:]
                afd_j = rel_s_by_s_permuted[species_j_idx,:]

                c_ij = np.corrcoef(afd_i, afd_j)[0][1]

                corr_dict[(species_i, species_j)]['correlation_null'].append(c_ij)


    for species_i_idx, species_i in enumerate(species):

        for species_j_idx, species_j in enumerate(species):

            if species_j_idx >= species_i_idx:
                continue

            corr_dict[(species_i, species_j)]['correlation_null_mean'] = np.mean(corr_dict[(species_i, species_j)]['correlation_null'])



    return corr_dict





def get_p_value(p_value, alpha=0.05):

    if p_value >= alpha:

        return r'$P \, \nless 0.05$'

    else:
        return r'$P < 0.05$'


def get_pair_stats(x, y):

    #calculate means
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    #get number of entries
    n = len(x)

    #calculate sums
    x_sum = np.sum(x)
    x_sum_square = np.sum([xi**2 for xi in x])
    y_sum = np.sum(y)
    y_sum_square = np.sum([yi**2 for yi in y])
    xy_sum = np.sum([xi*yi for xi, yi in zip(x, y)])

    #calculcate remainder of equations
    s_xx  = x_sum_square - (1/n)*(x_sum**2)
    s_yy = y_sum_square - (1/n)*(y_sum**2)
    s_xy = xy_sum - (1/n)*x_sum*y_sum

    return s_xx, s_yy, s_xy





def get_s_by_s(carbons, transfer=12, communities=None):

    otu = open(directory + '/data/otu_table.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')


    replica_id_list = []
    comm_rep_list = []
    comm_rep_replica_id_dict = {}

    for line in open(metadata_path):
        line = line.strip().split(',')
        #if line[-3].strip() == carbon:
        if line[-3].strip() in carbons:
            if line[-1].strip() == str(transfer):
                replica_id = line[0].strip()
                replica_id_list.append(replica_id)
                comm_rep = line[-5]+ '_' + line[-4]
                comm_rep_list.append(comm_rep)
                comm_rep_replica_id_dict[comm_rep] = replica_id

    if communities == None:

        communities_final = comm_rep_list
        communities_idx = [otu_first_line.index(comm_rep_replica_id_dict[c]) for c in comm_rep_list]


    else:

        communities_final = [str(k) for k in communities]
        communities_idx = [otu_first_line.index(comm_rep_replica_id_dict[c]) for c in communities]


    s_by_s = []
    species = []

    for line in otu:
        line = line.strip().split(',')
        line_idx = [int(line[i]) for i in communities_idx]

        if sum(line_idx) == 0:
            continue

        s_by_s.append(line_idx)
        species.append(line[0])

    s_by_s_np = np.asarray(s_by_s)

    return s_by_s_np, species, communities_final



def get_s_by_s_temporal(transfers, carbon='Glucose'):

    transfers = [str(transfer) for transfer in transfers]

    otu = open( '%s/data/otu_table.csv' % directory)
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    communities = []
    comm_rep_list = []

    for line in open(metadata_path):
        line = line.strip().split(',')
        if line[-3].strip() == carbon:
            if line[-1].strip() == str(1):
                communities.append(line[0].strip())
                comm_rep_list.append(line[-5]+ '_' + line[-4] )

    #communities_transfer = []
    comm_rep_transfer_list = []
    for line in open(metadata_path):
        line = line.strip().split(',')
        if line[-3].strip() == carbon:
            line_transfer = line[-1].strip()
            if line[-5]+ '_' + line[-4] in comm_rep_list:
                #communities_transfer.append(line[0].strip()+'_'+line_transfer)
                comm_rep_transfer_list.append(line[-5]+ '_' + line[-4]+'_'+line_transfer)

    communities_idx = [otu_first_line.index(community) for community in communities]



def get_species_means_and_variances(rel_s_by_s, species_list, min_observations=3, zeros=False):
    mean_rel_abundances = []
    var_rel_abundances = []
    species_to_keep = []

    for afd_idx, afd in enumerate(rel_s_by_s):

        afd_no_zeros = afd[afd>0]

        if len(afd_no_zeros) < min_observations:
            continue

        if zeros == True:

            mean_rel_abundances.append(np.mean(afd))
            var_rel_abundances.append(np.var(afd))

        else:

            mean_rel_abundances.append(np.mean(afd_no_zeros))
            var_rel_abundances.append(np.var(afd_no_zeros))

        species_to_keep.append(species_list[afd_idx])

    mean_rel_abundances = np.asarray(mean_rel_abundances)
    var_rel_abundances = np.asarray(var_rel_abundances)
    species_to_keep = np.asarray(species_to_keep)

    return mean_rel_abundances, var_rel_abundances, species_to_keep





def get_temporal_patterns(migration_innoculum, attractor=None):

    relative_abundance_dict_init = get_relative_abundance_dictionary_temporal_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])

    if attractor != None:
        attractor_dict = get_attractor_status(migration=migration_innoculum[0], inocula=migration_innoculum[1])
        to_keep = attractor_dict[attractor]
        relative_abundance_dict = {}
        for ESV, ESV_dict in relative_abundance_dict_init.items():

            for replicate in ESV_dict.keys():
                if replicate in to_keep:

                    if ESV not in relative_abundance_dict:
                        relative_abundance_dict[ESV] = {}
                    relative_abundance_dict[ESV][replicate] = ESV_dict[replicate]

    else:
        relative_abundance_dict = relative_abundance_dict_init


    species_mean_relative_abundances = []
    species_mean_absolute_differences = []
    species_mean_relative_abundances_for_width = []
    species_width_distribution_ratios = []
    species_width_distribution_transfers = []
    species_transfers = []
    species_names = []

    species_width_distribution_ratios_dict = {}

    for species, species_dict in relative_abundance_dict.items():

        species_abundance_difference_dict = {}

        for replicate, species_replicate_dict in species_dict.items():

            if len(species_replicate_dict['transfers']) < 2:
                continue

            transfers = np.asarray(species_replicate_dict['transfers'])

            relative_abundances = np.asarray(species_replicate_dict['relative_abundances'])

            transfer_differences = transfers[:-1]
            absolute_differences = np.abs(relative_abundances[1:] - relative_abundances[:-1])

            #width_distribution_ratios = np.abs(relative_abundances[1:] / relative_abundances[:-1])
            width_distribution_ratios = relative_abundances[1:] / relative_abundances[:-1]

            for transfer_difference_idx, transfer_difference in enumerate(transfer_differences):

                if str(transfer_difference) not in species_abundance_difference_dict:
                    species_abundance_difference_dict[str(transfer_difference)] = {}
                    species_abundance_difference_dict[str(transfer_difference)]['absolute_differences'] = []
                    species_abundance_difference_dict[str(transfer_difference)]['relative_abundances'] = []
                    species_abundance_difference_dict[str(transfer_difference)]['width_distribution_ratios'] = []

                species_abundance_difference_dict[str(transfer_difference)]['absolute_differences'].append(absolute_differences[transfer_difference_idx])
                species_abundance_difference_dict[str(transfer_difference)]['relative_abundances'].append(relative_abundances[transfer_difference_idx])
                species_abundance_difference_dict[str(transfer_difference)]['width_distribution_ratios'].append(width_distribution_ratios[transfer_difference_idx])


        for transfer, transfer_dict in species_abundance_difference_dict.items():

            if len(transfer_dict['relative_abundances']) < 3:
                continue

            #species_mean_relative_abundances.append(np.mean(np.log10(transfer_dict['relative_abundances'])))
            #species_mean_absolute_differences.append(np.mean(np.log10(transfer_dict['absolute_differences'])))
            #species_mean_width_distribution_ratios.append(np.mean(np.log10(transfer_dict['width_distribution_ratios'])))

            # figure this out tomorrow
            mean_relative_abundance = np.mean(transfer_dict['relative_abundances'])
            species_mean_relative_abundances.append(mean_relative_abundance)
            species_mean_absolute_differences.append(np.mean(transfer_dict['absolute_differences']))


            width_distribution_ratios = transfer_dict['width_distribution_ratios']
            species_width_distribution_ratios.extend(width_distribution_ratios)
            species_mean_relative_abundances_for_width.extend([mean_relative_abundance]*len(width_distribution_ratios))
            species_width_distribution_transfers.extend([int(transfer)]*len(width_distribution_ratios))

            species_transfers.append(int(transfer))
            species_names.append(species)


            if int(transfer) not in species_width_distribution_ratios_dict:
                species_width_distribution_ratios_dict[int(transfer)] = []

            species_width_distribution_ratios_dict[int(transfer)].extend(np.log10(transfer_dict['width_distribution_ratios']))


    # calcualte variance in width distribution
    variance_width_distribution_transfers = []
    variance_width_distribution = []

    #mean_width_distribution_transfers = []
    mean_width_distribution = []

    for transfer_, widths_ in species_width_distribution_ratios_dict.items():

        widths_ = np.asarray(widths_)

        if transfer_ == 6:
            widths_ = widths_[widths_<-1.8]

        if len(widths_) < 3:
            continue

        variance_width_distribution_transfers.append(transfer_)
        variance_width_distribution.append(np.sqrt(np.var(widths_)) / np.absolute(np.mean(widths_)) )

        mean_width_distribution.append(np.mean(widths_))

    variance_width_distribution_transfers = np.asarray(variance_width_distribution_transfers)
    variance_width_distribution = np.asarray(variance_width_distribution)
    mean_width_distribution = np.asarray(mean_width_distribution)


    variance_width_distribution_transfers_idx = np.argsort(variance_width_distribution_transfers)


    variance_width_distribution_transfers = variance_width_distribution_transfers[variance_width_distribution_transfers_idx]
    variance_width_distribution = variance_width_distribution[variance_width_distribution_transfers_idx]
    mean_width_distribution = mean_width_distribution[variance_width_distribution_transfers_idx]

    species_mean_relative_abundances = np.asarray(species_mean_relative_abundances)
    species_mean_absolute_differences = np.asarray(species_mean_absolute_differences)

    species_width_distribution_ratios = np.asarray(species_width_distribution_ratios)

    species_width_distribution_transfers = np.asarray(species_width_distribution_transfers)

    species_mean_relative_abundances_for_width = np.asarray(species_mean_relative_abundances_for_width)


    species_transfers = np.asarray(species_transfers)
    species_names = np.asarray(species_names)

    return species_names, species_transfers, species_mean_relative_abundances, species_mean_absolute_differences, species_width_distribution_transfers, species_width_distribution_ratios,species_mean_relative_abundances_for_width, variance_width_distribution_transfers, variance_width_distribution, mean_width_distribution




#class lognorm:

#    def __init__(self, obs):
#        self.obs = obs


#    def ppoints(self, n):
#        """ numpy analogue or `R`'s `ppoints` function
#            see details at http://stat.ethz.ch/R-manual/R-patched/library/stats/html/ppoints.html
#            :param n: array type or number
#            Obtained from: http://stackoverflow.com/questions/20292216/imitating-ppoints-r-function-in-python
#            on 5 April 2016
#            """
#        if n < 10:
#            a = 3/8
#        else:
#            a = 1/2

#        try:
#            n = np.float(len(n))
#        except TypeError:
#            n = np.float(n)
#        return (np.arange(n) + 1 - a)/(n + 1 - 2*a)



#    def get_rad_pln(self, S, mu, sigma, lower_trunc = True):
#        """Obtain the predicted RAD from a Poisson lognormal distribution"""
#        abundance = list(np.empty([S]))
#        rank = range(1, int(S) + 1)
#        cdf_obs = [(rank[i]-0.5) / S for i in range(0, int(S))]
#        j = 0
#        cdf_cum = 0
#        i = 1
#        while j < S:
#            cdf_cum += pln.pmf(i, mu, sigma, lower_trunc)
#            while cdf_cum >= cdf_obs[j]:
#                abundance[j] = i
#                j += 1
#                if j == S:
#                    abundance.reverse()
#                    return abundance
#            i += 1


#    def get_rad_from_obs(self):
#        mu, sigma = pln_solver(self.obs)
#        pred_rad = self.get_rad_pln(len(self.obs), mu, sigma)

#        return (pred_rad, mu, sigma)




#def fit_sad():

#    for carbon_idx, carbon in enumerate(carbons):
#
#        sys.stdout.write("Fitting lognormal for %s communities...\n" % (carbon))
#
#        s_by_s, species, comm_rep_list = get_s_by_s(carbon)

#        data = []

#        obs_list = []
#        pred_list = []

#        N_communities = s_by_s.shape[1]
#        for sad in s_by_s.T:
#            sys.stdout.write("%d communities to go!\n" % (N_communities))
#            N_communities -= 1
#
#            sad = sad[sad>0]
#            sad.sort()
#            sad = sad[::-1]
#            if len(sad) < 10:
#                continue

#            N = sum(sad)
#            S = len(sad)
#            Nmax = max(sad)

#            lognorm_pred = lognorm(sad)
#            sad_predicted, mu, sigma = lognorm_pred.get_rad_from_obs()
#            pln_Nmax = max(sad_predicted)

#            ll = pln_ll(sad, mu, sigma)
#            r2 = obs_pred_rsquare(np.log10(sad), np.log10(sad_predicted))

#            data.append([str(N),str(S),str(Nmax),str(pln_Nmax),str(ll),str(r2)])

#            obs_list.extend(sad)
#            pred_list.extend(sad_predicted)


#        df_sad_out = open((directory + '/data/pln_sad_counts_%s.csv') % (carbon) , 'w')
#        df_sad_out.write(','.join(['Observed','Predicted']) + '\n')
#        for i in range(len(obs_list)):
#            df_sad_out.write(','.join([str(obs_list[i]), str(pred_list[i])]) + '\n')
#        df_sad_out.close()


#        df_out = open((directory + '/data/pln_sad_%s.csv') % (carbon) , 'w')
#        df_out.write(','.join(['N','S','Nmax', 'pln_Nmax', 'll','r2_mod']) + '\n')
#        for i in range(len(data)):
#            df_out.write(','.join(data[i]) + '\n')
#        df_out.close()







def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #todo: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = np.log10(x), np.log10(y), np.log10(radius)
    for a, b in unique_points:
        if logscale == 1:
            loga, logb = np.log10(a), np.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 +
                                   (logy - logb) ** 2) <= logr ** 2])
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))
    #color_range = np.sqrt(sorted_plot_data[:, 2])
    #color_range = color_range / max(color_range)

    #colors = [ cm.YlOrRd(x) for x in color_range ]

    if plot_obj == None:
        plot_obj = plt.axes()

    if loglog == 1:
        plot_obj.set_xscale('log')
        plot_obj.set_yscale('log')
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                         c = np.sqrt(sorted_plot_data[:, 2]), edgecolors='none')
        plot_obj.set_xlim(min(x) * 0.5, max(x) * 2)
        plot_obj.set_ylim(min(y) * 0.5, max(y) * 2)
    else:
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                    c = np.log10(sorted_plot_data[:, 2]), edgecolors='none')
    return plot_obj





def predict_occupancy(s_by_s, species, totreads=np.asarray([])):

    # get squared inverse cv
    # assume that entries are read counts.
    rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

    beta_all = []
    mean_all = []

    for s in rel_s_by_s_np:

        var = np.var(s)
        mean = np.mean(s)

        beta = (mean**2)/var

        mean_all.append(mean)
        beta_all.append(beta)

    beta_all = np.asarray(beta_all)
    mean_all = np.asarray(mean_all)


    s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

    #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
    # calcualte total reads if no argument is passed
    # sloppy quick fix
    if len(totreads) == 0:
        totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(np.mean(afd_no_zeros/ totreads[afd>0]))
        #tf.append(np.mean(afd_no_zeros/s_by_s.sum(axis=0)[afd>0]))

    #tf = np.mean(rel_abundances)
    tf = np.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        N_reads = s_by_s.sum(axis=0)[np.nonzero(afd)[0]]
        #N_reads = s_by_s.sum(axis=0)[afd>0]

        tvpf_list.append(np.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    #tvpf = np.mean(tvpf_list)
    tvpf = np.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    predicted_occupancies = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    for beta_i, theta_i in zip(beta,theta):
        predicted_occupancies.append(1 - np.mean( ((1+theta_i*totreads)**(-1*beta_i ))   ))
        #1- mean( (1.+theta*totreads)^(-beta ) )

    predicted_occupancies = np.asarray(predicted_occupancies)

    idx_to_keep = (~np.isnan(occupancies)) & (~np.isnan(predicted_occupancies)) & (occupancies>0)

    occupancies_no_zeros = occupancies[idx_to_keep]
    predicted_occupancies_no_zeros = predicted_occupancies[idx_to_keep]

    species = np.asarray(species)
    species_no_zeros = species[idx_to_keep]

    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
    mad = np.mean(rel_s_by_s, axis=1)
    mad_no_zeros = mad[idx_to_keep]

    #beta_no_zeros = beta[idx_to_keep]
    #theta_no_zeros = theta[idx_to_keep]
    # beta/f
    #beta_div_f_no_zeros = 1/theta_no_zeros

    beta_no_zeros = beta_all[idx_to_keep]

    return occupancies_no_zeros, predicted_occupancies_no_zeros, mad_no_zeros, beta_no_zeros, species_no_zeros




def calculate_shape_and_rate_params(s_by_s):



    return mean_all, beta_all





def calculate_theta(s_by_s, totreads=np.asarray([])):

    s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

    #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
    # calcualte total reads if no argument is passed
    # sloppy quick fix
    if len(totreads) == 0:
        totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(np.mean(afd_no_zeros/ totreads[afd>0]))
        #tf.append(np.mean(afd_no_zeros/s_by_s.sum(axis=0)[afd>0]))

    #tf = np.mean(rel_abundances)
    tf = np.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        N_reads = s_by_s.sum(axis=0)[np.nonzero(afd)[0]]
        #N_reads = s_by_s.sum(axis=0)[afd>0]

        tvpf_list.append(np.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    #tvpf = np.mean(tvpf_list)
    tvpf = np.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    return theta**-1

    #predicted_occupancies = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    #for beta_i, theta_i in zip(beta,theta):

    #    predicted_occupancies.append(1 - np.mean( ((1+theta_i*totreads)**(-1*beta_i ))   ))



def group_ESVs(s_by_s, species, comm_rep_list, taxonomic_level='genus'):
    taxonomic_level_dict = {'genus':-1, 'family':-2, 'order':-3, 'class':-4}
    taxonomy_map = {}
    taxonomy = open(directory+'/data/taxonomy.csv')
    taxonomy_fitst_line = taxonomy.readline()
    for line in taxonomy:
        line = line.strip().split(',')
        if line[ taxonomic_level_dict[taxonomic_level] ] != 'NA':
            taxonomy_map[line[0]] = line[ taxonomic_level_dict[taxonomic_level] ]
    taxonomy.close()


    taxon_sample_dict = {}

    for afd_idx, afd in enumerate(s_by_s):

        esv = species[afd_idx]
        if esv not in taxonomy_map:
            continue

        taxon = taxonomy_map[esv]
        if taxon not in taxon_sample_dict:
            taxon_sample_dict[taxon] = {}

        for comm_rep_idx, comm_rep in enumerate(comm_rep_list):
            if comm_rep not in taxon_sample_dict[taxon]:
                taxon_sample_dict[taxon][comm_rep] = 0

            taxon_sample_dict[taxon][comm_rep] += afd[comm_rep_idx]


    s_by_s_df = pd.DataFrame(taxon_sample_dict)
    s_by_s = s_by_s_df.values.T

    #return s_by_s, list(taxon_sample_dict.keys()), comm_rep_list
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()






def get_relative_s_by_s_migration(transfer=18,migration='No_migration',inocula=40, communities=None):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/migration_ESV_data_table_SE.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    for line in otu:
        line = line.strip().split(',')
        if float(line[1]) == 0:
            continue

        if (int(line[5]) == transfer) and (line[2] == migration) and (int(line[3]) == inocula) :

            if line[0] not in count_dict:
                count_dict[line[0]] = {}

            if communities == None:

                if line[6] not in count_dict[line[0]]:
                    count_dict[line[0]][line[6]] = 0

                if count_dict[line[0]][line[6]] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[line[0]][line[6]] = float(line[1])

            else:

                if line[6] not in communities:
                    continue

                if line[6] not in count_dict[line[0]]:
                    count_dict[line[0]][line[6]] = 0

                if count_dict[line[0]][line[6]] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[line[0]][line[6]] = float(line[1])



    otu.close()

    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()






def get_s_by_s_migration_test_singleton(transfer=18,migration='No_migration',inocula=40, communities=None):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/migration_data_table_totabund_all_singleton_mapped_20210501.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    all_treatments = []
    for line in otu:
        line = line.strip().split(',')
        treatment_line = line[1]
        treatment_line = re.sub(r'["]', '', treatment_line)

        all_treatments.append(treatment_line)


        if (int(line[4]) == transfer) and (treatment_line == migration) and (int(line[3]) == inocula):

            sample_line = line[5]
            sample_line = sample_line.strip()
            sample_line = re.sub(r'["]', '', sample_line)

            #sample_line = int(sample_line)

            esv_line = line[6]
            esv_line = re.sub(r'["]', '', esv_line)

            if esv_line not in count_dict:
                count_dict[esv_line] = {}

            if communities == None:

                if sample_line not in count_dict[esv_line]:
                    count_dict[esv_line][sample_line] = 0

                if count_dict[esv_line][sample_line] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )


                count_dict[esv_line][sample_line] = float(line[7])

            else:

                if sample_line not in communities:
                    continue

                if sample_line not in count_dict[esv_line]:
                    count_dict[esv_line][sample_line] = 0

                if count_dict[esv_line][sample_line] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[esv_line][sample_line] = float(line[7])



    otu.close()



    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    #s_by_s = s_by_s_df.values
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()





# ,"Replicate","Carbon.Source.Group","Carbon.Source","Experiment_name","Transfer","nCS"





def get_s_by_s_resource(transfer=1,carbon_source_group='single',carbon_source='glucose',n_carbon_sources=1):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/NutrientDominance_data_table_totabund_all_mapped_singletons_20210510.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    all_treatments = []
    for line in otu:
        line = line.strip().split(',')

        print(line)
        continue

        treatment_line = line[1]
        treatment_line = re.sub(r'["]', '', treatment_line)

        all_treatments.append(treatment_line)


        if (int(line[4]) == transfer) and (treatment_line == migration) and (int(line[3]) == inocula):

            sample_line = line[5]
            sample_line = sample_line.strip()
            sample_line = re.sub(r'["]', '', sample_line)

            #sample_line = int(sample_line)

            esv_line = line[6]
            esv_line = re.sub(r'["]', '', esv_line)

            if esv_line not in count_dict:
                count_dict[esv_line] = {}

            if communities == None:

                if sample_line not in count_dict[esv_line]:
                    count_dict[esv_line][sample_line] = 0

                if count_dict[esv_line][sample_line] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )


                count_dict[esv_line][sample_line] = float(line[7])

            else:

                if sample_line not in communities:
                    continue

                if sample_line not in count_dict[esv_line]:
                    count_dict[esv_line][sample_line] = 0

                if count_dict[esv_line][sample_line] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[esv_line][sample_line] = float(line[7])



    otu.close()



    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    #s_by_s = s_by_s_df.values
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()










def get_relative_s_by_s_temporal_migration(migration='No_migration',inocula=40,communities=None):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if communities==None:

        communities_new = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities_new.append(line[6])

        communities_new = list(set(communities_new))

    else:

        communities_new = [str(k) for k in communities]

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        if float(line[1]) == 0:
            continue

        if (line[6] in communities_new) and (line[2] == migration) and (int(line[3]) == inocula):

            if line[0] not in count_dict:
                count_dict[line[0]] = {}


            community_time = '%s_%s' % (line[6], line[5])

            if community_time not in count_dict[line[0]]:
                count_dict[line[0]][community_time] = 0

            if count_dict[line[0]][community_time] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[line[0]][community_time] = float(line[1])


    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()





def get_s_by_s_temporal_migration(migration='No_migration',inocula=40,communities=None):

    otu_path = directory + '/data/migration_ESV_data_table_absabundance_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if communities==None:

        communities_new = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities_new.append(line[6])

        communities_new = list(set(communities_new))

    else:

        communities_new = [str(k) for k in communities]


    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        abundance = int(line[1])

        if abundance == 0:
            continue

        if (line[6] in communities_new) and (line[3] == migration) and (int(line[4]) == inocula):

            if line[0] not in count_dict:
                count_dict[line[0]] = {}

            community_time = '%s_%s' % (line[6], line[5])

            if community_time not in count_dict[line[0]]:
                count_dict[line[0]][community_time] = 0

            if count_dict[line[0]][community_time] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[line[0]][community_time] = abundance


    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()







def get_migration_time_series_community_names_old(migration='No_migration',inocula=40):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_path = directory + '/data/migration_data_table_totabund_all_singleton_mapped_20210501.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_community_dict = {}

    communities = []

    # line[5] = transfer
    # line[2] = migration
    # line[3] = inocula

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')
        if line[3] == 'NA':
            continue

        transfer_line = int(line[4])
        migration_line = re.sub(r'["]', '', line[1])
        inocula_line = int(line[3])


        if (migration_line == migration) and (inocula_line == inocula):

            replicate_community = int(line[5])

            if replicate_community not in count_community_dict:
                count_community_dict[replicate_community] = []
            transfer = int(line[4])
            if transfer not in count_community_dict[replicate_community]:
                count_community_dict[replicate_community].append(transfer)

        #if (int(line[5]) == 3) and (line[2] == migration) and (int(line[3]) == inocula):
        #    communities.append(line[6])

    return count_community_dict




def get_migration_time_series_community_names(migration='No_migration',inocula=40):

    #otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    otu_path = directory + '/data/migration_data_table_totabund_all_singleton_mapped_20210501.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_community_dict = {}

    communities = []

    # line[5] = transfer
    # line[2] = migration
    # line[3] = inocula

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')
        if line[3] == 'NA':
            continue

        transfer_line = int(line[4])
        migration_line = re.sub(r'["]', '', line[1])
        inocula_line = int(line[3])

        if (migration_line == migration) and (inocula_line == inocula):
            replicate_community = int(line[5])
            if replicate_community not in count_community_dict:
                count_community_dict[replicate_community] = []
            transfer = int(line[4])
            if transfer not in count_community_dict[replicate_community]:
                count_community_dict[replicate_community].append(transfer)

        #if (int(line[5]) == 3) and (line[2] == migration) and (int(line[3]) == inocula):
        #    communities.append(line[6])

    return count_community_dict





def get_s_by_s_migration(transfer=18, migration='No_migration',inocula=40, communities=None ):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/migration_ESV_data_table_absabundance_SE.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    if communities != None:
        communities = [str(k) for k in communities]

    count_dict = {}
    for line in otu:
        line = line.strip().split(',')
        if int(line[1]) == 0:
            continue

        if line[1] == 1:
            print("singleton!")

        if (int(line[5]) == transfer) and (line[3] == migration) and (int(line[4]) == inocula):

            rep = '%s_%s_%s' % (line[3], line[4], line[0])

            if rep not in count_dict:
                count_dict[rep] = {}

            if communities == None:

                if line[6] not in count_dict[rep]:
                    count_dict[rep][line[6]] = 0

                if count_dict[rep][line[6]] > 0:

                    sys.stdout.write("Species already in dictionary!!\n" )

                count_dict[rep][line[6]] += int(line[1])

            else:

                if line[6] not in communities:
                    continue

                if line[6] not in count_dict[rep]:
                    count_dict[rep][line[6]] = 0

                if count_dict[rep][line[6]] > 0:

                    sys.stdout.write("Species already in dictionary!!\n" )

                count_dict[rep][line[6]] += int(line[1])



    otu.close()

    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()




def get_relative_abundance_dictionary_temporal_migration(migration='No_migration',inocula=40,community=None):

    otu_path = directory + '/data/migration_data_table_totabund_all_singleton_mapped_20210501.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if community==None:

        communities = []

        transfers = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            transfer_line = int(line[4])
            migration_line = re.sub(r'["]', '', line[1])
            inocula_line = int(line[3])

            if (transfer_line == 1) and (migration_line == migration) and (inocula_line == inocula):
                communities.append(line[5])

        communities = list(set(communities))

    else:
        communities = [str(community)]


    for transfer in range(1, 19):

        s_by_s, species, comm_rep_list = get_s_by_s_migration_test_singleton(transfer=transfer, migration=migration, inocula=inocula)
        relative_s_by_s = (s_by_s/s_by_s.sum(axis=0))

        #comm_rep_list = np.asarray(comm_rep_list)

        #community_idx = [np.where(comm_rep_list == community)[0][0] for community in communities]
        #community_idx = np.asarray(community_idx)

        for afd_idx, afd in enumerate(relative_s_by_s):

            esv = species[afd_idx]

            for comm_rep_list_i, afd_i in zip(comm_rep_list, afd.tolist()):

                if afd_i == 0:
                    continue

                if comm_rep_list_i not in communities:
                    continue

                if esv not in count_dict:
                    count_dict[esv] = {}

                if comm_rep_list_i not in count_dict[esv]:
                    count_dict[esv][comm_rep_list_i] = {}
                    count_dict[esv][comm_rep_list_i]['transfers'] = []
                    count_dict[esv][comm_rep_list_i]['relative_abundances'] = []


                count_dict[esv][comm_rep_list_i]['transfers'].append(transfer)
                count_dict[esv][comm_rep_list_i]['relative_abundances'].append(afd_i)



    return count_dict






def get_relative_abundance_dictionary_temporal_migration_old(migration='No_migration',inocula=40,community=None):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if community==None:

        communities = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities.append(line[6])

        communities = list(set(communities))

    else:

        communities = [str(community)]

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        if float(line[1]) == 0:
            continue

        replicate = line[6]
        species = line[0]
        transfer = int(line[5])

        relative_abundance = float(line[1])

        if relative_abundance == 0:
            continue

        if (replicate in  communities) and (line[2] == migration) and (int(line[3]) == inocula):

            if species not in count_dict:
                count_dict[species] = {}

            if replicate not in  count_dict[species]:
                count_dict[species][replicate] = {}

                count_dict[species][replicate]['transfers'] = []

                count_dict[species][replicate]['relative_abundances'] = []


            bisect.insort(count_dict[species][replicate]['transfers'],  int(transfer))

            transfer_idx = count_dict[species][replicate]['transfers'].index(transfer)

            count_dict[species][replicate]['relative_abundances'].insert(transfer_idx, relative_abundance)



            #community_time = '%s_%s' % (line[6], line[5])

            #if community_time not in count_dict[line[0]]:
            #    count_dict[line[0]][community_time] = 0

            #if count_dict[line[0]][community_time] > 0:

            #    sys.stdout.write("Relative abundance already in dictionary!!\n" )

            #count_dict[line[0]][community_time] = float(line[1])

    return count_dict

    #s_by_s_df = pd.DataFrame(count_dict)
    #s_by_s_df = s_by_s_df.fillna(0)
    #s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    #return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()





def get_attractor_status(migration='No_migration', inocula=4):

    attractor_dict = {}

    attractor_file = open(directory+'/data/attractor_status.csv')
    attractor_file_fitst_line = attractor_file.readline()
    for line in attractor_file:
        line = line.strip().split(',')
        if (line[0] == migration) and (int(line[1]) == inocula):

            if line[-1] not in attractor_dict:
                attractor_dict[line[-1]] = []

            attractor_dict[line[-1]].append(str(line[-2]))
    attractor_file.close()

    return attractor_dict




def get_slopes_cutoffs(mean_relative_abundances, mean_absolute_differences):

    x_cutoffs = np.linspace(np.log10(0.01), np.log10(1), num=100)

    slopes_cutoff = []

    for x_cutoff in x_cutoffs:

        mean_relative_abundances_cutoff = mean_relative_abundances[mean_relative_abundances< x_cutoff]
        mean_absolute_differences_cutoff = mean_absolute_differences[mean_relative_abundances< x_cutoff]
        #mean_relative_abundances_cutoff = mean_relative_abundances[(mean_relative_abundances> x_cutoff-0.5) & (mean_relative_abundances< x_cutoff) ]
        #mean_absolute_differences_cutoff = mean_absolute_differences[(mean_relative_abundances> x_cutoff-0.5) & (mean_relative_abundances< x_cutoff) ]

        slope_cutoff, intercept_cutoff, r_value_cutoff, p_value_cutoff, std_err_cutoff = stats.linregress(mean_relative_abundances_cutoff, mean_absolute_differences_cutoff)
        slopes_cutoff.append(slope_cutoff)

    slopes_cutoff = np.asarray(slopes_cutoff)

    return x_cutoffs, slopes_cutoff




## Variance difference test

def test_difference_two_time_series(array_1, array_2):
    # two-sided test
    _matrix = np.array([array_1, array_2])
    #mean_ratio_observed = sum(_matrix[0,:] - _matrix[1,:])
    mean_ratio_observed = np.absolute(np.mean(_matrix[0,:] - _matrix[1,:]))

    null_values = []
    for i in range(10000):
        _matrix_copy = np.copy(_matrix)
        for i in range(_matrix_copy.shape[1]):
            np.random.shuffle(_matrix_copy[:,i])

        #null_values.append(sum(_matrix_copy[0,:] - _matrix_copy[1,:]))
        null_values.append(np.absolute(np.mean(_matrix_copy[0,:] - _matrix_copy[1,:])))

    null_values = np.asarray(null_values)
    p_value = len(null_values[null_values > mean_ratio_observed]) / 10000

    return mean_ratio_observed, p_value



def get_intersecting_timepoints(transfers_1, observations_1, transfers_2, observations_2):

    transfers_intersect = np.intersect1d(transfers_1, transfers_2)

    sorter_1 = np.argsort(transfers_1)
    sorted_idx_1 = sorter_1[np.searchsorted(transfers_1, transfers_intersect, sorter=sorter_1)]
    observations_1_intersect = observations_1[sorted_idx_1]

    sorter_2 = np.argsort(transfers_2)
    sorted_idx_2 = sorter_2[np.searchsorted(transfers_2, transfers_intersect, sorter=sorter_2)]
    observations_2_intersect = observations_2[sorted_idx_2]

    return transfers_intersect, observations_1_intersect, observations_2_intersect



def estimate_mean_abundances_parent():

    otu = open(directory + '/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    all_treatments = []

    for line in otu:
        line = line.strip().split(',')
        sample_line_name = line[0]
        sample_line_name = re.sub(r'["]', '', sample_line_name)
        treatment_line = line[1]
        treatment_line = re.sub(r'["]', '', treatment_line)
        inocula = line[3]

        transfer = int(line[4])

        if transfer == 0:

            sample_line = line[5]
            sample_line = sample_line.strip()
            sample_line = re.sub(r'["]', '', sample_line)

            #sample_line = int(sample_line)

            esv_line = line[6]
            esv_line = re.sub(r'["]', '', esv_line)



            if sample_line_name not in count_dict:
                count_dict[sample_line_name] = {}


            if esv_line not in count_dict[sample_line_name]:
                count_dict[sample_line_name][esv_line] = 0

            if count_dict[sample_line_name][esv_line] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[sample_line_name][esv_line] = int(line[7])


    abundances_dict = {}

    for sample, sample_dict in count_dict.items():

        N = sample_dict.values()

        species = list(sample_dict.keys())

        rel_abundances = np.asarray(list(sample_dict.values())) / sum(sample_dict.values())

        for s_idx, s in enumerate(species):

            if s not in abundances_dict:
                abundances_dict[s] = []

            abundances_dict[s].append(rel_abundances[s_idx])

    mean_rel_abundances_all = []
    species_all = []


    for species, rel_abundances in abundances_dict.items():

        if len(rel_abundances) < 3:

            rel_abundances.extend([0]*(3-len(rel_abundances)))


        mean_rel_abundances_all.append( np.mean(rel_abundances))
        species_all.append(species)

    return mean_rel_abundances_all, species_all




def get_otu_dict():

    otu = open(directory + '/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    all_treatments = []

    for line in otu:
        line = line.strip().split(',')
        sample_line_name = line[0]
        sample_line_name = re.sub(r'["]', '', sample_line_name)
        treatment_line = line[1]
        treatment_line = re.sub(r'["]', '', treatment_line)
        inocula = line[3]

        transfer = int(line[4])

        if (transfer == 18) or (transfer == 0):

            sample_line = line[5]
            sample_line = sample_line.strip()
            sample_line = re.sub(r'["]', '', sample_line)

            #sample_line = int(sample_line)

            esv_line = line[6]
            esv_line = re.sub(r'["]', '', esv_line)

            if sample_line_name not in count_dict:
                count_dict[sample_line_name] = {}


            if esv_line not in count_dict[sample_line_name]:
                count_dict[sample_line_name][esv_line] = 0

            if count_dict[sample_line_name][esv_line] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[sample_line_name][esv_line] = int(line[7])

    otu.close()

    return count_dict




def make_treatment_csv():

    count_dict = get_otu_dict()
    samples = list(count_dict.keys())
    #Parent_migration.4.T18, No_migration.4.T18, No_migration.40.T18, Global_migration.4. T18,Parent_migration.NA.T0
    # export each as rows = species, columns = samples

    for treatment in treatments:

        #samples_to_keep = [sample for sample in samples if treatment in sample]
        count_dict_to_keep = {key: value for key, value in count_dict.items() if treatment in key}
        s_by_s_df = pd.DataFrame(count_dict_to_keep)
        s_by_s_df = s_by_s_df.fillna(0)

        s_by_s_df = s_by_s_df.reset_index().rename({'index':'ASVs '}, axis = 'columns')
        # index = True,
        s_by_s_df.to_csv('%s/data/%s.csv' % (directory, treatment),  header=True)



def Klogn(emp_mad, c, mu0=-19,s0=5):
    # This function estimates the parameters (mu, s) of the lognormal distribution of K
    m1 = np.mean(np.log(emp_mad[emp_mad>c]))
    m2 = np.mean(np.log(emp_mad[emp_mad>c])**2)
    xmu = sp.symbols('xmu')
    xs = sp.symbols('xs')
    eq1 =- m1+xmu + np.sqrt(2/math.pi)*xs*sp.exp(-((np.log(c)-xmu)**2)/2/(xs**2))/(sp.erfc((np.log(c)-xmu)/np.sqrt(2)/xs))
    eq2 =- m2+xs**2+m1*xmu+np.log(c)*m1-xmu*np.log(c)

    sol = sp.nsolve([eq1,eq2],[xmu,xs],[mu0,s0])

    return(float(sol[0]),float(sol[1]))



def get_lognorma_mad_prediction(x, mu, sigma, c):

    return np.sqrt(2/math.pi)/sigma*np.exp(-(x-mu)**2 /2/(sigma**2))/special.erfc((np.log(c)-mu)/np.sqrt(2)/sigma)
