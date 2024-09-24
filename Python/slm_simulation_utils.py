from __future__ import division
import os
import sys
import copy
from math import factorial, sqrt
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate, stats, optimize, special
from scipy.stats import rv_discrete, rv_continuous, itemfreq
from scipy.optimize import bisect, fsolve
from scipy.integrate import quad
import warnings

import utils
import pickle

np.random.seed(123456789)
np.seterr(divide = 'ignore')



simulation_parent_rho_path = utils.directory + "/data/simulation_parent_rho.pickle"
simulation_parent_rho_abc_path = utils.directory + "/data/simulation_parent_rho_abc.pickle"
simulation_parent_rho_fixed_parameters_path = utils.directory + "/data/simulation_parent_rho_fixed_parameters.pickle"

simulation_global_rho_path = utils.directory + "/data/simulation_global_rho.pickle"
simulation_global_rho_abc_path = utils.directory + "/data/simulation_global_rho_abc.pickle"

#simulation_global_rho_fixed_parameters_path = utils.directory + "/data/simulation_global_rho_fixed_parameters.pickle"
simulation_global_rho_fixed_parameters_path = utils.directory + "/data/simulation_global_rho_fixed_parameters_%s.pickle"


simulation_migration_all_path = utils.directory + "/data/simulation_migration_all.pickle"
simulation_migration_all_afd_path = utils.directory + "/data/simulation_migration_all_afd.pickle"

simulation_migration_all_abc_path = utils.directory + "/data/simulation_migration_all_abc.pickle"
simulation_all_migration_fixed_parameters_path = utils.directory + "/data/simulation_all_migration_fixed_parameters_%s.pickle"


simulation_migration_all_abc_afd_path = utils.directory + "/data/simulation_migration_all_abc_afd.pickle"
simulation_all_migration_fixed_parameters_afd_path = utils.directory + "/data/simulation_all_migration_fixed_parameters_afd_%s.pickle"


# soil in parent mixed with 100ml, aliquots taken from this culture
n_cells_parent = 7.92*(10**6)
# innoculation of replicate populations done with 4ul, 0.004 / 100
# large innoculation = 40uL, 0.4/100
# descendent lines are 500ul in size
#n_cells_descendent = 1*(10**8)
n_cells_descendent = 1*(10**8)

D_global = 504/60000
D_parent = 504/60000
D_transfer = 4/500



# merge ancestral SADs to get probability vector for multinomial sampling
# Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
# redo later to draw carrying capacities from no migration distribution
#mu_pln=3.4533926814573506
#sigma_pln=2.6967286975393754


#sigma_pln=4
mu_pln = -12.91762817054296
sigma_pln = 8.892456341236272
#sigma_pln = 6.892456341236272



tau_min = 1.7
tau_max = 6.9
sigma_min = 0.01
sigma_max = 1.9


tau_all = np.linspace(tau_min, tau_max, num=10, endpoint=True)
#sigma_all = np.linspace(0.01, 1.9, num=10, endpoint=True)
sigma_all = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num=10, endpoint=True, base=10.0)


rep_number_and_read_count_dict = utils.get_rep_number_and_read_count_dict()
sample_intersect_12_18_dict = utils.get_sample_intersect_12_18_dict()


# attractor indices
attractor_dict = utils.get_attractor_status(migration='No_migration', inocula=4)
attractor_alcaligenaceae_idx = np.asarray(attractor_dict['Alcaligenaceae'])
attractor_alcaligenaceae_idx = attractor_alcaligenaceae_idx.astype(int) - 1
#attractor_alcaligenaceae_idx = attractor_alcaligenaceae_idx-1


sad_idx_12 = rep_number_and_read_count_dict[('No_migration', 4)][11]['community_reps'] - 1
sad_idx_18 = rep_number_and_read_count_dict[('No_migration', 4)][17]['community_reps'] - 1


# get indices *relatvie* to position in sad_idx_t
attractor_alcaligenaceae_iter_12_idx = [np.where(sad_idx_12==k)[0][0] for k in attractor_alcaligenaceae_idx if k in sad_idx_12]
attractor_alcaligenaceae_iter_18_idx = [np.where(sad_idx_18==k)[0][0] for k in attractor_alcaligenaceae_idx if k in sad_idx_18]



max_sigma_dict = {1.7:0.5920575338892576, 2.2777777777777777:0.5920575338892576, 2.8555555555555556:1.060617421311563}




#._rvs method is not currently available for pln.
class pln_gen(rv_discrete):
    """Poisson lognormal distribution

    Method derived from Bulmer 1974 Biometrics 30:101-110

    Bulmer equation 7 - approximation for large abundances
    Bulmer equation 2 - integral for small abundances

    Adapted from Brian McGill's MATLAB function of the same name that was
    originally developed as part of the Palamedes software package by the
    National Center for Ecological Analysis and Synthesis working group on
    Tools and Fresh Approaches for Species Abundance Distributions
    (http://www.nceas.ucsb.edu/projects/11121)

    """
    def _pmf(self, x, mu, sigma, lower_trunc, approx_cut = 10):
        def untrunc_pmf(x_i, mu, sigma):
            pmf_i = 1e-120
            if sigma > 0:
                if x_i > approx_cut:
                    #use approximation for large abundances
                    #Bulmer equation 7
                    #tested vs. integral below - matches to about 6 significant digits for
                    #intermediate abundances (10-50) - integral fails for higher
                    #abundances, approx fails for lower abundances -
                    #assume it gets better for abundance > 50
                    V = sigma ** 2
                    pmf_i = 1 / sqrt(2 * np.pi * V) / x_i * \
                           np.exp(-(np.log(x_i) - mu) ** 2 / (2 * V)) * \
                           (1 + 1 / (2 * x_i * V) * ((np.log(x_i) - mu) ** 2 / V + \
                                                    np.log(x_i) - mu- 1))
                else:
                    # Bulmer equation 2 -tested against Grundy Biometrika 38:427-434
                    # Table 1 & Table 2 and matched to the 4 decimals in the table except
                    # for very small mu (10^-2)
                    # having the /gamma(ab+1) inside the integrand is inefficient but
                    # avoids pseudo-singularities
                    # split integral into two so the quad function finds the peak
                    # peak apppears to be just below ab - for very small ab (ab<10)
                    # works better to leave entire peak in one integral and integrate
                    # the tail in the second integral
                    if x_i < 10:
                        ub = 10
                    else:
                        ub = x_i
                    term1 = ((2 * np.pi * sigma ** 2) ** -0.5)/ factorial(x_i)
                    #integrate low end where peak is so it finds peak
                    eq = lambda t: np.exp(t * x_i - np.exp(t) - ((t - mu) / sigma) ** 2 / 2)
                    term2a = integrate.quad(eq, -np.inf, np.log(ub), full_output = 0, limit = 500)
                    #integrate higher end for accuracy and in case peak moves
                    term2b = integrate.quad(eq, np.log(ub), np.inf, full_output = 0, limit = 500)
                    Pr = term1 * (term2a[0] + term2b[0])
                    if Pr > 0:
                    #likelihood shouldn't really be zero and causes problem taking
                    #log of it
                        pmf_i = Pr
            return pmf_i

        pmf = []
        for i, x_i in enumerate(x):
            if lower_trunc[i]: # distribution lowered truncated at 1 (not accouting for zero-abundance species)
                if x_i == 0:
                    pmf_i = 0
                else:
                    pmf_i = untrunc_pmf(x_i, mu[i], sigma[i]) / (1 - untrunc_pmf(0, mu[i], sigma[i]))
            else:
                pmf_i = untrunc_pmf(x_i, mu[i], sigma[i])
            pmf.append(pmf_i)
        return np.array(pmf)

    def _cdf(self, x, mu, sigma, lower_trunc, approx_cut = 10):
        x = np.array(x)
        cdf = []
        for x_i in x:
            cdf.append(sum(self.pmf(range(int(x_i) + 1), mu[0], sigma[0], lower_trunc[0])))
        return np.array(cdf)

    def _ppf(self, cdf, mu, sigma, lower_trunc, approx_cut = 10):
        cdf = np.array(cdf)
        ppf = []
        for cdf_i in cdf:
            ppf_i = 1
            while self.cdf(ppf_i, mu, sigma, lower_trunc, approx_cut = approx_cut) < cdf_i:
                ppf_i += 1
            ppf.append(ppf_i)
        return np.array(ppf)

    def _rvs(self, mu, sigma, lower_trunc, size=10):
        if not lower_trunc:
            pois_par = np.exp(stats.norm.rvs(loc = mu, scale = sigma, size = size))
            ab = stats.poisson.rvs(pois_par, size = size)
        else:
            ab = []
            while len(ab) < size:
                pois_par = np.exp(stats.norm.rvs(loc = mu, scale = sigma))
                ab_single = stats.poisson.rvs(pois_par)
                if ab_single: ab.append(ab_single)
        return np.array(ab)

    def _argcheck(self, mu, sigma, lower_trunc):
        if lower_trunc is True: self.a = 1
        else: self.a = 0
        return (sigma > 0)


pln = pln_gen(name='pln', longname='Poisson lognormal',
              shapes = 'mu, sigma, lower_trunc')



def pln_ll(x, mu, sigma, lower_trunc = True):
    """Log-likelihood of a truncated Poisson lognormal distribution

    Method derived from Bulmer 1974 Biometrics 30:101-110

    Bulmer equation A1

    Adapted from Brian McGill's MATLAB function of the same name that was
    originally developed as part of the Palamedes software package by the
    National Center for Ecological Analysis and Synthesis working group on
    Tools and Fresh Approaches for Species Abundance Distributions
    (http://www.nceas.ucsb.edu/projects/11121)

    """
    x = np.array(x)
    #uniq_counts = itemfreq(x)
    unique_vals, counts = np.unique(x, return_counts=True)
    #unique_vals, counts = zip(*uniq_counts)
    plik = pln.logpmf(unique_vals, mu, sigma, lower_trunc)
    ll = 0
    for i, count in enumerate(counts):
        ll += count * plik[i]
    return ll



def pln_solver(ab, lower_trunc = True):
    """Given abundance data, solve for MLE of pln parameters mu and sigma

    Adapted from MATLAB code by Brian McGill that was originally developed as
    part of the Palamedes software package by the National Center for Ecological
    Analysis and Synthesis working group on Tools and Fresh Approaches for
    Species Abundance Distributions (http://www.nceas.ucsb.edu/projects/11121)

    """
    ab = np.array(ab)
    if lower_trunc is True:
        ab = check_for_support(ab, lower = 1)
    else: ab = check_for_support(ab, lower = 0)
    mu0 = np.mean(np.log(ab[ab > 0]))
    sig0 = np.std(np.log(ab[ab > 0]))
    def pln_func(x):
        return -pln_ll(ab, x[0], np.exp(x[1]), lower_trunc)
    mu, logsigma = optimize.fmin_l_bfgs_b(pln_func, x0 = [mu0, np.log(sig0)], approx_grad = True, \
                                          bounds = [(None, None), (np.log(10**-16), None)])[0]
    sigma = np.exp(logsigma)
    ll_ = pln_ll(ab, mu, sigma, lower_trunc)
    return mu, sigma, ll_



def check_for_support(x, lower = 0, upper = np.inf, warning = True):
    """Check if x (list or array) contains values out of support [lower, upper]

    If it does, remove the values and optionally print out a warning.
    This function is used for solvers of distributions with support smaller than (-inf, inf).

    """
    if (min(x) < lower) or (max(x) > upper):
        if warning:
            print("Warning: Out-of-support values in the input are removed.")
    x = np.array([element for element in x if lower <= element <= upper])
    return x



def estimate_pln_lognorm_parameters():

    # ancestor richness = 1533
    # richness in treatments at transfer 18 = 29,18,30,26
    # 25/1533 = 0.016

    # fit PLN to MAD

    s_by_s, species, comm_rep_list = utils.get_s_by_s_migration_test_singleton(transfer=18,migration='No_migration',inocula=4)
    mean_abundance = np.mean(s_by_s, axis=1)
    mu, sigma, ll_pln = pln_solver(mean_abundance)

    #mu=3.4533926814573506
    #sigma=2.6967286975393754

    return mu, sigma



def prob_non_zero_k(log_10_x, intercept=0.7123794529201377, slope=0.9410534572734155):
    return 1 / (1 + np.exp( -1 * (intercept + slope*log_10_x) ))

# use it for log10





def run_simulation_initial_condition_migration(sigma = 0.5, tau = 0.9, dt = 48, T = 864, reps = 92, migration_treatment='global'):
    # just use units of generations for now
    # T = generations
    # tau units of number of generations
    # dt = gens. per-transfer
    #noise_term = np.sqrt( sigma*dt/tau  ) # compound paramter, for conveinance
    noise_term_per_generation = np.sqrt( sigma*dt/(tau*7) ) 

    # get parent mean relative abundances
    mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
    mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

    # divide by sum to use as probability
    prob_mean_rel_abundances_parent = mean_rel_abundances_parent/sum(mean_rel_abundances_parent)
    richness_parent = len(prob_mean_rel_abundances_parent)
    # delta = % of species that survive, used to draw non zero carrying capacities
    #delta = 0.016
    delta = 56/richness_parent
    # draw from binomial distribution
    n_non_zero_k = np.random.binomial(richness_parent, delta)
    #print(n_non_zero_k)


    #def discrete_slm_initial_condition_migration(q_array, m):
    #    # q_array = array of replicate and species
    #    q_array_new = np.log(np.exp(q_array) + m)
    #    return q_array_new + (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))


    def discrete_slm_initial_condition_migration(q_array, m):
            # q_array = array of replicate and species
            q_array_new = np.log(np.exp(q_array) + m)
            
            #for t in range(7):
            #    q_array_new += ((dt/7)*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term_per_generation * np.random.randn(reps, len(init_abund_rel)))

            # instead of for loop, we can generate (# reps, # ASVs, # generations) random matrix, sum over # generations
            std_normal_matrix = noise_term_per_generation * np.sum(np.random.randn(reps, len(init_abund_rel), 7), axis=2)
            # returns (# reps, # ASVs) matrix
            q_array_new += (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + std_normal_matrix

            return q_array_new


    n_time_steps = int(T / dt)  # Number of time steps.
    t_gen = np.arange(0, T, dt)

    # non zero carrying capacities
    k_to_keep = pln._rvs(mu_pln, sigma_pln, lower_trunc=True, size=n_non_zero_k)


    #n_cells_parent

    # randomize order
    k_to_keep = np.random.permutation(k_to_keep)

    # the probability that a species has a non-zero carrying capicity given its abundance in the parent community
    non_zero_k_weights = np.asarray([prob_non_zero_k(l) for l in np.log10(mean_rel_abundances_parent)])
    non_zero_k_prob = non_zero_k_weights/sum(non_zero_k_weights)
    idx_mean_rel_abundances_parent = np.arange(len(mean_rel_abundances_parent))

    # select index of parent abundances with non-zero k
    idx_non_zero_k = np.random.choice(idx_mean_rel_abundances_parent, size=n_non_zero_k, replace=False, p=non_zero_k_prob)

    # select index of parent abundances with zero k
    idx_zero_k = np.delete(idx_mean_rel_abundances_parent, idx_non_zero_k)

    # sort the initial abundances by non-zero/zero k status
    idx_non_zero_and_zero = np.concatenate([idx_non_zero_k, idx_zero_k])

    # sort parent relative abundances and probability
    init_abund_rel = mean_rel_abundances_parent[idx_non_zero_and_zero]
    prob_mean_rel_abundances_parent = prob_mean_rel_abundances_parent[idx_non_zero_and_zero]

    k_to_keep = np.append(k_to_keep, [0]*len(idx_zero_k))
    # turn into absolute abundances
    k_to_keep = np.random.multinomial(n_cells_descendent, k_to_keep/sum(k_to_keep))

    def run_experiment(migration_treatment_):

        # create three dimensional vector
        # spcies, replicate, and timestep
        # create array for innnocumum
        n_0_inoc = np.zeros((reps, len(init_abund_rel)))
        for rep in range(reps):
            n_cells_inoc = np.random.binomial(n_cells_parent, D_transfer) # number of cells that get transferred between descendent communities
            n_0_inoc[rep,:] = np.random.multinomial(n_cells_inoc, init_abund_rel)

        q_0_inoc = np.log(n_0_inoc)
        # migration only happens at transfers, no migration when experiment is set up
        m_0 = np.zeros((reps, len(init_abund_rel)))
        q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
        # transfer 1
        q[0,:,:] = discrete_slm_initial_condition_migration(q_0_inoc, m_0)
        # transfer 2 to 18
        for t in range(1, n_time_steps): # t = transfer number
            # perform transfer
            n_t_minus_1 = np.exp(q[t,:,:])
            # remove infs, nan
            n_t_minus_1[(np.isnan(n_t_minus_1)) | np.isneginf(n_t_minus_1) | np.isinf(n_t_minus_1)] = 0
            prob_n_t_minus_1 = (n_t_minus_1.T / np.sum(n_t_minus_1, axis=1)).T

            # make empty array for innoculation sampling
            n_t_inoc = np.zeros((reps, len(init_abund_rel)))
            for rep in range(reps):
                # draw number cells,
                n_cells_transfer = np.random.binomial(n_cells_descendent, D_transfer) # number of cells that get transferred between descendent communities
                n_t_inoc[rep,:] = np.random.multinomial(n_cells_transfer, prob_n_t_minus_1[rep,:])

            q_t_inoc = np.log(n_t_inoc)

            if t > 11:
                m = np.zeros((reps, len(init_abund_rel)))

            else:
                # get migration vector
                if migration_treatment_ == 'global':

                    n_global = np.zeros(len(init_abund_rel))
                    for r in range(reps):
                        q_t_r = q[t,r,:]
                        n_t_r = np.exp(q_t_r)
                        n_t_r[n_t_r<0] = 0
                        #n_cells_migration = np.random.binomial(n_cells_descendent, 0.4/0.5)
                        n_cells_migration = np.random.binomial(n_cells_descendent, 0.004/0.5)
                        n_t_r_sample = np.random.multinomial(n_cells_migration, n_t_r/sum(n_t_r))
                        n_global += n_t_r_sample

                    # diluted 10,000 fold
                    #8*(10**-5)
                    n_cells_global_diluted = np.random.binomial(sum(n_global), 8*(10**-5))
                    # sample to get abundances of diluted global samples
                    n_global_diluted = np.random.multinomial(n_cells_global_diluted, n_global/sum(n_global))

                    # get number of cells transferred from diluted global sample
                    n_cells_migration_transfer = np.random.binomial(n_cells_global_diluted, 504/60000, size=reps)
                    m = np.asarray([np.random.multinomial(l, n_global_diluted/sum(n_global_diluted)) for l in n_cells_migration_transfer])

                elif migration_treatment_ == 'parent':
                    n_cells_migration = np.random.binomial(n_cells_parent, D_parent, size=reps)
                    m = np.asarray([np.random.multinomial(l, prob_mean_rel_abundances_parent) for l in n_cells_migration])

                else:
                    m = np.zeros((reps, len(init_abund_rel)))

            q[t,:,:] = discrete_slm_initial_condition_migration(q_t_inoc, m)


        n = np.exp(q)
        # replace negative q values with zero
        n[(np.isnan(n)) | np.isneginf(n) | np.isinf(n)] = 0

        return n


    n_migration = run_experiment(migration_treatment)
    n_no_migration = run_experiment('no_migration')

    # sample using distribution of read counts....

    #n_migration_reads = sample_simulation_results(n_migration)
    #n_no_migration_reads = sample_simulation_results(n_no_migration)

    if migration_treatment == 'global':
        migration_label = ('Global_migration',4)
    else:
        migration_label = ('Parent_migration',4)

    n_migration_reads = sample_simulation_results(n_migration, migration_label)
    print(n_no_migration)
    n_no_migration_reads = sample_simulation_results(n_no_migration, ('No_migration',4))

    init_abund_rel_reads = multinomial_sample_reads(init_abund_rel)
    init_abund_rel_reads_rel = init_abund_rel_reads/sum(init_abund_rel_reads)

    return n_migration_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel_reads_rel








def run_simulation_initial_condition_all_migration(sigma = 0.5, tau = 0.9, dt = 48, T = 864, reps = 93):
    # T = 126 generations
    # dt = 7 generations ==> T/dt = 18

    # T = 18*48hrs = 864
    # dt = 48 hrs.

    # just use units of generations for now
    # T = generations
    # tau units of number of generations
    # dt = gens. per-transfer
    #noise_term = np.sqrt( sigma*dt/tau  ) # compound paramter, for conveinance
    noise_term_per_generation = np.sqrt( sigma*dt/(tau*7) ) 

    # get parent mean relative abundances
    mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
    mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)

    # divide by sum to use as probability
    prob_mean_rel_abundances_parent = mean_rel_abundances_parent/sum(mean_rel_abundances_parent)
    richness_parent = len(prob_mean_rel_abundances_parent)
    # delta = % of species that survive, used to draw non zero carrying capacities
    #delta = 0.016
    delta = 56/richness_parent
    # draw from binomial distribution
    n_non_zero_k = np.random.binomial(richness_parent, delta)


    #def discrete_slm_initial_condition_migration(q_array, m):
    #    # q_array = array of replicate and species
    #    q_array_new = np.log(np.exp(q_array) + m)
    #    return q_array_new + (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))

    # multiple time steps corresponding to log2(1/0.008) ~= 7 generations
    def discrete_slm_initial_condition_migration(q_array, m):
       
        # q_array = array of replicate and species
        q_array_new = np.log(np.exp(q_array) + m)
        
        #for t in range(7):
        #    q_array_new += ((dt/7)*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term_per_generation * np.random.randn(reps, len(init_abund_rel)))

        # instead of for loop, we can generate (# reps, # ASVs, # generations) random matrix, sum over # generations
        std_normal_matrix = noise_term_per_generation * np.sum(np.random.randn(reps, len(init_abund_rel), 7), axis=2)
        # returns (# reps, # ASVs) matrix
        q_array_new += (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + std_normal_matrix

        return q_array_new


    n_time_steps = int(T / dt)  # Number of time steps.
    t_gen = np.arange(0, T, dt)

    # non zero carrying capacities
    k_to_keep = pln._rvs(mu_pln, sigma_pln, lower_trunc=True, size=n_non_zero_k)

    # randomize order
    k_to_keep = np.random.permutation(k_to_keep)

    # the probability that a species has a non-zero carrying capicity given its abundance in the parent community
    non_zero_k_weights = np.asarray([prob_non_zero_k(l) for l in np.log10(mean_rel_abundances_parent)])
    non_zero_k_prob = non_zero_k_weights/sum(non_zero_k_weights)
    idx_mean_rel_abundances_parent = np.arange(len(mean_rel_abundances_parent))

    # select index of parent abundances with non-zero k
    idx_non_zero_k = np.random.choice(idx_mean_rel_abundances_parent, size=n_non_zero_k, replace=False, p=non_zero_k_prob)

    # select index of parent abundances with zero k
    idx_zero_k = np.delete(idx_mean_rel_abundances_parent, idx_non_zero_k)

    # sort the initial abundances by non-zero/zero k status
    idx_non_zero_and_zero = np.concatenate([idx_non_zero_k, idx_zero_k])

    # sort parent relative abundances and probability
    init_abund_rel = mean_rel_abundances_parent[idx_non_zero_and_zero]
    prob_mean_rel_abundances_parent = prob_mean_rel_abundances_parent[idx_non_zero_and_zero]

    k_to_keep = np.append(k_to_keep, [0]*len(idx_zero_k))
    # turn into absolute abundances
    k_to_keep = np.random.multinomial(n_cells_descendent, k_to_keep/sum(k_to_keep))

    def run_experiment(migration_treatment_):

        # create three dimensional vector
        # spcies, replicate, and timestep
        # create array for innnocumum
        n_0_inoc = np.zeros((reps, len(init_abund_rel)))
        for rep in range(reps):
            n_cells_inoc = np.random.binomial(n_cells_parent, D_transfer) # number of cells that get transferred between descendent communities
            n_0_inoc[rep,:] = np.random.multinomial(n_cells_inoc, init_abund_rel)

        q_0_inoc = np.log(n_0_inoc)
        # migration only happens at transfers, no migration when experiment is set up
        m_0 = np.zeros((reps, len(init_abund_rel)))
        q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
        # transfer 1
        q[0,:,:] = discrete_slm_initial_condition_migration(q_0_inoc, m_0)
        # transfer 2 to 18
        for t in range(1, n_time_steps): # t = transfer number
            # perform transfer
            n_t_minus_1 = np.exp(q[t,:,:])
            # remove infs, nan
            n_t_minus_1[(np.isnan(n_t_minus_1)) | np.isneginf(n_t_minus_1) | np.isinf(n_t_minus_1)] = 0
            prob_n_t_minus_1 = (n_t_minus_1.T / np.sum(n_t_minus_1, axis=1)).T

            # make empty array for innoculation sampling
            n_t_inoc = np.zeros((reps, len(init_abund_rel)))
            for rep in range(reps):
                # draw number cells,
                n_cells_transfer = np.random.binomial(n_cells_descendent, D_transfer) # number of cells that get transferred between descendent communities
                n_t_inoc[rep,:] = np.random.multinomial(n_cells_transfer, prob_n_t_minus_1[rep,:])

            q_t_inoc = np.log(n_t_inoc)
            
            # starts at transfer 12
            if t > 11:
                m = np.zeros((reps, len(init_abund_rel)))

            else:
                # get migration vector
                if migration_treatment_ == 'global':

                    n_global = np.zeros(len(init_abund_rel))
                    for r in range(reps):
                        q_t_r = q[t,r,:]
                        n_t_r = np.exp(q_t_r)
                        n_t_r[n_t_r<0] = 0
                        n_cells_migration = np.random.binomial(n_cells_descendent, 0.004/0.5)
                        n_t_r_sample = np.random.multinomial(n_cells_migration, n_t_r/sum(n_t_r))
                        n_global += n_t_r_sample

                    # diluted 10,000 fold
                    #8*(10**-5)
                    n_cells_global_diluted = np.random.binomial(sum(n_global), 8*(10**-5))
                    # sample to get abundances of diluted global samples
                    n_global_diluted = np.random.multinomial(n_cells_global_diluted, n_global/sum(n_global))

                    # get number of cells transferred from diluted global sample
                    n_cells_migration_transfer = np.random.binomial(n_cells_global_diluted, 504/60000, size=reps)
                    m = np.asarray([np.random.multinomial(l, n_global_diluted/sum(n_global_diluted)) for l in n_cells_migration_transfer])


                elif migration_treatment_ == 'parent':
                    n_cells_migration = np.random.binomial(n_cells_parent, D_parent, size=reps)
                    m = np.asarray([np.random.multinomial(l, prob_mean_rel_abundances_parent) for l in n_cells_migration])

                else:
                    m = np.zeros((reps, len(init_abund_rel)))
            
            q[t,:,:] = discrete_slm_initial_condition_migration(q_t_inoc, m)



        n = np.exp(q)
        # replace negative q values with zero
        n[(np.isnan(n)) | np.isneginf(n) | np.isinf(n)] = 0

        return n


    n_migration_global = run_experiment('global')
    n_migration_parent = run_experiment('parent')
    n_no_migration = run_experiment('no_migration')

    # sample using distribution of read counts....
    n_migration_global_reads = sample_simulation_results(n_migration_global, ('Global_migration',4))
    n_migration_parent_reads = sample_simulation_results(n_migration_parent, ('Parent_migration',4))
    n_no_migration_reads = sample_simulation_results(n_no_migration, ('No_migration',4))

    init_abund_rel_reads = multinomial_sample_reads(init_abund_rel)
    init_abund_rel_reads_rel = init_abund_rel_reads/sum(init_abund_rel_reads)

    return n_migration_global_reads, n_migration_parent_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel_reads_rel







def multinomial_sample_reads(array_):

    n_reads = utils.get_dist_read_counts()
    n_reads_ = np.random.choice(n_reads, size=1, replace=True, p=None)[0]

    rel_array_ = array_/sum(array_)

    rel_array_reads = np.random.multinomial(n_reads_, rel_array_)

    return rel_array_reads




def sample_simulation_results_random(s_by_s_by_t):

    n_reads = utils.get_dist_read_counts()

    n_trasnfers, n_replicates, n_species = s_by_s_by_t.shape

    t_by_s_by_s_reads = np.zeros((n_trasnfers, n_replicates, n_species))

    for t in range(n_trasnfers):

        n_reads_t = np.random.choice(n_reads, size=n_replicates, replace=True, p=None)

        for n_reads_t_i_idx, n_reads_t_i in enumerate(n_reads_t):

            sad = s_by_s_by_t[t,n_reads_t_i_idx,:]
            rel_sad = sad/sum(sad)

            t_by_s_by_s_reads[t,n_reads_t_i_idx,:] = np.random.multinomial(int(n_reads_t_i), rel_sad)

    return t_by_s_by_s_reads




def sample_simulation_results(s_by_s_by_t, treatment):

    #n_reads = utils.get_dist_read_counts()

    n_trasnfers, n_replicates, n_species = s_by_s_by_t.shape
    s_by_s_by_t_reads = np.zeros((n_trasnfers, n_replicates, n_species))



    for t in range(n_trasnfers):
        
        # skip if communities weren't sampled at that time point
        if t not in rep_number_and_read_count_dict[treatment]:
            continue

        #n_reads_t = np.random.choice(n_reads, size=n_replicates, replace=True, p=None)
        n_reads_t = rep_number_and_read_count_dict[treatment][t]['n_reads']
        # convert to idx
        sad_idx_t = rep_number_and_read_count_dict[treatment][t]['community_reps'] - 1
        #s_by_s_by_t_subset = s_by_s_by_t[t,sad_idx_t,:]

        #for n_reads_t_i_idx, n_reads_t_i in enumerate(n_reads_t):
        # loop through indices
        for sad_idx_t_i_idx, sad_idx_t_i in enumerate(sad_idx_t):

            #sad = s_by_s_by_t[t,n_reads_t_i_idx,:]
            # subset s_by_s using index of community sampled in the experiment
            sad = s_by_s_by_t[t, sad_idx_t_i,:]
            rel_sad = sad/sum(sad)
            # select the corresponding total read count (sampling depth)

            #if sum((rel_sad<0) | (rel_sad>1) | np.isnan(rel_sad)) > 0:

            s_by_s_by_t_reads[t,sad_idx_t_i,:] = np.random.multinomial(int(n_reads_t[sad_idx_t_i_idx]), rel_sad)

    return s_by_s_by_t_reads





def run_simulation_parent_rho_abc(n_iter=10000, tau=None, sigma=None, return_dict=False):

    if tau == None:
        tau_all = (np.random.uniform(tau_min, tau_max, size=n_iter*100)).tolist()
    else:
        tau_all = [tau]*(n_iter*100)

    if sigma == None:
        sigma_all = (10**(np.random.uniform(np.log10(sigma_min), np.log10(sigma_max), size=n_iter*100))).tolist()
    else:
        sigma_all = [sigma]*(n_iter*100)


    transfers = [11, 17]

    rho_dict = {}
    rho_dict['tau_all'] = []
    rho_dict['sigma_all'] = []

    rho_dict['rho_12_vs_18'] = {}
    rho_dict['rho_12_vs_18']['rho_12'] = []
    rho_dict['rho_12_vs_18']['rho_18'] = []
    rho_dict['rho_12_vs_18']['Z'] = []


    rho_dict['slope_12_vs_18'] = {}
    
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_18'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_18'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_rho_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_rho_18'] = []

    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_18'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_18'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_18'] = []

    rho_dict['slope_12_vs_18']['mad_slope_12'] = []
    rho_dict['slope_12_vs_18']['mad_slope_18'] = []
    rho_dict['slope_12_vs_18']['mad_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_12'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_18'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['mad_rho_12'] = []
    rho_dict['slope_12_vs_18']['mad_rho_18'] = []

    for t in transfers:
        rho_dict[t] = {}
        rho_dict[t]['slope_migration_vs_parent'] = []
        rho_dict[t]['slope_no_migration_vs_parent'] = []
        rho_dict[t]['slope_mad_ratio_vs_parent'] = []

        rho_dict[t]['rho_migration'] = []
        rho_dict[t]['rho_migration_vs_parent'] = []
        rho_dict[t]['rho_no_migration_vs_parent'] = []
        rho_dict[t]['rho_mad_ratio_vs_parent'] = []

    if return_dict == False:
        sys.stderr.write("Starting ABC simulation for parent migration treatment...\n")

    # run ABC 
    #for i in range(n_iter):
    while len(rho_dict['tau_all']) < n_iter:
        
        n_iter_successful = len(rho_dict['tau_all'])

        if (n_iter_successful+1) % 1000 == 0:
            print(n_iter_successful+1)

        tau_i = tau_all.pop()
        sigma_i = sigma_all.pop()

        n_migration_global_reads, n_migration_parent_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)
        s_by_s_migration = n_migration_parent_reads
        s_by_s_no_migration = n_no_migration_reads

        s_by_s_dict = {}
        for t in transfers:

            s_by_s_migration_t = s_by_s_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            community_reps_migration_t_idx = rep_number_and_read_count_dict[('Parent_migration',4)][t]['community_reps_idx']
            community_reps_no_migration_t_idx = rep_number_and_read_count_dict[('No_migration',4)][t]['community_reps_idx']
        
            # select communities actually sampled
            s_by_s_migration_t = s_by_s_migration_t[community_reps_migration_t_idx,:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[community_reps_no_migration_t_idx,:]

            rel_s_by_s_migration_t = s_by_s_migration_t.T/s_by_s_migration_t.sum(axis=1)
            rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

            idx_migration = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t == 0, axis=1))
            idx_migration_vs_parent = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (init_abund_rel>0)
            idx_no_migration_vs_parent = (~np.all(rel_s_by_s_no_migration_t == 0, axis=1)) &  (init_abund_rel>0)

            idx_ratio_vs_parent = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t == 0, axis=1)) &  (init_abund_rel>0)

            s_by_s_dict[t] = {}
            s_by_s_dict[t]['rel_s_by_s_migration_t'] = rel_s_by_s_migration_t
            s_by_s_dict[t]['rel_s_by_s_no_migration_t'] = rel_s_by_s_no_migration_t

            s_by_s_dict[t]['idx_migration'] = idx_migration
            s_by_s_dict[t]['idx_migration_vs_parent'] = idx_migration_vs_parent
            s_by_s_dict[t]['idx_no_migration_vs_parent'] = idx_no_migration_vs_parent
            s_by_s_dict[t]['idx_ratio_vs_parent'] = idx_ratio_vs_parent


        # at least give species present in both treatments
        if (sum(s_by_s_dict[transfers[0]]['idx_ratio_vs_parent']) < 5) or sum(s_by_s_dict[transfers[1]]['idx_ratio_vs_parent']) < 5:
            continue

        else:


            mad_dict = {}
            mad_ratio_dict = {}

            for t in transfers:

                rel_s_by_s_migration_t = s_by_s_dict[t]['rel_s_by_s_migration_t']
                rel_s_by_s_no_migration_t = s_by_s_dict[t]['rel_s_by_s_no_migration_t']

                idx_migration = s_by_s_dict[t]['idx_migration']
                idx_migration_vs_parent = s_by_s_dict[t]['idx_migration_vs_parent']
                idx_no_migration_vs_parent = s_by_s_dict[t]['idx_no_migration_vs_parent']
                idx_ratio_vs_parent = s_by_s_dict[t]['idx_ratio_vs_parent']

                mean_rel_migration_t = np.mean(rel_s_by_s_migration_t[idx_migration,:], axis=1)
                mean_rel_no_migration_t = np.mean(rel_s_by_s_no_migration_t[idx_migration,:], axis=1)

                mean_rel_migration_vs_parent_t = np.mean(rel_s_by_s_migration_t[idx_migration_vs_parent,:], axis=1)
                mean_rel_no_migration_vs_parent_t = np.mean(rel_s_by_s_no_migration_t[idx_no_migration_vs_parent,:], axis=1)

                init_abund_rel_migration_vs_parent_t = init_abund_rel[idx_migration_vs_parent]
                init_abund_rel_no_migration_vs_parent_t = init_abund_rel[idx_no_migration_vs_parent]


                log10_mean_rel_migration_t = np.log10(mean_rel_migration_t)
                log10_mean_rel_no_migration_t = np.log10(mean_rel_no_migration_t)

                mad_dict[t] = {}
                mad_dict[t]['log10_mean_rel_migration_t'] = log10_mean_rel_migration_t
                mad_dict[t]['log10_mean_rel_no_migration_t'] = log10_mean_rel_no_migration_t

                # use MAD ratio
                mad_ratio = np.mean(rel_s_by_s_migration_t[idx_ratio_vs_parent,:], axis=1) / np.mean(rel_s_by_s_no_migration_t[idx_ratio_vs_parent,:], axis=1)
                init_abund_mad_ratio_vs_parent_t = init_abund_rel[idx_ratio_vs_parent]

                log10_init_abund_rel_migration_vs_parent_t = np.log10(init_abund_rel_migration_vs_parent_t)
                log10_mean_rel_migration_vs_parent_t = np.log10(mean_rel_migration_vs_parent_t)
                log10_init_abund_rel_no_migration_vs_parent_t = np.log10(init_abund_rel_no_migration_vs_parent_t)
                log10_mean_rel_no_migration_vs_parent_t = np.log10(mean_rel_no_migration_vs_parent_t)
                log10_init_abund_mad_ratio_vs_parent_t = np.log10(init_abund_mad_ratio_vs_parent_t)
                log10_mad_ratio = np.log10(mad_ratio)
                
                mad_ratio_dict[t] = {}
                mad_ratio_dict[t]['log10_init_abund_rel_migration_vs_parent_t'] = log10_init_abund_rel_migration_vs_parent_t
                mad_ratio_dict[t]['log10_mean_rel_migration_vs_parent_t'] = log10_mean_rel_migration_vs_parent_t
                
                mad_ratio_dict[t]['log10_init_abund_rel_no_migration_vs_parent_t'] = log10_init_abund_rel_no_migration_vs_parent_t
                mad_ratio_dict[t]['log10_mean_rel_no_migration_vs_parent_t'] = log10_mean_rel_no_migration_vs_parent_t
                
                mad_ratio_dict[t]['log10_init_abund_mad_ratio_vs_parent_t'] = log10_init_abund_mad_ratio_vs_parent_t
                mad_ratio_dict[t]['log10_mad_ratio'] = log10_mad_ratio


            rho_dict['tau_all'].append(tau_i)
            rho_dict['sigma_all'].append(sigma_i)

            rho_parent_vs_no_18, rho_parent_vs_no_12, z_parent = utils.compare_rho_fisher_z(mad_dict[17]['log10_mean_rel_migration_t'], mad_dict[17]['log10_mean_rel_no_migration_t'], mad_dict[11]['log10_mean_rel_migration_t'], mad_dict[11]['log10_mean_rel_no_migration_t'])
            
            rho_dict['rho_12_vs_18']['rho_12'].append(rho_parent_vs_no_12)
            rho_dict['rho_12_vs_18']['rho_18'].append(rho_parent_vs_no_18)
            rho_dict['rho_12_vs_18']['Z'].append(z_parent)

            #########
            # t-tests
            #########
            
            slope_migration_18, slope_migration_12, t_slope_migration, intercept_migration_18, intercept_migration_12, t_intercept_migration, rho_migration_18, rho_migration_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_rel_migration_vs_parent_t'], mad_ratio_dict[17]['log10_mean_rel_migration_vs_parent_t'], mad_ratio_dict[11]['log10_init_abund_rel_migration_vs_parent_t'], mad_ratio_dict[11]['log10_mean_rel_migration_vs_parent_t'])
            slope_no_migration_18, slope_no_migration_12, t_slope_no_migration, intercept_no_migration_18, intercept_no_migration_12, t_intercept_no_migration, rho_no_migration_18, rho_no_migration_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_rel_no_migration_vs_parent_t'], mad_ratio_dict[17]['log10_mean_rel_no_migration_vs_parent_t'], mad_ratio_dict[11]['log10_init_abund_rel_no_migration_vs_parent_t'], mad_ratio_dict[11]['log10_mean_rel_no_migration_vs_parent_t'])
            slope_mad_18, slope_mad_12, t_slope_mad, intercept_mad_18, intercept_mad_12, t_intercept_mad, rho_mad_18, rho_mad_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_mad_ratio_vs_parent_t'], mad_ratio_dict[17]['log10_mad_ratio'], mad_ratio_dict[11]['log10_init_abund_mad_ratio_vs_parent_t'], mad_ratio_dict[11]['log10_mad_ratio'])

            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_18'].append(slope_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_12'].append(slope_migration_12)
            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_t_test'].append(t_slope_migration)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_18'].append(intercept_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_12'].append(intercept_migration_12)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_t_test'].append(t_intercept_migration)
            rho_dict['slope_12_vs_18']['migration_vs_parent_rho_18'].append(rho_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_rho_12'].append(rho_migration_12)

            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_18'].append(slope_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_12'].append(slope_no_migration_12)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_t_test'].append(t_slope_no_migration)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_18'].append(intercept_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_12'].append(intercept_no_migration_12)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_t_test'].append(t_intercept_no_migration)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_18'].append(rho_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_12'].append(rho_no_migration_12)

            rho_dict['slope_12_vs_18']['mad_slope_18'].append(slope_mad_18)
            rho_dict['slope_12_vs_18']['mad_slope_12'].append(slope_mad_12)
            rho_dict['slope_12_vs_18']['mad_slope_t_test'].append(t_slope_mad)
            rho_dict['slope_12_vs_18']['mad_intercept_18'].append(intercept_mad_18)
            rho_dict['slope_12_vs_18']['mad_intercept_12'].append(intercept_mad_12)
            rho_dict['slope_12_vs_18']['mad_intercept_t_test'].append(t_intercept_mad)
            rho_dict['slope_12_vs_18']['mad_rho_18'].append(rho_mad_18)
            rho_dict['slope_12_vs_18']['mad_rho_12'].append(rho_mad_12)


    if return_dict == True:
        
        return rho_dict

    else:

        if tau != None:
            path_ = simulation_parent_rho_fixed_parameters_path

        else:
            path_ = simulation_parent_rho_abc_path

        sys.stderr.write("Saving dictionary...\n")
        with open(path_, 'wb') as handle:
            pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)








def run_simulation_parent_rho_fixed_parameters(tau_i, sigma_i, n_iter=1000):

    transfers = [11, 17]

    #tau_all = np.linspace(1.7, 6.9, num=n_iter, endpoint=True)
    #sigma_all = np.logspace(np.log10(0.01), np.log10(1.9), num=n_iter, endpoint=True, base=10.0)

    rho_dict = {}

    rho_dict['tau_all'] = tau_i
    rho_dict['sigma_all'] = sigma_i

    rho_dict['rho_12_vs_18'] = {}
    rho_dict['rho_12_vs_18']['rho_12'] = []
    rho_dict['rho_12_vs_18']['rho_18'] = []
    rho_dict['rho_12_vs_18']['Z'] = []


    rho_dict['slope_12_vs_18'] = {}
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_18'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_18'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_rho_12'] = []
    rho_dict['slope_12_vs_18']['migration_vs_parent_rho_18'] = []

    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_18'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_18'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_12'] = []
    rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_18'] = []

    rho_dict['slope_12_vs_18']['mad_slope_12'] = []
    rho_dict['slope_12_vs_18']['mad_slope_18'] = []
    rho_dict['slope_12_vs_18']['mad_slope_t_test'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_12'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_18'] = []
    rho_dict['slope_12_vs_18']['mad_intercept_t_test'] = []
    rho_dict['slope_12_vs_18']['mad_rho_12'] = []
    rho_dict['slope_12_vs_18']['mad_rho_18'] = []

    for t in transfers:
        rho_dict[t] = {}
        rho_dict[t]['slope_migration_vs_parent'] = []
        rho_dict[t]['slope_no_migration_vs_parent'] = []
        rho_dict[t]['slope_mad_ratio_vs_parent'] = []

        rho_dict[t]['rho_migration'] = []
        rho_dict[t]['rho_migration_vs_parent'] = []
        rho_dict[t]['rho_no_migration_vs_parent'] = []
        rho_dict[t]['rho_mad_ratio_vs_parent'] = []



    # run ABC 
    #for i in range(n_iter):
    while len(rho_dict['rho_12_vs_18']['Z']) < n_iter:
        
        n_iter_successful = len(rho_dict['rho_12_vs_18']['Z'])

        if (n_iter_successful+1) % 1000 == 0:
            print(n_iter_successful+1)

        s_by_s_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_migration(sigma = sigma_i, tau = tau_i, migration_treatment='parent')

        s_by_s_dict = {}
        for t in transfers:

            s_by_s_migration_t = s_by_s_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            rel_s_by_s_migration_t = s_by_s_migration_t.T/s_by_s_migration_t.sum(axis=1)
            rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

            idx_migration = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t == 0, axis=1))
            idx_migration_vs_parent = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (init_abund_rel>0)
            idx_no_migration_vs_parent = (~np.all(rel_s_by_s_no_migration_t == 0, axis=1)) &  (init_abund_rel>0)

            idx_ratio_vs_parent = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t == 0, axis=1)) &  (init_abund_rel>0)

            s_by_s_dict[t] = {}
            s_by_s_dict[t]['rel_s_by_s_migration_t'] = rel_s_by_s_migration_t
            s_by_s_dict[t]['rel_s_by_s_no_migration_t'] = rel_s_by_s_no_migration_t

            s_by_s_dict[t]['idx_migration'] = idx_migration
            s_by_s_dict[t]['idx_migration_vs_parent'] = idx_migration_vs_parent
            s_by_s_dict[t]['idx_no_migration_vs_parent'] = idx_no_migration_vs_parent
            s_by_s_dict[t]['idx_ratio_vs_parent'] = idx_ratio_vs_parent


        if (sum(s_by_s_dict[transfers[0]]['idx_ratio_vs_parent']) < 5) or sum(s_by_s_dict[transfers[1]]['idx_ratio_vs_parent']) < 5:
            continue

        else:


            mad_dict = {}
            mad_ratio_dict = {}

            for t in transfers:

                rel_s_by_s_migration_t = s_by_s_dict[t]['rel_s_by_s_migration_t']
                rel_s_by_s_no_migration_t = s_by_s_dict[t]['rel_s_by_s_no_migration_t']

                idx_migration = s_by_s_dict[t]['idx_migration']
                idx_migration_vs_parent = s_by_s_dict[t]['idx_migration_vs_parent']
                idx_no_migration_vs_parent = s_by_s_dict[t]['idx_no_migration_vs_parent']
                idx_ratio_vs_parent = s_by_s_dict[t]['idx_ratio_vs_parent']

                mean_rel_migration_t = np.mean(rel_s_by_s_migration_t[idx_migration,:], axis=1)
                mean_rel_no_migration_t = np.mean(rel_s_by_s_no_migration_t[idx_migration,:], axis=1)

                mean_rel_migration_vs_parent_t = np.mean(rel_s_by_s_migration_t[idx_migration_vs_parent,:], axis=1)
                mean_rel_no_migration_vs_parent_t = np.mean(rel_s_by_s_no_migration_t[idx_no_migration_vs_parent,:], axis=1)

                init_abund_rel_migration_vs_parent_t = init_abund_rel[idx_migration_vs_parent]
                init_abund_rel_no_migration_vs_parent_t = init_abund_rel[idx_no_migration_vs_parent]


                log10_mean_rel_migration_t = np.log10(mean_rel_migration_t)
                log10_mean_rel_no_migration_t = np.log10(mean_rel_no_migration_t)

                mad_dict[t] = {}
                mad_dict[t]['log10_mean_rel_migration_t'] = log10_mean_rel_migration_t
                mad_dict[t]['log10_mean_rel_no_migration_t'] = log10_mean_rel_no_migration_t


                # use MAD ratio
                mad_ratio = np.mean(rel_s_by_s_migration_t[idx_ratio_vs_parent,:], axis=1) / np.mean(rel_s_by_s_no_migration_t[idx_ratio_vs_parent,:], axis=1)
                init_abund_mad_ratio_vs_parent_t = init_abund_rel[idx_ratio_vs_parent]

                log10_init_abund_rel_migration_vs_parent_t = np.log10(init_abund_rel_migration_vs_parent_t)
                log10_mean_rel_migration_vs_parent_t = np.log10(mean_rel_migration_vs_parent_t)
                log10_init_abund_rel_no_migration_vs_parent_t = np.log10(init_abund_rel_no_migration_vs_parent_t)
                log10_mean_rel_no_migration_vs_parent_t = np.log10(mean_rel_no_migration_vs_parent_t)
                log10_init_abund_mad_ratio_vs_parent_t = np.log10(init_abund_mad_ratio_vs_parent_t)
                log10_mad_ratio = np.log10(mad_ratio)
                
                mad_ratio_dict[t] = {}
                mad_ratio_dict[t]['log10_init_abund_rel_migration_vs_parent_t'] = log10_init_abund_rel_migration_vs_parent_t
                mad_ratio_dict[t]['log10_mean_rel_migration_vs_parent_t'] = log10_mean_rel_migration_vs_parent_t
                
                mad_ratio_dict[t]['log10_init_abund_rel_no_migration_vs_parent_t'] = log10_init_abund_rel_no_migration_vs_parent_t
                mad_ratio_dict[t]['log10_mean_rel_no_migration_vs_parent_t'] = log10_mean_rel_no_migration_vs_parent_t
                
                mad_ratio_dict[t]['log10_init_abund_mad_ratio_vs_parent_t'] = log10_init_abund_mad_ratio_vs_parent_t
                mad_ratio_dict[t]['log10_mad_ratio'] = log10_mad_ratio


            print(mad_dict[11]['log10_mean_rel_migration_t'], mad_dict[11]['log10_mean_rel_no_migration_t'])
            rho_parent_vs_no_18, rho_parent_vs_no_12, z_parent = utils.compare_rho_fisher_z(mad_dict[17]['log10_mean_rel_migration_t'], mad_dict[17]['log10_mean_rel_no_migration_t'], mad_dict[11]['log10_mean_rel_migration_t'], mad_dict[11]['log10_mean_rel_no_migration_t'])
            
            print(z_parent)
            rho_dict['rho_12_vs_18']['rho_12'].append(rho_parent_vs_no_12)
            rho_dict['rho_12_vs_18']['rho_18'].append(rho_parent_vs_no_18)
            rho_dict['rho_12_vs_18']['Z'].append(z_parent)

            #########
            # t-tests
            #########
            
            slope_migration_18, slope_migration_12, t_slope_migration, intercept_migration_18, intercept_migration_12, t_intercept_migration, rho_migration_18, rho_migration_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_rel_migration_vs_parent_t'], mad_ratio_dict[17]['log10_mean_rel_migration_vs_parent_t'], mad_ratio_dict[11]['log10_init_abund_rel_migration_vs_parent_t'], mad_ratio_dict[11]['log10_mean_rel_migration_vs_parent_t'])
            slope_no_migration_18, slope_no_migration_12, t_slope_no_migration, intercept_no_migration_18, intercept_no_migration_12, t_intercept_no_migration, rho_no_migration_18, rho_no_migration_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_rel_no_migration_vs_parent_t'], mad_ratio_dict[17]['log10_mean_rel_no_migration_vs_parent_t'], mad_ratio_dict[11]['log10_init_abund_rel_no_migration_vs_parent_t'], mad_ratio_dict[11]['log10_mean_rel_no_migration_vs_parent_t'])
            slope_mad_18, slope_mad_12, t_slope_mad, intercept_mad_18, intercept_mad_12, t_intercept_mad, rho_mad_18, rho_mad_12 = utils.t_statistic_two_slopes(mad_ratio_dict[17]['log10_init_abund_mad_ratio_vs_parent_t'], mad_ratio_dict[17]['log10_mad_ratio'], mad_ratio_dict[11]['log10_init_abund_mad_ratio_vs_parent_t'], mad_ratio_dict[11]['log10_mad_ratio'])

            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_18'].append(slope_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_12'].append(slope_migration_12)
            rho_dict['slope_12_vs_18']['migration_vs_parent_slope_t_test'].append(t_slope_migration)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_18'].append(intercept_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_12'].append(intercept_migration_12)
            rho_dict['slope_12_vs_18']['migration_vs_parent_intercept_t_test'].append(t_intercept_migration)
            rho_dict['slope_12_vs_18']['migration_vs_parent_rho_18'].append(rho_migration_18)
            rho_dict['slope_12_vs_18']['migration_vs_parent_rho_12'].append(rho_migration_12)

            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_18'].append(slope_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_12'].append(slope_no_migration_12)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_slope_t_test'].append(t_slope_no_migration)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_18'].append(intercept_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_12'].append(intercept_no_migration_12)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_intercept_t_test'].append(t_intercept_no_migration)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_18'].append(rho_no_migration_18)
            rho_dict['slope_12_vs_18']['no_migration_vs_parent_rho_12'].append(rho_no_migration_12)

            rho_dict['slope_12_vs_18']['mad_slope_18'].append(slope_mad_18)
            rho_dict['slope_12_vs_18']['mad_slope_12'].append(slope_mad_12)
            rho_dict['slope_12_vs_18']['mad_slope_t_test'].append(t_slope_mad)
            rho_dict['slope_12_vs_18']['mad_intercept_18'].append(intercept_mad_18)
            rho_dict['slope_12_vs_18']['mad_intercept_12'].append(intercept_mad_12)
            rho_dict['slope_12_vs_18']['mad_intercept_t_test'].append(t_intercept_mad)
            rho_dict['slope_12_vs_18']['mad_rho_18'].append(rho_mad_18)
            rho_dict['slope_12_vs_18']['mad_rho_12'].append(rho_mad_12)


    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_parent_rho_fixed_parameters_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def run_simulation_parent_rho_heatmap(n_iter=100):

    sys.stderr.write("Simulating parent migration treatments across parameter combinations...\n")

    transfers = [11, 17]
    rho_dict = {}
    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:

            print(tau_i, sigma_i)

            rho_dict[tau_i][sigma_i] = {}
            rho_dict[tau_i][sigma_i]['rho_12_vs_18'] = {}

            rho_12_vs_18_stats = ['rho_12', 'rho_18', 'Z']
            for r in rho_12_vs_18_stats:
                rho_dict[tau_i][sigma_i]['rho_12_vs_18'][r] = []


            rho_dict[tau_i][sigma_i]['slope_12_vs_18'] = {}        
            
            slope_12_vs_18_stats = ['migration_vs_parent_slope_12', 'migration_vs_parent_slope_18', \
                                    'migration_vs_parent_slope_t_test', 'migration_vs_parent_intercept_12', \
                                        'migration_vs_parent_intercept_18', 'migration_vs_parent_intercept_t_test', \
                                        'migration_vs_parent_rho_12', 'migration_vs_parent_rho_18', \
                                        'no_migration_vs_parent_slope_12', 'no_migration_vs_parent_slope_18', \
                                        'no_migration_vs_parent_slope_t_test', 'no_migration_vs_parent_intercept_12', \
                                        'no_migration_vs_parent_intercept_18', 'no_migration_vs_parent_intercept_t_test', \
                                        'no_migration_vs_parent_rho_12', 'no_migration_vs_parent_rho_18', \
                                        'mad_slope_12', 'mad_slope_18', 'mad_slope_t_test', 'mad_intercept_12', \
                                        'mad_intercept_18', 'mad_intercept_t_test', 'mad_rho_12', 'mad_rho_18']

            for r in slope_12_vs_18_stats:
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][r] = []


            for t in transfers:
                rho_dict[tau_i][sigma_i][t] = {}
                
                rho_dict[tau_i][sigma_i][t]['slope_migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['slope_no_migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['slope_mad_ratio_vs_parent'] = []

                rho_dict[tau_i][sigma_i][t]['rho_migration'] = []
                rho_dict[tau_i][sigma_i][t]['rho_migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['rho_no_migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['rho_mad_ratio_vs_parent'] = []


            #for i in range(iter):
            #while len(rho_dict[tau_i][sigma_i][transfers[0]]['rho_migration']) < iter:
            while len(rho_dict[tau_i][sigma_i]['slope_12_vs_18']['mad_slope_t_test']) < n_iter:

                try:
                    rho_dict_iter = run_simulation_parent_rho_abc(n_iter = 1, tau=tau_i, sigma=sigma_i, return_dict = True)
                except ValueError:
                    rho_dict_iter = None
                
                if rho_dict_iter == None:
                    continue

                if len(rho_dict_iter['rho_12_vs_18']['Z']) == 0:
                    continue

                for r in rho_12_vs_18_stats:     
                    rho_dict[tau_i][sigma_i]['rho_12_vs_18'][r].append(rho_dict_iter['rho_12_vs_18'][r][0])

                #########
                # t-tests
                #########

                for r in slope_12_vs_18_stats:     
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][r].append(rho_dict_iter['slope_12_vs_18'][r][0])
                

            # for transfer in...
            #meal_delta_rho = np.mean(np.asarray(rho_dict[tau_i][sigma_i][17]['rho_migration']) - np.asarray(rho_dict[tau_i][sigma_i][11]['rho_migration']))



    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_parent_rho_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





def run_simulation_global_rho_heatmap(n_iter=100):

    sys.stderr.write("Simulating global migration treatments across parameter combinations...\n")

    # run the whole range of transfers since we have this data for global migration
    transfers = range(18)

    rho_dict = {}

    #for tau_i in tau_all:
    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:

            print(tau_i, sigma_i)

            rho_dict[tau_i][sigma_i] = {}
            rho_dict[tau_i][sigma_i]['ratio_stats'] = {}
            rho_dict[tau_i][sigma_i]['per_transfer_stats'] = {}

            rho_dict[tau_i][sigma_i]['z_rho'] = {}
            rho_dict[tau_i][sigma_i]['z_rho']['mean_log10'] = {}
            rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['rho_mean_12'] = []
            rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['rho_mean_18'] = []
            rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['z_mean'] = []

            rho_dict[tau_i][sigma_i]['z_rho']['cv_log10'] = {}
            rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['rho_cv_12'] = []
            rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['rho_cv_18'] = []
            rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['z_cv'] = []

            for migration_status in ['global_migration', 'no_migration']:

                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status] = {}

                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_log_ratio_per_transfer'] = []
                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['cv_log_ratio'] = []

                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio'] = []

                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_mean'] = []
                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_cv'] = []
                
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_t_stat'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['pooled_t_stat'] = []

                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_before'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_after'] = []
                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_delta_cv_log_ratio'] = []


            if tau_i in max_sigma_dict:
                if sigma_i > max_sigma_dict[tau_i]:
                    continue


            if sigma_i > 1.060617421311563:
                continue


            for t in transfers:

                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t] = {}
                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['mean_rho'] = []
                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['cv_rho'] = []


            n_attempts = 0
            while (len(rho_dict[tau_i][sigma_i]['per_transfer_stats'][transfers[0]]['mean_rho']) < n_iter) and (n_attempts < 10*n_iter):

                try:
                    rho_dict_iter = run_simulation_global_rho_abc(n_iter = 1, tau=tau_i, sigma=sigma_i, return_dict = True)
                except ValueError:
                    continue
                
                if rho_dict_iter == None:
                    continue

                if len(rho_dict_iter['z_rho']['cv_log10']['z_cv']) == 0:
                    continue


                n_attempts+=1

                # z-test
                rho_mean_12 = rho_dict_iter['z_rho']['mean_log10']['rho_mean_12'][0]
                rho_mean_18 = rho_dict_iter['z_rho']['mean_log10']['rho_mean_18'][0]
                z_mean = rho_dict_iter['z_rho']['mean_log10']['z_mean'][0]
                rho_cv_12 = rho_dict_iter['z_rho']['cv_log10']['rho_cv_12'][0]
                rho_cv_18 = rho_dict_iter['z_rho']['cv_log10']['rho_cv_18'][0]
                z_cv = rho_dict_iter['z_rho']['cv_log10']['z_cv'][0]

                # skip nan
                if (np.isnan(z_mean) == True) or (np.isnan(z_cv) == True):
                    continue
                
                missing_t_test = False
                for migration_treatment in ['rel_s_by_s_migration_t', 'rel_s_by_s_no_migration_t']:

                    if migration_treatment == 'rel_s_by_s_migration_t':
                        migration_status = 'global_migration'
                    else:
                        migration_status = 'no_migration'

                    if len(rho_dict_iter['ratio_stats'][migration_status]['mean_t_stat']) == 0:
                        missing_t_test = True

                    if len(rho_dict_iter['ratio_stats'][migration_status]['pooled_t_stat']) == 0:
                        missing_t_test = True

                if missing_t_test == True:
                    continue


                if len(rho_dict[tau_i][sigma_i]['per_transfer_stats'][transfers[0]]['mean_rho']) % 10 == 0:
                    print(len(rho_dict[tau_i][sigma_i]['per_transfer_stats'][transfers[0]]['mean_rho']))


                rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['rho_mean_12'].append(rho_mean_12)
                rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['rho_mean_18'].append(rho_mean_18)
                rho_dict[tau_i][sigma_i]['z_rho']['mean_log10']['z_mean'].append(z_mean)

                rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['rho_cv_12'].append(rho_cv_12)
                rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['rho_cv_18'].append(rho_cv_18)
                rho_dict[tau_i][sigma_i]['z_rho']['cv_log10']['z_cv'].append(z_cv)

                for t in transfers:

                    rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['mean_rho'].append(rho_dict_iter['per_transfer_stats'][t]['mean_rho'][0])
                    rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['cv_rho'].append(rho_dict_iter['per_transfer_stats'][t]['cv_rho'][0])
                            
                # we are calculating the following log-ratio statistics
                # the mean of the mean log ratio across species per-transfer
                # the mean of the CV log ratio across species per-transfer
                # the mean of the mean log ratio across species before transfer 12
                # the mean of the CV log ratio across species after transfer 12
                for migration_treatment in ['rel_s_by_s_migration_t', 'rel_s_by_s_no_migration_t']:

                    if migration_treatment == 'rel_s_by_s_migration_t':
                        migration_status = 'global_migration'
                    else:
                        migration_status = 'no_migration'

                    #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['transfer'].append(t)
                    #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'].append(mean_mean_log_ratio)
                    #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio'].append(mean_cv_log_ratio)

                    #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_mean'].append(rho_dict_iter['ratio_stats'][migration_status]['ks_mean'][0])
                    #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_cv'].append(rho_dict_iter['ratio_stats'][migration_status]['ks_cv'][0])

                    rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_t_stat'].append(rho_dict_iter['ratio_stats'][migration_status]['mean_t_stat'][0])
                    rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['pooled_t_stat'].append(rho_dict_iter['ratio_stats'][migration_status]['pooled_t_stat'][0])



            #ks_cv_no = rho_dict[tau_i][sigma_i]['ratio_stats']['no_migration']['ks_cv']
            #ks_cv_global = rho_dict[tau_i][sigma_i]['ratio_stats']['global_migration']['ks_cv']
            #if (len(ks_cv_no) > 0) and (len(ks_cv_global)>0):

            #    print('no migration', np.mean(ks_cv_no), len(ks_cv_no))
            #    print('global migration', np.mean(ks_cv_global), len(ks_cv_global))




    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_global_rho_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





def run_simulation_global_rho_abc(n_iter = 10000, tau=None, sigma=None, return_dict = False):

    if tau == None:
        tau_all = (np.random.uniform(tau_min, tau_max, size=n_iter*100)).tolist()
    else:
        tau_all = [tau]*(n_iter*1000)


    if sigma == None:
        sigma_all = (10**(np.random.uniform(np.log10(sigma_min), np.log10(sigma_max), size=n_iter*100))).tolist()
    else:
        sigma_all = [sigma]*(n_iter*1000)


    # run the whole range of transfers since we have this data for global migration
    transfers = range(18)
    rho_dict = {}
    rho_dict['ratio_stats'] = {}
    rho_dict['per_transfer_stats'] = {}
    rho_dict['tau_all'] = []
    rho_dict['sigma_all'] = []

    rho_dict['z_rho'] = {}
    rho_dict['z_rho']['mean_log10'] = {}
    rho_dict['z_rho']['mean_log10']['rho_mean_12'] = []
    rho_dict['z_rho']['mean_log10']['rho_mean_18'] = []
    rho_dict['z_rho']['mean_log10']['z_mean'] = []

    rho_dict['z_rho']['cv_log10'] = {}
    rho_dict['z_rho']['cv_log10']['rho_cv_12'] = []
    rho_dict['z_rho']['cv_log10']['rho_cv_18'] = []
    rho_dict['z_rho']['cv_log10']['z_cv'] = []

    for migration_status in ['global_migration', 'no_migration']:

        rho_dict['ratio_stats'][migration_status] = {}
        rho_dict['ratio_stats'][migration_status]['transfer'] = []
        rho_dict['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'] = []
        rho_dict['ratio_stats'][migration_status]['mean_cv_log_ratio'] = []

        rho_dict['ratio_stats'][migration_status]['ks_mean'] = []
        rho_dict['ratio_stats'][migration_status]['ks_cv'] = []

        rho_dict['ratio_stats'][migration_status]['ks_mean_constrain_species'] = []
        rho_dict['ratio_stats'][migration_status]['ks_cv_constrain_species'] = []

        rho_dict['ratio_stats'][migration_status]['mean_cv_log_ratio_before'] = []
        rho_dict['ratio_stats'][migration_status]['mean_cv_log_ratio_after'] = []

        rho_dict['ratio_stats'][migration_status]['mean_t_stat'] = []
        rho_dict['ratio_stats'][migration_status]['pooled_t_stat'] = []


    for t in transfers:
        rho_dict['per_transfer_stats'][t] = {}
        rho_dict['per_transfer_stats'][t]['mean_rho'] = []
        rho_dict['per_transfer_stats'][t]['cv_rho'] = []


    if return_dict == False:
        sys.stderr.write("Starting ABC simulation for global migration treatment...\n")

    # get indexes for ratios
    communities_global = utils.get_migration_time_series_community_names(migration='Global_migration', inocula=4)
    communities_keep_global = np.asarray([int(key) for key, value in communities_global.items() if len(value) == 18])
    communities_keep_global.sort()
    
    communities_no = utils.get_migration_time_series_community_names(migration='No_migration', inocula=4)
    communities_keep_no = np.asarray([int(key) for key, value in communities_no.items() if len(value) == 18])
    communities_keep_no.sort()

    ratio_community_idx_dict = {}
    ratio_community_idx_dict['rel_s_by_s_migration_t'] = {}
    ratio_community_idx_dict['rel_s_by_s_no_migration_t'] = {}
    for t in transfers:

        community_reps_global_t = rep_number_and_read_count_dict[('Global_migration',4)][t]['community_reps']
        community_reps_no_t = rep_number_and_read_count_dict[('No_migration',4)][t]['community_reps']

        # the position of the elements in community_reps_global_t is the community label minus one
        global_subset_idx = (np.intersect1d(community_reps_global_t, communities_keep_global)) - 1
        no_subset_idx = (np.intersect1d(community_reps_no_t, communities_keep_no)) - 1

        # sort
        global_subset_idx.sort()
        no_subset_idx.sort()

        #miration_subset_idx = np.asarray([np.where(community_reps_global_t == k)[0][0] for k in communities_keep_global])
        #no_miration_subset_idx = np.asarray([np.where(community_reps_no_t == k)[0][0] for k in communities_keep_no])

        # we have sorted communities_keep_global and communities_keep_no earlier
        # this means that we are getting the indices of the sorted communities
        # so when we index the simulated communities to get the log-ratio between timepoints, 
        # miration_subset_idx and no_miration_subset_idx make sure we're taking the 
        # ratio of timepoints for the same sample.
        ratio_community_idx_dict['rel_s_by_s_migration_t'][t] = global_subset_idx
        ratio_community_idx_dict['rel_s_by_s_no_migration_t'][t] = no_subset_idx


    while len(rho_dict['tau_all']) < n_iter:
        
        n_iter_successful = len(rho_dict['tau_all'])


        tau_i = tau_all.pop()
        sigma_i = sigma_all.pop()


        try:
            n_migration_global_reads, n_migration_parent_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)
        except ValueError:
            continue
        
        #if n_migration_global_reads == None:
        #    continue
  

        s_by_s_migration = n_migration_global_reads
        s_by_s_no_migration = n_no_migration_reads

        s_by_s_dict = {}
        does_idx_ratio_vs_parent_meet_cutoff = True
        for t in transfers:

            s_by_s_migration_t = s_by_s_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            rel_s_by_s_migration_t = (s_by_s_migration_t.T/s_by_s_migration_t.sum(axis=1)).T
            rel_s_by_s_no_migration_t = (s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)).T

            idx_migration = (~np.all(rel_s_by_s_migration_t.T == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t.T == 0, axis=1))

            s_by_s_dict[t] = {}
            s_by_s_dict[t]['rel_s_by_s_migration_t'] = rel_s_by_s_migration_t
            s_by_s_dict[t]['rel_s_by_s_no_migration_t'] = rel_s_by_s_no_migration_t
            s_by_s_dict[t]['idx_migration'] = idx_migration

            # at least five species that are present in both treatments
            if sum(idx_migration) < 5:
                does_idx_ratio_vs_parent_meet_cutoff = False


        if does_idx_ratio_vs_parent_meet_cutoff == False:
            continue

        
        # check whether simulation meets critiera for CV log-ratio analysis


        else:

            measure_dict = {}
            measure_dict['mean_log10'] = {}
            measure_dict['cv_log10'] = {}
            measure_dict['mean_corr_t'] = {}
            measure_dict['cv_corr_t'] = {}

            for t in transfers:

                rel_s_by_s_migration_t = s_by_s_dict[t]['rel_s_by_s_migration_t']
                rel_s_by_s_no_migration_t = s_by_s_dict[t]['rel_s_by_s_no_migration_t']
                idx_migration = s_by_s_dict[t]['idx_migration']

                community_reps_global_t_idx = rep_number_and_read_count_dict[('Global_migration',4)][t]['community_reps_idx']
                community_reps_no_t_idx = rep_number_and_read_count_dict[('No_migration',4)][t]['community_reps_idx']

                # select communities that were actually sampled
                rel_s_by_s_migration_t = rel_s_by_s_migration_t[community_reps_global_t_idx,:]
                rel_s_by_s_no_migration_t = rel_s_by_s_no_migration_t[community_reps_no_t_idx,:]

                rel_s_by_s_migration_t_subset = rel_s_by_s_migration_t[:,idx_migration]
                rel_s_by_s_no_migration_t_subset = rel_s_by_s_no_migration_t[:,idx_migration]

                mean_rel_migration_t = np.mean(rel_s_by_s_migration_t_subset, axis=0)
                mean_rel_no_migration_t = np.mean(rel_s_by_s_no_migration_t_subset, axis=0)

                std_rel_migration_t = np.std(rel_s_by_s_migration_t_subset, axis=0)
                std_rel_no_migration_t = np.std(rel_s_by_s_no_migration_t_subset, axis=0)

                cv_rel_migration_t = std_rel_migration_t/mean_rel_migration_t
                cv_rel_no_migration_t = std_rel_no_migration_t/mean_rel_no_migration_t

                mean_rel_migration_t_log10 = np.log10(mean_rel_migration_t)
                mean_rel_no_migration_t_log10 = np.log10(mean_rel_no_migration_t)

                cv_rel_migration_t_log10 = np.log10(cv_rel_migration_t)
                cv_rel_no_migration_t_log10 = np.log10(cv_rel_no_migration_t)

                mean_corr_t = np.corrcoef(mean_rel_migration_t_log10, mean_rel_no_migration_t_log10)[0,1]
                cv_corr_t = np.corrcoef(cv_rel_migration_t_log10, cv_rel_no_migration_t_log10)[0,1]

                measure_dict['mean_log10'][t] = {}
                measure_dict['mean_log10'][t]['global_migration'] = mean_rel_migration_t_log10
                measure_dict['mean_log10'][t]['no_migration'] = mean_rel_no_migration_t_log10

                measure_dict['cv_log10'][t] = {}
                measure_dict['cv_log10'][t]['global_migration'] = cv_rel_migration_t_log10
                measure_dict['cv_log10'][t]['no_migration'] = cv_rel_no_migration_t_log10

                measure_dict['mean_corr_t'][t] = mean_corr_t
                measure_dict['cv_corr_t'][t] = cv_corr_t


            # z-test
            rho_mean_18, rho_mean_12, z_mean = utils.compare_rho_fisher_z(measure_dict['mean_log10'][17]['global_migration'], measure_dict['mean_log10'][17]['no_migration'], measure_dict['mean_log10'][11]['global_migration'], measure_dict['mean_log10'][11]['no_migration'])
            rho_cv_18, rho_cv_12, z_cv = utils.compare_rho_fisher_z(measure_dict['cv_log10'][17]['global_migration'], measure_dict['cv_log10'][17]['no_migration'], measure_dict['cv_log10'][11]['global_migration'], measure_dict['cv_log10'][11]['no_migration'])
            
            # check for nan
            if (np.isnan(z_mean) == True) or (np.isnan(z_cv) == True):
                continue

            # we are calculating the following log-ratio statistics
            # the mean of the mean log ratio across species per-transfer
            # the mean of the CV log ratio across species per-transfer
            # the mean of the mean log ratio across species before transfer 12
            # the mean of the CV log ratio across species after transfer 12
            
            skip_iter = False
            #ks_iter_dict = {}
            ratio_stats_dict = {}
            for migration_treatment in ['rel_s_by_s_migration_t', 'rel_s_by_s_no_migration_t']:

                if migration_treatment == 'rel_s_by_s_migration_t':
                    migration_status = 'global_migration'
                else:
                    migration_status = 'no_migration'

                log_ratio_before_after_dict = {}
                transfer_all_transfers= []
                mean_log_ratio_per_transfer_all_transfers = []
                cv_log_ratio_all_transfers = []

                ratio_stats_dict[migration_status] = {}
                ratio_stats_dict[migration_status]['transfer'] = []
                ratio_stats_dict[migration_status]['mean_mean_log_ratio_per_transfer'] = []
                ratio_stats_dict[migration_status]['mean_cv_log_ratio'] = []

                # one less index because we're examining the ratio between two timepoints
                t_range = np.arange(7, 18-1)
                # [0,...,16]
                for t in t_range:

                    rel_s_by_s_migration_t = s_by_s_dict[t][migration_treatment]
                    rel_s_by_s_migration_t_plus = s_by_s_dict[t+1][migration_treatment]

                    # subset samples that were actually sampled
                    sampled_t_idx = ratio_community_idx_dict[migration_treatment][t] 
                    sampled_t_plus_idx = ratio_community_idx_dict[migration_treatment][t+1]

                    # already sorted! We can not take the ratio.
                    rel_s_by_s_migration_t = rel_s_by_s_migration_t[sampled_t_idx,:]
                    rel_s_by_s_migration_t_plus = rel_s_by_s_migration_t_plus[sampled_t_plus_idx,:]

                    # log ratio dimensions are # ASVS by # replicates for a given pair of timepoints
                    log_ratio = np.log10(rel_s_by_s_migration_t_plus) - np.log10(rel_s_by_s_migration_t)

                    # get mean
                    mean_log_ratio_all = []
                    cv_log_ratio_all = []

                    # log-ratio distribution *across replicates* for a given pair of timepoints for a given species
                    # length should be equal to number of replicates
                    for log_ratio_dist_idx, log_ratio_dist in enumerate(log_ratio.T):

                        to_keep_dist_idx = (~np.isnan(log_ratio_dist)) & np.isfinite(log_ratio_dist)
                        log_ratio_dist = log_ratio_dist[to_keep_dist_idx]
                        # select timepoints that are used to calculate the statistics
                        # keep fluctuation dist of an ASV if there are observations from at least *five* replicate communities
                        if len(log_ratio_dist) >= 5:

                            mean_log_ratio = np.mean(log_ratio_dist)
                            std_log_ratio = np.std(log_ratio_dist)
                            cv_log_ratio = std_log_ratio/np.absolute(mean_log_ratio)

                            # Initiate ASV
                            if log_ratio_dist_idx not in log_ratio_before_after_dict:
                                log_ratio_before_after_dict[log_ratio_dist_idx] = {}
                            
                            log_ratio_before_after_dict[log_ratio_dist_idx][t] = {}
                            log_ratio_before_after_dict[log_ratio_dist_idx][t]['mean_log_ratio'] = mean_log_ratio
                            log_ratio_before_after_dict[log_ratio_dist_idx][t]['cv_log_ratio'] = cv_log_ratio
                            
                            mean_log_ratio_all.append(mean_log_ratio)
                            cv_log_ratio_all.append(cv_log_ratio)


                        #if log_ratio_dist_idx not in log_ratio_before_after_dict:
                        #    log_ratio_before_after_dict[log_ratio_dist_idx] = {}
                        #    log_ratio_before_after_dict[log_ratio_dist_idx]['transfer'] = []
                        #    log_ratio_before_after_dict[log_ratio_dist_idx]['log_ratio'] = []


                        #log_ratio_before_after_dict[log_ratio_dist_idx]['transfer'].extend([t]*len(log_ratio_dist))
                        #log_ratio_before_after_dict[log_ratio_dist_idx]['log_ratio'].extend(log_ratio_dist)


                    transfer_all_transfers.extend([t]*len(mean_log_ratio_all))
                    mean_log_ratio_per_transfer_all_transfers.extend(mean_log_ratio_all)
                    cv_log_ratio_all_transfers.extend(cv_log_ratio_all)

                    # mean across species of the mean/CV of log-ratio across reps, per-transfer
                    mean_mean_log_ratio = np.mean(mean_log_ratio_all)
                    mean_cv_log_ratio = np.mean(cv_log_ratio_all)

                    ratio_stats_dict[migration_status]['transfer'].append(t)
                    ratio_stats_dict[migration_status]['mean_mean_log_ratio_per_transfer'].append(mean_mean_log_ratio)
                    ratio_stats_dict[migration_status]['mean_cv_log_ratio'].append(mean_cv_log_ratio)

                    #rho_dict['ratio_stats'][migration_status]['transfer'].append(t)
                    #rho_dict['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'].append(mean_mean_log_ratio)
                    #rho_dict['ratio_stats'][migration_status]['mean_cv_log_ratio'].append(mean_cv_log_ratio)


                # log_ratio_before_after_dict contains the log-ratio dist per-ASV
                # find ASVs with log-ratio values at all required timepoints

                log_ratio_before_after_dict_keys = list(log_ratio_before_after_dict.keys())
                for key_ in log_ratio_before_after_dict_keys:

                    if len(log_ratio_before_after_dict[key_]) < len(t_range):
                        del log_ratio_before_after_dict[key_]

                # skip if not enough observations.
                if len(log_ratio_before_after_dict_keys) < 3:
                    ratio_stats_dict[migration_status]['mean_t_stat'] = np.nan
                    continue

                
                #t_all = []
                t_stat_all = []
                cv_before_all_pooled = []
                cv_after_all_pooled = []
                for key_ in log_ratio_before_after_dict.keys(): 

                    cv_before_all_key = []
                    cv_after_all_key = []
                    
                    for t_k in t_range:

                        if t_k <= 11:
                            cv_before_all_key.append(log_ratio_before_after_dict[key_][t_k]['cv_log_ratio'])
                            cv_before_all_pooled.append(log_ratio_before_after_dict[key_][t_k]['cv_log_ratio'])
                        else:
                            cv_after_all_key.append(log_ratio_before_after_dict[key_][t_k]['cv_log_ratio'])
                            cv_after_all_pooled.append(log_ratio_before_after_dict[key_][t_k]['cv_log_ratio'])


                    t_stat_all.append(stats.ttest_ind(cv_after_all_key, cv_before_all_key, equal_var=True)[0])


                ratio_stats_dict[migration_status]['mean_t_stat'] = np.mean(t_stat_all)
                ratio_stats_dict[migration_status]['pooled_t_stat'] = np.mean(stats.ttest_ind(cv_after_all_pooled, cv_before_all_pooled, equal_var=True)[0])

                
                # log-ratio stats
                #transfer_all_transfers = np.asarray(transfer_all_transfers)
                #mean_log_ratio_per_transfer_all_transfers = np.asarray(mean_log_ratio_per_transfer_all_transfers)
                #cv_log_ratio_all_transfers = np.asarray(cv_log_ratio_all_transfers)

                #mean_log_ratio_per_transfer_all_transfers_before = mean_log_ratio_per_transfer_all_transfers[(transfer_all_transfers>=7) & (transfer_all_transfers<=11)]
                #mean_log_ratio_per_transfer_all_transfers_after = mean_log_ratio_per_transfer_all_transfers[(transfer_all_transfers>11)]

                #cv_log_ratio_all_transfers_before = cv_log_ratio_all_transfers[(transfer_all_transfers>=7) & (transfer_all_transfers<=11)]
                #cv_log_ratio_all_transfers_after = cv_log_ratio_all_transfers[(transfer_all_transfers>11)]

                #n_species_ratio_dict = {'cv': {'no_migration':{'before': 80, 'after': 64, }, 'global_migration':{'before': 68, 'after':49} }}

                #if (len(mean_log_ratio_per_transfer_all_transfers_before)>0) & (len(mean_log_ratio_per_transfer_all_transfers_after)>0):
                #    ks_mean_log_ratio_per_transfer_all_transfers, p_value_mean_log_ratio_per_transfer_all_transfers = stats.ks_2samp(mean_log_ratio_per_transfer_all_transfers_before, mean_log_ratio_per_transfer_all_transfers_after)
                #    rho_dict['ratio_stats'][migration_status]['ks_mean'].append(ks_mean_log_ratio_per_transfer_all_transfers)


                #else:
                #    to_skip = True



                #if (len(cv_log_ratio_all_transfers_before)>0) & (len(cv_log_ratio_all_transfers_after)>0):
                #    ks_cv_log_ratio_all_transfers, p_value_cv_log_ratio_all_transfers = stats.ks_2samp(cv_log_ratio_all_transfers_before, cv_log_ratio_all_transfers_after)
                #    rho_dict['ratio_stats'][migration_status]['ks_cv'].append(ks_cv_log_ratio_all_transfers)

                #    # random_indices, high is exclusive.
                #    #num_to = min([len(cv_log_ratio_all_transfers_before), len(cv_log_ratio_all_transfers_after)])

                #    if (len(cv_log_ratio_all_transfers_before) > n_species_ratio_dict['cv'][migration_status]['before']) and (len(cv_log_ratio_all_transfers_after) > n_species_ratio_dict['cv'][migration_status]['after']):

                #        species_rndm_before_idx = np.random.randint(0, len(cv_log_ratio_all_transfers_before), n_species_ratio_dict['cv'][migration_status]['before'])
                #        species_rndm_after_idx = np.random.randint(0, len(cv_log_ratio_all_transfers_after), n_species_ratio_dict['cv'][migration_status]['after'])
                #        ks_cv_log_ratio_all_transfers_subset, p_value_cv_log_ratio_all_transfers_subset = stats.ks_2samp(cv_log_ratio_all_transfers_before[species_rndm_before_idx], cv_log_ratio_all_transfers_after[species_rndm_after_idx])
                    
                #    else:
                #        ks_cv_log_ratio_all_transfers_subset = ks_cv_log_ratio_all_transfers
                    
                    
                #    rho_dict['ratio_stats'][migration_status]['ks_cv_constrain_species'].append(ks_cv_log_ratio_all_transfers_subset)


                #else:
                #    to_skip = True


            # check whether to skip
            for migration_status in ['global_migration', 'no_migration']:

                if 'mean_t_stat' not in ratio_stats_dict[migration_status]:
                    skip_iter = True
                    continue

                if 'pooled_t_stat' not in ratio_stats_dict[migration_status]:
                    skip_iter = True
                    continue

                mean_t_stat = ratio_stats_dict[migration_status]['mean_t_stat']
                pooled_t_stat = ratio_stats_dict[migration_status]['pooled_t_stat']

                #print(mean_t_stat)

                if np.isnan(mean_t_stat) == True:
                    skip_iter = True

                if np.isnan(pooled_t_stat) == True:
                    skip_iter = True

            if skip_iter == True:
                continue

            # save to dictionary
            rho_dict['tau_all'].append(tau_i)
            rho_dict['sigma_all'].append(sigma_i)

            for t in transfers:
                rho_dict['per_transfer_stats'][t]['mean_rho'].append(measure_dict['mean_corr_t'][t])
                rho_dict['per_transfer_stats'][t]['cv_rho'].append(measure_dict['cv_corr_t'][t])

            rho_dict['z_rho']['mean_log10']['rho_mean_12'].append(rho_mean_12)
            rho_dict['z_rho']['mean_log10']['rho_mean_18'].append(rho_mean_18)
            rho_dict['z_rho']['mean_log10']['z_mean'].append(z_mean)

            rho_dict['z_rho']['cv_log10']['rho_cv_12'].append(rho_cv_12)
            rho_dict['z_rho']['cv_log10']['rho_cv_18'].append(rho_cv_18)
            rho_dict['z_rho']['cv_log10']['z_cv'].append(z_cv)


            for migration_status in ['global_migration', 'no_migration']:
                rho_dict['ratio_stats'][migration_status]['mean_t_stat'].append(ratio_stats_dict[migration_status]['mean_t_stat'])
                rho_dict['ratio_stats'][migration_status]['pooled_t_stat'].append(ratio_stats_dict[migration_status]['pooled_t_stat'])

            
            
        if (n_iter_successful+1) % 100 == 0:
            print(n_iter_successful+1)


    if return_dict == True:

        return rho_dict

    else:

        sys.stderr.write("Saving dictionary...\n")
        with open(simulation_global_rho_abc_path, 'wb') as handle:
            pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)









def run_simulation_global_rho_fixed_parameters(tau_i, sigma_i, label='', n_iter=1000):

    #tau_all = np.linspace(1.7, 6.9, num=n_iter, endpoint=True)
    #sigma_all = np.logspace(np.log10(0.01), np.log10(1.9), num=n_iter, endpoint=True, base=10.0)

    # run the whole range of transfers since we have this data for global migration

    rho_dict_iter = run_simulation_global_rho_abc(n_iter=n_iter, tau=tau_i, sigma=sigma_i, return_dict = True)

    simulation_global_rho_fixed_parameters_path_ = simulation_global_rho_fixed_parameters_path % label

    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_global_rho_fixed_parameters_path_, 'wb') as handle:
        pickle.dump(rho_dict_iter, handle, protocol=pickle.HIGHEST_PROTOCOL)







def run_simulation_all_migration_fixed_parameters(tau_i, sigma_i, label, n_iter=1000, n_div_to_print=1000):
    
    #tau_all = np.linspace(1.7, 6.9, num=n_iter, endpoint=True)
    #sigma_all = np.logspace(np.log10(0.01), np.log10(1.9), num=n_iter, endpoint=True, base=10.0)

    #transfers = range(18)
    transfers = [11,17]

    rho_dict = {}
    rho_dict['tau_all'] = tau_i
    rho_dict['sigma_all'] = sigma_i
    rho_dict['ks_12_vs_18'] = {}
    rho_dict['ks_rescaled_12_vs_18'] = {}
    rho_dict['slope_12_vs_18'] = {}

    for treatment in ['global_migration', 'parent_migration', 'no_migration']:

        rho_dict['ks_12_vs_18'][treatment] = []
        rho_dict['ks_rescaled_12_vs_18'][treatment] = []
        
        rho_dict['slope_12_vs_18'][treatment] = {}
        rho_dict['slope_12_vs_18'][treatment]['slope_12'] = []
        rho_dict['slope_12_vs_18'][treatment]['slope_18'] = []
        rho_dict['slope_12_vs_18'][treatment]['slope_t_test'] = []
        rho_dict['slope_12_vs_18'][treatment]['intercept_12'] = []
        rho_dict['slope_12_vs_18'][treatment]['intercept_18'] = []
        rho_dict['slope_12_vs_18'][treatment]['intercept_t_test'] = []
        rho_dict['slope_12_vs_18'][treatment]['rho_12'] = []
        rho_dict['slope_12_vs_18'][treatment]['rho_18'] = []


    for t in transfers:
        
        rho_dict[t] = {}
        rho_dict[t]['global_migration'] = {}
        rho_dict[t]['parent_migration'] = {}
        rho_dict[t]['no_migration'] = {}

        rho_dict[t]['global_migration']['taylors_slope'] = []
        rho_dict[t]['parent_migration']['taylors_slope'] = []
        rho_dict[t]['no_migration']['taylors_slope'] = []

        rho_dict[t]['global_migration']['taylors_intercept'] = []
        rho_dict[t]['parent_migration']['taylors_intercept'] = []
        rho_dict[t]['no_migration']['taylors_intercept'] = []

        rho_dict[t]['global_migration']['t_slope'] = []
        rho_dict[t]['parent_migration']['t_slope'] = []

        rho_dict[t]['global_migration']['t_intercept'] = []
        rho_dict[t]['parent_migration']['t_intercept'] = []

        rho_dict[t]['global_migration']['mean_log_error'] = []
        rho_dict[t]['parent_migration']['mean_log_error'] = []
        rho_dict[t]['no_migration']['mean_log_error'] = []

        # ks test of AFD
        rho_dict[t]['global_migration']['ks_global_vs_no'] = []
        rho_dict[t]['parent_migration']['ks_parent_vs_no'] = []
        rho_dict[t]['global_migration']['ks_rescaled_global_vs_no'] = []
        rho_dict[t]['parent_migration']['ks_rescaled_parent_vs_no'] = []


    #for i in range(n_iter):
    while len(rho_dict['slope_12_vs_18']['no_migration']['rho_18']) < n_iter:
        
        n_iter_successful = len(rho_dict['slope_12_vs_18']['no_migration']['rho_18'])

        if (n_iter_successful+1) % n_div_to_print == 0:
            print(n_iter_successful+1)


        #tau_i = np.random.uniform(1.7, 6.9, size=1)[0]
        #sigma_i = 10**(np.random.uniform(np.log10(0.01), np.log10(1.9), size=1)[0])

        #tau_i = tau_all[i]
        #sigma_i = sigma_all[i]

        s_by_s_global_migration, s_by_s_parent_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)

        afd_dict = {}
        afd_dict['global_migration'] = {}
        afd_dict['parent_migration'] = {}
        afd_dict['no_migration'] = {}

        mean_var_dict = {}
        mean_var_dict['global_migration'] = {}
        mean_var_dict['parent_migration'] = {}
        mean_var_dict['no_migration'] = {}
        
        for t in transfers:

            s_by_s_global_migration_t = s_by_s_global_migration[t,:,:]
            s_by_s_parent_migration_t = s_by_s_parent_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            # remove zeros
            s_by_s_global_migration_t = s_by_s_global_migration_t[(~np.all(s_by_s_global_migration_t == 0, axis=1)),:]
            s_by_s_parent_migration_t = s_by_s_parent_migration_t[(~np.all(s_by_s_parent_migration_t == 0, axis=1)),:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[(~np.all(s_by_s_no_migration_t == 0, axis=1)),:]

            occupancies_global_migration, predicted_occupancies_global_migration, mad_global_migration, beta_global_migration, species_occupancies_global_migration  = utils.predict_occupancy(s_by_s_global_migration_t.T, range(s_by_s_global_migration_t.shape[1]))
            occupancies_parent_migration, predicted_occupancies_parent_migration, mad_parent_migration, beta_parent_migration, species_occupancies_parent_migration  = utils.predict_occupancy(s_by_s_parent_migration_t.T, range(s_by_s_parent_migration_t.shape[1]))
            occupancies_no_migration, predicted_occupancies_no_migration, mad_no_migration, beta_no_migration, species_occupancies_no_migration  = utils.predict_occupancy(s_by_s_no_migration_t.T, range(s_by_s_no_migration_t.shape[1]))


            # KS test
            rel_s_by_s_global_migration_t = s_by_s_global_migration_t.T/s_by_s_global_migration_t.sum(axis=1)
            rel_s_by_s_parent_migration_t = s_by_s_parent_migration_t.T/s_by_s_parent_migration_t.sum(axis=1)
            rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

            #afd_global_migration_t = np.ndarray.flatten(rel_s_by_s_global_migration_t)
            #afd_parent_migration_t = np.ndarray.flatten(rel_s_by_s_parent_migration_t)
            #afd_no_migration_t = np.ndarray.flatten(rel_s_by_s_no_migration_t)

            #log_afd_global_migration_t = np.log10(afd_global_migration_t[afd_global_migration_t>0])
            #log_afd_parent_migration_t = np.log10(afd_parent_migration_t[afd_parent_migration_t>0])
            #log_afd_no_migration_t = np.log10(afd_no_migration_t[afd_no_migration_t>0])

            #rescaled_log_afd_global_migration_t = (log_afd_global_migration_t - np.mean(log_afd_global_migration_t)) / np.std(log_afd_global_migration_t)
            #rescaled_log_afd_parent_migration_t = (log_afd_parent_migration_t - np.mean(log_afd_parent_migration_t)) / np.std(log_afd_parent_migration_t)
            #rescaled_log_afd_no_migration_t = (log_afd_no_migration_t - np.mean(log_afd_no_migration_t)) / np.std(log_afd_no_migration_t)

            log_afd_global_migration_t, rescaled_log_afd_global_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_global_migration_t.T)
            log_afd_no_migration_t, rescaled_log_afd_no_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_no_migration_t.T)
            log_afd_parent_migration_t, rescaled_log_afd_parent_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_parent_migration_t.T)

            ks_global_vs_no, p_value_ks_global_vs_no = stats.ks_2samp(log_afd_global_migration_t, log_afd_no_migration_t)
            ks_parent_vs_no, p_value_ks_parent_vs_no = stats.ks_2samp(log_afd_parent_migration_t, log_afd_no_migration_t)

            ks_rescaled_global_vs_no, p_value_ks_rescaled_global_vs_no = stats.ks_2samp(rescaled_log_afd_global_migration_t, rescaled_log_afd_no_migration_t)
            ks_rescaled_parent_vs_no, p_value_ks_rescaled_parent_vs_no = stats.ks_2samp(rescaled_log_afd_parent_migration_t, rescaled_log_afd_no_migration_t)

            afd_dict['global_migration'][t] = {}
            afd_dict['parent_migration'][t] = {}
            afd_dict['no_migration'][t] = {}

            afd_dict['global_migration'][t]['afd'] = log_afd_global_migration_t
            afd_dict['parent_migration'][t]['afd'] = log_afd_parent_migration_t
            afd_dict['no_migration'][t]['afd'] = log_afd_no_migration_t

            afd_dict['global_migration'][t]['rescaled_afd'] = rescaled_log_afd_global_migration_t
            afd_dict['parent_migration'][t]['rescaled_afd'] = rescaled_log_afd_parent_migration_t
            afd_dict['no_migration'][t]['rescaled_afd'] = rescaled_log_afd_no_migration_t

            error_global_migration = np.absolute(occupancies_global_migration - predicted_occupancies_global_migration)/occupancies_global_migration
            error_parent_migration = np.absolute(occupancies_parent_migration - predicted_occupancies_parent_migration)/occupancies_parent_migration
            error_no_migration = np.absolute(occupancies_no_migration - predicted_occupancies_no_migration)/occupancies_no_migration

            mean_log_error_global_migration = np.mean(np.log10(error_global_migration[error_global_migration>0]))
            mean_log_error_parent_migration = np.mean(np.log10(error_parent_migration[error_parent_migration>0]))
            mean_log_error_no_migration = np.mean(np.log10(error_no_migration[error_no_migration>0]))

            # taylors law
            means_global_migration, variances_global_migration, species_to_keep_global_migration = utils.get_species_means_and_variances(rel_s_by_s_global_migration_t, range(rel_s_by_s_global_migration_t.shape[0]), zeros=True)
            means_parent_migration, variances_parent_migration, species_to_keep_parent_migration = utils.get_species_means_and_variances(rel_s_by_s_parent_migration_t, range(rel_s_by_s_parent_migration_t.shape[0]), zeros=True)
            means_no_migration, variances_no_migration, species_to_keep_no_migration = utils.get_species_means_and_variances(rel_s_by_s_no_migration_t, range(rel_s_by_s_no_migration_t.shape[0]), zeros=True)

            # filter observations with mean greter than 0.95
            idx_to_keep_global = (means_global_migration<0.95)
            idx_to_keep_parent = (means_parent_migration<0.95)
            idx_to_keep_no = (means_no_migration<0.95)

            means_global_migration = means_global_migration[idx_to_keep_global]
            variances_global_migration = variances_global_migration[idx_to_keep_global]

            means_parent_migration = means_parent_migration[idx_to_keep_parent]
            variances_parent_migration = variances_parent_migration[idx_to_keep_parent]

            means_no_migration = means_no_migration[idx_to_keep_no]
            variances_no_migration = variances_no_migration[idx_to_keep_no]

            # log transform
            means_global_migration_log10 = np.log10(means_global_migration)
            variances_global_migration_log10 = np.log10(variances_global_migration)

            means_parent_migration_log10 = np.log10(means_parent_migration)
            variances_parent_migration_log10 = np.log10(variances_parent_migration)

            means_no_migration_log10 = np.log10(means_no_migration)
            variances_no_migration_log10 = np.log10(variances_no_migration)


            # save means and variances for 12 vs. 18 tests
            mean_var_dict['global_migration'][t] = {}
            mean_var_dict['parent_migration'][t] = {}
            mean_var_dict['no_migration'][t] = {}

            mean_var_dict['global_migration'][t]['means_log10'] = means_global_migration_log10
            mean_var_dict['global_migration'][t]['variances_log10'] = variances_global_migration_log10

            mean_var_dict['parent_migration'][t]['means_log10'] = means_parent_migration_log10
            mean_var_dict['parent_migration'][t]['variances_log10'] = variances_parent_migration_log10

            mean_var_dict['no_migration'][t]['means_log10'] = means_no_migration_log10
            mean_var_dict['no_migration'][t]['variances_log10'] = variances_no_migration_log10

            #slope_global, intercept_global, r_value_global, p_value_global, std_err_global = stats.linregress(np.log10(means_global_migration), np.log10(variances_global_migration))
            #slope_parent, intercept_parent, r_value_parent, p_value_parent, std_err_parent = stats.linregress(np.log10(means_parent_migration), np.log10(variances_parent_migration))
            #slope_no, intercept_no, r_value_no, p_value_no, std_err_no = stats.linregress(np.log10(means_no_migration), np.log10(variances_no_migration))

            # t-test b/w migration and no migration 
            slope_global, slope_no, t_slope_global, intercept_global, intercept_no, t_intercept_global, r_value_global, r_value_no = utils.t_statistic_two_slopes(means_global_migration_log10, variances_global_migration_log10, means_no_migration_log10, variances_no_migration_log10)
            slope_parent, slope_no, t_slope_parent, intercept_parent, intercept_no, t_intercept_parent, r_value_parent, r_value_no = utils.t_statistic_two_slopes(means_parent_migration_log10, variances_parent_migration_log10, means_no_migration_log10, variances_no_migration_log10)

            rho_dict[t]['global_migration']['taylors_slope'].append(slope_global)
            rho_dict[t]['parent_migration']['taylors_slope'].append(slope_parent)
            rho_dict[t]['no_migration']['taylors_slope'].append(slope_no)

            rho_dict[t]['global_migration']['taylors_intercept'].append(intercept_global)
            rho_dict[t]['parent_migration']['taylors_intercept'].append(intercept_parent)
            rho_dict[t]['no_migration']['taylors_intercept'].append(intercept_no)

            rho_dict[t]['global_migration']['t_slope'].append(t_slope_global)
            rho_dict[t]['parent_migration']['t_slope'].append(t_slope_parent)

            rho_dict[t]['global_migration']['t_intercept'].append(t_intercept_global)
            rho_dict[t]['parent_migration']['t_intercept'].append(t_intercept_parent)

            rho_dict[t]['global_migration']['mean_log_error'].append(mean_log_error_global_migration)
            rho_dict[t]['parent_migration']['mean_log_error'].append(mean_log_error_parent_migration)
            rho_dict[t]['no_migration']['mean_log_error'].append(mean_log_error_no_migration)

            # KS test
            rho_dict[t]['global_migration']['ks_global_vs_no'].append(ks_global_vs_no)
            rho_dict[t]['parent_migration']['ks_parent_vs_no'].append(ks_parent_vs_no)
            rho_dict[t]['global_migration']['ks_rescaled_global_vs_no'].append(ks_rescaled_global_vs_no)
            rho_dict[t]['parent_migration']['ks_rescaled_parent_vs_no'].append(ks_rescaled_parent_vs_no)
        
        # 12 vs 18 trasnfers
        for treatment in ['global_migration', 'parent_migration', 'no_migration']:

            ks_12_18, p_value_ks_12_18 = stats.ks_2samp(afd_dict[treatment][17]['afd'], afd_dict[treatment][11]['afd'])
            ks_rescaled_12_18, p_value_ks_rescaled_12_18 = stats.ks_2samp(afd_dict[treatment][17]['rescaled_afd'], afd_dict[treatment][11]['rescaled_afd'])

            rho_dict['ks_12_vs_18'][treatment].append(ks_12_18)
            rho_dict['ks_rescaled_12_vs_18'][treatment].append(ks_rescaled_12_18)

            # 12 vs. 18 Taylors law slope
            means_log10_12 = mean_var_dict[treatment][11]['means_log10']
            variances_log10_12 = mean_var_dict[treatment][11]['variances_log10']
            means_log10_18 = mean_var_dict[treatment][17]['means_log10']
            variances_log10_18 = mean_var_dict[treatment][17]['variances_log10']
            slope_18, slope_12, t_slope_18_vs_12, intercept_18, intercept_12, t_intercept_18_vs_12, r_value_18, r_value_12 = utils.t_statistic_two_slopes(means_log10_18, variances_log10_18, means_log10_12, variances_log10_12)

            rho_dict['slope_12_vs_18'][treatment]['slope_12'].append(slope_12)
            rho_dict['slope_12_vs_18'][treatment]['slope_18'].append(slope_18)
            rho_dict['slope_12_vs_18'][treatment]['slope_t_test'].append(t_slope_18_vs_12)
            rho_dict['slope_12_vs_18'][treatment]['intercept_12'].append(intercept_12)
            rho_dict['slope_12_vs_18'][treatment]['intercept_18'].append(intercept_18)
            rho_dict['slope_12_vs_18'][treatment]['intercept_t_test'].append(t_intercept_18_vs_12)
            rho_dict['slope_12_vs_18'][treatment]['rho_12'].append(r_value_12)
            rho_dict['slope_12_vs_18'][treatment]['rho_18'].append(r_value_18)

            
            #if treatment == 'parent_migration':
            #print(treatment, t_slope_18_vs_12)



    simulation_all_migration_fixed_parameters_path_ = simulation_all_migration_fixed_parameters_path % label

    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_all_migration_fixed_parameters_path_, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





def run_simulation_all_migration_abc(n_iter=10000, tau=None, sigma=None, label=None, return_dict = False):

    # randomly sample parameters if none are provided
    if tau == None:
        tau_all = (np.random.uniform(tau_min, tau_max, size=n_iter*1000)).tolist()
    else:
        tau_all = [tau]*(n_iter*100)

    if sigma == None:
        sigma_all = (10**(np.random.uniform(np.log10(sigma_min), np.log10(sigma_max), size=n_iter*1000))).tolist()
    else:
        sigma_all = [sigma]*(n_iter*100)


    transfers = range(18)

    rho_dict = {}
    rho_dict['tau_all'] = []
    rho_dict['sigma_all'] = []

    if return_dict == False:
        sys.stderr.write("Starting ABC simulation for all migration treatments...\n")
    
    # statistics for all treatments
    all_stats_labels = ['ks_12_vs_18', 'ks_rescaled_12_vs_18', 'slope_12_vs_18', 'ks_12_vs_18_intersect', 'ks_rescaled_12_vs_18_intersect', 'mean_over_asv_ks_12_vs_18', 'mean_over_asv_ks_rescaled_12_vs_18']
    for s in all_stats_labels:
        rho_dict[s] = {}

        for treatment in utils.experiments_no_inocula:
            
            if s != 'slope_12_vs_18': 
                rho_dict[s][treatment] = []

            else:
                rho_dict[s][treatment] = {}
                rho_dict[s][treatment]['slope_12'] = []
                rho_dict[s][treatment]['slope_18'] = []
                rho_dict[s][treatment]['slope_t_test'] = []
                rho_dict[s][treatment]['intercept_12'] = []
                rho_dict[s][treatment]['intercept_18'] = []
                rho_dict[s][treatment]['intercept_t_test'] = []
                rho_dict[s][treatment]['rho_12'] = []
                rho_dict[s][treatment]['rho_18'] = []


    for t in transfers:
        
        rho_dict[t] = {}

        for treatment in utils.experiments_no_inocula:
            rho_dict[t][treatment] = {}
            rho_dict[t][treatment]['taylors_slope'] = []
            rho_dict[t][treatment]['taylors_intercept'] = []
            rho_dict[t][treatment]['mean_log_error'] = []

            # only global and parent migration
            if treatment != 'no_migration':

                rho_dict[t][treatment]['t_slope'] = []
                rho_dict[t][treatment]['t_intercept'] = []

                # ks test of AFD
                rho_dict[t][treatment]['ks_migration_vs_no'] = []
                rho_dict[t][treatment]['ks_rescaled_migration_vs_no'] = []


    while len(rho_dict['tau_all']) < n_iter:
        
        n_iter_successful = len(rho_dict['tau_all'])

        if (n_iter_successful+1) % 1000 == 0:
            print(n_iter_successful+1)

        
        tau_i = tau_all.pop()
        sigma_i = sigma_all.pop()

        s_by_s_global_migration, s_by_s_parent_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)

        afd_dict = {}
        afd_dict['global_migration'] = {}
        afd_dict['parent_migration'] = {}
        afd_dict['no_migration'] = {}

        mean_var_dict = {}
        mean_var_dict['global_migration'] = {}
        mean_var_dict['parent_migration'] = {}
        mean_var_dict['no_migration'] = {}


        #community_reps_global_12_idx = rep_number_and_read_count_dict[('Global_migration',4)][11]['community_reps_idx']
        #community_reps_parent_12_idx = rep_number_and_read_count_dict[('Parent_migration',4)][11]['community_reps_idx']
        #community_reps_no_12_idx = rep_number_and_read_count_dict[('No_migration',4)][11]['community_reps_idx']

        # select communities actually sampled
        #s_by_s_global_migration_12 = s_by_s_global_migration_12[community_reps_global_12_idx,:]
        #s_by_s_parent_migration_12 = s_by_s_parent_migration_12[community_reps_parent_12_idx,:]
        #s_by_s_no_migration_12 = s_by_s_no_migration_12[community_reps_no_12_idx,:]


        #s_by_s_global_migration_18 = s_by_s_global_migration[17,:,:]
        #s_by_s_parent_migration_18 = s_by_s_parent_migration[17,:,:]
        #s_by_s_no_migration_18 = s_by_s_no_migration[17,:,:]

        #community_reps_global_18_idx = rep_number_and_read_count_dict[('Global_migration',4)][17]['community_reps_idx']
        #community_reps_parent_18_idx = rep_number_and_read_count_dict[('Parent_migration',4)][17]['community_reps_idx']
        #community_reps_no_18_idx = rep_number_and_read_count_dict[('No_migration',4)][17]['community_reps_idx']

        # select communities actually sampled
        #s_by_s_global_migration_18 = s_by_s_global_migration_18[community_reps_global_18_idx,:]
        #s_by_s_parent_migration_18 = s_by_s_parent_migration_18[community_reps_parent_18_idx,:]
        #s_by_s_no_migration_18 = s_by_s_no_migration_18[community_reps_no_18_idx,:]

        #occupancy_global_12 = np.sum(s_by_s_global_migration_12>0, axis=0)/s_by_s_global_migration_12.shape[0]
        #occupancy_parent_12 = np.sum(s_by_s_parent_migration_12>0, axis=0)/s_by_s_parent_migration_12.shape[0]
        #occupancy_no_12 = np.sum(s_by_s_no_migration_12>0, axis=0)/s_by_s_no_migration_12.shape[0]

        #occupancy_global_18 = np.sum(s_by_s_global_migration_18>0, axis=0)/s_by_s_global_migration_18.shape[0]
        #occupancy_parent_18 = np.sum(s_by_s_parent_migration_18>0, axis=0)/s_by_s_parent_migration_18.shape[0]
        #occupancy_no_18 = np.sum(s_by_s_no_migration_18>0, axis=0)/s_by_s_no_migration_18.shape[0]

        #n_occupancy_one_global = sum((occupancy_global_12==1) & (occupancy_global_18==1))
        #n_occupancy_one_parent = sum((occupancy_parent_12==1) & (occupancy_parent_18==1))
        #n_occupancy_one_no = sum((occupancy_no_12==1) & (occupancy_no_18==1))


        #if (n_occupancy_one_global==0) or (n_occupancy_one_parent==0) or (n_occupancy_one_no==0):
        #    continue


        s_by_s_dict = {}
        #for t in transfers:
        for t in [11, 17]:

            s_by_s_global_migration_t = s_by_s_global_migration[t,:,:]
            s_by_s_parent_migration_t = s_by_s_parent_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            community_reps_global_t_idx = rep_number_and_read_count_dict[('Global_migration',4)][t]['community_reps_idx']
            community_reps_parent_t_idx = rep_number_and_read_count_dict[('Parent_migration',4)][t]['community_reps_idx']
            community_reps_no_t_idx = rep_number_and_read_count_dict[('No_migration',4)][t]['community_reps_idx']

            # select communities actually sampled
            s_by_s_global_migration_t = s_by_s_global_migration_t[community_reps_global_t_idx,:]
            s_by_s_parent_migration_t = s_by_s_parent_migration_t[community_reps_parent_t_idx,:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[community_reps_no_t_idx,:]

            s_by_s_dict[t] = {}
            s_by_s_dict[t]['global_migration'] = s_by_s_global_migration_t
            s_by_s_dict[t]['parent_migration'] = s_by_s_parent_migration_t
            s_by_s_dict[t]['no_migration'] = s_by_s_no_migration_t

            # remove zeros
            s_by_s_global_migration_t = s_by_s_global_migration_t[(~np.all(s_by_s_global_migration_t == 0, axis=1)),:]
            s_by_s_parent_migration_t = s_by_s_parent_migration_t[(~np.all(s_by_s_parent_migration_t == 0, axis=1)),:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[(~np.all(s_by_s_no_migration_t == 0, axis=1)),:]

            occupancies_global_migration, predicted_occupancies_global_migration, mad_global_migration, beta_global_migration, species_occupancies_global_migration  = utils.predict_occupancy(s_by_s_global_migration_t.T, range(s_by_s_global_migration_t.shape[1]))
            occupancies_parent_migration, predicted_occupancies_parent_migration, mad_parent_migration, beta_parent_migration, species_occupancies_parent_migration  = utils.predict_occupancy(s_by_s_parent_migration_t.T, range(s_by_s_parent_migration_t.shape[1]))
            occupancies_no_migration, predicted_occupancies_no_migration, mad_no_migration, beta_no_migration, species_occupancies_no_migration  = utils.predict_occupancy(s_by_s_no_migration_t.T, range(s_by_s_no_migration_t.shape[1]))

            # KS test
            rel_s_by_s_global_migration_t = s_by_s_global_migration_t.T/s_by_s_global_migration_t.sum(axis=1)
            rel_s_by_s_parent_migration_t = s_by_s_parent_migration_t.T/s_by_s_parent_migration_t.sum(axis=1)
            rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)


            # we need the transpose.
            log_afd_global_migration_t, rescaled_log_afd_global_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_global_migration_t.T)
            log_afd_no_migration_t, rescaled_log_afd_no_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_no_migration_t.T)
            log_afd_parent_migration_t, rescaled_log_afd_parent_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_parent_migration_t.T)

            # intersection of communities in both transfer 12 and 18
            intersect_global_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('Global_migration',4)][t+1]
            intersect_no_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('No_migration',4)][t+1]
            intersect_parent_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('Parent_migration',4)][t+1]

            # subtract by one
            rel_s_by_s_global_migration_t_intersect = rel_s_by_s_global_migration_t[:,(intersect_global_migration_idx)]
            rel_s_by_s_no_migration_t_intersect = rel_s_by_s_no_migration_t[:,(intersect_no_migration_idx)]
            rel_s_by_s_parent_migration_t_intersect = rel_s_by_s_parent_migration_t[:,(intersect_parent_migration_idx)]

            log_afd_global_migration_t_intersect, rescaled_log_afd_global_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_global_migration_t_intersect.T)
            log_afd_parent_migration_t_intersect, rescaled_log_afd_parent_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_parent_migration_t_intersect.T)
            log_afd_no_migration_t_intersect, rescaled_log_afd_no_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_no_migration_t_intersect.T)

            ks_global_vs_no, p_value_ks_global_vs_no = stats.ks_2samp(log_afd_global_migration_t, log_afd_no_migration_t)
            ks_parent_vs_no, p_value_ks_parent_vs_no = stats.ks_2samp(log_afd_parent_migration_t, log_afd_no_migration_t)

            ks_rescaled_global_vs_no, p_value_ks_rescaled_global_vs_no = stats.ks_2samp(rescaled_log_afd_global_migration_t, rescaled_log_afd_no_migration_t)
            ks_rescaled_parent_vs_no, p_value_ks_rescaled_parent_vs_no = stats.ks_2samp(rescaled_log_afd_parent_migration_t, rescaled_log_afd_no_migration_t)


            afd_dict['global_migration'][t] = {}
            afd_dict['parent_migration'][t] = {}
            afd_dict['no_migration'][t] = {}

            afd_dict['global_migration'][t]['afd'] = log_afd_global_migration_t
            afd_dict['parent_migration'][t]['afd'] = log_afd_parent_migration_t
            afd_dict['no_migration'][t]['afd'] = log_afd_no_migration_t

            afd_dict['global_migration'][t]['rescaled_afd'] = rescaled_log_afd_global_migration_t
            afd_dict['parent_migration'][t]['rescaled_afd'] = rescaled_log_afd_parent_migration_t
            afd_dict['no_migration'][t]['rescaled_afd'] = rescaled_log_afd_no_migration_t

            afd_dict['global_migration'][t]['afd_intersect'] = log_afd_global_migration_t_intersect
            afd_dict['parent_migration'][t]['afd_intersect'] = log_afd_parent_migration_t_intersect
            afd_dict['no_migration'][t]['afd_intersect'] = log_afd_no_migration_t_intersect

            afd_dict['global_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_global_migration_t_intersect
            afd_dict['parent_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_parent_migration_t_intersect
            afd_dict['no_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_no_migration_t_intersect

            error_global_migration = np.absolute(occupancies_global_migration - predicted_occupancies_global_migration)/occupancies_global_migration
            error_parent_migration = np.absolute(occupancies_parent_migration - predicted_occupancies_parent_migration)/occupancies_parent_migration
            error_no_migration = np.absolute(occupancies_no_migration - predicted_occupancies_no_migration)/occupancies_no_migration

            mean_log_error_global_migration = np.mean(np.log10(error_global_migration[error_global_migration>0]))
            mean_log_error_parent_migration = np.mean(np.log10(error_parent_migration[error_parent_migration>0]))
            mean_log_error_no_migration = np.mean(np.log10(error_no_migration[error_no_migration>0]))

            # taylors law
            means_global_migration, variances_global_migration, species_to_keep_global_migration = utils.get_species_means_and_variances(rel_s_by_s_global_migration_t, range(rel_s_by_s_global_migration_t.shape[0]), zeros=True)
            means_parent_migration, variances_parent_migration, species_to_keep_parent_migration = utils.get_species_means_and_variances(rel_s_by_s_parent_migration_t, range(rel_s_by_s_parent_migration_t.shape[0]), zeros=True)
            means_no_migration, variances_no_migration, species_to_keep_no_migration = utils.get_species_means_and_variances(rel_s_by_s_no_migration_t, range(rel_s_by_s_no_migration_t.shape[0]), zeros=True)

            # filter observations with mean greter than 0.95 for taylor's law
            idx_to_keep_global = (means_global_migration<0.95)
            idx_to_keep_parent = (means_parent_migration<0.95)
            idx_to_keep_no = (means_no_migration<0.95)

            means_global_migration = means_global_migration[idx_to_keep_global]
            variances_global_migration = variances_global_migration[idx_to_keep_global]

            means_parent_migration = means_parent_migration[idx_to_keep_parent]
            variances_parent_migration = variances_parent_migration[idx_to_keep_parent]

            means_no_migration = means_no_migration[idx_to_keep_no]
            variances_no_migration = variances_no_migration[idx_to_keep_no]

            # log transform
            means_global_migration_log10 = np.log10(means_global_migration)
            variances_global_migration_log10 = np.log10(variances_global_migration)

            means_parent_migration_log10 = np.log10(means_parent_migration)
            variances_parent_migration_log10 = np.log10(variances_parent_migration)

            means_no_migration_log10 = np.log10(means_no_migration)
            variances_no_migration_log10 = np.log10(variances_no_migration)


            # save means and variances for 12 vs. 18 tests
            mean_var_dict['global_migration'][t] = {}
            mean_var_dict['parent_migration'][t] = {}
            mean_var_dict['no_migration'][t] = {}

            mean_var_dict['global_migration'][t]['means_log10'] = means_global_migration_log10
            mean_var_dict['global_migration'][t]['variances_log10'] = variances_global_migration_log10

            mean_var_dict['parent_migration'][t]['means_log10'] = means_parent_migration_log10
            mean_var_dict['parent_migration'][t]['variances_log10'] = variances_parent_migration_log10

            mean_var_dict['no_migration'][t]['means_log10'] = means_no_migration_log10
            mean_var_dict['no_migration'][t]['variances_log10'] = variances_no_migration_log10


        # check whether you need to skip this iteration
        skip_iter = False
        for t in [11, 17]:
            # t-test b/w migration and no migration 
            # need at least three data points
            if (len(mean_var_dict['global_migration'][t]['means_log10']) < 3) or (len(mean_var_dict['parent_migration'][t]['means_log10']) < 3) or (len(mean_var_dict['no_migration'][t]['means_log10']) < 3):
                skip_iter = True
            

        for treatment in utils.experiments_no_inocula:

            if (len(mean_var_dict[treatment][11]['means_log10']) < 3) or (len(mean_var_dict[treatment][17]['means_log10']) < 3):
                skip_iter = True


        # skip for both transfers 12 and 18
        if skip_iter == True:
            continue


        # now run the analyses
        for t in [11, 17]:

            slope_global, slope_no, t_slope_global, intercept_global, intercept_no, t_intercept_global, r_value_global, r_value_no = utils.t_statistic_two_slopes(mean_var_dict['global_migration'][t]['means_log10'], mean_var_dict['global_migration'][t]['variances_log10'], mean_var_dict['no_migration'][t]['means_log10'], mean_var_dict['no_migration'][t]['variances_log10'])
            slope_parent, slope_no, t_slope_parent, intercept_parent, intercept_no, t_intercept_parent, r_value_parent, r_value_no = utils.t_statistic_two_slopes(mean_var_dict['parent_migration'][t]['means_log10'], mean_var_dict['parent_migration'][t]['variances_log10'], mean_var_dict['no_migration'][t]['means_log10'], mean_var_dict['no_migration'][t]['variances_log10'])

            rho_dict[t]['global_migration']['taylors_slope'].append(slope_global)
            rho_dict[t]['parent_migration']['taylors_slope'].append(slope_parent)
            rho_dict[t]['no_migration']['taylors_slope'].append(slope_no)

            rho_dict[t]['global_migration']['taylors_intercept'].append(intercept_global)
            rho_dict[t]['parent_migration']['taylors_intercept'].append(intercept_parent)
            rho_dict[t]['no_migration']['taylors_intercept'].append(intercept_no)

            rho_dict[t]['global_migration']['t_slope'].append(t_slope_global)
            rho_dict[t]['parent_migration']['t_slope'].append(t_slope_parent)

            rho_dict[t]['global_migration']['t_intercept'].append(t_intercept_global)
            rho_dict[t]['parent_migration']['t_intercept'].append(t_intercept_parent)


            rho_dict[t]['global_migration']['mean_log_error'].append(mean_log_error_global_migration)
            rho_dict[t]['parent_migration']['mean_log_error'].append(mean_log_error_parent_migration)
            rho_dict[t]['no_migration']['mean_log_error'].append(mean_log_error_no_migration)

            # KS test
            rho_dict[t]['global_migration']['ks_migration_vs_no'].append(ks_global_vs_no)
            rho_dict[t]['parent_migration']['ks_migration_vs_no'].append(ks_parent_vs_no)
            rho_dict[t]['global_migration']['ks_rescaled_migration_vs_no'].append(ks_rescaled_global_vs_no)
            rho_dict[t]['parent_migration']['ks_rescaled_migration_vs_no'].append(ks_rescaled_parent_vs_no)


            


        # 12 vs 18 trasnfers
        for treatment in utils.experiments_no_inocula:

            ks_12_18, p_value_ks_12_18 = stats.ks_2samp(afd_dict[treatment][17]['afd'], afd_dict[treatment][11]['afd'])
            ks_rescaled_12_18, p_value_ks_rescaled_12_18 = stats.ks_2samp(afd_dict[treatment][17]['rescaled_afd'], afd_dict[treatment][11]['rescaled_afd'])

            ks_12_18_intersect, p_value_ks_12_18 = stats.ks_2samp(afd_dict[treatment][17]['afd_intersect'], afd_dict[treatment][11]['afd_intersect'])
            ks_rescaled_12_18_intersect, p_value_ks_rescaled_12_18 = stats.ks_2samp(afd_dict[treatment][17]['rescaled_afd_intersect'], afd_dict[treatment][11]['rescaled_afd_intersect'])

            rho_dict['ks_12_vs_18'][treatment].append(ks_12_18)
            rho_dict['ks_rescaled_12_vs_18'][treatment].append(ks_rescaled_12_18)
            rho_dict['ks_12_vs_18_intersect'][treatment].append(ks_12_18_intersect)
            rho_dict['ks_rescaled_12_vs_18_intersect'][treatment].append(ks_rescaled_12_18_intersect)

            # 12 vs. 18 Taylors law slope
            means_log10_12 = mean_var_dict[treatment][11]['means_log10']
            variances_log10_12 = mean_var_dict[treatment][11]['variances_log10']
            means_log10_18 = mean_var_dict[treatment][17]['means_log10']
            variances_log10_18 = mean_var_dict[treatment][17]['variances_log10']

            slope_18, slope_12, t_slope_18_vs_12, intercept_18, intercept_12, t_intercept_18_vs_12, r_value_18, r_value_12 = utils.t_statistic_two_slopes(means_log10_18, variances_log10_18, means_log10_12, variances_log10_12)

            rho_dict['slope_12_vs_18'][treatment]['slope_12'].append(slope_12)
            rho_dict['slope_12_vs_18'][treatment]['slope_18'].append(slope_18)
            rho_dict['slope_12_vs_18'][treatment]['slope_t_test'].append(t_slope_18_vs_12)
            rho_dict['slope_12_vs_18'][treatment]['intercept_12'].append(intercept_12)
            rho_dict['slope_12_vs_18'][treatment]['intercept_18'].append(intercept_18)
            rho_dict['slope_12_vs_18'][treatment]['intercept_t_test'].append(t_intercept_18_vs_12)
            rho_dict['slope_12_vs_18'][treatment]['rho_12'].append(r_value_12)
            rho_dict['slope_12_vs_18'][treatment]['rho_18'].append(r_value_18)


            # Mean KS test...
            s_by_s_12 = s_by_s_dict[11][treatment]
            s_by_s_18 = s_by_s_dict[17][treatment]

            rel_s_by_s_12 = s_by_s_12/s_by_s_12.sum(axis=0)
            rel_s_by_s_18 = s_by_s_18/s_by_s_18.sum(axis=0)

            # identfify taxa present in all samples
            occupancy_12 = np.sum(rel_s_by_s_12>0, axis=0)/rel_s_by_s_12.shape[0]
            occupancy_18 = np.sum(rel_s_by_s_18>0, axis=0)/rel_s_by_s_18.shape[0]

            # occupancy of one in both samples
            to_keep_idx = (occupancy_12==1) & (occupancy_18==1)

            rel_s_by_s_12_subset = rel_s_by_s_12[:,to_keep_idx]
            rel_s_by_s_18_subset = rel_s_by_s_18[:,to_keep_idx]

            rel_s_by_s_12_subset_log10 = np.log10(rel_s_by_s_12_subset)
            rel_s_by_s_18_subset_log10 = np.log10(rel_s_by_s_18_subset)

            ks_statistic_afd_all = []
            ks_statistic_rescaled_afd_all = []

            for afd_idx in range(rel_s_by_s_12_subset.shape[1]):

                afd_12 = rel_s_by_s_12_subset_log10[:,afd_idx]
                afd_18 = rel_s_by_s_18_subset_log10[:,afd_idx]

                rescaled_afd_12 = (afd_12 - np.mean(afd_12))/np.std(afd_12)
                rescaled_afd_18 = (afd_18 - np.mean(afd_18))/np.std(afd_18)

                ks_statistic_afd, p_value = stats.ks_2samp(afd_12, afd_18)
                ks_statistic_rescaled_afd, p_value = stats.ks_2samp(rescaled_afd_12, rescaled_afd_18)

                scaled_ks_statistic_afd = ks_statistic_afd*np.sqrt(len(afd_12)*len(afd_18)/(len(afd_12) + len(afd_18)))
                scaled_ks_statistic_rescaled_afd = ks_statistic_rescaled_afd*np.sqrt(len(rescaled_afd_12)*len(rescaled_afd_18)/(len(rescaled_afd_12) + len(rescaled_afd_18)))

                ks_statistic_afd_all.append(scaled_ks_statistic_afd)
                ks_statistic_rescaled_afd_all.append(scaled_ks_statistic_rescaled_afd)


            rho_dict['mean_over_asv_ks_12_vs_18'][treatment].append(np.mean(ks_statistic_afd_all))
            rho_dict['mean_over_asv_ks_rescaled_12_vs_18'][treatment].append(np.mean(ks_statistic_rescaled_afd_all))


        rho_dict['tau_all'].append(tau_i)
        rho_dict['sigma_all'].append(sigma_i)


    if return_dict == True:
        return rho_dict
    
    else:

        if label == None:
            simulation_migration_all_path_ = simulation_migration_all_abc_path
        
        else:
            simulation_migration_all_path_ = simulation_all_migration_fixed_parameters_path % label

        sys.stderr.write("Saving dictionary...\n")
        with open(simulation_migration_all_path_, 'wb') as handle:
            pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write("Done!\n")







def run_simulation_all_migration_afd_abc(n_iter=10000, tau=None, sigma=None, label=None, return_dict = False):

    # ABC simulation for AFD

    # randomly sample parameters if none are provided
    if tau == None:
        tau_all = (np.random.uniform(tau_min, tau_max, size=n_iter*1000)).tolist()
    else:
        tau_all = [tau]*(n_iter*100)

    if sigma == None:
        sigma_all = (10**(np.random.uniform(np.log10(sigma_min), np.log10(sigma_max), size=n_iter*1000))).tolist()
    else:
        sigma_all = [sigma]*(n_iter*100)


    transfers = range(18)

    rho_dict = {}
    rho_dict['tau_all'] = []
    rho_dict['sigma_all'] = []

    if return_dict == False:
        sys.stderr.write("Starting ABC simulation for all migration treatments...\n")
    
    # statistics for all treatments
    all_stats_labels = ['ks_12_vs_18', 'ks_rescaled_12_vs_18', 'slope_12_vs_18', 'ks_12_vs_18_intersect', 'ks_rescaled_12_vs_18_intersect', 'mean_over_asv_ks_12_vs_18', 'mean_over_asv_ks_rescaled_12_vs_18', 'mean_over_asv_ks_12_vs_18_attractor', 'mean_over_asv_ks_rescaled_12_vs_18_attractor', 'n_occupancy_one', 'n_occupancy_one_attractor']
    for s in all_stats_labels:
        rho_dict[s] = {}

        for treatment in utils.experiments_no_inocula:
            
            if s != 'slope_12_vs_18': 
                rho_dict[s][treatment] = []

            else:
                rho_dict[s][treatment] = {}
                rho_dict[s][treatment]['slope_12'] = []
                rho_dict[s][treatment]['slope_18'] = []
                rho_dict[s][treatment]['slope_t_test'] = []
                rho_dict[s][treatment]['intercept_12'] = []
                rho_dict[s][treatment]['intercept_18'] = []
                rho_dict[s][treatment]['intercept_t_test'] = []
                rho_dict[s][treatment]['rho_12'] = []
                rho_dict[s][treatment]['rho_18'] = []


    for t in transfers:
        
        rho_dict[t] = {}

        for treatment in utils.experiments_no_inocula:
            rho_dict[t][treatment] = {}
            rho_dict[t][treatment]['taylors_slope'] = []
            rho_dict[t][treatment]['taylors_intercept'] = []
            rho_dict[t][treatment]['mean_log_error'] = []

            # only global and parent migration
            if treatment != 'no_migration':

                rho_dict[t][treatment]['t_slope'] = []
                rho_dict[t][treatment]['t_intercept'] = []

                # ks test of AFD
                rho_dict[t][treatment]['ks_migration_vs_no'] = []
                rho_dict[t][treatment]['ks_rescaled_migration_vs_no'] = []


    while len(rho_dict['tau_all']) < n_iter:
        
        n_iter_successful = len(rho_dict['tau_all'])

        if (n_iter_successful+1) % 1000 == 0:
            print(n_iter_successful+1)


        if len(tau_all) == 0:
            break

        
        tau_i = tau_all.pop()
        sigma_i = sigma_all.pop()

        s_by_s_global_migration, s_by_s_parent_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)

        afd_dict = {}
        afd_dict['global_migration'] = {}
        afd_dict['parent_migration'] = {}
        afd_dict['no_migration'] = {}

        mean_var_dict = {}
        mean_var_dict['global_migration'] = {}
        mean_var_dict['parent_migration'] = {}
        mean_var_dict['no_migration'] = {}


        # make sure there are ASVs with occupancy 1 in both transfers 11 and 17.
        #occupancy_one_status = True
        s_by_s_global_migration_12 = s_by_s_global_migration[11,:,:]
        s_by_s_parent_migration_12 = s_by_s_parent_migration[11,:,:]
        s_by_s_no_migration_12 = s_by_s_no_migration[11,:,:]

        community_reps_global_12_idx = rep_number_and_read_count_dict[('Global_migration',4)][11]['community_reps_idx']
        community_reps_parent_12_idx = rep_number_and_read_count_dict[('Parent_migration',4)][11]['community_reps_idx']
        community_reps_no_12_idx = rep_number_and_read_count_dict[('No_migration',4)][11]['community_reps_idx']

        # select communities actually sampled
        s_by_s_global_migration_12 = s_by_s_global_migration_12[community_reps_global_12_idx,:]
        s_by_s_parent_migration_12 = s_by_s_parent_migration_12[community_reps_parent_12_idx,:]
        s_by_s_no_migration_12 = s_by_s_no_migration_12[community_reps_no_12_idx,:]


        s_by_s_global_migration_18 = s_by_s_global_migration[17,:,:]
        s_by_s_parent_migration_18 = s_by_s_parent_migration[17,:,:]
        s_by_s_no_migration_18 = s_by_s_no_migration[17,:,:]

        community_reps_global_18_idx = rep_number_and_read_count_dict[('Global_migration',4)][17]['community_reps_idx']
        community_reps_parent_18_idx = rep_number_and_read_count_dict[('Parent_migration',4)][17]['community_reps_idx']
        community_reps_no_18_idx = rep_number_and_read_count_dict[('No_migration',4)][17]['community_reps_idx']

        # select communities actually sampled
        s_by_s_global_migration_18 = s_by_s_global_migration_18[community_reps_global_18_idx,:]
        s_by_s_parent_migration_18 = s_by_s_parent_migration_18[community_reps_parent_18_idx,:]
        s_by_s_no_migration_18 = s_by_s_no_migration_18[community_reps_no_18_idx,:]

        occupancy_global_12 = np.sum(s_by_s_global_migration_12>0, axis=0)/s_by_s_global_migration_12.shape[0]
        occupancy_parent_12 = np.sum(s_by_s_parent_migration_12>0, axis=0)/s_by_s_parent_migration_12.shape[0]
        occupancy_no_12 = np.sum(s_by_s_no_migration_12>0, axis=0)/s_by_s_no_migration_12.shape[0]

        occupancy_global_18 = np.sum(s_by_s_global_migration_18>0, axis=0)/s_by_s_global_migration_18.shape[0]
        occupancy_parent_18 = np.sum(s_by_s_parent_migration_18>0, axis=0)/s_by_s_parent_migration_18.shape[0]
        occupancy_no_18 = np.sum(s_by_s_no_migration_18>0, axis=0)/s_by_s_no_migration_18.shape[0]

        n_occupancy_one_global = sum((occupancy_global_12==1) & (occupancy_global_18==1))
        n_occupancy_one_parent = sum((occupancy_parent_12==1) & (occupancy_parent_18==1))
        n_occupancy_one_no = sum((occupancy_no_12==1) & (occupancy_no_18==1))


        # need at leat one ASV with an occupancy of one for cross-comparisons.
        if (n_occupancy_one_global==0) or (n_occupancy_one_parent==0) or (n_occupancy_one_no==0):
            continue


        s_by_s_dict = {}
        #for t in transfers:
        for t in [11, 17]:

            s_by_s_global_migration_t = s_by_s_global_migration[t,:,:]
            s_by_s_parent_migration_t = s_by_s_parent_migration[t,:,:]
            s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

            community_reps_global_t_idx = rep_number_and_read_count_dict[('Global_migration',4)][t]['community_reps_idx']
            community_reps_parent_t_idx = rep_number_and_read_count_dict[('Parent_migration',4)][t]['community_reps_idx']
            community_reps_no_t_idx = rep_number_and_read_count_dict[('No_migration',4)][t]['community_reps_idx']

            # select communities actually sampled
            s_by_s_global_migration_t = s_by_s_global_migration_t[community_reps_global_t_idx,:]
            s_by_s_parent_migration_t = s_by_s_parent_migration_t[community_reps_parent_t_idx,:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[community_reps_no_t_idx,:]

            s_by_s_dict[t] = {}
            s_by_s_dict[t]['global_migration'] = s_by_s_global_migration_t
            s_by_s_dict[t]['parent_migration'] = s_by_s_parent_migration_t
            s_by_s_dict[t]['no_migration'] = s_by_s_no_migration_t

            # remove zeros
            s_by_s_global_migration_t = s_by_s_global_migration_t[(~np.all(s_by_s_global_migration_t == 0, axis=1)),:]
            s_by_s_parent_migration_t = s_by_s_parent_migration_t[(~np.all(s_by_s_parent_migration_t == 0, axis=1)),:]
            s_by_s_no_migration_t = s_by_s_no_migration_t[(~np.all(s_by_s_no_migration_t == 0, axis=1)),:]

            occupancies_global_migration, predicted_occupancies_global_migration, mad_global_migration, beta_global_migration, species_occupancies_global_migration  = utils.predict_occupancy(s_by_s_global_migration_t.T, range(s_by_s_global_migration_t.shape[1]))
            occupancies_parent_migration, predicted_occupancies_parent_migration, mad_parent_migration, beta_parent_migration, species_occupancies_parent_migration  = utils.predict_occupancy(s_by_s_parent_migration_t.T, range(s_by_s_parent_migration_t.shape[1]))
            occupancies_no_migration, predicted_occupancies_no_migration, mad_no_migration, beta_no_migration, species_occupancies_no_migration  = utils.predict_occupancy(s_by_s_no_migration_t.T, range(s_by_s_no_migration_t.shape[1]))

            # KS test
            rel_s_by_s_global_migration_t = s_by_s_global_migration_t.T/s_by_s_global_migration_t.sum(axis=1)
            rel_s_by_s_parent_migration_t = s_by_s_parent_migration_t.T/s_by_s_parent_migration_t.sum(axis=1)
            rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

            

            # we need the transpose.
            log_afd_global_migration_t, rescaled_log_afd_global_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_global_migration_t.T)
            log_afd_no_migration_t, rescaled_log_afd_no_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_no_migration_t.T)
            log_afd_parent_migration_t, rescaled_log_afd_parent_migration_t = utils.get_flat_rescaled_afd(rel_s_by_s_parent_migration_t.T)

            # intersection of communities in both transfer 12 and 18
            intersect_global_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('Global_migration',4)][t+1]
            intersect_no_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('No_migration',4)][t+1]
            intersect_parent_migration_idx = sample_intersect_12_18_dict['intersection_idx'][('Parent_migration',4)][t+1]

            # subtract by one
            rel_s_by_s_global_migration_t_intersect = rel_s_by_s_global_migration_t[:,(intersect_global_migration_idx)]
            rel_s_by_s_no_migration_t_intersect = rel_s_by_s_no_migration_t[:,(intersect_no_migration_idx)]
            rel_s_by_s_parent_migration_t_intersect = rel_s_by_s_parent_migration_t[:,(intersect_parent_migration_idx)]

            log_afd_global_migration_t_intersect, rescaled_log_afd_global_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_global_migration_t_intersect.T)
            log_afd_parent_migration_t_intersect, rescaled_log_afd_parent_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_parent_migration_t_intersect.T)
            log_afd_no_migration_t_intersect, rescaled_log_afd_no_migration_t_intersect, = utils.get_flat_rescaled_afd(rel_s_by_s_no_migration_t_intersect.T)

            ks_global_vs_no, p_value_ks_global_vs_no = stats.ks_2samp(log_afd_global_migration_t, log_afd_no_migration_t)
            ks_parent_vs_no, p_value_ks_parent_vs_no = stats.ks_2samp(log_afd_parent_migration_t, log_afd_no_migration_t)

            ks_rescaled_global_vs_no, p_value_ks_rescaled_global_vs_no = stats.ks_2samp(rescaled_log_afd_global_migration_t, rescaled_log_afd_no_migration_t)
            ks_rescaled_parent_vs_no, p_value_ks_rescaled_parent_vs_no = stats.ks_2samp(rescaled_log_afd_parent_migration_t, rescaled_log_afd_no_migration_t)


            afd_dict['global_migration'][t] = {}
            afd_dict['parent_migration'][t] = {}
            afd_dict['no_migration'][t] = {}

            afd_dict['global_migration'][t]['afd'] = log_afd_global_migration_t
            afd_dict['parent_migration'][t]['afd'] = log_afd_parent_migration_t
            afd_dict['no_migration'][t]['afd'] = log_afd_no_migration_t

            afd_dict['global_migration'][t]['rescaled_afd'] = rescaled_log_afd_global_migration_t
            afd_dict['parent_migration'][t]['rescaled_afd'] = rescaled_log_afd_parent_migration_t
            afd_dict['no_migration'][t]['rescaled_afd'] = rescaled_log_afd_no_migration_t

            afd_dict['global_migration'][t]['afd_intersect'] = log_afd_global_migration_t_intersect
            afd_dict['parent_migration'][t]['afd_intersect'] = log_afd_parent_migration_t_intersect
            afd_dict['no_migration'][t]['afd_intersect'] = log_afd_no_migration_t_intersect

            afd_dict['global_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_global_migration_t_intersect
            afd_dict['parent_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_parent_migration_t_intersect
            afd_dict['no_migration'][t]['rescaled_afd_intersect'] = rescaled_log_afd_no_migration_t_intersect

            error_global_migration = np.absolute(occupancies_global_migration - predicted_occupancies_global_migration)/occupancies_global_migration
            error_parent_migration = np.absolute(occupancies_parent_migration - predicted_occupancies_parent_migration)/occupancies_parent_migration
            error_no_migration = np.absolute(occupancies_no_migration - predicted_occupancies_no_migration)/occupancies_no_migration

            # taylors law
            means_global_migration, variances_global_migration, species_to_keep_global_migration = utils.get_species_means_and_variances(rel_s_by_s_global_migration_t, range(rel_s_by_s_global_migration_t.shape[0]), zeros=True)
            means_parent_migration, variances_parent_migration, species_to_keep_parent_migration = utils.get_species_means_and_variances(rel_s_by_s_parent_migration_t, range(rel_s_by_s_parent_migration_t.shape[0]), zeros=True)
            means_no_migration, variances_no_migration, species_to_keep_no_migration = utils.get_species_means_and_variances(rel_s_by_s_no_migration_t, range(rel_s_by_s_no_migration_t.shape[0]), zeros=True)

            # filter observations with mean greter than 0.95 for taylor's law
            idx_to_keep_global = (means_global_migration<0.95)
            idx_to_keep_parent = (means_parent_migration<0.95)
            idx_to_keep_no = (means_no_migration<0.95)

            means_global_migration = means_global_migration[idx_to_keep_global]
            variances_global_migration = variances_global_migration[idx_to_keep_global]

            means_parent_migration = means_parent_migration[idx_to_keep_parent]
            variances_parent_migration = variances_parent_migration[idx_to_keep_parent]

            means_no_migration = means_no_migration[idx_to_keep_no]
            variances_no_migration = variances_no_migration[idx_to_keep_no]

            # log transform
            means_global_migration_log10 = np.log10(means_global_migration)
            variances_global_migration_log10 = np.log10(variances_global_migration)

            means_parent_migration_log10 = np.log10(means_parent_migration)
            variances_parent_migration_log10 = np.log10(variances_parent_migration)

            means_no_migration_log10 = np.log10(means_no_migration)
            variances_no_migration_log10 = np.log10(variances_no_migration)


            # save means and variances for 12 vs. 18 tests
            mean_var_dict['global_migration'][t] = {}
            mean_var_dict['parent_migration'][t] = {}
            mean_var_dict['no_migration'][t] = {}

            mean_var_dict['global_migration'][t]['means_log10'] = means_global_migration_log10
            mean_var_dict['global_migration'][t]['variances_log10'] = variances_global_migration_log10

            mean_var_dict['parent_migration'][t]['means_log10'] = means_parent_migration_log10
            mean_var_dict['parent_migration'][t]['variances_log10'] = variances_parent_migration_log10

            mean_var_dict['no_migration'][t]['means_log10'] = means_no_migration_log10
            mean_var_dict['no_migration'][t]['variances_log10'] = variances_no_migration_log10


            # KS test
            rho_dict[t]['global_migration']['ks_migration_vs_no'].append(ks_global_vs_no)
            rho_dict[t]['parent_migration']['ks_migration_vs_no'].append(ks_parent_vs_no)
            rho_dict[t]['global_migration']['ks_rescaled_migration_vs_no'].append(ks_rescaled_global_vs_no)
            rho_dict[t]['parent_migration']['ks_rescaled_migration_vs_no'].append(ks_rescaled_parent_vs_no)


        # 12 vs 18 trasnfers
        for treatment in utils.experiments_no_inocula:

            ks_12_18, p_value_ks_12_18 = stats.ks_2samp(afd_dict[treatment][17]['afd'], afd_dict[treatment][11]['afd'])
            ks_rescaled_12_18, p_value_ks_rescaled_12_18 = stats.ks_2samp(afd_dict[treatment][17]['rescaled_afd'], afd_dict[treatment][11]['rescaled_afd'])

            ks_12_18_intersect, p_value_ks_12_18 = stats.ks_2samp(afd_dict[treatment][17]['afd_intersect'], afd_dict[treatment][11]['afd_intersect'])
            ks_rescaled_12_18_intersect, p_value_ks_rescaled_12_18 = stats.ks_2samp(afd_dict[treatment][17]['rescaled_afd_intersect'], afd_dict[treatment][11]['rescaled_afd_intersect'])

            rho_dict['ks_12_vs_18'][treatment].append(ks_12_18)
            rho_dict['ks_rescaled_12_vs_18'][treatment].append(ks_rescaled_12_18)
            rho_dict['ks_12_vs_18_intersect'][treatment].append(ks_12_18_intersect)
            rho_dict['ks_rescaled_12_vs_18_intersect'][treatment].append(ks_rescaled_12_18_intersect)


            # Mean KS test...
            s_by_s_12 = s_by_s_dict[11][treatment]
            s_by_s_18 = s_by_s_dict[17][treatment]

            rel_s_by_s_12 = s_by_s_12/s_by_s_12.sum(axis=0)
            rel_s_by_s_18 = s_by_s_18/s_by_s_18.sum(axis=0)

            # identfify taxa present in all samples
            occupancy_12 = np.sum(rel_s_by_s_12>0, axis=0)/rel_s_by_s_12.shape[0]
            occupancy_18 = np.sum(rel_s_by_s_18>0, axis=0)/rel_s_by_s_18.shape[0]

            # occupancy of one in both samples
            to_keep_idx = (occupancy_12==1) & (occupancy_18==1)

            rel_s_by_s_12_subset = rel_s_by_s_12[:,to_keep_idx]
            rel_s_by_s_18_subset = rel_s_by_s_18[:,to_keep_idx]

            rel_s_by_s_12_subset_log10 = np.log10(rel_s_by_s_12_subset)
            rel_s_by_s_18_subset_log10 = np.log10(rel_s_by_s_18_subset)

            ks_statistic_afd_all = []
            ks_statistic_rescaled_afd_all = []

            for afd_idx in range(rel_s_by_s_12_subset.shape[1]):

                afd_12 = rel_s_by_s_12_subset_log10[:,afd_idx]
                afd_18 = rel_s_by_s_18_subset_log10[:,afd_idx]

                rescaled_afd_12 = (afd_12 - np.mean(afd_12))/np.std(afd_12)
                rescaled_afd_18 = (afd_18 - np.mean(afd_18))/np.std(afd_18)

                ks_statistic_afd, p_value = stats.ks_2samp(afd_12, afd_18)
                ks_statistic_rescaled_afd, p_value = stats.ks_2samp(rescaled_afd_12, rescaled_afd_18)

                scaled_ks_statistic_afd = ks_statistic_afd*np.sqrt(len(afd_12)*len(afd_18)/(len(afd_12) + len(afd_18)))
                scaled_ks_statistic_rescaled_afd = ks_statistic_rescaled_afd*np.sqrt(len(rescaled_afd_12)*len(rescaled_afd_18)/(len(rescaled_afd_12) + len(rescaled_afd_18)))

                ks_statistic_afd_all.append(scaled_ks_statistic_afd)
                ks_statistic_rescaled_afd_all.append(scaled_ks_statistic_rescaled_afd)


            rho_dict['mean_over_asv_ks_12_vs_18'][treatment].append(np.mean(ks_statistic_afd_all))
            rho_dict['mean_over_asv_ks_rescaled_12_vs_18'][treatment].append(np.mean(ks_statistic_rescaled_afd_all))
            rho_dict['n_occupancy_one'][treatment].append(len(ks_statistic_rescaled_afd_all))

            
            ks_statistic_afd_attractor_all = []
            ks_statistic_rescaled_afd_attractor_all = []
            # occupancy of one *within* the attractor
            if treatment == 'no_migration':

                # subset by attractor
                rel_s_by_s_12_attractor = rel_s_by_s_12[attractor_alcaligenaceae_iter_12_idx,:]
                rel_s_by_s_18_attractor = rel_s_by_s_18[attractor_alcaligenaceae_iter_18_idx,:]

                # identfify taxa present in all samples
                occupancy_12_attractor = np.sum(rel_s_by_s_12_attractor>0, axis=0)/rel_s_by_s_12_attractor.shape[0]
                occupancy_18_attractor = np.sum(rel_s_by_s_18_attractor>0, axis=0)/rel_s_by_s_18_attractor.shape[0]

                # occupancy of one in both samples
                to_keep_attractor_idx = (occupancy_12_attractor==1) & (occupancy_18_attractor==1)

                rel_s_by_s_12_subset_attractor = rel_s_by_s_12_attractor[:,to_keep_attractor_idx]
                rel_s_by_s_18_subset_attractor = rel_s_by_s_18_attractor[:,to_keep_attractor_idx]

                rel_s_by_s_12_subset_log10_attractor = np.log10(rel_s_by_s_12_subset_attractor)
                rel_s_by_s_18_subset_log10_attractor = np.log10(rel_s_by_s_18_subset_attractor)

                for afd_idx in range(rel_s_by_s_12_subset_attractor.shape[1]):

                    afd_12_attractor = rel_s_by_s_12_subset_log10_attractor[:,afd_idx]
                    afd_18_attractor = rel_s_by_s_18_subset_log10_attractor[:,afd_idx]

                    rescaled_afd_12_attractor = (afd_12_attractor - np.mean(afd_12_attractor))/np.std(afd_12_attractor)
                    rescaled_afd_18_attractor = (afd_18_attractor - np.mean(afd_18_attractor))/np.std(afd_18_attractor)

                    ks_statistic_afd_attractor, p_value = stats.ks_2samp(afd_12_attractor, afd_18_attractor)
                    ks_statistic_rescaled_afd_attractor, p_value = stats.ks_2samp(rescaled_afd_12_attractor, rescaled_afd_18_attractor)

                    scaled_ks_statistic_afd_attractor = ks_statistic_afd_attractor*np.sqrt(len(afd_12_attractor)*len(afd_18_attractor)/(len(afd_12_attractor) + len(afd_18_attractor)))
                    scaled_ks_statistic_rescaled_afd_attractor = ks_statistic_rescaled_afd_attractor*np.sqrt(len(rescaled_afd_12_attractor)*len(rescaled_afd_18_attractor)/(len(rescaled_afd_12_attractor) + len(rescaled_afd_18_attractor)))

                    ks_statistic_afd_attractor_all.append(scaled_ks_statistic_afd_attractor)
                    ks_statistic_rescaled_afd_attractor_all.append(scaled_ks_statistic_rescaled_afd_attractor)


                rho_dict['mean_over_asv_ks_12_vs_18_attractor'][treatment].append(np.mean(ks_statistic_afd_attractor_all))
                rho_dict['mean_over_asv_ks_rescaled_12_vs_18_attractor'][treatment].append(np.mean(ks_statistic_rescaled_afd_attractor_all))
                rho_dict['n_occupancy_one_attractor'][treatment].append(len(ks_statistic_afd_attractor_all))
            



        rho_dict['tau_all'].append(tau_i)
        rho_dict['sigma_all'].append(sigma_i)


    if return_dict == True:
        return rho_dict
    
    else:

        if label == None:
            simulation_migration_all_path_ = simulation_migration_all_abc_afd_path
        
        else:
            simulation_migration_all_path_ = simulation_all_migration_fixed_parameters_afd_path % label

        sys.stderr.write("Saving dictionary...\n")
        with open(simulation_migration_all_path_, 'wb') as handle:
            pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write("Done!\n")







def run_simulation_all_migration_heatmap(n_iter=100):

    sys.stderr.write("Simulating all migration treatments across parameter combinations...\n")

    transfers = range(18)
    rho_dict = {}
    #for tau_i in [5.744444444444445]:
    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        #for sigma_i in [0.03209147576450548]:
        for sigma_i in sigma_all:

            print(tau_i, sigma_i)

            rho_dict[tau_i][sigma_i] = {}
            rho_dict[tau_i][sigma_i]['ks_12_vs_18'] = {}
            rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18'] = {}
            rho_dict[tau_i][sigma_i]['slope_12_vs_18'] = {}

            for treatment in ['global_migration', 'parent_migration', 'no_migration']:

                rho_dict[tau_i][sigma_i]['ks_12_vs_18'][treatment] = []
                rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18'][treatment] = []
                
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment] = {}
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_12'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_18'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_t_test'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_12'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_18'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_t_test'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['rho_12'] = []
                rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['rho_18'] = []



            for t in transfers:
                rho_dict[tau_i][sigma_i][t] = {}
                rho_dict[tau_i][sigma_i][t]['global_migration'] = {}
                rho_dict[tau_i][sigma_i][t]['parent_migration'] = {}
                rho_dict[tau_i][sigma_i][t]['no_migration'] = {}

                rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_slope'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_slope'] = []
                rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_slope'] = []

                rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_intercept'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_intercept'] = []
                rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_intercept'] = []

                rho_dict[tau_i][sigma_i][t]['global_migration']['t_slope'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['t_slope'] = []

                rho_dict[tau_i][sigma_i][t]['global_migration']['t_intercept'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['t_intercept'] = []

                rho_dict[tau_i][sigma_i][t]['global_migration']['mean_log_error'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['mean_log_error'] = []
                rho_dict[tau_i][sigma_i][t]['no_migration']['mean_log_error'] = []

                # ks test of AFD
                rho_dict[tau_i][sigma_i][t]['global_migration']['ks_migration_vs_no'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['ks_migration_vs_no'] = []
                rho_dict[tau_i][sigma_i][t]['global_migration']['ks_rescaled_migration_vs_no'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['ks_rescaled_migration_vs_no'] = []



            while len(rho_dict[tau_i][sigma_i][transfers[11]]['global_migration']['taylors_intercept']) < n_iter:

                rho_dict_iter = run_simulation_all_migration_abc(n_iter=1, sigma = sigma_i, tau = tau_i, return_dict=True)

                for t in transfers:

                    rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_slope'].append(rho_dict_iter[t]['global_migration']['taylors_slope'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_slope'].append(rho_dict_iter[t]['parent_migration']['taylors_slope'])
                    rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_slope'].append(rho_dict_iter[t]['no_migration']['taylors_slope'])

                    rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_intercept'].append(rho_dict_iter[t]['global_migration']['taylors_intercept'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_intercept'].append(rho_dict_iter[t]['parent_migration']['taylors_intercept'])
                    rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_intercept'].append(rho_dict_iter[t]['no_migration']['taylors_intercept'])

                    rho_dict[tau_i][sigma_i][t]['global_migration']['t_slope'].append(rho_dict_iter[t]['global_migration']['t_slope'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['t_slope'].append(rho_dict_iter[t]['parent_migration']['t_slope'])

                    rho_dict[tau_i][sigma_i][t]['global_migration']['t_intercept'].append(rho_dict_iter[t]['global_migration']['t_intercept'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['t_intercept'].append(rho_dict_iter[t]['parent_migration']['t_intercept'])


                    rho_dict[tau_i][sigma_i][t]['global_migration']['mean_log_error'].append(rho_dict_iter[t]['global_migration']['mean_log_error'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['mean_log_error'].append(rho_dict_iter[t]['parent_migration']['mean_log_error'])
                    rho_dict[tau_i][sigma_i][t]['no_migration']['mean_log_error'].append(rho_dict_iter[t]['no_migration']['mean_log_error'])

                    # KS test 
                    rho_dict[tau_i][sigma_i][t]['global_migration']['ks_migration_vs_no'].append(rho_dict_iter[t]['global_migration']['ks_migration_vs_no'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['ks_migration_vs_no'].append(rho_dict_iter[t]['parent_migration']['ks_migration_vs_no'])
                    
                    rho_dict[tau_i][sigma_i][t]['global_migration']['ks_rescaled_migration_vs_no'].append(rho_dict_iter[t]['global_migration']['ks_rescaled_migration_vs_no'])
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['ks_rescaled_migration_vs_no'].append(rho_dict_iter[t]['parent_migration']['ks_rescaled_migration_vs_no'])


                
                # 12 vs 18 trasnfers
                for treatment in utils.experiments_no_inocula:

                    rho_dict[tau_i][sigma_i]['ks_12_vs_18'][treatment].append(rho_dict_iter['ks_12_vs_18'][treatment])
                    rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18'][treatment].append(rho_dict_iter['ks_rescaled_12_vs_18'][treatment])

                    # 12 vs. 18 Taylors law slope
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_12'].append(rho_dict_iter['slope_12_vs_18'][treatment]['slope_12'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_18'].append(rho_dict_iter['slope_12_vs_18'][treatment]['slope_18'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['slope_t_test'].append(rho_dict_iter['slope_12_vs_18'][treatment]['slope_t_test'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_12'].append(rho_dict_iter['slope_12_vs_18'][treatment]['intercept_12'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_18'].append(rho_dict_iter['slope_12_vs_18'][treatment]['intercept_18'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['intercept_t_test'].append(rho_dict_iter['slope_12_vs_18'][treatment]['intercept_t_test'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['rho_12'].append(rho_dict_iter['slope_12_vs_18'][treatment]['rho_12'])
                    rho_dict[tau_i][sigma_i]['slope_12_vs_18'][treatment]['rho_18'].append(rho_dict_iter['slope_12_vs_18'][treatment]['rho_18'])




    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_migration_all_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def run_simulation_all_migration_afd_heatmap(n_iter=100):

    sys.stderr.write("Simulating AFDs for all migration treatments across parameter combinations...\n")

    rho_dict = {}
    #for tau_i in [tau_all[3]]:
    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:
            #for sigma_i in [0.33049817624300415]:

            if tau_i in max_sigma_dict:
                if sigma_i > max_sigma_dict[tau_i]:
                    continue

            #if sigma_i < 0.33049817624300415:
            #    continue

            print(tau_i, sigma_i)

            statistics_all = ['ks_12_vs_18', 'ks_rescaled_12_vs_18', 'ks_12_vs_18_intersect', 'ks_rescaled_12_vs_18_intersect', 'mean_over_asv_ks_12_vs_18', 'mean_over_asv_ks_rescaled_12_vs_18', 'n_occupancy_one']

            rho_dict[tau_i][sigma_i] = {}
            for s in statistics_all:
                rho_dict[tau_i][sigma_i][s] = {}
                for treatment in ['global_migration', 'parent_migration', 'no_migration']:
                    rho_dict[tau_i][sigma_i][s][treatment] = []


            while len(rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18']['global_migration']) < n_iter:

                try:
                    rho_dict_iter = run_simulation_all_migration_afd_abc(n_iter=1, tau=tau_i, sigma=sigma_i, return_dict=True)
                except ValueError:
                    rho_dict_iter = None
                
                if rho_dict_iter == None:
                    continue

                if len(rho_dict_iter['ks_12_vs_18']['global_migration']) == 0:
                    continue

                if len(rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18']['global_migration']) % 10 == 0:
                    print(len(rho_dict[tau_i][sigma_i]['ks_rescaled_12_vs_18']['global_migration']))

                for s in statistics_all:

                    for treatment in ['global_migration', 'parent_migration', 'no_migration']:
                        rho_dict[tau_i][sigma_i][s][treatment].append(rho_dict_iter[s][treatment][0])


    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_migration_all_afd_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def load_simulation_parent_rho_dict():

    with open(simulation_parent_rho_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



def load_simulation_all_migration_dict():

    with open(simulation_migration_all_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



def load_simulation_global_rho_dict():

    with open(simulation_global_rho_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_





def load_simulation_parent_rho_abc_dict():

    with open(simulation_parent_rho_abc_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



def load_simulation_all_migration_abc_dict():

    with open(simulation_migration_all_abc_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



def load_simulation_all_migration_abc_afd_dict():

    with open(simulation_migration_all_abc_afd_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_



def load_simulation_global_rho_abc_dict():

    with open(simulation_global_rho_abc_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_




def load_simulation_parent_rho_fixed_parameters_dict():

    with open(simulation_parent_rho_fixed_parameters_path, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


#def load_simulation_global_rho_fixed_parameters_dict():

#    with open(simulation_global_rho_fixed_parameters_path, 'rb') as handle:
#        dict_ = pickle.load(handle)
#    return dict_



def load_simulation_global_rho_fixed_parameters_dict(label):

    simulation_global_rho_fixed_parameters_path_ = simulation_global_rho_fixed_parameters_path % label

    with open(simulation_global_rho_fixed_parameters_path_, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


def load_simulation_all_migration_fixed_parameters_dict(label):

    simulation_all_migration_fixed_parameters_path_ = simulation_all_migration_fixed_parameters_path % label

    with open(simulation_all_migration_fixed_parameters_path_, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


def load_simulation_all_migration_fixed_parameters_afd_dict(label):

    simulation_all_migration_fixed_parameters_path_ = simulation_all_migration_fixed_parameters_afd_path % label

    with open(simulation_all_migration_fixed_parameters_path_, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_




if __name__=='__main__':

    print("running simulations")

    # DONE

    # general ABC simultions
    #run_simulation_all_migration_abc(n_iter=1000)

    # ABC simulations for AFD 
    #run_simulation_all_migration_afd_abc(n_iter=1000)

    # parent migration
    #run_simulation_parent_rho_abc(n_iter=1000)

    # global migration
    #run_simulation_global_rho_abc(n_iter=1000)
    

    # heatmaps
    #run_simulation_all_migration_heatmap(n_iter=100)
    #run_simulation_global_rho_heatmap(n_iter=100)
    
    
    # RUNNING    
    
    #run_simulation_all_migration_afd_heatmap(n_iter=100)
    
    



