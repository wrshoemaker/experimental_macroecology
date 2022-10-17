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
simulation_global_rho_path = utils.directory + "/data/simulation_global_rho.pickle"
simulation_migration_all_path = utils.directory + "/data/simulation_migration_all.pickle"



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









def run_simulation_continous_migration(sigma = 0.9, tau = 0.5, dt = 48, T = 864, reps = 92, migration_treatment='none'):
    # just use units of generations for now
    # T = generations
    # tau units of number of generations
    # dt = gens. per-transfer
    noise_term = np.sqrt( sigma*dt/tau  ) # compound paramter, for conveinance

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

    # soil in parent mixed with 100ml, aliquots taken from this culture
    n_cells_parent = 10**12
    # innoculation of replicate populations done with 4ul, 0.004 / 100
    # large innoculation = 40uL, 0.4/100
    # descendent lines are 500ul in size
    n_cells_descendent = 10**10

    n_time_steps = int(T / dt)  # Number of time steps.
    t_gen = np.arange(0, T, dt)
    # merge ancestral SADs to get probability vector for multinomial sampling
    # Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
    # redo later to draw carrying capacities from no migration distribution
    mu_pln=3.4533926814573506
    sigma_pln=2.6967286975393754
    # non zero carrying capacities
    k_to_keep = pln._rvs(mu_pln, sigma_pln, lower_trunc=True, size=n_non_zero_k)
    # randomize order
    k_to_keep = np.random.permutation(k_to_keep)
    #inoc_abund_non_zero = inoc_abund[inoc_abund>0] # get nonzero


    def discrete_slm_continous_migration(q_array, m):
        # q_array = array of replicate and species
        return q_array + (dt*(m/np.exp(q_array))*(1/tau)*(1 - (sigma/2) - (np.exp(q_array) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))


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


    # make vector of initial relative abundances, match with carrying capacities, and add zeros
    #inoc_abund_non_zero_set = list(set(inoc_abund_non_zero))
    #for i_idx, i in enumerate(inoc_abund_non_zero_set):
    #    n_i = sum(inoc_abund_non_zero==i)
    #    n_non_zero_k_i = sum(init_abund_non_zero_with_non_zero_k==i)
    #    n_zero_k_i = n_i - n_non_zero_k_i
    #    # add zeros for species that have non zero initial abundances, but carrying capacities of zerp
    #    k_to_keep = np.append(k_to_keep, [0]*n_zero_k_i)
    #    init_abund_non_zero_with_non_zero_k = np.append(init_abund_non_zero_with_non_zero_k, [0]*n_zero_k_i)

    k_to_keep = np.append(k_to_keep, [0]*len(idx_zero_k))
    # turn into absolute abundances
    k_to_keep = np.random.multinomial(n_cells_descendent, k_to_keep/sum(k_to_keep))

    # create three dimensional vector
    # spcies, replicate, and timestep
    # create array for innnocumum
    n_0_inoc = np.zeros((reps, len(init_abund_rel)))
    for rep in range(reps):
        n_cells_inoc = np.random.binomial(n_cells_parent, 0.004/0.5) # number of cells that get transferred between descendent communities
        n_0_inoc[rep,:] = np.random.multinomial(n_cells_inoc, init_abund_rel)

    q_0_inoc = np.log(n_0_inoc)
    # migration only happens at transfers, no migration when experiment is set up
    m_0 = np.zeros((reps, len(init_abund_rel)))
    q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
    q[0,:,:] = discrete_slm_migration(q_0_inoc, m_0)

    for t in range(1, n_time_steps): # t = transfer number
        # perform transfer
        n_t_minus_1 = np.exp(q[t-1,:,:])
        # remove infs, nan
        n_t_minus_1[(np.isnan(n_t_minus_1)) | np.isneginf(n_t_minus_1) | np.isinf(n_t_minus_1)] = 0
        prob_n_t_minus_1 = (n_t_minus_1.T / np.sum(n_t_minus_1, axis=1)).T

        # make empty array for innoculation sampling
        n_t_inoc = np.zeros((reps, len(init_abund_rel)))
        for rep in range(reps):
            # draw number cells,
            n_cells_transfer = np.random.binomial(n_cells_descendent, 0.004/0.5) # number of cells that get transferred between descendent communities

            n_t_inoc[rep,:] = np.random.multinomial(n_cells_transfer, prob_n_t_minus_1[rep,:])

        q_t_inoc = np.log(n_t_inoc)

        if t > 12:
            m = np.zeros((reps, len(init_abund_rel)))

        else:
            # get migration vector
            if migration_treatment == 'global':

                n_global = np.zeros(len(init_abund_rel))
                for r in range(reps):
                    q_t_r = q[t,r,:]
                    n_t_r = np.exp(q_t_r)
                    n_t_r[n_t_r<0] = 0
                    n_cells_migration = np.random.binomial(n_cells_descendent, 0.004/0.5)
                    n_t_r_sample = np.random.multinomial(n_cells_migration, n_t_r/sum(n_t_r))
                    n_global += n_t_r_sample

                # diluted 10,000 fold
                n_cells_global_diluted = np.random.binomial(sum(n_global), 10**-4)
                # sample to get abundances of diluted global samples
                n_global_diluted = np.random.multinomial(n_cells_global_diluted, n_global/sum(n_global))

                # get number of cells transferred from diluted global sample
                # ask alvaro, but assume for now that the 10,000 fold diluated community is in a volume of size 500uL
                #n_cells_migration_transfer = np.random.binomial(n_cells_global_diluted, 0.004/0.5, size=reps)
                n_cells_migration_transfer = np.random.binomial(n_cells_global_diluted, 0.004/(reps*0.004), size=reps)
                m = np.asarray([np.random.multinomial(l, n_global_diluted/sum(n_global_diluted)) for l in n_cells_migration_transfer])


            elif migration_treatment == 'parent':
                # not sure! ask Alvaro!
                # 0.004/0.5
                # size of PARENT community, n = 100 mL
                n_cells_migration = np.random.binomial(n_cells_parent, 0.004/109900, size=reps)
                m = np.asarray([np.random.multinomial(l, prob_mean_rel_abundances_parent) for l in n_cells_migration])


            else:
                m = np.zeros((reps, len(init_abund_rel)))


        #q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:]))*(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund)))
        q[t,:,:] = discrete_slm_migration(q_t_inoc, m)


    n = np.exp(q)
    # replace negative q values with zero
    n[(np.isnan(n)) | np.isneginf(n) | np.isinf(n)] = 0
    #test = n[-1,0,:]

    return n, k_to_keep, t_gen









def run_simulation_initial_condition_migration(sigma = 0.5, tau = 0.9, dt = 48, T = 864, reps = 92, migration_treatment='global'):
    # just use units of generations for now
    # T = generations
    # tau units of number of generations
    # dt = gens. per-transfer
    noise_term = np.sqrt( sigma*dt/tau  ) # compound paramter, for conveinance

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

    # soil in parent mixed with 100ml, aliquots taken from this culture
    n_cells_parent = 7*(10**7)
    # innoculation of replicate populations done with 4ul, 0.004 / 100
    # large innoculation = 40uL, 0.4/100
    # descendent lines are 500ul in size
    n_cells_descendent = 5*(10**8)
    D_global = 504/60000
    D_parent = 504/60000
    D_transfer = 4/500


    def discrete_slm_initial_condition_migration(q_array, m):
        # q_array = array of replicate and species
        q_array_new = np.log(np.exp(q_array) + m)
        return q_array_new + (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))


    n_time_steps = int(T / dt)  # Number of time steps.
    t_gen = np.arange(0, T, dt)

    # merge ancestral SADs to get probability vector for multinomial sampling
    # Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
    # redo later to draw carrying capacities from no migration distribution
    mu_pln=3.4533926814573506
    sigma_pln=2.6967286975393754

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

                    print(m)

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

    n_migration_reads = sample_simulation_results(n_migration)
    n_no_migration_reads = sample_simulation_results(n_no_migration)

    init_abund_rel_reads = multinomial_sample_reads(init_abund_rel)
    init_abund_rel_reads_rel = init_abund_rel_reads/sum(init_abund_rel_reads)

    return n_migration_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel_reads_rel








def run_simulation_initial_condition_all_migration(sigma = 0.5, tau = 0.9, dt = 48, T = 864, reps = 92):
    # T = 126 generations
    # dt = 7 generations ==> T/dt = 18

    # T = 18*48hrs = 864
    # dt = 48 hrs.

    # just use units of generations for now
    # T = generations
    # tau units of number of generations
    # dt = gens. per-transfer
    noise_term = np.sqrt( sigma*dt/tau  ) # compound paramter, for conveinance

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

    # soil in parent mixed with 100ml, aliquots taken from this culture
    n_cells_parent = 7*(10**7)
    # innoculation of replicate populations done with 4ul, 0.004 / 100
    # large innoculation = 40uL, 0.4/100
    # descendent lines are 500ul in size
    n_cells_descendent = 5*(10**8)
    D_global = 504/60000
    D_parent = 504/60000
    D_transfer = 4/500


    def discrete_slm_initial_condition_migration(q_array, m):
        # q_array = array of replicate and species
        q_array_new = np.log(np.exp(q_array) + m)
        return q_array_new + (dt*(1/tau)*(1 - (sigma/2) - (np.exp(q_array_new) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))


    n_time_steps = int(T / dt)  # Number of time steps.
    t_gen = np.arange(0, T, dt)

    # merge ancestral SADs to get probability vector for multinomial sampling
    # Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
    # redo later to draw carrying capacities from no migration distribution
    mu_pln=3.4533926814573506
    sigma_pln=2.6967286975393754

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

    n_migration_global_reads = sample_simulation_results(n_migration_global)
    n_migration_parent_reads = sample_simulation_results(n_migration_parent)
    n_no_migration_reads = sample_simulation_results(n_no_migration)

    init_abund_rel_reads = multinomial_sample_reads(init_abund_rel)
    init_abund_rel_reads_rel = init_abund_rel_reads/sum(init_abund_rel_reads)

    return n_migration_global_reads, n_migration_parent_reads, n_no_migration_reads, k_to_keep, t_gen, init_abund_rel_reads_rel







def multinomial_sample_reads(array_):

    n_reads = utils.get_dist_read_counts()
    n_reads_ = np.random.choice(n_reads, size=1, replace=True, p=None)[0]

    rel_array_ = array_/sum(array_)

    rel_array_reads = np.random.multinomial(n_reads_, rel_array_)

    return rel_array_reads




def sample_simulation_results(s_by_s_by_t):

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




def run_simulation_parent_rho(iter=100):

    transfers = [11, 17]

    rho_dict = {}

    #rho_dict['migration'] = {}
    #rho_dict['migration_vs_parent'] = {}
    #for t in transfers:
    #    rho_dict['migration'][t] = []
    #    rho_dict['migration_vs_parent'][t] = []

    tau_all = np.linspace(0.5, 6, num=10, endpoint=True)
    sigma_all = np.linspace(0.1, 1.9, num=10, endpoint=True)

    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:

            rho_dict[tau_i][sigma_i] = {}
            for t in transfers:
                rho_dict[tau_i][sigma_i][t] = {}
                rho_dict[tau_i][sigma_i][t]['migration'] = []
                rho_dict[tau_i][sigma_i][t]['migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['no_migration_vs_parent'] = []
                rho_dict[tau_i][sigma_i][t]['mad_ratio_vs_parent'] = []

                #rho_dict[tau_i][sigma_i][t]['migration_taylor_slope'] = []
                #rho_dict[tau_i][sigma_i][t]['migration_taylor_intercept'] = []

                #rho_dict[tau_i][sigma_i][t]['no_migration_taylor_slope'] = []
                #rho_dict[tau_i][sigma_i][t]['no_migration_taylor_intercept'] = []

            #mean_corr_migration_t_all = []
            #mean_corr_migration_vs_parent_t_all = []

            #for i in range(iter):
            while len(rho_dict[tau_i][sigma_i][transfers[0]]['migration']) < iter:

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

                        mean_corr_migration_t = np.corrcoef(np.log10(mean_rel_migration_t), np.log10(mean_rel_no_migration_t))[0,1]

                        slope_migration, intercept_migration, r_value_migration, p_value_migration, std_err_migration = stats.linregress(np.log10(init_abund_rel_migration_vs_parent_t), np.log10(mean_rel_migration_vs_parent_t))
                        slope_no_migration, intercept_no_migration, r_value_no_migration, p_value_no_migration, std_err_no_migration = stats.linregress(np.log10(init_abund_rel_no_migration_vs_parent_t), np.log10(mean_rel_no_migration_vs_parent_t))

                        # use MAD ratio
                        mad_ratio = np.mean(rel_s_by_s_migration_t[idx_ratio_vs_parent,:], axis=1) / np.mean(rel_s_by_s_no_migration_t[idx_ratio_vs_parent,:], axis=1)
                        init_abund_mad_ratio_vs_parent_t = init_abund_rel[idx_ratio_vs_parent]
                        slope_mad, intercept_mad, r_value_mad, p_value_mad, std_err_mad = stats.linregress(np.log10(init_abund_mad_ratio_vs_parent_t), np.log10(mad_ratio))

                        # taylors law - migration
                        #mean_rel_migration_t = np.mean(rel_s_by_s_migration_t, axis=1)
                        #var_rel_migration_t = np.var(rel_s_by_s_migration_t, axis=1)

                        #idx_taylor_migration = (mean_rel_migration_t > 0) & (var_rel_migration_t > 0)
                        #mean_rel_migration_t = mean_rel_migration_t[idx_taylor_migration]
                        #var_rel_migration_t = var_rel_migration_t[idx_taylor_migration]

                        #slope_taylor_migration, intercept_taylor_migration, r_value_taylor_migration, p_value_taylor_migration, std_err_taylor_migration = stats.linregress(np.log10(mean_rel_migration_t), np.log10(var_rel_migration_t))

                        # taylors law - no migration
                        #mean_rel_no_migration_t = np.mean(rel_s_by_s_no_migration_t, axis=1)
                        #var_rel_no_migration_t = np.var(rel_s_by_s_no_migration_t, axis=1)

                        #idx_taylor_no_migration = (mean_rel_no_migration_t > 0) & (var_rel_no_migration_t > 0)
                        #mean_rel_no_migration_t = mean_rel_no_migration_t[idx_taylor_no_migration]
                        #var_rel_no_migration_t = var_rel_no_migration_t[idx_taylor_no_migration]

                        #slope_taylor_no_migration, intercept_taylor_no_migration, r_value_taylor_no_migration, p_value_taylor_no_migration, std_err_taylor_no_migration = stats.linregress(np.log10(mean_rel_no_migration_t), np.log10(var_rel_no_migration_t))

                        rho_dict[tau_i][sigma_i][t]['migration'].append(mean_corr_migration_t)
                        rho_dict[tau_i][sigma_i][t]['migration_vs_parent'].append(r_value_migration)
                        rho_dict[tau_i][sigma_i][t]['no_migration_vs_parent'].append(r_value_no_migration)
                        rho_dict[tau_i][sigma_i][t]['mad_ratio_vs_parent'].append(r_value_mad)

                        #rho_dict[tau_i][sigma_i][t]['migration_taylor_slope'].append(slope_taylor_migration)
                        #rho_dict[tau_i][sigma_i][t]['migration_taylor_intercept'].append(intercept_taylor_migration)

                        #rho_dict[tau_i][sigma_i][t]['no_migration_taylor_slope'].append(slope_taylor_no_migration)
                        #rho_dict[tau_i][sigma_i][t]['no_migration_taylor_intercept'].append(intercept_taylor_no_migration)

                        #var_rel_migration_t = np.var(rel_s_by_s_migration_t, axis=1)
                        #var_rel_no_migration_t = np.var(rel_s_by_s_no_migration_t, axis=1)

                        #cv_rel_migration_t = np.sqrt(var_rel_migration_t)/mean_rel_migration_t
                        #cv_rel_no_migration_t = np.sqrt(var_rel_no_migration_t)/mean_rel_no_migration_t

                        #mean_corr = np.corrcoef(np.log10(mean_rel_migration_t), np.log10(mean_rel_no_migration_t))[0,1]
                        #cv_corr = np.corrcoef(np.log10(cv_rel_migration_t), np.log10(cv_rel_no_migration_t))[0,1]

                        #init_abund_rel_to_keep = init_abund_rel[idx_migration]

            for t in transfers:
                print(tau_i, sigma_i, t, np.mean(rho_dict[tau_i][sigma_i][t]['migration']))



    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_parent_rho_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





def run_simulation_global_rho(iter=100):

    # run the whole range of transfers since we have this data for global migration
    transfers = range(18)
    #transfers = [11, 17]

    rho_dict = {}

    tau_all = np.linspace(0.5, 6, num=10, endpoint=True)
    sigma_all = np.linspace(0.1, 1.9, num=10, endpoint=True)

    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:

            print(tau_i, sigma_i)

            rho_dict[tau_i][sigma_i] = {}
            rho_dict[tau_i][sigma_i]['ratio_stats'] = {}
            rho_dict[tau_i][sigma_i]['per_transfer_stats'] = {}

            for migration_status in ['global_migration', 'no_migration']:

                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status] = {}

                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_log_ratio_per_transfer'] = []
                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['cv_log_ratio'] = []

                #rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio'] = []

                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_mean'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_cv'] = []

                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_before'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_after'] = []
                rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_delta_cv_log_ratio'] = []


            for t in transfers:
                #rho_dict[tau_i][sigma_i][t] = {}

                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t] = {}

                #rho_dict[tau_i][sigma_i][t]['mean_rho'] = []
                #rho_dict[tau_i][sigma_i][t]['cv_rho'] = []

                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['mean_rho'] = []
                rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['cv_rho'] = []


            #for i in range(iter):
            while (len(rho_dict[tau_i][sigma_i]['per_transfer_stats'][transfers[0]]['mean_rho']) < iter):
                # rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio']

                s_by_s_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_migration(sigma = sigma_i, tau = tau_i, migration_treatment='global')

                s_by_s_dict = {}
                does_idx_ratio_vs_parent_meet_cutoff = True
                for t in transfers:

                    s_by_s_migration_t = s_by_s_migration[t,:,:]
                    s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

                    rel_s_by_s_migration_t = s_by_s_migration_t.T/s_by_s_migration_t.sum(axis=1)
                    rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

                    idx_migration = (~np.all(rel_s_by_s_migration_t == 0, axis=1)) & (~np.all(rel_s_by_s_no_migration_t == 0, axis=1))

                    s_by_s_dict[t] = {}
                    s_by_s_dict[t]['rel_s_by_s_migration_t'] = rel_s_by_s_migration_t
                    s_by_s_dict[t]['rel_s_by_s_no_migration_t'] = rel_s_by_s_no_migration_t

                    s_by_s_dict[t]['idx_migration'] = idx_migration

                    if sum(idx_migration) < 5:
                        does_idx_ratio_vs_parent_meet_cutoff = False


                if does_idx_ratio_vs_parent_meet_cutoff == False:
                    continue


                else:

                    for t in transfers:

                        rel_s_by_s_migration_t = s_by_s_dict[t]['rel_s_by_s_migration_t']
                        rel_s_by_s_no_migration_t = s_by_s_dict[t]['rel_s_by_s_no_migration_t']
                        idx_migration = s_by_s_dict[t]['idx_migration']

                        mean_rel_migration_t = np.mean(rel_s_by_s_migration_t[idx_migration,:], axis=1)
                        mean_rel_no_migration_t = np.mean(rel_s_by_s_no_migration_t[idx_migration,:], axis=1)

                        std_rel_migration_t = np.std(rel_s_by_s_migration_t[idx_migration,:], axis=1)
                        std_rel_no_migration_t = np.std(rel_s_by_s_no_migration_t[idx_migration,:], axis=1)

                        cv_rel_migration_t = std_rel_migration_t/mean_rel_migration_t
                        cv_rel_no_migration_t = std_rel_no_migration_t/mean_rel_no_migration_t

                        mean_corr_t = np.corrcoef(np.log10(mean_rel_migration_t), np.log10(mean_rel_no_migration_t))[0,1]
                        cv_corr_t = np.corrcoef(np.log10(cv_rel_migration_t), np.log10(cv_rel_no_migration_t))[0,1]

                        rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['mean_rho'].append(mean_corr_t)
                        rho_dict[tau_i][sigma_i]['per_transfer_stats'][t]['cv_rho'].append(cv_corr_t)


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

                        log_ratio_before_after_dict = {}
                        transfer_all_transfers= []
                        mean_log_ratio_per_transfer_all_transfers = []
                        cv_log_ratio_all_transfers = []
                        for t in range(18-1):

                            rel_s_by_s_migration_t = s_by_s_dict[t][migration_treatment]
                            rel_s_by_s_migration_t_plus = s_by_s_dict[t+1][migration_treatment]

                            ratio  = np.divide(rel_s_by_s_migration_t_plus, rel_s_by_s_migration_t)
                            log_ratio = np.log10(ratio)

                            # get mean
                            mean_log_ratio_all = []
                            cv_log_ratio_all = []

                            # log-ratio distribution across reps for a given timepoint
                            for log_ratio_dist_idx, log_ratio_dist in enumerate(log_ratio):
                                log_ratio_dist = log_ratio_dist[~np.isnan(log_ratio_dist)]
                                log_ratio_dist = log_ratio_dist[np.isfinite(log_ratio_dist)]

                                # keep fluctuation dist if at least five values
                                if len(log_ratio_dist) >= 5:

                                    mean_log_ratio = np.mean(log_ratio_dist)
                                    std_log_ratio = np.std(log_ratio_dist)
                                    cv_log_ratio = std_log_ratio/np.absolute(mean_log_ratio)

                                    mean_log_ratio_all.append(mean_log_ratio)
                                    cv_log_ratio_all.append(cv_log_ratio)

                                if log_ratio_dist_idx not in log_ratio_before_after_dict:
                                    log_ratio_before_after_dict[log_ratio_dist_idx] = {}
                                    log_ratio_before_after_dict[log_ratio_dist_idx]['transfer'] = []
                                    log_ratio_before_after_dict[log_ratio_dist_idx]['log_ratio'] = []

                                log_ratio_before_after_dict[log_ratio_dist_idx]['transfer'].extend([t]*len(log_ratio_dist))
                                log_ratio_before_after_dict[log_ratio_dist_idx]['log_ratio'].extend(log_ratio_dist)


                            transfer_all_transfers.extend([t]*len(mean_log_ratio_all))
                            mean_log_ratio_per_transfer_all_transfers.extend(mean_log_ratio_all)
                            cv_log_ratio_all_transfers.extend(cv_log_ratio_all)


                            # mean across species of the mean/CV of log-ratio across reps, per-transfer
                            mean_mean_log_ratio = np.mean(mean_log_ratio_all)
                            mean_cv_log_ratio = np.mean(cv_log_ratio_all)

                            rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['transfer'].append(t)
                            rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_mean_log_ratio_per_transfer'].append(mean_mean_log_ratio)
                            rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio'].append(mean_cv_log_ratio)


                        # KS test
                        transfer_all_transfers = np.asarray(transfer_all_transfers)
                        mean_log_ratio_per_transfer_all_transfers = np.asarray(mean_log_ratio_per_transfer_all_transfers)
                        cv_log_ratio_all_transfers = np.asarray(cv_log_ratio_all_transfers)

                        mean_log_ratio_per_transfer_all_transfers_before = mean_log_ratio_per_transfer_all_transfers[(transfer_all_transfers>5) & (transfer_all_transfers<12)]
                        mean_log_ratio_per_transfer_all_transfers_after = mean_log_ratio_per_transfer_all_transfers[(transfer_all_transfers>5) & (transfer_all_transfers>=12)]

                        cv_log_ratio_all_transfers_before = cv_log_ratio_all_transfers[(transfer_all_transfers>5) & (transfer_all_transfers<12)]
                        cv_log_ratio_all_transfers_after = cv_log_ratio_all_transfers[(transfer_all_transfers>5) & (transfer_all_transfers>=12)]

                        if (len(mean_log_ratio_per_transfer_all_transfers_before)>0) & (len(mean_log_ratio_per_transfer_all_transfers_after)>0):
                            ks_mean_log_ratio_per_transfer_all_transfers, p_value_mean_log_ratio_per_transfer_all_transfers = stats.ks_2samp(mean_log_ratio_per_transfer_all_transfers_before, mean_log_ratio_per_transfer_all_transfers_after)
                            rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_mean'].append(ks_mean_log_ratio_per_transfer_all_transfers)


                        if (len(cv_log_ratio_all_transfers_before)>0) & (len(cv_log_ratio_all_transfers_after)>0):
                            ks_cv_log_ratio_all_transfers, p_value_cv_log_ratio_all_transfers = stats.ks_2samp(cv_log_ratio_all_transfers_before, cv_log_ratio_all_transfers_after)
                            rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['ks_cv'].append(ks_cv_log_ratio_all_transfers)


                        # calculate mean/CV before across time before and after the treatment ends
                        #species_idx = list(log_ratio_before_after_dict.keys())
                        #mean_before_all = []
                        #mean_after_all = []
                        #cv_before_all = []
                        #cv_after_all = []
                        #delta_cv_all = []
                        #for s in species_idx:
                        #    transfer_s  = np.asarray(log_ratio_before_after_dict[s]['transfer'])
                        #    if len(transfer_s) == 0:
                        #        continue
                        #    log_ratio_s = np.asarray(log_ratio_before_after_dict[s]['log_ratio'])

                        #    log_ratio_s_before = log_ratio_s[(transfer_s>5) & (transfer_s<12)]
                        #    log_ratio_s_after = log_ratio_s[(transfer_s>5) & (transfer_s>=12)]

                        #    if (len(log_ratio_s_before)>=10) and (len(log_ratio_s_after)>=10):

                        #        mean_log_ratio_s_before = np.mean(log_ratio_s_before)
                        #        cv_log_ratio_s_before = np.std(log_ratio_s_before)/np.absolute(mean_log_ratio_s_before)

                        #        mean_log_ratio_s_after = np.mean(log_ratio_s_after)
                        #        cv_log_ratio_s_after = np.std(log_ratio_s_after)/np.absolute(mean_log_ratio_s_after)

                        #        mean_before_all.append(mean_log_ratio_s_before)
                        #        mean_after_all.append(mean_log_ratio_s_after)

                        #        cv_before_all.append(cv_log_ratio_s_before)
                        #        cv_after_all.append(cv_log_ratio_s_after)

                        #        delta_cv_all.append(cv_log_ratio_s_after - cv_log_ratio_s_before)


                        # data from at least 10 species
                        #if len(delta_cv_all) >= 10:
                        #    rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_before'].append(np.mean(cv_before_all))
                        #    rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_cv_log_ratio_after'].append(np.mean(cv_after_all))
                        #    rho_dict[tau_i][sigma_i]['ratio_stats'][migration_status]['mean_delta_cv_log_ratio'].append(np.mean(delta_cv_all))


            #print(migration_treatment, ks_cv_log_ratio_all_transfers)

            ks_cv_no = rho_dict[tau_i][sigma_i]['ratio_stats']['no_migration']['ks_cv']
            ks_cv_global = rho_dict[tau_i][sigma_i]['ratio_stats']['global_migration']['ks_cv']
            if (len(ks_cv_no) > 0) and (len(ks_cv_global)>0):

                print('no migration', np.mean(ks_cv_no), len(ks_cv_no))
                print('global migration', np.mean(ks_cv_global), len(ks_cv_global))




    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_global_rho_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)






def run_simulation_all_migration(iter=100):

    transfers = range(18)

    rho_dict = {}

    tau_all = np.linspace(0.5, 6, num=10, endpoint=True)
    sigma_all = np.linspace(0.1, 1.9, num=10, endpoint=True)

    for tau_i in tau_all:

        rho_dict[tau_i] = {}

        for sigma_i in sigma_all:

            print(tau_i, sigma_i)

            rho_dict[tau_i][sigma_i] = {}
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

                rho_dict[tau_i][sigma_i][t]['global_migration']['mean_log_error'] = []
                rho_dict[tau_i][sigma_i][t]['parent_migration']['mean_log_error'] = []
                rho_dict[tau_i][sigma_i][t]['no_migration']['mean_log_error'] = []


            while len(rho_dict[tau_i][sigma_i][transfers[0]]['global_migration']['taylors_intercept']) < iter:

                s_by_s_global_migration, s_by_s_parent_migration, s_by_s_no_migration, k_to_keep, t_gen, init_abund_rel = run_simulation_initial_condition_all_migration(sigma = sigma_i, tau = tau_i)

                for t in transfers:

                    s_by_s_global_migration_t = s_by_s_global_migration[t,:,:]
                    s_by_s_parent_migration_t = s_by_s_parent_migration[t,:,:]
                    s_by_s_no_migration_t = s_by_s_no_migration[t,:,:]

                    # remove zeros
                    s_by_s_global_migration_t = s_by_s_global_migration_t[(~np.all(s_by_s_global_migration_t == 0, axis=1)),:]
                    s_by_s_parent_migration_t = s_by_s_parent_migration_t[(~np.all(s_by_s_parent_migration_t == 0, axis=1)),:]
                    s_by_s_no_migration_t = s_by_s_no_migration_t[(~np.all(s_by_s_no_migration_t == 0, axis=1)),:]

                    rel_s_by_s_global_migration_t = s_by_s_global_migration_t.T/s_by_s_global_migration_t.sum(axis=1)
                    rel_s_by_s_parent_migration_t = s_by_s_parent_migration_t.T/s_by_s_parent_migration_t.sum(axis=1)
                    rel_s_by_s_no_migration_t = s_by_s_no_migration_t.T/s_by_s_no_migration_t.sum(axis=1)

                    occupancies_global_migration, predicted_occupancies_global_migration, mad_global_migration, beta_global_migration, species_occupancies_global_migration  = utils.predict_occupancy(s_by_s_global_migration_t.T, range(s_by_s_global_migration_t.shape[1]))
                    occupancies_parent_migration, predicted_occupancies_parent_migration, mad_parent_migration, beta_parent_migration, species_occupancies_parent_migration  = utils.predict_occupancy(s_by_s_parent_migration_t.T, range(s_by_s_parent_migration_t.shape[1]))
                    occupancies_no_migration, predicted_occupancies_no_migration, mad_no_migration, beta_no_migration, species_occupancies_no_migration  = utils.predict_occupancy(s_by_s_no_migration_t.T, range(s_by_s_no_migration_t.shape[1]))

                    error_global_migration = np.absolute(occupancies_global_migration - predicted_occupancies_global_migration)/occupancies_global_migration
                    error_parent_migration = np.absolute(occupancies_parent_migration - predicted_occupancies_parent_migration)/occupancies_parent_migration
                    error_no_migration = np.absolute(occupancies_no_migration - predicted_occupancies_no_migration)/occupancies_no_migration

                    mean_log_error_global_migration = np.mean(np.log10(error_global_migration[error_global_migration>0]))
                    mean_log_error_parent_migration = np.mean(np.log10(error_parent_migration[error_parent_migration>0]))
                    mean_log_error_no_migration = np.mean(np.log10(error_no_migration[error_no_migration>0]))

                    #print(t, mean_log_error_global_migration, mean_log_error_parent_migration, mean_log_error_no_migration)
                    # taylors law
                    means_global_migration, variances_global_migration, species_to_keep_global_migration = utils.get_species_means_and_variances(rel_s_by_s_global_migration_t, range(rel_s_by_s_global_migration_t.shape[1]), zeros=True)
                    means_parent_migration, variances_parent_migration, species_to_keep_parent_migration = utils.get_species_means_and_variances(rel_s_by_s_parent_migration_t, range(rel_s_by_s_parent_migration_t.shape[1]), zeros=True)
                    means_no_migration, variances_no_migration, species_to_keep_no_migration = utils.get_species_means_and_variances(rel_s_by_s_no_migration_t, range(rel_s_by_s_no_migration_t.shape[1]), zeros=True)


                    slope_global, intercept_global, r_value_global, p_value_global, std_err_global = stats.linregress(np.log10(means_global_migration), np.log10(variances_global_migration))
                    slope_parent, intercept_parent, r_value_parent, p_value_parent, std_err_parent = stats.linregress(np.log10(means_parent_migration), np.log10(variances_parent_migration))
                    slope_no, intercept_no, r_value_no, p_value_no, std_err_no = stats.linregress(np.log10(means_no_migration), np.log10(variances_no_migration))


                    rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_slope'].append(slope_global)
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_slope'].append(slope_parent)
                    rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_slope'].append(slope_no)

                    rho_dict[tau_i][sigma_i][t]['global_migration']['taylors_intercept'].append(intercept_global)
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['taylors_intercept'].append(intercept_parent)
                    rho_dict[tau_i][sigma_i][t]['no_migration']['taylors_intercept'].append(intercept_no)

                    rho_dict[tau_i][sigma_i][t]['global_migration']['mean_log_error'].append(mean_log_error_global_migration)
                    rho_dict[tau_i][sigma_i][t]['parent_migration']['mean_log_error'].append(mean_log_error_parent_migration)
                    rho_dict[tau_i][sigma_i][t]['no_migration']['mean_log_error'].append(mean_log_error_no_migration)


    sys.stderr.write("Saving dictionary...\n")
    with open(simulation_migration_all_path, 'wb') as handle:
        pickle.dump(rho_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)






#run_simulation_global_rho()

#run_simulation_all_migration()

#run_simulation_parent_rho()
