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

np.random.seed(123456789)
np.seterr(divide = 'ignore')


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





def run_simulation(sigma = 0.9, tau = 0.8, dt = 7, T = 126, reps = 92, migration_treatment='none'):
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

    def discrete_slm(q_array, m):
        # q_array = array of replicate and species
        return q_array + (dt*(m/np.exp(q_array))*(1/tau)*(1 - (sigma/2) - (np.exp(q_array) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund_rel)))

    q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
    q[0,:,:] = discrete_slm(q_0_inoc, m_0)

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


        #if migration_treatment == 'global':
        #    if t > 12:
        #        print(m)

        # update array
        #q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:]))*(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k_to_keep))) + (noise_term * np.random.randn(reps, len(init_abund)))
        q[t,:,:] = discrete_slm(q_t_inoc, m)


    n = np.exp(q)
    # replace negative q values with zero
    n[(np.isnan(n)) | np.isneginf(n) | np.isinf(n)] = 0
    #test = n[-1,0,:]

    return n, k_to_keep, t_gen
