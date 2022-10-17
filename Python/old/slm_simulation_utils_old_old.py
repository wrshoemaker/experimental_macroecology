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



def prob_non_zero_k(log_10_x, intercept=1.6207715018444455, slope=1.086147398369049):
    return 1 / (1 + np.exp( -1 * (intercept + slope*log_10_x) ))

# use it for log10




def run_simulation_relative(sigma = 0.02, dt = 7, T = 126, reps = 100, migration_treatment='none'):
    # just use units of generations for now
    # T = generations
    # dt = gens. per-transfer

    # get parent mean relative abundances
    mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
    mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)
    # divide by sum to use as probability
    prob_mean_rel_abundances_parent = mean_rel_abundances_parent/sum(mean_rel_abundances_parent)
    s_parent = len(prob_mean_rel_abundances_parent)
    delta = 0.016
    # number of cells in inoculum
    n_cells_inoc = 10**8
    # total community size
    n_cells_total = 10**11
    n_non_zero_k = int(delta*len(prob_mean_rel_abundances_parent))
    tau = 3
    sigma = 0.02
    n_time_steps = int(T / dt)  # Number of time steps.
    #t_gen = np.linspace(0., T, n_time_steps)  # Vector of times.
    t_gen = np.arange(0, T, dt)
    # merge ancestral SADs to get probability vector for multinomial sampling
    # delta% survive, get non zero carrying capacities
    # Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
    # redo later to draw carrying capacities from no migration distribution
    mu_pln=3.4533926814573506
    sigma_pln=2.6967286975393754
    # non zero carrying capacities
    k_to_keep = pln._rvs(mu_pln, sigma_pln, lower_trunc=True, size=n_non_zero_k)
    # pretty sure the array order is random but let's make sure
    k_to_keep = np.random.permutation(k_to_keep)
    inoc_abund = np.random.multinomial(n_cells_inoc, prob_mean_rel_abundances_parent)
    #inoc_rel_abund = inoc_abund / sum(inoc_abund)
    # get nonzero
    inoc_abund_non_zero = inoc_abund[inoc_abund>0]
    # permute
    inoc_abund_non_zero = np.random.permutation(inoc_abund_non_zero)
    non_zero_k_weights = np.asarray([prob_non_zero_k(l) for l in np.log10(inoc_abund_non_zero)])
    non_zero_k_prob = non_zero_k_weights/sum(non_zero_k_weights)

    init_abund_non_zero_with_non_zero_k = np.random.choice(inoc_abund_non_zero, size=len(k_to_keep), replace=False, p=non_zero_k_prob)
    # probably have initial abundances that are the same... (1/#reads),
    # make new vector of initial relative abundances, and match with carrying capacities
    #n_non_zero_k_copy = copy.copy(n_non_zero_k)
    # add zeros
    inoc_abund_non_zero_set = list(set(inoc_abund_non_zero))
    for i_idx, i in enumerate(inoc_abund_non_zero_set):
        n_i = sum(inoc_abund_non_zero==i)
        n_non_zero_k_i = sum(init_abund_non_zero_with_non_zero_k==i)
        n_zero_k_i = n_i - n_non_zero_k_i
        # add zeros for species that have non zero initial abundances, but carrying capacities of zerp
        k_to_keep = np.append(k_to_keep, [0]*n_zero_k_i)
        init_abund_non_zero_with_non_zero_k = np.append(init_abund_non_zero_with_non_zero_k, [0]*n_zero_k_i)


    k_to_keep = np.append(k_to_keep, [0]*sum(inoc_abund==0))
    init_abund = np.append(init_abund_non_zero_with_non_zero_k, [0]*sum(inoc_abund==0))

    k_to_keep_rel = k_to_keep / sum(k_to_keep)
    init_abund_rel = init_abund / sum(init_abund)

    # how to handle relative abundances
    #  renormalized variables
    noise_variable = np.sqrt( sigma*dt/tau  )

    # create three dimensional vector
    # spcies, replicate, and timestep
    q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
    for rep in range(reps):
        x_0 = np.random.multinomial(n_cells_inoc, init_abund_rel)
        x_0 = x_0/sum(x_0)
        q_0 = np.log(x_0)
        q[0,rep,:] = np.asarray(q_0)

    #print(range(n_time_steps - 1))
    #print(n_time_steps)
    # t = transfer number
    for t in range(n_time_steps - 1):
        # migration only happens at transfers
        # no migration during first transfer
        if t == 0:
            m = np.zeros((reps, len(init_abund_rel)))

        else:
            if migration_treatment == 'global':

                x_global = np.zeros(len(init_abund_rel))

                for r in range(reps):
                    q_t_r = q[t, r, :]
                    x_t_r = np.exp(q_t_r)
                    x_t_r[x_t_r<0] = 0
                    print(x_t_r)
                    x_t_r_sample = np.random.multinomial(1000, x_t_r/sum(x_t_r))
                    x_global += x_t_r_sample

                m = np.random.multinomial(n_cells_inoc, x_global/sum(x_global), size=reps)
                #m = m/sum(m)
                print(x_global/sum(x_global))
                m = m/n_cells_total


            elif migration_treatment == 'parent':
                # columns = species
                # rows = reps
                m = np.random.multinomial(n_cells_inoc, init_abund_rel, size=reps)
                #m = m/sum(m)
                m = m/n_cells_total

            else:
                m = np.zeros((reps, len(init_abund_rel)))

        # update array
        #q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:])) *(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k_to_keep_rel))) + (noise_variable * np.random.randn(reps, len(init_abund_rel)))
        q[t + 1,:,:] = q[t,:,:] + dt*(m/np.exp(q[t,:,:]) + (1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k_to_keep_rel))) + (noise_variable * np.random.randn(reps, len(init_abund_rel)))




    x = np.exp(q)
    # replace negative q values with zero
    x[x<0] = 0
    x[(np.isnan(x)) | np.isneginf(x) | np.isinf(x)] = 0

    return x, t_gen





def run_simulation(sigma = 0.02, dt = 7, T = 126, reps = 100, migration_treatment='none'):
    # just use units of generations for now
    # T = generations
    # dt = gens. per-transfer
    #n_cells = 10**9

    # get parent mean relative abundances
    mean_rel_abundances_parent, species_parent = utils.estimate_mean_abundances_parent()
    mean_rel_abundances_parent = np.asarray(mean_rel_abundances_parent)
    # divide by sum to use as probability
    prob_mean_rel_abundances_parent = mean_rel_abundances_parent/sum(mean_rel_abundances_parent)
    s_parent = len(prob_mean_rel_abundances_parent)
    delta = 0.016
    n_cells_inoc = 10**8
    n_non_zero_k = int(delta*len(prob_mean_rel_abundances_parent))
    tau = 3
    sigma = 0.02
    n_time_steps = int(T / dt)  # Number of time steps.
    #t_gen = np.linspace(0., T, n_time_steps)  # Vector of times.
    t_gen = np.arange(0, T, dt)
    # merge ancestral SADs to get probability vector for multinomial sampling
    # delta% survive, get non zero carrying capacities
    # Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration
    # redo later to draw carrying capacities from no migration distribution
    mu_pln=3.4533926814573506
    sigma_pln=2.6967286975393754
    # non zero carrying capacities
    k_to_keep = pln._rvs(mu_pln, sigma_pln, lower_trunc=True, size=n_non_zero_k)
    print(k_to_keep)
    # pretty sure the array order is random but let's make sure
    k_to_keep = np.random.permutation(k_to_keep)
    inoc_abund = np.random.multinomial(n_cells_inoc, prob_mean_rel_abundances_parent)
    #inoc_rel_abund = inoc_abund / sum(inoc_abund)
    # get nonzero
    inoc_abund_non_zero = inoc_abund[inoc_abund>0]
    # permute
    inoc_abund_non_zero = np.random.permutation(inoc_abund_non_zero)
    non_zero_k_weights = np.asarray([prob_non_zero_k(l) for l in np.log10(inoc_abund_non_zero)])
    non_zero_k_prob = non_zero_k_weights/sum(non_zero_k_weights)

    init_abund_non_zero_with_non_zero_k = np.random.choice(inoc_abund_non_zero, size=len(k_to_keep), replace=False, p=non_zero_k_prob)
    # probably have initial abundances that are the same... (1/#reads),
    # make new vector of initial relative abundances, and match with carrying capacities
    # add zeros
    inoc_abund_non_zero_set = list(set(inoc_abund_non_zero))
    for i_idx, i in enumerate(inoc_abund_non_zero_set):
        n_i = sum(inoc_abund_non_zero==i)
        n_non_zero_k_i = sum(init_abund_non_zero_with_non_zero_k==i)
        n_zero_k_i = n_i - n_non_zero_k_i
        # add zeros for species that have non zero initial abundances, but carrying capacities of zerp
        k_to_keep = np.append(k_to_keep, [0]*n_zero_k_i)
        init_abund_non_zero_with_non_zero_k = np.append(init_abund_non_zero_with_non_zero_k, [0]*n_zero_k_i)


    k_to_keep = np.append(k_to_keep, [0]*sum(inoc_abund==0))
    init_abund = np.append(init_abund_non_zero_with_non_zero_k, [0]*sum(inoc_abund==0))

    k_to_keep_rel = k_to_keep / sum(k_to_keep)
    init_abund_rel = init_abund / sum(init_abund)

    # how to handle relative abundances
    #  renormalized variables
    noise_variable = np.sqrt( sigma*dt/tau  )

    # create three dimensional vector
    # spcies, replicate, and timestep
    q = np.zeros((n_time_steps, reps, len(init_abund_rel)))
    for rep in range(reps):
        x_0 = np.random.multinomial(n_cells_inoc, init_abund_rel)
        x_0 = x_0/sum(x_0)
        q_0 = np.log(x_0)
        q[0,rep,:] = np.asarray(q_0)

    # t = transfer number
    for t in range(n_time_steps - 1):
        # migration only happens at transfers
        # no migration during first transfer
        if t == 0:
            m = np.zeros((reps, len(init_abund_rel)))

        else:
            if migration_treatment == 'global':

                x_global = np.zeros(len(init_abund_rel))

                for r in range(reps):
                    q_t_r = q[t, r, :]
                    x_t_r = np.exp(q_t_r)
                    x_t_r[x_t_r<0] = 0
                    x_t_r_sample = np.random.multinomial(1000, x_t_r/sum(x_t_r))
                    x_global += x_t_r_sample

                m = np.random.multinomial(n_cells_inoc, x_global/sum(x_global), size=reps)
                m = m/sum(m)

            elif migration_treatment == 'parent':
                # columns = species
                # rows = reps
                m = np.random.multinomial(n_cells_inoc, init_abund_rel, size=reps)
                m = m/sum(m)

            else:
                m = np.zeros((reps, len(init_abund_rel)))

        # update array
        q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:]))*(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k_to_keep_rel))) + (noise_variable * np.random.randn(reps, len(init_abund_rel)))


    x = np.exp(q)
    # replace negative q values with zero
    x[x<0] = 0
    x[(np.isnan(x)) | np.isneginf(x) | np.isinf(x)] = 0

    return x, t_gen
