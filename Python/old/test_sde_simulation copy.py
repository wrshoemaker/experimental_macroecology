from __future__ import division
import os
import sys
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



# write code to get mu and sigma by fitting to jacopo's MAD
# get SADs
#ab = [400, 40, 30, 10,4,3,2,1,1,1,1,1,1,1,1]
#ab = np.asarray(ab)

# get parent mean relative abundances
mean_rel_abundances = utils.estimate_mean_abundances_parent()
# divide by sum to use as probability
prob_mean_rel_abundances = mean_rel_abundances/sum(mean_rel_abundances)

s_parent = len(prob_mean_rel_abundances)
delta = 0.016
n_reads = 10**4.4
n_non_zero_k = int(delta*len(prob_mean_rel_abundances))
reps = 100

# merge ancestral SADs to get probability vector for multinomial sampling
# delta% survive, get non zero carrying capacities
# Non-zero carrying capacities assigned from lognormal distribution obtained by fitting PLN to transfer 18 SADs no migration

mu=3.4533926814573506
sigma=2.6967286975393754
# non zero carrying capacities
k_to_keep = pln._rvs(mu, sigma, lower_trunc=True, size=n_non_zero_k)
# pretty sure the array order is random but let's make sure
k_to_keep = np.random.permutation(k_to_keep)

inoc_abund = np.random.multinomial(n_reads, prob_mean_rel_abundances)
inoc_rel_abund = inoc_abund / sum(inoc_abund)
# get nonzero
inoc_rel_abund_non_zero = inoc_rel_abund[inoc_rel_abund>0]
# permute
inoc_rel_abund_non_zero = np.random.permutation(inoc_rel_abund_non_zero)
# add the zeros back
inoc_rel_abund = np.append(inoc_rel_abund_non_zero, np.zeros(len(inoc_rel_abund) - len(inoc_rel_abund_non_zero)))
# make the vector of carrying capacities
k = np.append(k_to_keep, np.zeros(len(inoc_rel_abund) - len(k_to_keep)))


#unique_vals, counts = np.unique(ab, return_counts=True)
#S_0=200
#sad_0 = pln._rvs(0.3, 3, lower_trunc=True, size=S_0)




def simulate_slm(migration_treatment='none'):

    tau = 1
    sigma = 0.02

    # 7 generations per transfer
    # just use units of generations for now
    dt = 1
    # T = generations
    T = 126
    n_time_steps = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n_time_steps)  # Vector of times.


    #  renormalized variables
    noise_variable = np.sqrt( sigma*dt/tau  )

    #np.random.multinomial(ab_to_keep/sum(ab_to_keep))

    # create three dimensional vector
    # spcies, replicate, and timestep

    q = np.zeros((n_time_steps, reps, len(ab_to_keep)))

    for rep in range(reps):
        x_0 = np.random.multinomial(1000, ab_to_keep/sum(ab_to_keep))
        #x_0 = x_0/sum(x_0)

        q_0 = np.log(x_0)

        q[0,rep,:] = np.asarray(q_0)


    #q[0,:] = np.asarray([x_0]*reps)

    for t in range(n_time_steps - 1):

        if migration_treatment == 'global':

            x_global = np.zeros(len(ab_to_keep))

            for r in range(reps):
                q_t_r = q[t, r, :]
                x_t_r = np.exp(q_t_r)
                x_t_r[x_t_r<0] = 0
                x_t_r_sample = np.random.multinomial(1000, x_t_r/sum(x_t_r))
                x_global += x_t_r_sample

            m = np.random.multinomial(1000, x_global/sum(x_global), size=reps)

        elif migration_treatment == 'parent':
            # columns = species
            # rows = reps
            m = np.random.multinomial(1000, ab_to_keep/sum(ab_to_keep), size=reps)

        else:
            m = np.zeros((reps, len(ab_to_keep)))

        # update array
        q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:]))*(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / k))) + (noise_variable * np.random.randn(reps, len(ab_to_keep)))



    x = np.exp(q)
    # replace negative q values with zero
    x[x<0] = 0
    x[(np.isnan(x)) | np.isneginf(x) | np.isinf(x)] = 0


    print(x)

    #fig, ax = plt.subplots(figsize=(4,4))
    #ax.scatter(t, x[:,4, 5], alpha=0.7, s=4)
    #ax.set_yscale('log', basey=10)

    #fig.subplots_adjust(hspace=0.35,wspace=0.3)
    #fig_name = "%s/figs/test_simulation.png" % utils.directory
    #fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    #plt.close()



simulate_slm()


#simulate_slm()
