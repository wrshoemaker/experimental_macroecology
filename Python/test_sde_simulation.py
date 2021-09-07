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
                    pmf_i = 1 / sqrt(2 * pi * V) / x_i * \
                           exp(-(np.log(x_i) - mu) ** 2 / (2 * V)) * \
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
                    term1 = ((2 * pi * sigma ** 2) ** -0.5)/ factorial(x_i)
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


ab = [400, 40, 30, 10,4,3,2,1,1,1,1,1,1,1,1]
ab = np.asarray(ab)

# get number of specis to keep
delta = 0.9
S_to_keep = np.random.binomial(len(ab), p=delta)
# get abundances of those species
ab_to_keep = np.random.choice(ab, size=S_to_keep, replace=False)
# assign carrying capacities
K_to_keep = pln._rvs(2, 3, lower_trunc=True, size=S_to_keep)
### to-do fit pln to sads to get idea of parameter range


# number replicates
reps = 10

#unique_vals, counts = np.unique(ab, return_counts=True)
#S_0=200
#sad_0 = pln._rvs(0.3, 3, lower_trunc=True, size=S_0)




def simulate_slm(migration_treatment='parent'):

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
        q[t + 1,:,:] = q[t,:,:] + (dt*(m/np.exp(q[t,:,:]))*(1/tau)*(1 - (sigma/2) - (np.exp(q[t,:,:]) / K_to_keep))) + (noise_variable * np.random.randn(reps, len(ab_to_keep)))


    
    x = np.exp(q)
    # replace negative q values with zero
    x[x<0] = 0
    x[(np.isnan(x)) | np.isneginf(x) | np.isinf(x)] = 0


    #print(x)

    #fig, ax = plt.subplots(figsize=(4,4))
    #ax.scatter(t, x[:,4, 5], alpha=0.7, s=4)
    #ax.set_yscale('log', basey=10)

    #fig.subplots_adjust(hspace=0.35,wspace=0.3)
    #fig_name = "%s/figs/test_simulation.png" % utils.directory
    #fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    #plt.close()





simulate_slm()
