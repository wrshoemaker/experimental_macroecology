from __future__ import division
from functools import partial
import math, sys
import numpy as np
import sympy
from scipy import special, optimize, stats

import utils

np.random.seed(123456789)
#np.seterr(divide = 'ignore')


'''
Code to calculate the likelihood ratio for the gamma and zero-inflated gamma
Eq. 11 in Grilli 2020 https://doi.org/10.1038/s41467-020-18529-y

Script loops over a range of initial conditions and selects the highest log-likelihood
for the gamma distribution

For the zero-inflated gamma, for each initial condition the code loops through
different means of the prior distribution N times. The mean log-likelihood is
calculated over the prior. An initial condition is skipped if it fails to converge
for any of the prior means.
'''



# kl = vector of counts (non zeros)
# Nl = tot number of reads (corresponding to nonzeros)
# nrall = vector of number of redas (including zeros)

kl = np.asarray([100,100,400, 50, 30,20,10,10, 0,0,0])
#nl = np.asarray([1000, 1000,500, 90, 80,50,30,20])
nrall = np.asarray([1000, 1000,500, 90, 80,50,30,20,100,150,200])


# mean of prior distribution of species absence = alpha / (alpha+beta)
# parameters chosen to reflect range in Supp Fig 6, Grilli 2020
#alfapriorq_list = np.asarray([0.01, 0.05, 0.1, 0.15, 0.3, 0.7, 1.1, 1.5, 2.2, 3.1, 5.7, 9, 20])
alfapriorq_list = np.asarray([0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.7, 1.1, 1.5, 2.2, 3.1, 5.7, 9, 20])

betapriorq_all = 0.4

s_by_s_np, species, communities_final = utils.get_s_by_s('Leucine', transfer=12, communities=None)
nrall = np.sum(s_by_s_np, axis=0)

# N_tries = number times to try optimization
N_tries = 500
# N_sucesses = number of successful optimizations for a given prior
# loop continues to next prior after N_sucesses is hit
N_sucesses = 50


# initialize all combinations of initial conditions
x_start_list = []
for x_mean in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 0.1, 0.2, 0.3]:

    for x_1 in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5]:

        x_start_list.append((np.log(x_mean), x_1))




def likelihoodgamma(data, x):

    xstart, kl, nl = data

    if (x[1] > 0) and (x[0]<100):

        _lambda = math.exp(x[0])

        f_1 = np.mean((kl - nl*_lambda) / (x[1] +nl *_lambda ) )
        f_2 = -1 * np.mean((kl - nl*_lambda) / (x[1] +nl *_lambda ) ) + np.mean(special.digamma(kl+x[1])) - special.digamma(x[1]) - np.mean( np.log(1+nl*(_lambda/x[1])) )

    else:
        f_1 = 10**5
        f_2 = 10**5

    return (f_1, f_2)



def maxlikelihoodgamma(xstart, kl, nrall):

    # go over initial conditions.....
    #xstart = (np.log(np.mean(kl/nrall)), min((np.mean(kl/nrall)**2)/np.mean( ((kl**2) - kl)/(nrall**2) ), 3))

    data = [xstart, kl, nrall]
    likelihood_partial = partial(likelihoodgamma, data)

    #https://stackoverflow.com/questions/50040461/passing-extra-arguments-to-broyden1
    #nleqslv(xstart, likelihoodgamma, method="Broyden", global="dbldog", kl = kl, nl = nl )
    x = optimize.broyden1(likelihood_partial, xin=xstart, f_tol=1e-14)

    _llambda = x[0]
    _beta = x[1]
    _lambda = math.exp(_llambda)
    p = _lambda*nrall/(_beta + _lambda*nrall)

    # log factorial(x) = special.gammaln(x+1)
    logL = sum( -1*special.gammaln(kl+1) + special.gammaln(_beta+kl) - special.gammaln(_beta) + kl*np.log(p) + _beta*np.log(1-p) )

    return _llambda, _beta, logL, xstart




def likelihoodinflatedgamma(data, x):

    xstart, kl, nl, q = data

    if (x[1] > 0) and (x[0]<100):

        _lambda = math.exp(x[0])
        _beta = x[1]

        f_1 = np.mean( (kl - nl*_lambda)/(_beta+nl*_lambda) + [kl==0][0]*(nl*(q-1)*(_beta**(1 + _beta)) )/((_beta + nl*_lambda)*((1 - q) * (_beta**_beta) + q*(_beta + nl*_lambda)**_beta) ) )

        #with np.errstate(divide='ignore'):

        f_2 = -1*np.mean( (kl - nl*_lambda)/(_beta+nl*_lambda) ) \
                + np.mean( special.digamma(kl+_beta) ) - special.digamma(_beta) - np.mean( np.log(1+nl*_lambda/_beta) ) \
                + np.mean( [kl==0][0]*((q-1)*(_beta**_beta) * (nl*_lambda + (_beta + nl*_lambda) * np.log( _beta/(_beta + nl*_lambda)) )) / ((_beta + nl*_lambda)*( (q-1)*(_beta**_beta) - q*(_beta + nl*_lambda)**_beta)) )

    else:
        f_1 = 10**5
        f_2 = 10**5


    return (f_1, f_2)





def maxlikelihoodinflatedgamma(xstart, kl, nrall, alfapriorq, betapriorq):

    rv = stats.beta(a=alfapriorq, b=betapriorq)
    # equivalent of rbeta
    q = rv.rvs(1)[0]

    #xstart = (np.log(np.mean(kl/nrall)), min((np.mean(kl/nrall)**2)/np.mean( ((kl**2) - kl)/(nrall**2) ), 3))
    #xstart = (-1, 0.7)

    data = [xstart, kl, nrall, q]
    likelihood_partial = partial(likelihoodinflatedgamma, data)

    x = optimize.broyden1(likelihood_partial, xin=xstart, f_tol=1e-14)

    _llambda = x[0]
    _beta = x[1]
    _lambda = math.exp(_llambda)
    p = _lambda*nrall/(_beta + _lambda*nrall)

    logL = sum( np.log( q*[kl==0][0] + (1-q) * np.exp( (-1*special.gammaln(kl+1)) + special.gammaln(_beta+kl) - special.gammaln(_beta) + kl*np.log(p) + _beta*np.log(1-p) ) ) )

    return _llambda, _beta, logL, q, xstart



priors_mean = alfapriorq_list / (alfapriorq_list +betapriorq_all)

header = ['ESV', 'maxlikelihoodgamma', 'maxlikelihoodinflatedgamma']
for prior_mean in priors_mean:
    header.append('maxlikelihoodinflatedgamma-%s'%str(prior_mean))

record_strs = [", ".join(header)]






x_start_list_i = x_start_list.copy()
x_start_list_i.append((np.log(np.mean(afd_i/nrall)), min((np.mean(afd_i/nrall)**2)/np.mean( ((afd_i**2) - afd_i)/(nrall**2) ), 3)))

logL_0 = -1000000000

llambda_best = 0
beta_best = 0
logL_best = logL_0
xstart_best = (1,1)

llambda_inflated_best = 0
beta_inflated_best = 0
logL_inflated_best = logL_0
xstart_inflated_best = (1,1)
alpha_dict_best = {}

for x_start_i in x_start_list_i:


    try:

        llambda, beta, logL, xstart = maxlikelihoodgamma(x_start_i, afd_i, nrall)
        converged = True

        # keep maximum log likelihood
        if logL > logL_best:

            llambda_best = llambda
            beta_best = beta
            logL_best = logL
            xstart_best = xstart

    except optimize.nonlin.NoConvergence as e:

        x = e.args[0]
        converged = False


    # for zero inflated gamma, go over a range of starting conditions
    # for each starting condition, go through each prior fifty times
    # compute the weighted average of the maximum likelihood for all
    # initial conditions that successfully converge

    # try to save memory by skipping initial conditions where
    # less than k alphas successfully optimize

    alpha_dict = {}

    num_failures = 0
    for alpha in alfapriorq_list:

        # if we can't get enough sucessful iterations to perform the averaging,
        # skip to the next set of initial conditions
        if num_failures > 2:
            continue

        logL_inflated_alpha = []
        N_tries_i = 0
        N_sucesses_i = 0

        while (N_tries_i < N_tries) and (N_sucesses_i < N_sucesses):

            try:

                llambda_inflated, beta_inflated, logL_inflated, q_inflated, xstart_inflatd = maxlikelihoodinflatedgamma(x_start_i, afd_i, nrall, alpha, betapriorq_all)
                converged = True
                logL_inflated_alpha.append(logL_inflated)
                N_sucesses_i += 1

            except optimize.nonlin.NoConvergence as e:

                x = e.args[0]
                converged = False

            N_tries_i += 1


        if len(logL_inflated_alpha) < 10:
            num_failures += 1
            continue


        alpha_dict[alpha] = (np.mean(logL_inflated_alpha), len(logL_inflated_alpha))

        print(afd_i_idx, x_start_i, alpha/(alpha+betapriorq_all), len(logL_inflated_alpha))

    # skip to next iteration if optimization did not succeed for all priors
    if len(alpha_dict) < len(alfapriorq_list):
        continue

    weighted_mean_likelihood = sum([k[0] * k[1] for k in alpha_dict.values()]) / sum([k[1] for k in alpha_dict.values()])

    if weighted_mean_likelihood > logL_inflated_best:

        # keep maximum log likelihood, parameters, dictionary likelihoods for each prior
        logL_inflated_best = weighted_mean_likelihood
        #beta_inflated_best = beta
        #llambda_inflated_best = logL
        #xstart_inflated_best = xstart
        alpha_dict_best = alpha_dict



print(logL_best, logL_inflated_best)
