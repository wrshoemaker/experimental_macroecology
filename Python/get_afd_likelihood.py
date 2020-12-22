from __future__ import division
from functools import partial
import math, sys
import numpy as np
import sympy
from scipy import special, optimize, stats
from decimal import Decimal

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

#18 species with >=10 non zero occurances in glucos


# kl = vector of counts (non zeros)
# Nl = tot number of reads (corresponding to nonzeros)
# nrall = vector of number of redas (including zeros)

#kl = np.asarray([100,100,400, 50, 30,20,10,10])
#nl = np.asarray([1000, 1000,500, 90, 80,50,30,20])
#nrall = np.asarray([1000, 1000,500, 90, 80,50,30,20,100,150,200])


# mean of prior distribution of species absence = alpha / (alpha+beta)
# parameters chosen to reflect range in Supp Fig 6, Grilli 2020
#alfapriorq_list = np.asarray([0.01, 0.05, 0.1, 0.15, 0.3, 0.7, 1.1, 1.5, 2.2, 3.1, 5.7, 9, 20])
alfapriorq_list = np.asarray([0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.7, 1.1, 1.5, 2.2, 3.1, 5.7, 9, 20])

betapriorq_all = 0.4
#prior_probability = alfapriorq_list/(alfapriorq_list+betapriorq_all)

N_tries = 100
N_sucesses = 50

min_nonzero = 10


# initialize all combinations of initial conditions
x_start_list = []

for x_mean in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 0.1, 0.2, 0.3]:

    for x_1 in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]:

        x_start_list.append((np.log(x_mean), x_1))

#x_start_list = [(-9.210340371976182, 1)]


#kl = np.asarray([100, 100,400, 50, 30,20,10,10,1,1,1])
#xstart <- c( log(mean(kl/nl) ), min( mean(kl/nl)^2/mean( (kl^2 - kl)/nl^2 ), 3) )


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

    return _llambda, _beta, p, logL, q, xstart







def calculate_all_likelihoods(carbon_source, alfapriorq_list, betapriorq_all):

    s_by_s_np, species, communities_final = utils.get_s_by_s(carbon_source, transfer=12, communities=None)
    nrall = np.sum(s_by_s_np, axis=0)

    priors_mean = alfapriorq_list / (alfapriorq_list +betapriorq_all)

    header = ['ESV', 'N_obs', 'N_obs_nonzero', 'maxloglikelihoodgamma', 'maxloglikelihoodinflatedgamma']
    for prior_mean in priors_mean:
        header.append('maxlikelihoodinflatedgamma-%s'%str(prior_mean))

    record_strs = [",".join(header)]

    for afd_i_idx, afd_i in enumerate(s_by_s_np):

        #if afd_i_idx > 0:
        #    continue

        if len(afd_i[afd_i>0]) < min_nonzero:
            continue

        x_start_list_i = x_start_list.copy()
        x_start_list_i.append((np.log(np.mean(afd_i/nrall)), min((np.mean(afd_i/nrall)**2)/np.mean( ((afd_i**2) - afd_i)/(nrall**2) ), 3)))

        logL_0 = -1000000000

        llambda_best = 0
        beta_best = 0
        logL_best = logL_0
        xstart_best = (1,1)

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
        # for each prior fifty times
        # compute the weighted average of the maximum likelihood for all
        # initial conditions that successfully converge for a given prior

        # try to save time by skipping initial conditions where
        # less than k alphas successfully optimize

        prior_dict = {}

        num_failures = 0
        for alpha in alfapriorq_list:

            if num_failures > 0:
                continue

            prior_probability = alpha / (alpha+betapriorq_all)

            mean_weighted_logL_inflated_best = -1000000000
            logL_inflated_best_array = np.asarray([])
            q_inflated_best_array = np.asarray([])

            for x_start_i in x_start_list_i:

                # if we can't get enough sucessful iterations to perform the averaging,
                # skip to the next set of initial conditions
                #if num_failure s > 0:
                #    continue

                logL_inflated_i = []
                #p_inflated_alpha = []
                q_inflated_i = []
                N_tries_i = 0
                N_sucesses_i = 0

                while (N_tries_i < N_tries) and (N_sucesses_i < N_sucesses):

                    try:

                        llambda_inflated, beta_inflated, p_inflated, logL_inflated, q_inflated, xstart_inflatd = maxlikelihoodinflatedgamma(x_start_i, afd_i, nrall, alpha, betapriorq_all)
                        converged = True
                        logL_inflated_i.append(logL_inflated)
                        #p_inflated_alpha.append(p_inflated)
                        # get the prior
                        q_inflated_i.append(q_inflated)
                        N_sucesses_i += 1

                    except optimize.nonlin.NoConvergence as e:

                        x = e.args[0]
                        converged = False

                    N_tries_i += 1

                # skip to next iteration if optimization did not succeed for all priors
                if len(logL_inflated_i) < N_sucesses:
                    continue

                logL_inflated_i = np.asarray(logL_inflated_i)
                q_inflated_i  = np.asarray(q_inflated_i )
                # multiply the mean by prior probability because the prior
                # is a random variable, i.e., different values in different iterations
                #mean_weighted_logL_inflated_array_i = np.mean(logL_inflated_i * q_inflated_i)
                mean_weighted_logL_inflated_array_i = logL_inflated_i + np.log(q_inflated_i)

                mean_weighted_logL_inflated_list_i = mean_weighted_logL_inflated_array_i.tolist()
                mean_weighted_logL_inflated_list_i = [Decimal(l).exp() for l in mean_weighted_logL_inflated_list_i]
                mean_weighted_logL_inflated_i = float( (sum(mean_weighted_logL_inflated_list_i) / len(mean_weighted_logL_inflated_list_i)).ln() )

                if mean_weighted_logL_inflated_i > mean_weighted_logL_inflated_best:

                    # keep maximum log likelihood
                    mean_weighted_logL_inflated_best = mean_weighted_logL_inflated_i
                    logL_inflated_best_array = logL_inflated_i
                    q_inflated_best_array = q_inflated_i


                print(carbon_source, afd_i_idx, prior_probability, x_start_i, mean_weighted_logL_inflated_best)


            if len(logL_inflated_best_array) < N_sucesses:
                num_failures+=1


            prior_dict[prior_probability] = {}

            prior_dict[prior_probability]['logL_inflated_best_array'] = logL_inflated_best_array

            prior_dict[prior_probability]['q_inflated_best_array'] = q_inflated_best_array

            print(prior_probability)


        # skip the species if we could not get a MLE for all priors
        if len(prior_dict) < len(alfapriorq_list):
            continue


        #logL_inflated_all = [ prior_dict[k]['logL_inflated_best_array'] * prior_dict[k]['q_inflated_best_array'] for k in prior_dict.keys() ]
        # add together the log likelihood and the log of the probability, log(prob * likelihood) = log(prob) + log(likelihood)
        logL_inflated_all = [prior_dict[k]['logL_inflated_best_array'] + np.log(prior_dict[k]['q_inflated_best_array']) for k in prior_dict.keys() ]
        logL_inflated_all = np.concatenate(logL_inflated_all).ravel()
        logL_inflated_all_list = logL_inflated_all.tolist()
        # convert back from log scale and turn into decimal to prevent overflow
        logL_inflated_all_list_decimal = [Decimal(l).exp() for l in logL_inflated_all_list]
        # now calculate the mean
        mean_weighted_logL_inflated_all  = float( (sum(logL_inflated_all_list_decimal) / len(logL_inflated_all_list_decimal)).ln() )
        #mean_weighted_logL_inflated_all = math.log(np.mean(logL_inflated_all_list_decimal))

        record_list = [species[afd_i_idx], str(len(afd_i)), str(len(afd_i[afd_i>0])), str(logL_best), str(mean_weighted_logL_inflated_all)]

        for prior_prob_i in sorted(list(prior_dict.keys())):

            logL_i = prior_dict[prior_prob_i]['logL_inflated_best_array'] + np.log(prior_dict[prior_prob_i]['q_inflated_best_array'])
            logL_i_list = logL_i.tolist()
            logL_i_list_decimal = [Decimal(l).exp() for l in logL_i_list]

            record_list.append(str( float( (sum(logL_i_list_decimal) / len(logL_i_list_decimal)).ln() ) ))

        record_str = ",".join(record_list)
        record_strs.append(record_str)


    sys.stderr.write("Done with %s species!\n" % carbon_source)
    sys.stderr.write("Writing intermediate file...\n")
    intermediate_filename = "%s/data/%s_gamma_likelihood.csv" % (utils.directory, carbon_source)
    file = open(intermediate_filename,"w")
    record_str = "\n".join(record_strs)
    file.write(record_str)
    file.close()
    sys.stderr.write("Done!\n")




for carbon in utils.carbons:

    calculate_all_likelihoods(carbon, alfapriorq_list, betapriorq_all)
