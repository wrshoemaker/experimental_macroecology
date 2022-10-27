import os
#from Bio import Phylo
import random
import copy
import sys
import numpy
import random
import pickle
import scipy.stats as stats
from scipy.special import loggamma, hyperu

import matplotlib.pyplot as plt
import functools
import operator

import mpmath

from statsmodels.base.model import GenericLikelihoodModel




import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning
warnings.simplefilter('ignore', HessianInversionWarning)

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)



#float(mpmath.hyperu(2, 1/2, 30))

def mean_truncated_gaussian(mu, sigma):

    Z = 1 - stats.norm.cdf(0, loc=mu, scale=sigma)
    mean = mu + sigma*stats.norm.pdf(0, mu, sigma)/Z

    return mean


#def calculate_hyp1f1(N, n, mu, sigma):



def get_hyp1f1_bounds(n, N):

    max_n = max(n)
    N_for_max_n = N[numpy.where(n == max_n)[0][0]]

    x = n/N
    min_x = min(x[x>0])
    max_x = max(x)

    max_mu = n/N
    max_sigma = max_mu*(1-max_mu)

    #mu_range = numpy.logspace(numpy.log10(min_x), numpy.log10(max_x), num=10**2)
    mu_range = numpy.logspace(numpy.log10(min_x), numpy.log10(max_x), num=10**2)
    mu_range = numpy.concatenate((-1*mu_range, mu_range))
    sigma_range = numpy.logspace(numpy.log10(min_x*(1-min_x)), numpy.log10(max_x*(1-max_x)), num=10**2)
    #since we are looking at the maximum observed n_i for a given AFD, mu cannot be larger than that
    mu_to_keep = []
    sigma_to_keep = []
    for mu_i_idx, mu_i in enumerate(mu_range):
        for sigma_i in sigma_range:

            hyp1f1_i = float(mpmath.hyp1f1((max_n+1)/2, 1/2, (1/2)*((N_for_max_n*(sigma_i**2) - mu_i)**2), asymp_tol=1e-100) )
            if numpy.isinf(hyp1f1_i) == True:
                break

            mu_to_keep.append(mu_i)
            sigma_to_keep.append(sigma_i)


    mu_to_keep = numpy.asarray(mu_to_keep)
    sigma_to_keep = numpy.asarray(sigma_to_keep)

    max_mu = max(mu_to_keep)
    min_mu = min(mu_to_keep)

    max_sigma = max(sigma_to_keep[(mu_to_keep==max_mu)])
    min_sigma = min(sigma_to_keep[(mu_to_keep==min_mu)])


    return max_mu, max_sigma, min_mu, min_sigma





def _ll_truncated_gaussian(n, N, mu, sigma):

    x = n/N

    #rv = stats.truncnorm.logpdf()

    ll = stats.truncnorm.logpdf(x, 0, numpy.inf, loc=mu, scale=sigma)


    return ll






def _ll_truncated_gaussian_sampling(n, N, mu, sigma):


    #hyperu_all = numpy.asarray([float(mpmath.hyperu((n[i]+1)/2, 1/2, (1/2)*((N[i]*(sigma**2) - mu)**2))) for i in range(len(n))])
    hyperu_all = numpy.asarray([float(mpmath.hyp1f1((n[i]+1)/2, 1/2, (1/2)*((N[i]*(sigma**2) - mu)**2))) for i in range(len(n))])
    #hyperu_all = numpy.asarray([hyperu((n[i]+1)/2, 1/2, (1/2)*((N[i]*(sigma**2) - mu)**2)) for i in range(len(n))])
    #hyperu_log_all = numpy.asarray([float(mpmath.log(mpmath.hyperu((n[i]+1)/2, 1/2, (1/2)*((N[i]*(sigma**2) - mu)**2)))) for i in range(len(n))])

    # what if you did the whole calculation using mpmath?????

    ll = -1*numpy.log((1 - stats.norm.cdf(0, loc=mu, scale=sigma))) - (1/2)*numpy.log(numpy.pi) \
        + n*numpy.log(N) + n*numpy.log(sigma) + n*numpy.log(numpy.sqrt(2)) - (mu**2)/(2*(sigma**2)) \
        + numpy.log(hyperu_all)


    return ll



class mle_truncated_gaussian_sampling(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_truncated_gaussian_sampling, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        sigma = params[1]
        ll = -1*_ll_truncated_gaussian_sampling(self.exog.flatten(), self.endog, mu, sigma)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        #if start_params == None:
        if (type(start_params).__module__ == numpy.__name__ ) == False:
            mu_start = 0.01
            sigma_start = (mu_start**2)
            start_params = numpy.array([mu_start, sigma_start])


        return super(mle_truncated_gaussian_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)




class mle_truncated_gaussian(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_truncated_gaussian, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        sigma = params[1]
        ll = -1*_ll_truncated_gaussian(self.exog.flatten(), self.endog, mu, sigma)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        #if start_params == None:
        if (type(start_params).__module__ == numpy.__name__ ) == False:
            #rel_ = self.exog/self.endog
            mu_start = 0.01
            sigma_start = (mu_start**2)
            start_params = numpy.array([mu_start, sigma_start])


        return super(mle_truncated_gaussian, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)





def _ll_gamma(n, N, beta, x_mean):

    x = n/N

    ll = stats.gamma.logpdf(x, beta, loc=0, scale=x_mean/beta)

    return ll


def _ll_truncatd_gamma(n, N, beta, x_mean):

    x = n/N

    #prob = 1/(1-CDF(0)) * PDF
    #log(prob) = log(PDF) + -1*log([1 - CDF])
    ll = stats.gamma.logpdf(x, beta, loc=0, scale=x_mean/beta) - numpy.log((1-stats.gamma.cdf(0, beta, loc=0, scale=x_mean/beta)))

    return ll




def _ll_gamma_sampling(n, N, x_mean, x_var):
    # n = exogenous
    # N = endogenous

    beta = (x_mean**2)/x_var
    # gamma(x) = (x-1)!
    #ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))

    return ll




class mle_gamma_sampling(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma_sampling, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        x_mean = params[0]
        x_var = params[1]
        ll = -1*_ll_gamma_sampling(self.exog.flatten(), self.endog, x_mean, x_var)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        #print(type(start_params).__module__, numpy.__name__ )
        #if (type(start_params).__module__ == numpy.__name__ ) == False:

        if type(start_params) == type(None):
            x_mean_start = 0.001
            x_var_start = 0.0001
            start_params = numpy.array([x_mean_start, x_var_start])


        return super(mle_gamma_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)



class mle_gamma(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        x_mean = params[0]
        x_var = params[1]
        ll = -1*_ll_gamma(self.exog.flatten(), self.endog, x_mean, x_var)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if (type(start_params).__module__ == numpy.__name__ ) == False:
            x_mean_start = 0.001
            x_var_start = 0.00001
            start_params = numpy.array([x_mean_start, x_var_start])


        return super(mle_gamma, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)



class mle_truncated_gamma(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_truncated_gamma, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        beta = params[0]
        x_mean = params[1]
        ll = -1*_ll_truncatd_gamma(self.exog.flatten(), self.endog, beta, x_mean)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if (type(start_params).__module__ == numpy.__name__ ) == False:
            beta_start = 10
            x_mean_start = 0.001
            start_params = numpy.array([beta_start, x_mean_start])


        return super(mle_truncated_gamma, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)




def get_ll_ratio(ll_1, ll_2):

    return -2*(ll_1 - ll_2)



#def fit_trun



def test_mle():

    #n = numpy.asarray([24,30,30,28,4,0,24,21,20,19,0,22,24,22,23,25,26,19,20,19,18,16])
    #N = numpy.asarray([1000]* len(n))


    n = numpy.asarray([3.000e+00, 1.000e+00, 0.000e+00, 9.000e+00, 6.000e+00, 4.000e+00,
                    5.000e+00, 0.000e+00, 1.000e+00, 1.000e+00, 2.000e+00, 1.000e+00,
                    2.000e+00, 3.000e+00, 1.600e+01, 0.000e+00, 0.000e+00, 7.000e+00,
                    0.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.036e+03, 6.400e+01])


    N = numpy.asarray([8344, 7107, 8644, 8226, 7104, 7213, 7753, 5525, 8556, 6594, 8805, 8629,
                        8293, 7596, 5507, 3397, 7961, 6312, 7572, 6432, 8435, 7746, 8650, 8557])


    mu_start = numpy.mean(n/N)
    sigma_start = numpy.std(n/N)
    start_params = numpy.asarray([mu_start, sigma_start])

    #max_mu, max_sigma, min_mu, min_sigma = get_hyp1f1_bounds(n, N)

    #trunc_gaussian_sampling_model = mle_truncated_gaussian_sampling(N, n)
    #trunc_gaussian_sampling_result = trunc_gaussian_sampling_model.fit(method="lbfgs", disp = False, bounds=[(min_mu, max_mu), (min_sigma, max_sigma)])
    #trunc_gaussian_sampling_model_ll = trunc_gaussian_sampling_model.loglike(trunc_gaussian_sampling_result.params)

    gamma_sampling_model = mle_gamma_sampling(N, n)
    gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", disp = False, bounds= [(0.0000001,10000000), (0.000001,1)])
    gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)


    idx_ = (n>0)

    n = n[idx_]
    N = N[idx_]


    trunc_gaussian_model = mle_truncated_gaussian(N, n)
    trunc_gaussian_result = trunc_gaussian_model.fit(method="lbfgs", disp = False, bounds=[(0.00000001, 1), (0.00000001, 1)], start_params=start_params)
    trunc_gaussian_model_ll = trunc_gaussian_model.loglike(trunc_gaussian_result.params)




    gamma_model = mle_gamma(N, n)
    gamma_result = gamma_model.fit(method="lbfgs", disp = False, bounds= [(0.0000001,10000000), (0.000001,1)])
    gamma_model_ll = gamma_model.loglike(gamma_result.params)







#run_mle_gamma_taxon_dict()
