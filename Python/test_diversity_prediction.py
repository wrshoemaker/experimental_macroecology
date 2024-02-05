from __future__ import division
import os, pickle, sys
#from Bio import SeqIO
import numpy
#import ete3

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

import utils
import scipy.integrate as integrate
import scipy.special as special
#import dbd_utils



def calculate_shannon_diversity(sad):

    relative_sad = sad / sum(sad)
    relative_sad = relative_sad[relative_sad>0]
    shannon_diversity = -1*sum(relative_sad*numpy.log(relative_sad) )

    return shannon_diversity


def prob_n_reads(n, N, mean_, beta_):

    # exp( gammaln(beta+n) - gammaln(n+1) - gammaln(beta) )
    # gamma of factorial results in numerical overflow, do logamma trick instead
    # gamma(beta+n) and gamma(n+1) are large, but their ratio is not, so gammaln(beta+n) - gammaln(n+1) is ok and can be exponentiated

    return numpy.exp( special.gammaln(beta_+n) - special.gammaln(n+1) - special.gammaln(beta_) )   * (((mean_*N)/(beta_ + mean_*N))**n) * ((beta_/(beta_ + mean_*N))**beta_)


def integrand_first_moment(n, N, mean_, beta_):
    return (n/N)*numpy.log(n/N) * prob_n_reads(n, N, mean_, beta_)



def predict_mean_and_var_diversity_analytic(s_by_s):

    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
    diversity_observed = numpy.apply_along_axis(calculate_shannon_diversity, 0, rel_s_by_s)
    mean_diversity_observed = numpy.mean(diversity_observed)
    
    n_reads = s_by_s.sum(axis=0)

    mean_rel_s_by_s = numpy.mean(rel_s_by_s, axis=1)
    var_rel_s_by_s = numpy.var(rel_s_by_s, axis=1)
    beta_rel_s_by_s = (mean_rel_s_by_s**2)/var_rel_s_by_s

    mean_all = []
    # dict with integral for each species for each sample
    for m in range(len(n_reads)):

        N_m = int(n_reads[m])

        integrand_first_moment_all = []
        for i in range(len(mean_rel_s_by_s)):
            
            mean_i = mean_rel_s_by_s[i]
            beta_i = beta_rel_s_by_s[i]

            integral_first_moment_result = integrate.quad(integrand_first_moment, 0, N_m, args=(N_m, mean_i, beta_i), epsabs=1e-20)
            integrand_first_moment_all.append(integral_first_moment_result[0])


        integrand_first_moment_all = numpy.absolute(integrand_first_moment_all)
        mean_m = sum(integrand_first_moment_all)
        mean_all.append(mean_m)

    mean_diversity_predicted = numpy.mean(mean_all) 

    return mean_diversity_observed, mean_diversity_predicted



s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration_test_singleton(migration='No_migration', inocula=4, transfer=18)


mean_diversity_observed, mean_diversity_predicted = predict_mean_and_var_diversity_analytic(s_by_s)
rel_error = numpy.absolute(mean_diversity_observed - mean_diversity_predicted)/mean_diversity_observed

print(mean_diversity_observed, mean_diversity_predicted, rel_error)

