from __future__ import division
import numpy
from math import log,exp,sqrt
from scipy.optimize import fmin, minimize, least_squares, brentq

x_mean = numpy.asarray([0.01, 0.02, 0.03, 0.04, 0.035, 0.023])
x_parent = 0.1

n_cells_parent = 10**12
n_cells_descendent = 10**8

D_d = 0.004/0.5
D_p = 0.004/0.5
#D_m_tilde = D_m * n_cells_parent / n_cells_descendent
#t_div_tau = 1000
#K =  0.025
bounds = ((0.0001, 0.999), (1, 10))


def stationary_rel_abund_transfer(x_mean, K, t_div_tau):
    result = K / (1 + (   (K*(D_d*n_cells_descendent + D_p*n_cells_parent)/(x_mean*D_d*n_cells_descendent + x_parent*D_p*n_cells_parent))  -1) * exp(-1*t_div_tau) )
    return result


def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""
    def gradient(x):
        fx = fun(x)
        grad = numpy.zeros(len(x))
        for k in range(len(x)):
            d = numpy.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad
    return gradient


def fit_stationary_rel_abund_transfer(x_mean, params):
    # fits to obtain  K, t_div_tau
    K_0 = params[0]
    t_div_tau_0 = params[1]

    #fxn = lambda params: numpy.square(x_mean-stationary_rel_abund_transfer(x_mean, params[0], params[1])).sum()
    fxn = lambda params: numpy.absolute(numpy.log(x_mean)- numpy.log(stationary_rel_abund_transfer(x_mean, params[0], params[1])) ).sum()

    #result = minimize(fxn, numpy.array([K_0,t_div_tau_0]), bounds=bounds,  jac=gradient_respecting_bounds(bounds, fxn))
    result = minimize(fxn, numpy.array([K_0,t_div_tau_0]), bounds=bounds, method="L-BFGS-B") # ,  jac=gradient_respecting_bounds(bounds, fxn))
    print(result)
    return result


fit_stationary_rel_abund_transfer(x_mean,[0.25, 2])

print(numpy.mean(x_mean))
