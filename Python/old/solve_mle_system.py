from __future__ import division

import numpy
from scipy.optimize import fsolve
from scipy.special import erf

m_1 = -5.36
m_2 = 31.3
c = 10**-4

def hg(mu_sigma):
    mu = mu_sigma[0]
    sigma = mu_sigma[1]

    #h = y + 2*z
    #g = numpy.sin(y)/z


    f_1 = (numpy.sqrt(2/numpy.pi) * sigma * numpy.exp(-1* ((numpy.log(c)-mu)**2) / (2*(sigma**2)) ) / erf((numpy.log(c) - mu) / (numpy.sqrt(2)*sigma) )) + mu - m_1

    #f_1 = (numpy.sqrt(2/numpy.pi) * sigma * (numpy.exp(-1* ((numpy.log(c)-mu)**2))/(2*(sigma**2)))) / ) - m_1
    f_2 = (sigma**2) + m_1*mu + c*m_1 - mu*c - m_2

    print(mu, sigma)

    return numpy.array([f_1,f_2])


mu_sigma_0 = numpy.array([1,1])
yz = fsolve(hg, mu_sigma_0)

print(yz)
