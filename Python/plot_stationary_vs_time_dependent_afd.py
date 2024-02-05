from __future__ import division
import numpy as np
from scipy.special import gamma, factorial, genlaguerre, kv
import scipy.integrate as integrate
import mpmath
import utils
import matplotlib.pyplot as plt




def A(sigma, k):

    return ((2/(k*sigma))**(1 + (2/sigma)*(1-sigma))) / gamma(((2/sigma)*(1-sigma)) + 1)


def kappa(sigma):
    return 1 + (1-sigma)*(1/sigma)


def v_n(sigma, n):

    return (2/sigma)*(1-sigma) - 2*n + 1 


def lambda_n(sigma, tau, n):

    return (n/2)*(sigma/tau)*( (2/sigma) * (1-sigma)  + 1 - n) 


def lambda_eta(sigma, tau, eta):

    #return (sigma/(2*tau)) * ( ((((1-sigma)/sigma) + 0.5 )  **2 ) +  (eta**2))
    return (sigma/(2*tau)) * ( (((1/sigma)-0.5)**2) + (eta**2) )



def G_eta(sigma, tau, k, eta):

    kappa =  1 + (1-sigma)*(1/sigma)

    return np.sqrt(((sigma*k) / (2*(np.pi**2)*A(sigma, k))) * eta * np.sinh(2*np.pi*eta) * gamma(0.5 - kappa + 1j*eta) *  gamma(0.5 - kappa - 1j*eta) )


def h(y, sigma, tau, k, eta):

    kappa = 1 + (1-sigma)*(1/sigma)

    whitw = mpmath.whitw(kappa, 1j*eta, 2/(k*sigma*y))
    whitw_real = mpmath.re(whitw)
    whitw_im = mpmath.im(whitw)
    whitw_python = complex(whitw_real, whitw_im)

    return G_eta(sigma, tau, k, eta)* (y**kappa) * np.exp(1/(k*sigma*y)) * whitw_python




def integrand(eta, x, t, x_0, sigma, tau, k):
    # eta is what you integrate over 
    #print(  h(x**-1, sigma, tau, k, eta)* h(x_0**-1, sigma, tau, k, eta) * np.exp(-1*lambda_eta(sigma, tau, eta) * t ))

    # take real part of the product of two imaginary numbers
    real_part_h_product = np.real(h(x**-1, sigma, tau, k, eta)* h(x_0**-1, sigma, tau, k, eta))


    exp_term = np.exp(-1*lambda_eta(sigma, tau, eta) * t )
    
    if (np.isnan(real_part_h_product) == True) and (exp_term == float(0)):
        product = 0
    else:
        product = exp_term*real_part_h_product

    return product



def prob_x_time_dependent(x, t, x_0, sigma, tau, k, M):

    # M = dimensionality of Wiener process

    mult_term = (x**((2/sigma) * (1-sigma))) * np.exp(-2*x/(k*sigma))

    sum_ = 0
    for m in range(M):

        v_m = v_n(sigma, m)
        genlaguerre_m = genlaguerre(m, v_m)
        sum_ += ((2/(k*sigma))**v_m) * (factorial(m)*v_m/ gamma(v_m + m + 1) ) * np.exp(-1*lambda_n(sigma, tau, m)*t) * ((x*x_0)**-m) * genlaguerre_m((2*x/(k*sigma))) * genlaguerre_m((2*x_0/(k*sigma)))


    integral_ = integrate.quad(integrand, 0, np.inf, args=(x, t, x_0, sigma, tau, k))[0]


    return mult_term*(sum_ + A(sigma, k)*integral_)



def prob_x_migration(x, sigma, tau, k):

    m = 10*k

    m_tilde =  m*tau

    const = (2*((m_tilde*k)**(0.5*((2/sigma) - 1))) * kv((2/sigma)-1, (4/sigma)*np.sqrt(m_tilde/k)))**-1
    

    return const * np.exp(((-2)/(sigma*x)) * (m_tilde + ((x**2)/k)) ) * (x**( (2/sigma) - 2))





def prob_x(x, sigma, k):


    return (1/gamma( (2/sigma)-1 )) * ((2/(k*sigma))**((2/sigma)-1)) * np.exp(- x*2/(k*sigma)) * (x**((2/sigma)-2) )




x_0 = 10000
#x = 0.04
#t = 4
sigma = 0.7
tau = 1
k = 1000
M = 2

N_total = 10**6




fig, ax = plt.subplots(figsize=(4,4))


#x_range = list(range(1, 1001))

#x_range = np.logspace(np.log10(1), np.log10(10**5), num=50, endpoint=True, base=10.0)
x_range = np.logspace(np.log10(1), np.log10(10**5), num=500, endpoint=True, base=10.0)
x_range = [int(x) for x in x_range]


#print(prob_x_time_dependent(1, t, x_0, sigma, tau, k, M))


prob_x_t_1 = []
prob_x_t_100 = []
prob_x_t_10000 = []
prob_x_migration_all = []

prob_x_t_infty = []
for x in x_range:
    #print(x)
    prob_x_t_1.append(prob_x_time_dependent(x, 1, x_0, sigma, tau, k, M))
    prob_x_t_100.append(prob_x_time_dependent(x, 10, x_0, sigma, tau, k, M))
    prob_x_t_10000.append(prob_x_time_dependent(x, 100, x_0, sigma, tau, k, M))

    prob_x_migration_all.append(prob_x_migration(x, sigma, tau, k))


    prob_x_t_infty.append(prob_x(x, sigma, k))


print(prob_x_t_infty)

#p_t_4  = np.asarray([])


prob_x_t_1 = np.asarray(prob_x_t_1)
prob_x_t_100 = np.asarray(prob_x_t_100)
prob_x_t_10000 = np.asarray(prob_x_t_10000)
prob_x_migration_all = np.asarray(prob_x_migration_all)


#prob_x_t_1 = prob_x_t_1/sum(prob_x_t_1)
#prob_x_t_100 = prob_x_t_1/sum(prob_x_t_100)
#prob_x_t_10000 = prob_x_t_10000/sum(prob_x_t_10000)
#prob_x_migration_all = prob_x_migration_all/sum(prob_x_migration_all)



x_range = np.asarray(x_range)
ax.plot(x_range/N_total, prob_x_t_1, ls='-', c='lightskyblue', label= "Migration as an initial condition, " + r'$\frac{t}{\tau} = 10^{0}$' )
ax.plot(x_range/N_total, prob_x_t_100, ls='-', c='cornflowerblue', label= "Migration as an initial condition, " + r'$\frac{t}{\tau} = 10^{1}$')
ax.plot(x_range/N_total, prob_x_t_10000, ls='-', c='royalblue', label="Migration as an initial condition, " + r'$\frac{t}{\tau} = 10^{2}$')

ax.plot(x_range/N_total, prob_x_migration_all, ls='-', c='firebrick', label="Stationary AFD, constant migration")



#prob_x_t_infty = 

ax.plot(x_range/N_total, prob_x_t_infty, ls=':',  c='k', label= "Stationary AFD, no migration")


ax.set_ylim([1/N_total, 1])
#ax.set_ylim([1e-4, 1])
ax.set_ylim([1e-8, 5e-3])
ax.legend(loc="lower left", fontsize=7)

ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

ax.set_xlabel('Relative abundance, ' + r'$x_{i}$', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)


fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(utils.directory + "/figs/stationary_vs_time_dependent_afd.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/stationary_vs_time_dependent_afd.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()


