import calculate_species_relative_abundance
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf

mean_abundance_dict = calculate_species_relative_abundance.load_species_mean_abun_dict()

af_abun_all = []
na_abun_all = []
for species_abundance in mean_abundance_dict.keys():
    af_abun_all.append(mean_abundance_dict[species_abundance]['Africa'])
    na_abun_all.append(mean_abundance_dict[species_abundance]['North America'])
    
af_abun_all = np.array(af_abun_all)
na_abun_all = np.array(na_abun_all)

n = float(na_abun_all.shape[0])

na_m1 = (1/n)*np.log10(na_abun_all).sum()
na_m2 = (1/n)*((np.log10(na_abun_all))**2).sum()

af_m1 = (1/n)*np.log10(af_abun_all).sum()
af_m2 = (1/n)*((np.log10(af_abun_all))**2).sum()

c_na = 10**-4
c_africa = 10**-4


def get_estimates(init):
    sigma = init[0]
    mu = init[1]
    a = -m1 + mu + ((np.sqrt(2/np.pi)*sigma*np.exp(-1*((np.log(c)-mu)**2)/(2*(sigma**2))))/erf((np.log(c)-mu)/(np.sqrt(2)*sigma)))
    b = -m2 + (sigma**2) + m1*mu + c*m1 - mu*c
    return np.array([a,b])

init = [1,1]
m1 = na_m1
m2 = na_m2
c = c_na
na_dist = fsolve(get_estimates, init)

m1 = af_m1
m2 = af_m2
c = c_africa
af_dist = fsolve(get_estimates, init)

na_sigma = na_dist[0]
na_mu = na_dist[1]
af_sigma = af_dist[0]
af_mu = af_dist[1]