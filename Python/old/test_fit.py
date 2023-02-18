import numpy as np
import math
from scipy import optimize

#import curve_fit, least_squares, minimize




yobs = np.array([0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004,
                    0.048, 0.119, 0.199, 0.277, 0.346, 0.395, 0.444, 0.469,
                    0.502, 0.527, 0.553, 0.582, 0.595, 0.603, 0.612, 0.599])
xobs = np.array([0.013, 0.088, 0.159, 0.230, 0.292, 0.362, 0.419, 0.471,
                    0.528, 0.585, 0.639, 0.687, 0.726, 0.772, 0.814, 0.854,
                    0.889, 0.924, 0.958, 0.989, 1.015, 1.045, 1.076, 1.078])


def fit_underdamped_oscillator(t_star, y, y_0):

    gamma_init = 1
    omega_0_init = 1
    phi_init = 0.5

    def old_underdamped_oscillator(t_star, parameters):
        gamma, omega_0, phi = parameters
        #gamma = b/2m
        t_star = np.asarray(t_star)

        return y_0 * math.exp(-1*t_star*gamma) * math.cos(omega_0*t_star + phi)

    pars_init = [gamma_init, omega_0_init, phi_init]
    res = optimize.minimize(old_underdamped_oscillator, pars_init, args=(t_star, y), method='nelder-mead')

    print(res)




def fit_underdamped_oscillator(ts,xs):
    ts = np.asarray(ts)
    # shift to zero
    ts = ts - ts[0]
    x0 = xs[0]

    gamma_init = 1
    omega_0_init =1
    phi_init = 0.3


    def underdamped_oscillator(ts, gamma, omega_0, phi):

        return x0 * np.exp(-1*ts*gamma) * np.cos(omega_0*ts + phi)


    xmin = optimize.fmin(lambda x: np.square(xs-underdamped_oscillator(ts, x[0],x[1], x[2])).sum(), np.array([gamma_init,omega_0_init, phi_init]))
    gamma = xmin[0]
    omega_0 = xmin[1]
    phi = xmin[2]

    return gamma, omega_0, phi



#fit_underdamped_oscillator(xobs, yobs, y_0=0.3)




#res = optimize.curve_fit(CCER, f_exp, e_exp, p0=(ig_fc, ig_alpha))

#res = optimize.minimize(CCER, f_exp, e_exp, args=(ig_fc, ig_alpha), method='nelder-mead')
