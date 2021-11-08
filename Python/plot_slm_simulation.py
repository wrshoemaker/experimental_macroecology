from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.colors as clr

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

import test_sde_simulation


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])





def mut_freq_colormap():
    #cmap = clr.LinearSegmentedColormap.from_list('Zissou1', ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"], N=256)
    cmap = clr.LinearSegmentedColormap.from_list('Darjeeling1', ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"], N=50)

    # sample from cmap using uniform dist b/w 0 and 1
    u = np.random.uniform()
    rgb = '#%02x%02x%02x' % tuple([int(x * 100) for x in list(cmap(u))[:-1]])
    #tuple([int(x * 100) for x in list(cmap(u))[:-1]])
    # RGB six digit code
    return rgb




x, t = test_sde_simulation.run_simulation()


x_r = x[:,2,:]
x_r_rel = (x_r.T/x_r.sum(axis=1)).T
x_r_rel_no_zeros = x_r_rel[:,~np.all(x_r_rel == 0, axis=0)]
# sort by initial relative abundance
x_r_rel_no_zeros = x_r_rel_no_zeros[:,np.argsort(x_r_rel_no_zeros[0,:])[::-1]]



#print(x_r_rel_no_zeros[np.argsort(x_0)])

fig, ax = plt.subplots(figsize=(4,4))

y_upper_bound = np.asarray([1]*x_r_rel_no_zeros.shape[0])

for species_trajectory_idx, species_trajectory in enumerate( x_r_rel_no_zeros.T):

    if species_trajectory_idx == x_r_rel_no_zeros.shape[1]-1:
        y_lower_bound = np.asarray([0]*x_r_rel_no_zeros.shape[0])

    else:
        y_lower_bound = y_upper_bound-species_trajectory

    # fill between these two
    # get color map
    #print(y_upper_bound[0], y_lower_bound[0], species_trajectory[0])
    rgb = mut_freq_colormap()
    rgb = lighten_color(rgb, amount=0.5)
    # color=rgb,
    ax.fill_between(t, y_lower_bound, y_upper_bound, alpha=1)

    ax.set_xlim(0, max(t ))
    ax.set_ylim(0, 1)


    #ax.plot(times, freqs, '.-', c=rgb, alpha=0.4)


    y_upper_bound = y_upper_bound - species_trajectory





fig_name = utils.directory + '/figs/slm_timeseries.png'
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
