from __future__ import division
import utils
from scipy import stats, signal

import numpy as np

import matplotlib.pyplot as plt


from smooth_spline import get_natural_cubic_spline_model

#s_by_s, species_all, comm_rep_list_all = s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()

slope_dict = {}

for treatment in ['No_migration', 'Global_migration']:

    s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_temporal_migration(migration=treatment,inocula=4,communities=None)

    count = 0
    for afd_idx, afd in enumerate(s_by_s):
        #afd = afd[afd>0]
        if len(afd[afd>0]) < 35:
            continue

        afd_species = species[afd_idx]

        if afd_species not in slope_dict:
            slope_dict[afd_species] = {}

        # Fourier transform
        frq, f = signal.periodogram(afd)

        frq = np.log10(frq[1:])  # cut zero -> log(0) = -INF
        f = np.log10(np.abs(f.astype(complex))[1:])

        frq = frq[~np.isnan(f)]
        f = f[~np.isnan(f)]

        frq = frq[np.isfinite(f)]
        f = f[np.isfinite(f)]


        p_spline = get_natural_cubic_spline_model(frq, f, minval=min(frq), maxval=max(frq), n_knots=4)

        y = p_spline.predict(frq)

        deriv = (y[1:] - y[:-1]) / (frq[1:] - frq[:-1])

        slope_spline = min(deriv)
        #results.loc[c]['slope_spline'] = min(deriv)

        # only consider frequencies which correspond to periods that are smaller than (length_timeseries/10)
        # otherwise effects from windowing
        f = f[frq >= min(frq) + 1]
        frq = frq[frq >= min(frq) + 1]

        # linear fit
        #print(frq, f)
        p_lin, cov = np.polyfit(frq, f, deg=1, cov=True)

        slope_linear = p_lin[0]
        std_slope_linear = np.sqrt(cov[0, 0])

        slope_dict[afd_species][treatment] = {}
        slope_dict[afd_species][treatment]['slope_linear'] = slope_linear
        slope_dict[afd_species][treatment]['std_slope_linear'] = std_slope_linear


        #print(slope_linear)



#fig, ax = plt.subplots(figsize=(4,4))
fig = plt.figure()
#ax.axhline(1, lw=1.5, ls=':',color='k', zorder=1)
plt.plot([-20,2],[-20, 2], c='k', ls='--', zorder=2)

for key, value in slope_dict.items():
    if len(value)<2:
        continue

    # x axis no migration
    # y axis with migration

    plt.errorbar(value['No_migration']['slope_linear'], value['Global_migration']['slope_linear'],
        xerr=value['No_migration']['std_slope_linear'] , yerr=value['Global_migration']['std_slope_linear'], c='k')

    plt.scatter(value['No_migration']['slope_linear'], value['Global_migration']['slope_linear'], c='k', s=30, zorder=3)#, c='#87CEEB')



# Slope power spectral density

plt.hlines(y=0, xmin=-3, xmax=0, color='k', linestyle=':', lw = 2, zorder=1)
plt.vlines(x=0, ymin=-3, ymax=0, color='k', linestyle=':', lw = 2, zorder=1)

plt.hlines(y=-1, xmin=-3, xmax=-1, color='k', linestyle=':', lw = 2, zorder=1)
plt.vlines(x=-1, ymin=-3, ymax=-1, color='k', linestyle=':', lw = 2, zorder=1)

plt.hlines(y=-2, xmin=-3, xmax=-2, color='k', linestyle=':', lw = 2, zorder=1)
plt.vlines(x=-2, ymin=-3, ymax=-2, color='k', linestyle=':', lw = 2, zorder=1)



plt.xlabel('Slope power spectral density, no migration', fontsize=12)
plt.ylabel('Slope power spectral density, global migration', fontsize=12)

plt.xlim(-2.7, 0.8 )
plt.ylim(-2.7, 0.8 )
fig.subplots_adjust(wspace=0.3)
fig.savefig(utils.directory + "/figs/spectrum_slope.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
