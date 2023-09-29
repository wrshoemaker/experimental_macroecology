
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

np.random.seed(123456789)

x_0 = 0.1
K = 0.8
sigma = 0.1
tau = 2
delta_t =1

x_T_all = []
for i in range(100):

    q_T = [np.log(x_0)]

    Z = np.random.standard_normal(21)
    for t in range(1, 22):

        if t%7 == 0:

            q_t_minus_1 = q_T[t-1] + np.log(np.random.uniform(low=0.2, high=1)) + np.log(0.1)

        else:

            q_t_minus_1 = q_T[t-1]

        q_t_new = q_t_minus_1 + (1/tau)*(1 - sigma/2 - np.exp(q_t_minus_1)/K)*delta_t + np.sqrt(delta_t*sigma/tau)*Z[t-1]
        q_T.append(q_t_new)
        #x_T.append(x_t)

        #q_t = np.asarray(q_t)
        #x_t = np.exp(q_t)
        #x_T.append(x_t)
        
        #x_t_all.append(x_t)


    q_T = np.asarray(q_T)
    x_T = np.exp(q_T)
    x_T[x_T>1] = 1

    x_T_all.append(x_T)


fig, ax = plt.subplots(figsize=(5,4))

x_T_all = np.asarray(x_T_all)
mean_x_T = np.mean(x_T_all, axis=0)
t_range = range(len(mean_x_T))

for i in range(x_T_all.shape[0]):

    x_T_i = x_T_all[i,:]

    ax.plot(t_range, x_T_i, ls='-', c='lightblue', lw=1, alpha=0.6)

    
 

ax.plot(t_range, mean_x_T, ls='-', c='steelblue', lw=2)
ax.set_xlim([0, len(mean_x_T)-1]) 
ax.set_ylim([4e-2, 1])
ax.set_yscale('log', basey=10)
ax.set_xlabel("Cumulative generations", fontsize=12)
ax.set_ylabel('Relative abundance, ' + r'$x_{i}$', fontsize=12)

ax.axhline(y=K, lw=2, ls=':', c='k')
ax.axvline(x=7, lw=2, ls='-', c='k')
ax.axvline(x=14, lw=2, ls='-', c='k')

ax.set_xticks([0, 7, 14, 21])
ax.set_xticklabels(['0', '7', '14', '21'])

#ax.axhline(1, lw=1.5, ls=':',color='k', zorder=1)


legend_elements = [Line2D([0], [0], color='lightblue', lw=2, label='Replicate'),
                   Line2D([0], [0], color='steelblue', lw=2, label='Ensemble mean')]

# Line2D([0], [0], color='k', lw=2, ls='--', label='Carrying capacity, ' + r'$K_{i}$')

ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


# ax.annotate("", xy=(0, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
#ax.arrow(2, 0.3, 0, sigma, head_width=0.05, head_length=0.1, fc='k', ec='k')

#ax.annotate(r'$\sigma$', xy=(0.28,0.2), xytext=(0.2,0.4), va='center', multialignment='right',
#            arrowprops={'arrowstyle': '|-|', 'lw': 2, 'ec': 'k'})


#ax.text(0.5,1.5, r'$\tau$', fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes )


fig.subplots_adjust(wspace=0.25, hspace=0.2)
#fig.savefig(utils.directory + "/figs/example_langevin.png", format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
fig.savefig(utils.directory + "/figs/example_langevin.eps", format='eps', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)

plt.close()