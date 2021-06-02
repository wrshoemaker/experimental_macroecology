from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils


path_name = '%s/data/otu_table_psn_v13.txt' % utils.directory

otu = open(path_name)

#df = pd.read_csv(path_name, sep='\t')

#print(df.shape)
otu_first_line = otu.readline()
otu_first_line = otu_first_line.strip().split(',')


#count_dict = {}
#each line is a species
#

otus = []

for line in otu:
    line = line.strip().split('\t')
    line = line[1:-1]
    line = [int(x) for x in line]
    otus.append(line)
    #if int(line[1]) == 0:
    #    continue



otu.close()



otu_np = np.array(otus)
otu_np_no_zeros = np.where(otu_np == 1, 0, otu_np)


totreads = otu_np.sum(axis=0)
totreads_no_zeros = otu_np_no_zeros.sum(axis=0)




otu_np_subset = otu_np[:50, ]
otu_np_no_zeros_subset = otu_np_no_zeros[:50, ]



occupancies, predicted_occupancies = utils.predict_occupancy(otu_np_subset, totreads=totreads)

occupancies_no_zeros, predicted_occupancies_no_zeros = utils.predict_occupancy(otu_np_no_zeros_subset, totreads=totreads_no_zeros)


print(np.mean(np.absolute(predicted_occupancies-occupancies)))

print(np.mean(np.absolute(predicted_occupancies_no_zeros-occupancies_no_zeros)))






fig, ax = plt.subplots(figsize=(4,4))


ax.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
ax.scatter(occupancies, predicted_occupancies, c='b', alpha=0.8, label='Singletons', zorder=2)#, c='#87CEEB')
ax.scatter(occupancies_no_zeros, predicted_occupancies_no_zeros, c='r',  label='No singletons', alpha=0.8, zorder=2)#, c='#87CEEB')

ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)
ax.set_xlabel('Observed occupancy', fontsize=12)
ax.set_ylabel('Predicted occupancy', fontsize=10)

ax.set_title('Occupancy predicted by gamma distribution, HMP', fontsize=12, fontweight='bold' )
ax.legend(loc="lower right", fontsize=8)



fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(utils.directory + "/figs/predicted_occupancies_HMP.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
plt.close()
