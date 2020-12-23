from __future__ import division
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bisect

from matplotlib import cm

color_range =  np.linspace(0.0, 1.0, 18)
rgb = cm.get_cmap('Blues')( color_range )
rgb_red = cm.get_cmap('Reds')( color_range )
rgb_green = cm.get_cmap('Greens')( color_range )

color_dict = {('No_migration',4):rgb, ('Global_migration',4):rgb_red, ('Parent_migration', 4):rgb_green }


#from macroeco_distributions import pln, pln_solver, pln_ll
#from macroecotools import obs_pred_rsquare

directory = os.path.expanduser("~/GitHub/experimental_macroecology")
metadata_path = directory + '/data/metadata.csv'
otu_path = directory + '/data/otu_table.csv'


carbons = ['Glucose', 'Citrate', 'Leucine']
carbons_colors = ['royalblue', 'forestgreen', 'darkred']
carbons_shapes = ['o', 'D', 'X']

titles = ['No migration, low inoculum', 'No migration, high inoculum', 'Global migration, low inoculum', 'Parent migration, low inoculum' ]



migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]

titles_dict = {('No_migration',4): 'No migration, low inoculum',
                ('No_migration',40): 'No migration, high inoculum',
                ('Global_migration',4): 'Global migration, low inoculum',
                ('Parent_migration',4): 'Parent migration, low inoculum',
                ('Glucose', np.nan): 'Glucose' }


titles_no_inocula_dict = {('No_migration',4): 'No migration',
                ('Global_migration',4): 'Global migration',
                ('Parent_migration',4): 'Parent migration',
                ('Glucose', np.nan): 'Glucose' }


family_colors = {'Alcaligenaceae':'darkorange', 'Comamonadaceae': 'darkred',
                'Enterobacteriaceae':'dodgerblue', 'Enterococcaceae':'limegreen',
                'Lachnospiraceae':'deepskyblue', 'Pseudomonadaceae':'darkviolet'}



def get_p_value(p_value, alpha=0.05):

    if p_value >= alpha:

        return r'$P \nless 0.05$'

    else:
        return r'$P < 0.05$'


def get_pair_stats(x, y):

    #calculate means
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    #get number of entries
    n = len(x)

    #calculate sums
    x_sum = np.sum(x)
    x_sum_square = np.sum([xi**2 for xi in x])
    y_sum = np.sum(y)
    y_sum_square = np.sum([yi**2 for yi in y])
    xy_sum = np.sum([xi*yi for xi, yi in zip(x, y)])

    #calculcate remainder of equations
    s_xx  = x_sum_square - (1/n)*(x_sum**2)
    s_yy = y_sum_square - (1/n)*(y_sum**2)
    s_xy = xy_sum - (1/n)*x_sum*y_sum

    return s_xx, s_yy, s_xy





def get_s_by_s(carbons, transfer=12, communities=None):

    otu = open(directory + '/data/otu_table.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')


    replica_id_list = []
    comm_rep_list = []
    comm_rep_replica_id_dict = {}

    for line in open(metadata_path):
        line = line.strip().split(',')
        #if line[-3].strip() == carbon:
        if line[-3].strip() in carbons:
            if line[-1].strip() == str(transfer):
                replica_id = line[0].strip()
                replica_id_list.append(replica_id)
                comm_rep = line[-5]+ '_' + line[-4]
                comm_rep_list.append(comm_rep)
                comm_rep_replica_id_dict[comm_rep] = replica_id

    if communities == None:

        communities_final = comm_rep_list
        communities_idx = [otu_first_line.index(comm_rep_replica_id_dict[c]) for c in comm_rep_list]


    else:

        communities_final = [str(k) for k in communities]
        print(comm_rep_replica_id_dict)
        communities_idx = [otu_first_line.index(comm_rep_replica_id_dict[c]) for c in communities]


    s_by_s = []
    species = []

    for line in otu:
        line = line.strip().split(',')
        line_idx = [int(line[i]) for i in communities_idx]

        if sum(line_idx) == 0:
            continue

        s_by_s.append(line_idx)
        species.append(line[0])

    s_by_s_np = np.asarray(s_by_s)

    return s_by_s_np, species, communities_final



def get_s_by_s_temporal(transfers, carbon='Glucose'):

    transfers = [str(transfer) for transfer in transfers]

    otu = open( '%s/data/otu_table.csv' % directory)
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    communities = []
    comm_rep_list = []

    for line in open(metadata_path):
        line = line.strip().split(',')
        if line[-3].strip() == carbon:
            if line[-1].strip() == str(1):
                communities.append(line[0].strip())
                comm_rep_list.append(line[-5]+ '_' + line[-4] )

    #communities_transfer = []
    comm_rep_transfer_list = []
    for line in open(metadata_path):
        line = line.strip().split(',')
        if line[-3].strip() == carbon:
            line_transfer = line[-1].strip()
            if line[-5]+ '_' + line[-4] in comm_rep_list:
                #communities_transfer.append(line[0].strip()+'_'+line_transfer)
                comm_rep_transfer_list.append(line[-5]+ '_' + line[-4]+'_'+line_transfer)

    communities_idx = [otu_first_line.index(community) for community in communities]


    #print(communities_idx)
    #print(comm_rep_transfer_list)


def get_species_means_and_variances(rel_s_by_s, species_list, min_observations=3, zeros=False):

    mean_rel_abundances = []
    var_rel_abundances = []
    species_to_keep = []

    for afd_idx, afd in enumerate(rel_s_by_s):

        afd_no_zeros = afd[afd>0]

        if len(afd_no_zeros) < min_observations:
            continue

        if zeros == True:

            mean_rel_abundances.append(np.mean(afd))
            var_rel_abundances.append(np.var(afd))

        else:

            mean_rel_abundances.append(np.mean(afd_no_zeros))
            var_rel_abundances.append(np.var(afd_no_zeros))

        species_to_keep.append(species_list[afd_idx])

    mean_rel_abundances = np.asarray(mean_rel_abundances)
    var_rel_abundances = np.asarray(var_rel_abundances)
    species_to_keep = np.asarray(species_to_keep)

    return mean_rel_abundances, var_rel_abundances, species_to_keep




#class lognorm:

#    def __init__(self, obs):
#        self.obs = obs


#    def ppoints(self, n):
#        """ numpy analogue or `R`'s `ppoints` function
#            see details at http://stat.ethz.ch/R-manual/R-patched/library/stats/html/ppoints.html
#            :param n: array type or number
#            Obtained from: http://stackoverflow.com/questions/20292216/imitating-ppoints-r-function-in-python
#            on 5 April 2016
#            """
#        if n < 10:
#            a = 3/8
#        else:
#            a = 1/2

#        try:
#            n = np.float(len(n))
#        except TypeError:
#            n = np.float(n)
#        return (np.arange(n) + 1 - a)/(n + 1 - 2*a)



#    def get_rad_pln(self, S, mu, sigma, lower_trunc = True):
#        """Obtain the predicted RAD from a Poisson lognormal distribution"""
#        abundance = list(np.empty([S]))
#        rank = range(1, int(S) + 1)
#        cdf_obs = [(rank[i]-0.5) / S for i in range(0, int(S))]
#        j = 0
#        cdf_cum = 0
#        i = 1
#        while j < S:
#            cdf_cum += pln.pmf(i, mu, sigma, lower_trunc)
#            while cdf_cum >= cdf_obs[j]:
#                abundance[j] = i
#                j += 1
#                if j == S:
#                    abundance.reverse()
#                    return abundance
#            i += 1


#    def get_rad_from_obs(self):
#        mu, sigma = pln_solver(self.obs)
#        pred_rad = self.get_rad_pln(len(self.obs), mu, sigma)

#        return (pred_rad, mu, sigma)




#def fit_sad():

#    for carbon_idx, carbon in enumerate(carbons):
#
#        sys.stdout.write("Fitting lognormal for %s communities...\n" % (carbon))
#
#        s_by_s, species, comm_rep_list = get_s_by_s(carbon)

#        data = []

#        obs_list = []
#        pred_list = []

#        N_communities = s_by_s.shape[1]
#        for sad in s_by_s.T:
#            sys.stdout.write("%d communities to go!\n" % (N_communities))
#            N_communities -= 1
#
#            sad = sad[sad>0]
#            sad.sort()
#            sad = sad[::-1]
#            if len(sad) < 10:
#                continue

#            N = sum(sad)
#            S = len(sad)
#            Nmax = max(sad)

#            lognorm_pred = lognorm(sad)
#            sad_predicted, mu, sigma = lognorm_pred.get_rad_from_obs()
#            pln_Nmax = max(sad_predicted)

#            ll = pln_ll(sad, mu, sigma)
#            r2 = obs_pred_rsquare(np.log10(sad), np.log10(sad_predicted))

#            data.append([str(N),str(S),str(Nmax),str(pln_Nmax),str(ll),str(r2)])

#            obs_list.extend(sad)
#            pred_list.extend(sad_predicted)


#        df_sad_out = open((directory + '/data/pln_sad_counts_%s.csv') % (carbon) , 'w')
#        df_sad_out.write(','.join(['Observed','Predicted']) + '\n')
#        for i in range(len(obs_list)):
#            df_sad_out.write(','.join([str(obs_list[i]), str(pred_list[i])]) + '\n')
#        df_sad_out.close()


#        df_out = open((directory + '/data/pln_sad_%s.csv') % (carbon) , 'w')
#        df_out.write(','.join(['N','S','Nmax', 'pln_Nmax', 'll','r2_mod']) + '\n')
#        for i in range(len(data)):
#            df_out.write(','.join(data[i]) + '\n')
#        df_out.close()







def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #todo: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = np.log10(x), np.log10(y), np.log10(radius)
    for a, b in unique_points:
        if logscale == 1:
            loga, logb = np.log10(a), np.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 +
                                   (logy - logb) ** 2) <= logr ** 2])
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))
    #color_range = np.sqrt(sorted_plot_data[:, 2])
    #color_range = color_range / max(color_range)

    #colors = [ cm.YlOrRd(x) for x in color_range ]

    if plot_obj == None:
        plot_obj = plt.axes()

    if loglog == 1:
        plot_obj.set_xscale('log')
        plot_obj.set_yscale('log')
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                         c = np.sqrt(sorted_plot_data[:, 2]), edgecolors='none')
        plot_obj.set_xlim(min(x) * 0.5, max(x) * 2)
        plot_obj.set_ylim(min(y) * 0.5, max(y) * 2)
    else:
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                    c = np.log10(sorted_plot_data[:, 2]), edgecolors='none')
    return plot_obj



def predict_occupancy(s_by_s):

    s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

    #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
    totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(np.mean(afd_no_zeros/ totreads[afd>0]))
        #tf.append(np.mean(afd_no_zeros/s_by_s.sum(axis=0)[afd>0]))

    #tf = np.mean(rel_abundances)
    tf = np.asarray(tf)
    # go through and calculate the variance for each species



    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        N_reads = s_by_s.sum(axis=0)[np.nonzero(afd)[0]]
        #N_reads = s_by_s.sum(axis=0)[afd>0]

        tvpf_list.append(np.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    #tvpf = np.mean(tvpf_list)
    tvpf = np.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    predicted_occupancies = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    for beta_i, theta_i in zip(beta,theta):

        predicted_occupancies.append(1 - np.mean( (1+theta_i*totreads) **(-1*beta_i ) )  )

        #1- mean( (1.+theta*totreads)^(-beta ) )

    predicted_occupancies = np.asarray(predicted_occupancies)
    occupancies_no_zeros = occupancies[occupancies>0]
    predicted_occupancies_no_zeros = predicted_occupancies[occupancies>0]

    return occupancies_no_zeros, predicted_occupancies_no_zeros


def group_ESVs(s_by_s, species, comm_rep_list, taxonomic_level='genus'):
    taxonomic_level_dict = {'genus':-1, 'family':-2, 'order':-3, 'class':-4}
    taxonomy_map = {}
    taxonomy = open(directory+'/data/taxonomy.csv')
    taxonomy_fitst_line = taxonomy.readline()
    for line in taxonomy:
        line = line.strip().split(',')
        if line[ taxonomic_level_dict[taxonomic_level] ] != 'NA':
            taxonomy_map[line[0]] = line[ taxonomic_level_dict[taxonomic_level] ]
    taxonomy.close()


    taxon_sample_dict = {}

    for afd_idx, afd in enumerate(s_by_s):

        esv = species[afd_idx]
        if esv not in taxonomy_map:
            continue

        taxon = taxonomy_map[esv]
        if taxon not in taxon_sample_dict:
            taxon_sample_dict[taxon] = {}

        for comm_rep_idx, comm_rep in enumerate(comm_rep_list):
            if comm_rep not in taxon_sample_dict[taxon]:
                taxon_sample_dict[taxon][comm_rep] = 0

            taxon_sample_dict[taxon][comm_rep] += afd[comm_rep_idx]


    s_by_s_df = pd.DataFrame(taxon_sample_dict)
    s_by_s = s_by_s_df.values.T

    #return s_by_s, list(taxon_sample_dict.keys()), comm_rep_list
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()






def get_relative_s_by_s_migration(transfer=18,migration='No_migration',inocula=40, communities=None):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/migration_ESV_data_table_SE.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    count_dict = {}
    for line in otu:
        line = line.strip().split(',')
        if float(line[1]) == 0:
            continue

        if (int(line[5]) == transfer) and (line[2] == migration) and (int(line[3]) == inocula) :

            if line[0] not in count_dict:
                count_dict[line[0]] = {}

            if communities == None:

                if line[6] not in count_dict[line[0]]:
                    count_dict[line[0]][line[6]] = 0

                if count_dict[line[0]][line[6]] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[line[0]][line[6]] = float(line[1])

            else:

                if line[6] not in communities:
                    continue

                if line[6] not in count_dict[line[0]]:
                    count_dict[line[0]][line[6]] = 0

                if count_dict[line[0]][line[6]] > 0:

                    sys.stdout.write("Relative abundance already in dictionary!!\n" )

                count_dict[line[0]][line[6]] = float(line[1])



    otu.close()

    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()



def get_relative_s_by_s_temporal_migration(migration='No_migration',inocula=40,communities=None):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if communities==None:

        communities_new = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities_new.append(line[6])

        communities_new = list(set(communities_new))

    else:

        communities_new = [str(k) for k in communities]

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        if float(line[1]) == 0:
            continue

        if (line[6] in communities_new) and (line[2] == migration) and (int(line[3]) == inocula):

            if line[0] not in count_dict:
                count_dict[line[0]] = {}


            community_time = '%s_%s' % (line[6], line[5])

            if community_time not in count_dict[line[0]]:
                count_dict[line[0]][community_time] = 0

            if count_dict[line[0]][community_time] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[line[0]][community_time] = float(line[1])


    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()





def get_s_by_s_temporal_migration(migration='No_migration',inocula=40,communities=None):

    otu_path = directory + '/data/migration_ESV_data_table_absabundance_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if communities==None:

        communities_new = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities_new.append(line[6])

        communities_new = list(set(communities_new))

    else:

        communities_new = [str(k) for k in communities]


    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        abundance = int(line[1])

        if abundance == 0:
            continue

        if (line[6] in communities_new) and (line[3] == migration) and (int(line[4]) == inocula):

            if line[0] not in count_dict:
                count_dict[line[0]] = {}

            community_time = '%s_%s' % (line[6], line[5])

            if community_time not in count_dict[line[0]]:
                count_dict[line[0]][community_time] = 0

            if count_dict[line[0]][community_time] > 0:

                sys.stdout.write("Relative abundance already in dictionary!!\n" )

            count_dict[line[0]][community_time] = abundance


    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()







def get_migration_time_series_community_names(migration='No_migration',inocula=40):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_community_dict = {}

    communities = []

    # line[5] = transfer
    # line[2] = migration
    # line[3] = inocula

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')
        if line[3] == 'NA':
            continue

        if (line[2] == migration) and (int(line[3]) == inocula):
            replicate_community = int(line[6])
            if replicate_community not in count_community_dict:
                count_community_dict[replicate_community] = []
            transfer = int(line[5])
            if transfer not in count_community_dict[replicate_community]:
                count_community_dict[replicate_community].append(transfer)

        #if (int(line[5]) == 3) and (line[2] == migration) and (int(line[3]) == inocula):
        #    communities.append(line[6])

    return count_community_dict






def get_s_by_s_migration(transfer=18, migration='No_migration',inocula=40, communities=None ):
    # migration options, 'No_migration', 'Parent_migration', 'Global_migration'

    otu = open(directory + '/data/migration_ESV_data_table_absabundance_SE.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    if communities != None:
        communities = [str(k) for k in communities]

    count_dict = {}
    for line in otu:
        line = line.strip().split(',')
        if int(line[1]) == 0:
            continue

        if (int(line[5]) == transfer) and (line[3] == migration) and (int(line[4]) == inocula):

            rep = '%s_%s_%s' % (line[3], line[4], line[0])

            if rep not in count_dict:
                count_dict[rep] = {}

            if communities == None:

                if line[6] not in count_dict[rep]:
                    count_dict[rep][line[6]] = 0

                if count_dict[rep][line[6]] > 0:

                    sys.stdout.write("Species already in dictionary!!\n" )

                count_dict[rep][line[6]] += int(line[1])

            else:

                if line[6] not in communities:
                    continue

                if line[6] not in count_dict[rep]:
                    count_dict[rep][line[6]] = 0

                if count_dict[rep][line[6]] > 0:

                    sys.stdout.write("Species already in dictionary!!\n" )

                count_dict[rep][line[6]] += int(line[1])



    otu.close()

    s_by_s_df = pd.DataFrame(count_dict)
    s_by_s_df = s_by_s_df.fillna(0)
    s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()






def get_relative_abundance_dictionary_temporal_migration(migration='No_migration',inocula=40,community=None):

    otu_path = directory + '/data/migration_ESV_data_table_SE.csv'
    #otu_first_line = otu.readline()
    #otu_first_line = otu_first_line.strip().split(',')
    count_dict = {}

    if community==None:

        communities = []

        for line_idx, line in enumerate(open(otu_path, 'r')):
            if line_idx == 0:
                continue
            line = line.strip().split(',')
            if (int(line[5]) == 1) and (line[2] == migration) and (int(line[3]) == inocula):
                communities.append(line[6])

        communities = list(set(communities))

    else:

        communities = [str(community)]

    for line_idx, line in enumerate(open(otu_path, 'r')):
        if line_idx == 0:
            continue
        line = line.strip().split(',')

        if float(line[1]) == 0:
            continue

        replicate = line[6]
        species = line[0]
        transfer = int(line[5])

        relative_abundance = float(line[1])

        if relative_abundance == 0:
            continue

        if (replicate in  communities) and (line[2] == migration) and (int(line[3]) == inocula):

            if species not in count_dict:
                count_dict[species] = {}

            if replicate not in  count_dict[species]:
                count_dict[species][replicate] = {}

                count_dict[species][replicate]['transfers'] = []

                count_dict[species][replicate]['relative_abundances'] = []


            bisect.insort(count_dict[species][replicate]['transfers'],  int(transfer))

            transfer_idx = count_dict[species][replicate]['transfers'].index(transfer)

            count_dict[species][replicate]['relative_abundances'].insert(transfer_idx, relative_abundance)



            #community_time = '%s_%s' % (line[6], line[5])

            #if community_time not in count_dict[line[0]]:
            #    count_dict[line[0]][community_time] = 0

            #if count_dict[line[0]][community_time] > 0:

            #    sys.stdout.write("Relative abundance already in dictionary!!\n" )

            #count_dict[line[0]][community_time] = float(line[1])

    return count_dict

    #s_by_s_df = pd.DataFrame(count_dict)
    #s_by_s_df = s_by_s_df.fillna(0)
    #s_by_s = s_by_s_df.values.T
    # it's a relativized S by S matrix
    #return s_by_s, s_by_s_df.columns.to_list(), s_by_s_df.index.to_list()





def get_attractor_status(migration='No_migration', inocula=4):

    attractor_dict = {}

    attractor_file = open(directory+'/data/attractor_status.csv')
    attractor_file_fitst_line = attractor_file.readline()
    for line in attractor_file:
        line = line.strip().split(',')
        if (line[0] == migration) and (int(line[1]) == inocula):

            if line[-1] not in attractor_dict:
                attractor_dict[line[-1]] = []

            attractor_dict[line[-1]].append(str(line[-2]))
    attractor_file.close()

    return attractor_dict
