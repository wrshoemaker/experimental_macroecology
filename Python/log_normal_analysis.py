from __future__ import division
import os, sys, signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.stats as stats
from scipy import optimize
from scipy.stats import itemfreq, gamma

from macroeco_distributions import pln, pln_solver, pln_ll
from macroecotools import obs_pred_rsquare

mydir = os.path.expanduser("~/GitHub/experimental_macroecology")

metadata_path = mydir + '/data/metadata.csv'
otu_path = mydir + '/data/otu_table.csv'

carbons = ['Glucose', 'Citrate', 'Leucine']


def get_s_by_s(carbon, transfer=12):

    otu = open(mydir + '/data/otu_table.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    communities = []

    comm_rep_list = []

    for line in open(metadata_path):
        line = line.strip().split(',')
        if line[-3].strip() == carbon:
            if line[-1].strip() == str(transfer):
                communities.append(line[0].strip())

                comm_rep_list.append(line[-5]+ '_' + line[-4] )


    communities_idx = [otu_first_line.index(community) for community in communities]
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

    return s_by_s_np, species, comm_rep_list


def get_time_by_species_matrix(comm,rep):

    otu = open(mydir + '/data/otu_table.csv')
    otu_first_line = otu.readline()
    otu_first_line = otu_first_line.strip().split(',')

    communities = []

    #print(otu_first_line)

    for line in open(metadata_path):
        line = line.strip().split(',')
        #if line[-3].strip() == carbon:
            #if line[-1].strip() == str(transfer):
            #    communities.append(line[0].strip())

    #range(1, 13)




def plot_afd(n_bins=50):

    AFDs = []

    number_communities = []

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)
        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
        log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_np, axis=1))

        AFDs.extend(log_mean_rel_abundances)

        number_communities.append(len(log_mean_rel_abundances))

    AFDs = np.asarray(AFDs)
    rescaled_AFDs = (AFDs - np.mean(AFDs)) / np.std(AFDs)
    #print(rescaled_AFDs)
    #print(min(rescaled_AFDs), max(rescaled_AFDs))
    #print((max(rescaled_AFDs) - min(rescaled_AFDs)) / n_bins)

    ag,bg,cg = gamma.fit(rescaled_AFDs)

    x_range = np.linspace(min(rescaled_AFDs) , max(rescaled_AFDs) , 10000)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.hist(rescaled_AFDs, alpha=0.8, bins= 20, density=True)

    # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

    ax.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', lw=2)

    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Rescaled log relative\naverage abundance', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    fig_name = mydir + '/figs/AFD.pdf'
    fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







def plot_taylors(slope_null=2):

    fig = plt.figure(figsize = (4*len(carbons), 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)

        # rows are species, columns are sites
        # calculate relative abundance for each site

        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        for zeros_idx, zeros in enumerate(['yes', 'no']):

            mean_rel_abundances = []
            var_rel_abundances = []

            for sad in rel_s_by_s_np:

                sad_no_zeros = sad[sad>0]

                if len(sad_no_zeros) < 3:
                    continue

                if zeros == 'yes':

                    mean_rel_abundances.append(np.mean(sad))
                    var_rel_abundances.append(np.var(sad))

                else:

                    mean_rel_abundances.append(np.mean(sad_no_zeros))
                    var_rel_abundances.append(np.var(sad_no_zeros))


            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))


            ax_plot = plt.subplot2grid((2, 1*len(carbons)), (zeros_idx, carbon_idx), colspan=1)

            ax_plot.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.8)#, c='#87CEEB')

            x_log10_range =  np.linspace(min(np.log10(mean_rel_abundances)) , max(np.log10(mean_rel_abundances)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

            if zeros_idx ==0:

                ax_plot.set_title(carbon, fontsize=14, fontweight='bold' )

            ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
            ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")


            ax_plot.set_xscale('log', basex=10)
            ax_plot.set_yscale('log', basey=10)

            ax_plot.set_xlabel('Average relative\nabundance', fontsize=12)
            ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)

            ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )


            # run slope test
            #t, p = stats.ttest_ind(dnds_treatment[0], dnds_treatment[1], equal_var=False)
            t_value = (slope - (slope_null))/std_err
            p_value = stats.t.sf(np.abs(t_value), len(mean_rel_abundances)-2)

            sys.stdout.write("Slope = %g, t = %g, P= %g\n" % (slope, t_value, p_value))

            ax_plot.legend(loc="lower right", fontsize=8)



    fig.text(0.02, 0.7, "Zeros", va='center', fontweight='bold', rotation='vertical', fontsize=16)

    fig.text(0.02, 0.3, "No zeros", va='center', fontweight='bold',rotation='vertical', fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(mydir + "/figs/taylors_law.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def plot_N_Nmax():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)

        # rows are species, columns are sites
        # calculate relative abundance for each site
        #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        N_list = []
        Nmax_list = []

        for sad in s_by_s:

            N_list.append(sum(sad))

            Nmax_list.append(max(sad))


        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(N_list), np.log10(Nmax_list))


        ax_plot = plt.subplot2grid((1, 1*len(carbons)), (0, carbon_idx), colspan=1)

        ax_plot.scatter(N_list, Nmax_list, alpha=0.8)#, c='#87CEEB')
        ax_plot.set_title(carbon, fontsize=14, fontweight='bold' )

        x_log10_range =  np.linspace(min(np.log10(N_list)) , max(np.log10(N_list)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)

        ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=3, linestyle='-', zorder=2)

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        ax_plot.set_xlabel('Total abundance, ' + r'$N$', fontsize=12)
        ax_plot.set_ylabel('Dominance, ' + r'$N_{max}$', fontsize=12)

        ax_plot.text(0.2,0.9, r'$N_{{max}} \sim N^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )


        # run slope test
        #t, p = stats.ttest_ind(dnds_treatment[0], dnds_treatment[1], equal_var=False)
        #t_value = (slope - (slope_null))/std_err
        #p_value = stats.t.sf(np.abs(t_value), len(mean_rel_abundances)-2)

        #sys.stdout.write("Slope = %g, t = %g, P= %g\n" % (slope, t_value, p_value))


    fig.subplots_adjust(wspace=0.3)
    fig.savefig(mydir + "/figs/N_Nmax.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



class lognorm:

    def __init__(self, obs):
        self.obs = obs


    def ppoints(self, n):
        """ numpy analogue or `R`'s `ppoints` function
            see details at http://stat.ethz.ch/R-manual/R-patched/library/stats/html/ppoints.html
            :param n: array type or number
            Obtained from: http://stackoverflow.com/questions/20292216/imitating-ppoints-r-function-in-python
            on 5 April 2016
            """
        if n < 10:
            a = 3/8
        else:
            a = 1/2

        try:
            n = np.float(len(n))
        except TypeError:
            n = np.float(n)
        return (np.arange(n) + 1 - a)/(n + 1 - 2*a)



    def get_rad_pln(self, S, mu, sigma, lower_trunc = True):
        """Obtain the predicted RAD from a Poisson lognormal distribution"""
        abundance = list(np.empty([S]))
        rank = range(1, int(S) + 1)
        cdf_obs = [(rank[i]-0.5) / S for i in range(0, int(S))]
        j = 0
        cdf_cum = 0
        i = 1
        while j < S:
            cdf_cum += pln.pmf(i, mu, sigma, lower_trunc)
            while cdf_cum >= cdf_obs[j]:
                abundance[j] = i
                j += 1
                if j == S:
                    abundance.reverse()
                    return abundance
            i += 1


    def get_rad_from_obs(self):
        mu, sigma = pln_solver(self.obs)
        pred_rad = self.get_rad_pln(len(self.obs), mu, sigma)

        return (pred_rad, mu, sigma)



class TimeoutException(Exception):   # Custom exception class
    pass


def fit_sad():

    for carbon_idx, carbon in enumerate(carbons):

        sys.stdout.write("Fitting lognormal for %s communities...\n" % (carbon))

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)

        data = []

        obs_list = []
        pred_list = []

        N_communities = s_by_s.shape[1]
        for sad in s_by_s.T:
            sys.stdout.write("%d communities to go!\n" % (N_communities))
            N_communities -= 1

            sad = sad[sad>0]
            sad.sort()
            sad = sad[::-1]
            if len(sad) < 10:
                continue

            N = sum(sad)
            S = len(sad)
            Nmax = max(sad)

            lognorm_pred = lognorm(sad)
            sad_predicted, mu, sigma = lognorm_pred.get_rad_from_obs()
            pln_Nmax = max(sad_predicted)

            ll = pln_ll(sad, mu, sigma)
            r2 = obs_pred_rsquare(np.log10(sad), np.log10(sad_predicted))

            data.append([str(N),str(S),str(Nmax),str(pln_Nmax),str(ll),str(r2)])

            obs_list.extend(sad)
            pred_list.extend(sad_predicted)


        df_sad_out = open((mydir + '/data/pln_sad_counts_%s.csv') % (carbon) , 'w')
        df_sad_out.write(','.join(['Observed','Predicted']) + '\n')
        for i in range(len(obs_list)):
            df_sad_out.write(','.join([str(obs_list[i]), str(pred_list[i])]) + '\n')
        df_sad_out.close()


        df_out = open((mydir + '/data/pln_sad_%s.csv') % (carbon) , 'w')
        df_out.write(','.join(['N','S','Nmax', 'pln_Nmax', 'll','r2_mod']) + '\n')
        for i in range(len(data)):
            df_out.write(','.join(data[i]) + '\n')
        df_out.close()




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
    #print(sorted_plot_data)
    #color_range = np.sqrt(sorted_plot_data[:, 2])
    #print(color_range)
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


def plot_pln():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        obs = []
        pred = []

        fileeee = open(mydir+('/data/pln_sad_counts_%s.csv') % (carbon) )
        fileeee_first_line = fileeee.readline()
        for line in fileeee:
            line=line.strip().split(',')
            if line[0] == 'Observed':
                continue

            obs.append(int(line[0]))
            pred.append(int(line[1]))

        obs=np.asarray(obs)
        pred=np.asarray(pred)


        fileeee.close()

        # get mean r2
        fileeee_r2 = open(mydir+('/data/pln_sad_%s.csv') % (carbon))
        fileeee_r2_first_line = fileeee_r2.readline()
        r2_list = []
        for line in fileeee_r2:
            line=line.strip().split(',')
            r2_list.append(float(line[4]))

        r2_mean=np.mean(r2_list)
        fileeee_r2.close()

        ax_i = plt.subplot2grid((1, 1*len(carbons)), (0, carbon_idx), colspan=1)


        plot_color_by_pt_dens(obs, pred, 2, loglog=1,
                                plot_obj=ax_i)

        ax_i.plot([0.6, 2*max(obs)],[0.6, 2 * max(obs)], 'k-')
        ax_i.set_xlim(0.6, 2*max(obs))
        ax_i.set_ylim(0.6, 2*max(obs))

        ax_i.set_xlabel('Observed abundance', fontsize=14)
        ax_i.set_ylabel('Predicted Poisson-lognormal abundance', fontsize=12)

        ax_i.set_title(carbon, fontsize=14, fontweight='bold' )

        ax_i.text(0.2,0.9, r'$r_{m}^{2}=$' + str(round(r2_mean,3)), fontsize=11, color='k', ha='center', va='center', transform=ax_i.transAxes )


    fig.subplots_adjust(wspace=0.3)
    fig.savefig(mydir + "/figs/obs_pred_lognorm.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()


def examine_cv(taxonomic_level):
    taxonomic_level_dict = {'genus':-1, 'family':-2, 'order':-3, 'class':-4}
    esv_genus_map = {}
    taxonomy = open(mydir+'/data/taxonomy.csv')
    taxonomy_fitst_line = taxonomy.readline()
    for line in taxonomy:
        line = line.strip().split(',')
        if line[ taxonomic_level_dict[taxonomic_level] ] != 'NA':
            esv_genus_map[line[0]] = line[ taxonomic_level_dict[taxonomic_level] ]
    taxonomy.close()

    genus_cv_time_dict = {}

    comm_rep_list_all = []

    for transfer in range(1, 13):

        #cv_dict[transfer] = []

        s_by_s, species, comm_rep_list = get_s_by_s("Glucose", transfer=transfer)
        #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        # so we dont include the communities with no temporal samples
        if transfer == 1:
            comm_rep_list_all = comm_rep_list

        for afd_idx, afd in enumerate(s_by_s):

            esv = species[afd_idx]
            if esv not in esv_genus_map:
                continue

            genus = esv_genus_map[esv]

            if genus not in genus_cv_time_dict:
                genus_cv_time_dict[genus] = {}
            if transfer not in genus_cv_time_dict[genus]:
                genus_cv_time_dict[genus][transfer] = {}

            for comm_rep_idx, comm_rep in enumerate(comm_rep_list):
                # dont include extra ~70 communities at 12th trasnfer
                if comm_rep not in comm_rep_list_all:
                    continue
                if comm_rep not in genus_cv_time_dict[genus][transfer]:
                    genus_cv_time_dict[genus][transfer][comm_rep] = 0

                genus_cv_time_dict[genus][transfer][comm_rep] += afd[comm_rep_idx]

            #for abundance, comm_rep in zip(afd,comm_rep_list):
            #    genus_cv_time_dict[genus][transfer][comm_rep] += abundance

            #print(genus_cv_time_dict[genus][transfer])

        # go back through and divide by total reads

        N_total = s_by_s.sum(axis=0)

        for comm_rep_idx, comm_rep in enumerate(comm_rep_list):
            for genus in genus_cv_time_dict.keys():
                #print(genus, genus_cv_time_dict[genus])
                if transfer in genus_cv_time_dict[genus]:
                    if comm_rep in genus_cv_time_dict[genus][transfer]:
                        genus_cv_time_dict[genus][transfer][comm_rep] = genus_cv_time_dict[genus][transfer][comm_rep] / N_total[comm_rep_idx]

    #print(genus_cv_time_dict)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.axhline(1, lw=1.5, ls=':',color='k', zorder=1)

    time_cv_dict = {}

    for genus, genus_dict in genus_cv_time_dict.items():
        genus_transfers = []
        genus_cvs = []
        for transfer in range(1, 13):
            if transfer in genus_dict:
                relative_abundances = np.asarray(list(genus_dict[transfer].values()))
                relative_abundances = relative_abundances[relative_abundances>0]
                if len(relative_abundances) > 2:
                    genus_transfers.append(transfer)
                    cv = np.std(relative_abundances) / np.mean(relative_abundances)
                    genus_cvs.append( cv )

                    if transfer not in time_cv_dict:
                        time_cv_dict[transfer] = []

                    time_cv_dict[transfer].append(cv)

        if len(genus_transfers) > 2:

            ax.plot(genus_transfers, genus_cvs, c='#87CEEB', alpha=0.8)

    transfer_list = []
    mean_cv_list = []
    for transfer, cv_list in time_cv_dict.items():
        transfer_list.append(transfer)
        mean_cv_list.append(np.mean(cv_list))

    ax.plot(transfer_list, mean_cv_list, c='k', alpha=0.8)

    ax.set_title("ESVs grouped at %s level " % taxonomic_level, fontsize=14)

    ax.set_xlabel('Transfer', fontsize=12)
    ax.set_ylabel('Coefficient of variation\nof relative abundance', fontsize=12)

    fig_name = mydir + '/figs/cv_time_%s.pdf' % taxonomic_level
    fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_Nmax_vs_pln_Nmax():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        obs = []
        pred = []

        fileeee = open(mydir+('/data/pln_sad_%s.csv') % (carbon) )
        fileeee_first_line = fileeee.readline()
        for line in fileeee:
            line=line.strip().split(',')
            if line[0] == 'N':
                continue

            obs.append(int(line[2]))
            pred.append(int(line[3]))

        obs=np.asarray(obs)
        pred=np.asarray(pred)

        fileeee.close()

        ax_i = plt.subplot2grid((1, 1*len(carbons)), (0, carbon_idx), colspan=1)

        #plot_color_by_pt_dens(obs, pred, 1, loglog=1,
        #                        plot_obj=ax_i)

        ax_i.scatter(obs, pred, alpha=0.8)#, c='#87CEEB')

        ax_i.plot([0.6*min(obs), 2*max(obs)],[0.6*min(obs), 2 * max(obs)], 'k-')
        ax_i.set_xlim(0.6*min(obs), 2*max(obs))
        ax_i.set_ylim(0.6*min(obs), 2*max(obs))

        ax_i.set_xscale('log', basex=10)
        ax_i.set_yscale('log', basey=10)

        ax_i.set_xlabel('Observed ' + r'$N_{max}$', fontsize=14)
        ax_i.set_ylabel('Predicted Poisson-lognormal ' + r'$N_{max}$', fontsize=12)

        ax_i.set_title(carbon, fontsize=14, fontweight='bold' )

        RMSE = (sum( ((np.log10(pred) - np.log10(obs)) ** 2) ) / len(obs) ) ** 0.5
        NRMSE = RMSE / ( max(np.log10(obs)) - min(np.log10(obs)) )

        r2 = obs_pred_rsquare(np.log10(obs), np.log10(pred))

        ax_i.text(0.2,0.9, r'$r_{m}^{2}=$' + str(round(r2,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_i.transAxes )
        ax_i.text(0.25,0.8, r'$\mathrm{NRMSD}=$' + str(round(NRMSE,3)), fontsize=10, color='k', ha='center', va='center', transform=ax_i.transAxes )


    fig.subplots_adjust(wspace=0.3)
    fig.savefig(mydir + "/figs/Nmax_vs_pln_Nmax.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def plot_abundance_occupy_dist():

    fig = plt.figure(figsize = (4*len(carbons), 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)
        # rows are species, columns are sites
        # calculate relative abundance for each site

        s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

        species_occurance_dict = {}
        proportion_species_occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        mean_rel_abundances = []

        for afd in rel_s_by_s_np:

            afd_no_zeros = afd[afd>0]

            print(afd_no_zeros)

            mean_rel_abundances.append(np.mean(afd_no_zeros))


        ax_plot = plt.subplot2grid((2, 1*len(carbons)), (0, carbon_idx), colspan=1)

        ax_plot.scatter(mean_rel_abundances, proportion_species_occupancies, alpha=0.8)#, c='#87CEEB')


        ax_plot.set_title(carbon, fontsize=14, fontweight='bold' )

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        print(len(comm_rep_list))

        ax_plot.set_ylim([1/(len(comm_rep_list)+10) , 1.1])

        ax_plot.set_xlabel('Average relative abundance\nexcluding zeros', fontsize=12)
        ax_plot.set_ylabel('Occupancy', fontsize=10)


    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(mydir + "/figs/AOD.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_lognormal():

    MADs = []

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = get_s_by_s(carbon)
        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
        log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_np, axis=1))

        MADs.extend(log_mean_rel_abundances)

    MADs = np.asarray(MADs)
    rescaled_MADs = (MADs - np.mean(MADs)) / np.std(MADs)

    shape,loc,scale = stats.lognorm.fit(rescaled_MADs)

    x_range = np.linspace(min(rescaled_MADs) , max(rescaled_MADs) , 10000)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.hist(rescaled_MADs, alpha=0.8, bins= 20, density=True)

    # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

    ax.plot(x_range, stats.lognorm.pdf(x_range, shape,loc,scale), 'k', label='Lognormal fit', lw=2)

    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Rescaled log average\nrelative abundance', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    ax.legend(loc="upper left", fontsize=8)


    fig.savefig(mydir + '/figs/MAD.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







plot_lognormal()


#plot_abundance_occupy_dist()

#plot_Nmax_vs_pln_Nmax()

#examine_cv('family')
#examine_cv('order')
#examine_cv('genus')

#plot_pln()

#fit_sad()

#plot_afd()
#plot_taylors()

#plot_N_Nmax()
