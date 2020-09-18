from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

#from macroecotools import obs_pred_rsquare
import utils

carbons = utils.carbons

slope_null=2
alpha=0.05

titles = ['No migration, low inoculum', 'No migration, high inoculum', 'Global migration, low inoculum', 'Parent migration, low inoculum' ]
migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]
plot_idxs = [(0,0), (0,1), (1,0), (1,1)]

family_colors = {'Alcaligenaceae':'darkorange', 'Comamonadaceae': 'darkred',
                'Enterobacteriaceae':'dodgerblue', 'Enterococcaceae':'limegreen',
                'Lachnospiraceae':'deepskyblue', 'Pseudomonadaceae':'darkviolet'}



def plot_afd(n_bins=50):

    fig = plt.figure(figsize = (4*len(utils.carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15,  wspace=0.25)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)
        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
        #log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_np, axis=1))

        AFDs = []
        number_communities = []

        for rel_sad in rel_s_by_s_np:
            rel_sad = rel_sad[rel_sad>0]

            log_mean_rel_abundances = np.log(rel_sad)

            AFDs.extend(log_mean_rel_abundances)

        number_communities.append(len(log_mean_rel_abundances))

        AFDs = np.asarray(AFDs)
        #AFDs = AFDs.flatten()

        rescaled_AFDs = (AFDs - np.mean(AFDs)) / np.std(AFDs)

        ag,bg,cg = gamma.fit(rescaled_AFDs)

        x_range = np.linspace(min(rescaled_AFDs) , max(rescaled_AFDs) , 10000)


        ax_i = plt.subplot2grid((1, 1*len(carbons)), (0, carbon_idx), colspan=1)


        ax_i.hist(rescaled_AFDs, alpha=0.8, bins= 20, density=True)

        # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

        ax_i.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', lw=2)

        ax_i.set_yscale('log', basey=10)

        ax_i.set_xlabel('Rescaled log\nrelative abundance', fontsize=12)
        ax_i.set_ylabel('Probability density', fontsize=12)

        ax_i.set_title(carbon, fontsize=14, fontweight='bold' )

    #wspace=0.3, hspace=0.3
    fig_name = utils.directory + '/figs/AFD.pdf'
    fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







def plot_taylors(slope_null=2):

    fig = plt.figure(figsize = (4*len(utils.carbons), 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(utils.carbons):

        s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)

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
    fig.savefig(utils.directory + "/figs/taylors_law.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def plot_N_Nmax():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)

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
    fig.savefig(utils.directory + "/figs/N_Nmax.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()






def plot_pln():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        obs = []
        pred = []

        fileeee = open(utils.directory+('/data/pln_sad_counts_%s.csv') % (carbon) )
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
        fileeee_r2 = open(utils.directory+('/data/pln_sad_%s.csv') % (carbon))
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
    fig.savefig(utils.directory + "/figs/obs_pred_lognorm.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def examine_cv(taxonomic_level):
    taxonomic_level_dict = {'genus':-1, 'family':-2, 'order':-3, 'class':-4}
    esv_genus_map = {}
    taxonomy = open(utils.directory+'/data/taxonomy.csv')
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

        s_by_s, species, comm_rep_list = utils.get_s_by_s("Glucose", transfer=transfer)
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

    fig_name = utils.directory + '/figs/cv_time_%s.pdf' % taxonomic_level
    fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_Nmax_vs_pln_Nmax():

    fig = plt.figure(figsize = (4*len(carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(carbons):

        obs = []
        pred = []

        fileeee = open(utils.directory+('/data/pln_sad_%s.csv') % (carbon) )
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
    fig.savefig(utils.directory + "/figs/Nmax_vs_pln_Nmax.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def plot_abundance_occupy_dist():

    fig = plt.figure(figsize = (4*len(utils.carbons), 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_idx, carbon in enumerate(utils.carbons):

        s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)
        # rows are species, columns are sites
        # calculate relative abundance for each site

        s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

        species_occurance_dict = {}
        proportion_species_occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        mean_rel_abundances = []

        for afd in rel_s_by_s_np:

            afd_no_zeros = afd[afd>0]

            #print(afd_no_zeros)

            mean_rel_abundances.append(np.mean(afd_no_zeros))


        ax_plot = plt.subplot2grid((2, 1*len(utils.carbons)), (0, carbon_idx), colspan=1)

        ax_plot.scatter(mean_rel_abundances, proportion_species_occupancies, alpha=0.8)#, c='#87CEEB')


        ax_plot.set_title(carbon, fontsize=14, fontweight='bold' )

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)

        print(len(comm_rep_list))

        ax_plot.set_ylim([1/(len(comm_rep_list)+10) , 1.1])

        ax_plot.set_xlabel('Average relative abundance\nexcluding zeros', fontsize=12)
        ax_plot.set_ylabel('Occupancy', fontsize=10)


    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/AOD.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_mad():

    #MADs = []

    fig = plt.figure(figsize = (4*len(utils.carbons), 4)) #
    fig.subplots_adjust(bottom= 0.15,  wspace=0.25)

    for carbon_idx, carbon in enumerate(carbons):

        s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)
        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
        #log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_np, axis=1))
        MADs = np.log(np.mean(rel_s_by_s_np, axis=1))

        #MADs.extend(log_mean_rel_abundances)

        MADs = np.asarray(MADs)
        rescaled_MADs = (MADs - np.mean(MADs)) / np.std(MADs)

        shape,loc,scale = stats.lognorm.fit(rescaled_MADs)

        x_range = np.linspace(min(rescaled_MADs) , max(rescaled_MADs) , 10000)


        ax_i = plt.subplot2grid((1, 1*len(carbons)), (0, carbon_idx), colspan=1)

        ax_i.hist(rescaled_MADs, alpha=0.8, bins= 15, density=True)

        # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

        ax_i.plot(x_range, stats.lognorm.pdf(x_range, shape,loc,scale), 'k', label='Lognormal fit', lw=2)

        ax_i.set_yscale('log', basey=10)

        ax_i.set_xlabel('Rescaled log average\nrelative abundance', fontsize=12)
        ax_i.set_ylabel('Probability density', fontsize=12)

        ax_i.legend(loc="upper left", fontsize=8)

        ax_i.set_title(carbon, fontsize=14, fontweight='bold' )



    fig.savefig(utils.directory + '/figs/MAD.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()








def plot_predicted_occupancies():

    fig = plt.figure(figsize = (4*len(utils.carbons), 4*len(utils.carbons))) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_row_idx, carbon_row in enumerate(utils.carbons):

        for carbon_column_idx, carbon_column in enumerate(utils.carbons):

            if carbon_row_idx > carbon_column_idx:
                continue

            if carbon_row != carbon_column:
                carbon_title = carbon_row + ' + ' + carbon_column
            else:
                carbon_title = carbon_row

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s(list(set([carbon_row,carbon_column])))

            occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)

            ax_plot = plt.subplot2grid((1*len(utils.carbons), 1*len(utils.carbons)), (carbon_row_idx, carbon_column_idx), colspan=1)
            ax_plot.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
            ax_plot.scatter(occupancies, predicted_occupancies, alpha=0.8,zorder=2)#, c='#87CEEB')

            ax_plot.set_xscale('log', basex=10)
            ax_plot.set_yscale('log', basey=10)
            ax_plot.set_xlabel('Observed occupancy', fontsize=12)
            ax_plot.set_ylabel('Predicted occupancy', fontsize=10)

            ax_plot.set_title(carbon_title, fontsize=14, fontweight='bold' )

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/predicted_occupancies.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_taylors_time_series(slope_null=2):

    s_by_s_all_transfers_dict = {}

    all_samples = []
    for transfer in range(1, 13):

        s_by_s, species, comm_rep_list = utils.get_s_by_s("Glucose", transfer=transfer)
        #rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        # so we dont include the communities with no temporal samples
        if transfer == 1:
            comm_rep_list_all = comm_rep_list

            #for comm_rep in  comm_rep_list:
            #    s_by_s_all_transfers_dict[comm_rep] = {}

        for sad_idx, sad in enumerate(s_by_s.T):

            if comm_rep_list[sad_idx] not in comm_rep_list_all:
                continue

            comm_rep = comm_rep_list[sad_idx] + '_' + str(transfer)

            all_samples.append(comm_rep)

            s_by_s_all_transfers_dict[comm_rep] = {}

            for n_i_idx, n_i in enumerate(sad):
                species_i = species[n_i_idx]

                s_by_s_all_transfers_dict[comm_rep][species_i] = n_i

    s_by_s_all_transfers_df = pd.DataFrame(s_by_s_all_transfers_dict)
    s_by_s_all_transfers_df = s_by_s_all_transfers_df.fillna(0)

    s_by_s_all_transfers_df = s_by_s_all_transfers_df[(s_by_s_all_transfers_df.T != 0).any()]
    s_by_s_all_transfers = s_by_s_all_transfers_df.values

    species_all_transfers = s_by_s_all_transfers_df.index.values
    comm_rep_list_all_transfers = s_by_s_all_transfers_df.columns.values


    occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s_all_transfers)

    fig, ax = plt.subplots(figsize=(4,4))

    fig.subplots_adjust(bottom= 0.15)

    ax.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
    ax.scatter(occupancies, predicted_occupancies, alpha=0.8,zorder=2)#, c='#87CEEB')

    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)
    ax.set_xlabel('Observed occupancy', fontsize=12)
    ax.set_ylabel('Predicted occupancy', fontsize=10)

    ax.set_title('Merged Glucose time-series', fontsize=14, fontweight='bold' )

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/predicted_occupancies_merged_transfers.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()


    rel_s_by_s_all_transfers = (s_by_s_all_transfers/s_by_s_all_transfers.sum(axis=0))


    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for zeros_idx, zeros in enumerate(['yes', 'no']):

        mean_rel_abundances = []
        var_rel_abundances = []

        for sad in rel_s_by_s_all_transfers:

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

        ax_plot = plt.subplot2grid((2, 2), (zeros_idx, 0), colspan=1)

        ax_plot.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.8)#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(mean_rel_abundances)) , max(np.log10(mean_rel_abundances)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

        #if zeros_idx ==0:
        #    ax_plot.set_title('Merged Glucose time-series', fontsize=14, fontweight='bold' )

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


    ax_sad = plt.subplot2grid((2, 2), (0, 1), colspan=1)

    log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_all_transfers, axis=1))
    print(len(log_mean_rel_abundances), len(species_all_transfers))

    rescaled_MADs = (log_mean_rel_abundances - np.mean(log_mean_rel_abundances)) / np.std(log_mean_rel_abundances)

    shape,loc,scale = stats.lognorm.fit(rescaled_MADs)

    x_range = np.linspace(min(rescaled_MADs) , max(rescaled_MADs) , 10000)

    ax_sad.hist(rescaled_MADs, alpha=0.8, bins= 20, density=True)

    # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

    ax_sad.plot(x_range, stats.lognorm.pdf(x_range, shape,loc,scale), 'k', label='Lognormal fit', lw=2)

    ax_sad.set_yscale('log', basey=10)

    ax_sad.set_xlabel('Rescaled log average\nrelative abundance', fontsize=12)
    ax_sad.set_ylabel('Probability density', fontsize=12)

    ax_sad.legend(loc="upper right", fontsize=8)



    ax_afd = plt.subplot2grid((2, 2), (1, 1), colspan=1)


    #for carbon_idx, carbon in enumerate(carbons):

    #    s_by_s, species, comm_rep_list = get_s_by_s(carbon)
    #    rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))
    #log_mean_rel_abundances = np.log(np.mean(rel_s_by_s_all_transfers, axis=0))


    AFDs = rel_s_by_s_all_transfers.flatten()
    AFDs = AFDs[AFDs>0]
    AFDs = np.log(AFDs)
    rescaled_AFDs = (AFDs - np.mean(AFDs)) / np.std(AFDs)


    ag,bg,cg = gamma.fit(rescaled_AFDs)

    x_range = np.linspace(min(rescaled_AFDs) , max(rescaled_AFDs) , 10000)


    ax_afd.hist(rescaled_AFDs, alpha=0.8, bins= 20, density=True)

    # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

    ax_afd.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', label='Gamma fit', lw=2)

    ax_afd.set_yscale('log', basey=10)

    ax_afd.set_xlabel('Rescaled log\nrelative abundance', fontsize=12)
    ax_afd.set_ylabel('Probability density', fontsize=12)

    ax_afd.legend(loc="upper right", fontsize=8)


    fig.text(0, 0.7, "Zeros", va='center', fontweight='bold', rotation='vertical', fontsize=16)
    fig.text(0, 0.3, "No zeros", va='center', fontweight='bold',rotation='vertical', fontsize=16)

    fig.text(0.5, 1, "Merged glucose time-series", va='center',ha='center', fontweight='bold',fontsize=16)



    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/taylors_law_merged_transfers.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def reformat_site_by_species():

    carbon = 'Glucose'

    s_by_s, species, comm_rep_list = utils.get_s_by_s(carbon)
    # rows are species, columns are sites
    # calculate relative abundance for each site

    #s_by_s_presence_absence = np.where(s_by_s > 0, 1, 0)

    df_sad_out = open(utils.directory + '/data/reformat_s_by_s.csv' , 'w')
    df_sad_out.write(','.join(['condition','sample','sp','count','totreads']) + '\n')


    for sad_idx, sad in enumerate(np.transpose(s_by_s)):

        comm_rep = comm_rep_list[sad_idx]

        N = sum(sad)

        for sad_species_idx, sad_species in enumerate(sad):

            df_sad_out.write(','.join([carbon,comm_rep,species[sad_species_idx], str(sad_species), str(N)]) + '\n')


    df_sad_out.close()



def plot_taylors_law_migration(zeros=False, transfer=18):

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)


    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])

        mean_rel_abundances = []
        var_rel_abundances = []

        for afd in s_by_s:

            afd_no_zeros = afd[afd>0]

            if len(afd_no_zeros) < 3:
                continue

            if zeros == True:

                mean_rel_abundances.append(np.mean(afd))
                var_rel_abundances.append(np.var(afd))

            else:

                mean_rel_abundances.append(np.mean(afd_no_zeros))
                var_rel_abundances.append(np.var(afd_no_zeros))


        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))

        ax_plot = plt.subplot2grid((2, 2), plot_idxs[migration_innoculum_idx])

        ax_plot.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.8)#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(mean_rel_abundances)) , max(np.log10(mean_rel_abundances)) , 10000)
        y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
        y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)


        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )

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

        sys.stdout.write("Slope = %g, t = %g, P = %g\n" % (slope, t_value, p_value))

        ax_plot.legend(loc="lower right", fontsize=8)



    fig.text(0.5, 0.95, "Transfer %s"%str(transfer), va='center', ha='center', fontweight='bold',fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/taylors_law_migration.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def plot_mad_migration(zeros=False, transfer=18):

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)


    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        rel_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])
        mean_rel_abundances = np.mean(rel_s_by_s, axis=0)

        #mean_rel_abundances = mean_rel_abundances[mean_rel_abundances>0]
        log_mean_rel_abundances = np.log(mean_rel_abundances)

        rescaled_MADs = (log_mean_rel_abundances - np.mean(log_mean_rel_abundances)) / np.std(log_mean_rel_abundances)

        shape,loc,scale = stats.lognorm.fit(rescaled_MADs)

        x_range = np.linspace(min(rescaled_MADs) , max(rescaled_MADs) , 10000)

        ax_plot = plt.subplot2grid((2, 2), plot_idxs[migration_innoculum_idx])

        ax_plot.hist(rescaled_MADs, alpha=0.8, bins= 20, density=True)

        # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

        ax_plot.plot(x_range, stats.lognorm.pdf(x_range, shape,loc,scale), 'k', label='Lognormal fit', lw=2)

        ax_plot.set_yscale('log', basey=10)

        ax_plot.set_xlabel('Rescaled log average\nrelative abundance', fontsize=12)
        ax_plot.set_ylabel('Probability density', fontsize=12)

        ax_plot.legend(loc="upper right", fontsize=8)

        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )


    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/mad_migration.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def plot_afd_migration(zeros=False, transfer=18):

    fig = plt.figure(figsize = (8, 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        rel_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=migration_innoculum[0],inocula=migration_innoculum[1])

        AFDs = rel_s_by_s.flatten()
        AFDs = AFDs[AFDs>0]
        AFDs = np.log(AFDs)
        rescaled_AFDs = (AFDs - np.mean(AFDs)) / np.std(AFDs)


        ag,bg,cg = gamma.fit(rescaled_AFDs)

        x_range = np.linspace(min(rescaled_AFDs) , max(rescaled_AFDs) , 10000)

        ax_plot = plt.subplot2grid((2, 2), plot_idxs[migration_innoculum_idx])


        ax_plot.hist(rescaled_AFDs, alpha=0.8, bins= 20, density=True)

        # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

        ax_plot.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', label='Gamma fit', lw=2)

        ax_plot.set_yscale('log', basey=10)

        ax_plot.set_xlabel('Rescaled log\nrelative abundance', fontsize=12)
        ax_plot.set_ylabel('Probability density', fontsize=12)

        ax_plot.legend(loc="upper right", fontsize=8)

        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )


    fig.text(0.5, 0.95, "Transfer %s"%str(transfer), va='center', ha='center', fontweight='bold',fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/afd_migration.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_taylors_migration_time_series():

    fig = plt.figure(figsize = (4*len(migration_innocula), 8)) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula[:-1]):

        s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_temporal_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])

        # rows are species, columns are sites
        # calculate relative abundance for each site

        for zeros_idx, zeros in enumerate(['yes', 'no']):

            mean_rel_abundances = []
            var_rel_abundances = []

            for afd in s_by_s:

                afd_no_zeros = afd[afd>0]

                if len(afd_no_zeros) < 3:
                    continue

                if zeros == 'yes':

                    mean_rel_abundances.append(np.mean(afd))
                    var_rel_abundances.append(np.var(afd))

                else:

                    mean_rel_abundances.append(np.mean(afd_no_zeros))
                    var_rel_abundances.append(np.var(afd_no_zeros))

            #print(migration_innoculum, mean_rel_abundances)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_rel_abundances), np.log10(var_rel_abundances))

            ax_plot = plt.subplot2grid((2, 1*len(migration_innocula)), (zeros_idx, migration_innoculum_idx))

            ax_plot.scatter(mean_rel_abundances, var_rel_abundances, alpha=0.8)#, c='#87CEEB')

            x_log10_range =  np.linspace(min(np.log10(mean_rel_abundances)) , max(np.log10(mean_rel_abundances)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

            if zeros_idx ==0:

                ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )

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

    fig.text(0.5, 0.96, "Merged time series", va='center', ha='center', fontweight='bold', fontsize=18)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/taylors_law_migration_merged_transfers.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def plot_afd_migration_time_series():

    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula[:-1]):

        rel_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_temporal_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])

        AFDs = rel_s_by_s.flatten()
        AFDs = AFDs[AFDs>0]
        AFDs = np.log(AFDs)
        rescaled_AFDs = (AFDs - np.mean(AFDs)) / np.std(AFDs)

        ag,bg,cg = gamma.fit(rescaled_AFDs)

        x_range = np.linspace(min(rescaled_AFDs) , max(rescaled_AFDs) , 10000)


        ax_plot = plt.subplot2grid((1, 3), (0,migration_innoculum_idx))

        ax_plot.hist(rescaled_AFDs, alpha=0.8, bins= 20, density=True)

        # weights=np.zeros_like(rescaled_AFDs) + 1. / len(rescaled_AFDs)

        ax_plot.plot(x_range, gamma.pdf(x_range, ag, bg,cg), 'k', label='Gamma fit', lw=2)

        ax_plot.set_yscale('log', basey=10)

        ax_plot.set_xlabel('Rescaled log\nrelative abundance', fontsize=12)
        ax_plot.set_ylabel('Probability density', fontsize=12)

        ax_plot.legend(loc="upper right", fontsize=8)

        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )


    fig.text(0.5, 0.96, "Merged time series", va='center', ha='center', fontweight='bold',fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/afd_migration_transfers_merged.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()






def plot_mad_migration_time_series(zeros=False, transfer=18):

    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.15)


    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        rel_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_temporal_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])
        mean_rel_abundances = np.mean(rel_s_by_s, axis=1)

        #mean_rel_abundances = mean_rel_abundances[mean_rel_abundances>0]
        log_mean_rel_abundances = np.log(mean_rel_abundances)

        rescaled_MADs = (log_mean_rel_abundances - np.mean(log_mean_rel_abundances)) / np.std(log_mean_rel_abundances)
        print(len(rescaled_MADs), len(species))

        shape,loc,scale = stats.lognorm.fit(rescaled_MADs)

        x_range = np.linspace(min(rescaled_MADs) , max(rescaled_MADs) , 10000)


        ax_plot = plt.subplot2grid((1, 3), (0,migration_innoculum_idx))

        ax_plot.hist(rescaled_MADs, alpha=0.8, bins= 15, density=True)

        ax_plot.plot(x_range, stats.lognorm.pdf(x_range, shape,loc,scale), 'k', label='Lognormal fit', lw=2)

        ax_plot.set_yscale('log', basey=10)

        ax_plot.set_xlabel('Rescaled log average\nrelative abundance', fontsize=12)
        ax_plot.set_ylabel('Probability density', fontsize=12)

        ax_plot.legend(loc="upper right", fontsize=8)

        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=12, fontweight='bold' )


    fig.text(0.5, 0.97, "Merged time series", va='center', ha='center', fontweight='bold',fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(utils.directory + "/figs/mad_migration_merged_transfers.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def plot_predicted_occupancies_migration():

    fig = plt.figure(figsize = (12, 12)) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_idx, migration_innoculum in enumerate(migration_innocula):

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration(migration_inocula=[migration_innoculum])

        occupancies, predicted_occupancies = utils.predict_occupancy(s_by_s)

        ax_plot = plt.subplot2grid((2, 2),plot_idxs[migration_innoculum_idx], colspan=1)
        ax_plot.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
        ax_plot.scatter(occupancies, predicted_occupancies, alpha=0.8,zorder=2)#, c='#87CEEB')

        ax_plot.set_xscale('log', basex=10)
        ax_plot.set_yscale('log', basey=10)
        ax_plot.set_xlabel('Observed occupancy', fontsize=14)
        ax_plot.set_ylabel('Predicted occupancy', fontsize=14)

        ax_plot.set_title(titles[migration_innoculum_idx], fontsize=16, fontweight='bold' )


    fig.text(0.5, 0.97, "Transfer 18", va='center', ha='center', fontweight='bold',fontsize=16)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/predicted_occupancies_migration.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





def taylors_law_attractor(zeros=False, transfer=18, migration='No_migration', inocula=4):

    attractor_dict = {}

    attractor_file = open(utils.directory+'/data/attractor_status.csv')
    attractor_file_fitst_line = attractor_file.readline()
    for line in attractor_file:
        line = line.strip().split(',')
        if (line[0] == migration) and (int(line[1]) == inocula):

            if line[-1] not in attractor_dict:
                attractor_dict[line[-1]] = []

            attractor_dict[line[-1]].append(str(line[-2]))
    attractor_file.close()

    rel_s_by_s, species, comm_rep_list = utils.get_relative_s_by_s_migration(transfer=transfer,migration=migration,inocula=inocula)
    # assumes that
    # rows = species
    # columns = sites

    means, variances = utils.get_species_means_and_variances(rel_s_by_s, zeros=zeros)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))


    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.15)

    ax_0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)

    ax_0.scatter(means, variances, c='k', alpha=0.8)#, c='#87CEEB')

    x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
    y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
    y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

    ax_0.set_title("All replicate populations", fontsize=14, fontweight='bold' )

    ax_0.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
    ax_0.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

    ax_0.set_xscale('log', basex=10)
    ax_0.set_yscale('log', basey=10)

    ax_0.set_xlabel('Average relative\nabundance', fontsize=12)
    ax_0.set_ylabel('Variance of relative abundance', fontsize=10)

    ax_0.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_0.transAxes  )


    t_value = (slope - (slope_null))/std_err
    p_value = stats.t.sf(np.abs(t_value), len(means)-2)

    sys.stdout.write("All replicate populations\n")
    sys.stdout.write("Slope = %g, t = %g, P = %g\n" % (slope, t_value, p_value))


    ax_count = 1

    for attractor in attractor_dict.keys():

        attractor_idxs = [attractor_dict[attractor].index(comm_rep) for comm_rep in comm_rep_list if comm_rep in attractor_dict[attractor] ]

        rel_s_by_s_attractor = rel_s_by_s[:, attractor_idxs]
        rel_s_by_s_attractor = rel_s_by_s_attractor[~np.all(rel_s_by_s_attractor == 0, axis=1)]

        means_attractor, variances_attractor = utils.get_species_means_and_variances(rel_s_by_s_attractor, zeros=zeros)

        slope_attractor, intercept_attractor, r_value_attractor, p_value_attractor, std_err_attractor = stats.linregress(np.log10(means_attractor), np.log10(variances_attractor))


        ax_i = plt.subplot2grid((1, 3), (0, ax_count), colspan=1)

        ax_i.scatter(means_attractor, variances_attractor, alpha=0.8, c=family_colors[attractor])#, c='#87CEEB')

        x_log10_range =  np.linspace(min(np.log10(means_attractor)) , max(np.log10(means_attractor)) , 10000)
        y_log10_fit_range = 10 ** (slope_attractor*x_log10_range + intercept_attractor)
        y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept_attractor)

        ax_i.set_title(attractor, fontsize=14, fontweight='bold' )

        ax_i.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
        ax_i.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

        ax_i.set_xscale('log', basex=10)
        ax_i.set_yscale('log', basey=10)

        ax_i.set_xlabel('Average relative\nabundance', fontsize=12)
        ax_i.set_ylabel('Variance of relative abundance', fontsize=10)

        ax_i.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_attractor, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_i.transAxes  )

        ax_i.legend(loc="lower right", fontsize=8)

        # run slope test
        #t, p = stats.ttest_ind(dnds_treatment[0], dnds_treatment[1], equal_var=False)
        t_value = (slope_attractor - (slope_null))/std_err_attractor
        p_value = stats.t.sf(np.abs(t_value), len(means_attractor)-2)

        sys.stdout.write("Populations with attractor %s\n" % (attractor))
        sys.stdout.write("Slope = %g, t = %g, P = %g\n" % (slope, t_value, p_value))


        # slope of attractor vs merged slope test

        t_attractor = (slope_attractor - slope) / np.sqrt(std_err_attractor**2 + std_err**2)
        p_value_attractor = stats.t.sf(np.abs(t_attractor), len(means_attractor)+len(means)-4)

        sys.stdout.write("All attractors vs attractor %s\n" % (attractor))
        sys.stdout.write("t = %g, P = %g\n" % (t_attractor, p_value_attractor))



        ax_count+=1


    #wspace=0.3, hspace=0.3
    fig_name = utils.directory + '/figs/taylors_law_attractor.pdf'
    fig.text(0.5, 1, "No migration, low inoculum, transfer 18", ha='center', fontweight='bold', fontsize=16)
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_taylors_law_merged_treatments(zeros=False):

    fig = plt.figure(figsize = (4*len(utils.carbons), 4*len(utils.carbons))) #
    fig.subplots_adjust(bottom= 0.15)

    for carbon_row_idx, carbon_row in enumerate(utils.carbons):

        for carbon_column_idx, carbon_column in enumerate(utils.carbons):

            if carbon_row_idx > carbon_column_idx:
                continue

            if carbon_row != carbon_column:
                carbon_title = carbon_row + ' + ' + carbon_column
            else:
                carbon_title = carbon_row

            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s(list(set([carbon_row,carbon_column])))
            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            means, variances = utils.get_species_means_and_variances(rel_s_by_s, zeros=zeros)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

            ax_plot = plt.subplot2grid((1*len(utils.carbons), 1*len(utils.carbons)), (carbon_row_idx, carbon_column_idx), colspan=1)


            ax_plot.scatter(means, variances, alpha=0.8)#, c='#87CEEB')

            x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

            ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
            ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

            ax_plot.set_xscale('log', basex=10)
            ax_plot.set_yscale('log', basey=10)

            ax_plot.set_xlabel('Average relative\nabundance', fontsize=12)
            ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)

            ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )

            ax_plot.set_xlabel('Average relative\nabundance', fontsize=12)
            ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)

            ax_plot.set_title(carbon_title, fontsize=14, fontweight='bold' )

            ax_plot.legend(loc="lower right", fontsize=8)


    fig.subplots_adjust(wspace=0.3, hspace=0.45)
    fig.savefig(utils.directory + "/figs/taylors_law_merged_treatments.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()




def plot_taylors_law_migration_merged_treatments(zeros=False):

    fig = plt.figure(figsize = (4*len(migration_innocula), 4*len(migration_innocula))) #
    fig.subplots_adjust(bottom= 0.15)

    for migration_innoculum_row_idx, migration_innoculum_row in enumerate(migration_innocula):

        for migration_innoculum_column_idx, migration_innoculum_column in enumerate(migration_innocula):

            if migration_innoculum_row_idx > migration_innoculum_column_idx:
                continue

            if migration_innoculum_row != migration_innoculum_column:
                title = titles[migration_innoculum_row_idx] + ' +\n' + titles[migration_innoculum_column_idx]
            else:
                title = titles[migration_innoculum_row_idx]


            s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration(migration_innocula=list(set([migration_innoculum_row,migration_innoculum_column])))

            rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

            means, variances = utils.get_species_means_and_variances(rel_s_by_s, zeros=zeros)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

            ax_plot = plt.subplot2grid((1*len(migration_innocula), 1*len(migration_innocula)), (migration_innoculum_row_idx, migration_innoculum_column_idx), colspan=1)

            ax_plot.scatter(means, variances, alpha=0.8)#, c='#87CEEB')

            x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
            y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
            y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

            ax_plot.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
            ax_plot.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

            ax_plot.set_xscale('log', basex=10)
            ax_plot.set_yscale('log', basey=10)

            ax_plot.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_plot.transAxes  )

            ax_plot.set_xlabel('Average relative\nabundance', fontsize=12)
            ax_plot.set_ylabel('Variance of relative abundance', fontsize=10)

            ax_plot.set_title(title, fontsize=12, fontweight='bold' )

            ax_plot.legend(loc="lower right", fontsize=8)



    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(utils.directory + "/figs/taylors_law_migration_merged_treatments.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()



def getPairStats(x, y):

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




def taylors_law_time_series():

    from matplotlib import cm

    color_range =  np.linspace(0.0, 1.0, 12)

    #color_range_transfers = [x-1]

    rgb = cm.get_cmap('Blues')( color_range )


    means = []
    variances = []
    colors = []

    intercepts = []
    slopes = []

    slops_CIs = []

    for transfer in range(1, 13):

        s_by_s, species, comm_rep_list = utils.get_s_by_s("Glucose", transfer=transfer)
        #print(transfer, len(comm_rep_list))
        rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

        # so we dont include the communities with no temporal samples
        means_transfer = []
        variances_transfer = []
        if transfer == 1:
            comm_rep_list_all = comm_rep_list

        for afd_idx, afd in enumerate(rel_s_by_s_np):

            afd = afd[afd>0]


            if len(afd) < 4:
                continue


            means_transfer.append(np.mean(afd))
            variances_transfer.append(np.var(afd))


            means.append(np.mean(afd))
            variances.append(np.var(afd))

            colors.append(color_range[transfer-1])


        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means_transfer), np.log10(variances_transfer))

        s_xx, s_yy, s_xy = getPairStats(np.log10(means_transfer), np.log10(variances_transfer))

        t = stats.t.ppf(1-(alpha/2), len(means_transfer)-2)

        #maximim likelihood estimator
        sigma_hat = np.sqrt((1/len(means_transfer))*(s_yy-slope*s_xy))

        interval_val = t*sigma_hat*np.sqrt(len(means_transfer)/((len(means_transfer)-2)*s_xx))

        slopes.append(slope)
        intercepts.append(intercept)

        slops_CIs.append(interval_val)


    means = np.asarray(means)
    variances = np.asarray(variances)
    colors = np.asarray(colors)

    intercepts = np.asarray(intercepts)
    slopes = np.asarray(slopes)
    slops_CIs = np.asarray(slops_CIs)



    #fig, ax = plt.subplots(figsize=(4,4))

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.15)

    fig.text(0.5, 1.07, 'Glucose time-series merged across replicates', fontsize=14, fontweight='bold', ha='center', va='center')


    ax_scatter = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax_slopes = plt.subplot2grid((1, 2), (0, 1), colspan=1)

    #fig.subplots_adjust(bottom= 0.15)

    #slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))

    #x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
    #y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
    #y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)

    for slope_idx, slope in enumerate(slopes):

        x_log10_range =  np.linspace(min(np.log10(means)) , max(np.log10(means)) , 10000)
        y_log10_fit_range = 10 ** (slopes[slope_idx]*x_log10_range + intercepts[slope_idx])
        #y_log10_null_range = 10 ** (slope_null*x_log10_range + intercept)
        ax_scatter.plot(10**x_log10_range, y_log10_fit_range, c=rgb[slope_idx], lw=2.5, linestyle='-', zorder=1, label='transfer=%d, slope=%s' % (slope_idx+1 , ('%f' % round(slope,2)).rstrip('0').rstrip('.')))




    #ax.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression")
    #ax.plot(10**x_log10_range, y_log10_null_range, c='k', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

    #ax.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax.transAxes  )


    # run slope test
    #t, p = stats.ttest_ind(dnds_treatment[0], dnds_treatment[1], equal_var=False)
    #t_value = (slope - (slope_null))/std_err
    #p_value = stats.t.sf(np.abs(t_value), len(means)-2)

    #sys.stdout.write("Slope = %g, t = %g, P= %g\n" % (slope, t_value, p_value))

    ax_scatter.legend(loc="lower right", fontsize=6)

    ax_scatter.scatter(means, variances, c= colors, cmap='Blues', alpha=1,zorder=2)#, c='#87CEEB')

    ax_scatter.set_xscale('log', basex=10)
    ax_scatter.set_yscale('log', basey=10)
    ax_scatter.set_xlabel('Average relative\nabundance', fontsize=12)
    ax_scatter.set_ylabel('Variance of relative abundance', fontsize=10)



    #ax_slopes.plot(list(range(1, 13)), slopes, \
    #    'b-',  c = '#FF6347')

    print(color_range)

    ax_slopes.axhline(y=2, color='darkgrey', linestyle=':', lw = 3, zorder=1)


    ax_slopes.errorbar(list(range(1, 13)), slopes, slops_CIs,linestyle='-', marker='o', c='k', elinewidth=1.5, alpha=1, zorder=2)
    ax_slopes.scatter(list(range(1, 13)), slopes, c= color_range, cmap='Blues', alpha=2, zorder=3)#, c='#87CEEB')

    ax_slopes.set_xlabel('Transfer', fontsize=12)
    ax_slopes.set_ylabel('Slope', fontsize=10)



    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig(utils.directory + "/figs/taylors_law_time_series.pdf", format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





#taylors_law_time_series()



#plot_taylors_law_migration_merged_treatments()

taylors_law_attractor()



#taylors_law_attractor()



#plot_predicted_occupancies_migration()

#plot_mad_migration_time_series()


#plot_taylors_time_series()

#plot_taylors_law_migration()






#plot_afd_migration_time_series()



#plot_taylors_migration_time_series()



#plot_taylors_time_series()

#plot_afd()


#plot_afd_migration()


#plot_taylors_law_migration()

#plot_taylors_time_series()
#plot_predicted_occupancies()


#plot_mad()


#plot_abundance_occupy_dist()

#plot_Nmax_vs_pln_Nmax()

#examine_cv('family')
#examine_cv('order')
#examine_cv('genus')

#plot_pln()

#plot_afd()
#plot_taylors()

#plot_N_Nmax()
