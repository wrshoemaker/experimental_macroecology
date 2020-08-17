from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import gamma

from macroecotools import obs_pred_rsquare
import utils

carbons = utils.carbons

slope_null=2

titles = ['No migration, low inoculum', 'No migration, high inoculum', 'Global migration, low inoculum', 'Parent migration, low inoculum' ]
migration_innocula = [('No_migration',4), ('No_migration',40), ('Global_migration',4), ('Parent_migration',4)]
plot_idxs = [(0,0), (0,1), (1,0), (1,1)]



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

        s_by_s, ESVs, comm_rep_list = utils.get_s_by_s_migration(migration=migration_innoculum[0],inocula=migration_innoculum[1])

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




plot_predicted_occupancies_migration()

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
