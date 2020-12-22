
def calculate_all_likelihoods_old(carbon_source, alfapriorq_list, betapriorq_all):

    '''
    this old script selects the maximum mean likelihood for the inflated model over all
    initial conditions, I think what I should do instead is maximize the mean likelihood
    within each prior across all initial conditions,
    '''

    s_by_s_np, species, communities_final = utils.get_s_by_s(carbon_source, transfer=12, communities=None)
    nrall = np.sum(s_by_s_np, axis=0)

    priors_mean = alfapriorq_list / (alfapriorq_list +betapriorq_all)

    header = ['ESV', 'N_obs', 'N_obs_nonzero', 'maxlikelihoodgamma', 'maxlikelihoodinflatedgamma']
    for prior_mean in priors_mean:
        header.append('maxlikelihoodinflatedgamma-%s'%str(prior_mean))

    record_strs = [", ".join(header)]

    for afd_i_idx, afd_i in enumerate(s_by_s_np):

        #if afd_i_idx > 1:
        #    continue

        #prop_zeros = sum(afd_i==0) / len(communities_final)

        if len(afd_i[afd_i>0]) < min_nonzero:
            continue

        #if prop_zeros > 0.8:
        #    continue

        x_start_list_i = x_start_list.copy()
        x_start_list_i.append((np.log(np.mean(afd_i/nrall)), min((np.mean(afd_i/nrall)**2)/np.mean( ((afd_i**2) - afd_i)/(nrall**2) ), 3)))

        logL_0 = -1000000000

        llambda_best = 0
        beta_best = 0
        logL_best = logL_0
        xstart_best = (1,1)

        llambda_inflated_best = 0
        beta_inflated_best = 0
        p_inflated_best = 0
        logL_inflated_best = logL_0
        xstart_inflated_best = (1,1)
        alpha_dict_best = {}

        for x_start_i in x_start_list_i:

            print(afd_i_idx, x_start_i)

            try:

                llambda, beta, logL, xstart = maxlikelihoodgamma(x_start_i, afd_i, nrall)
                converged = True

                # keep maximum log likelihood
                if logL > logL_best:

                    llambda_best = llambda
                    beta_best = beta
                    logL_best = logL
                    xstart_best = xstart

            except optimize.nonlin.NoConvergence as e:

                x = e.args[0]
                converged = False


            # for zero inflated gamma, go over a range of starting conditions
            # for each starting condition, go through each prior fifty times
            # compute the weighted average of the maximum likelihood for all
            # initial conditions that successfully converge

            # try to save memory by skipping initial conditions where
            # less than k alphas successfully optimize

            alpha_dict = {}

            num_failures = 0
            for alpha in alfapriorq_list:

                # if we can't get enough sucessful iterations to perform the averaging,
                # skip to the next set of initial conditions
                if num_failures > 0:
                    continue

                logL_inflated_alpha = []
                #p_inflated_alpha = []
                q_inflated_alpha = []
                N_tries_i = 0
                N_sucesses_i = 0

                while (N_tries_i < N_tries) and (N_sucesses_i < N_sucesses):

                    try:

                        llambda_inflated, beta_inflated, p_inflated, logL_inflated, q_inflated, xstart_inflatd = maxlikelihoodinflatedgamma(x_start_i, afd_i, nrall, alpha, betapriorq_all)
                        converged = True
                        logL_inflated_alpha.append(logL_inflated)
                        #p_inflated_alpha.append(p_inflated)
                        # get the prior
                        q_inflated_alpha.append(q_inflated)
                        N_sucesses_i += 1

                    except optimize.nonlin.NoConvergence as e:

                        x = e.args[0]
                        converged = False

                    N_tries_i += 1


                if len(logL_inflated_alpha) < N_sucesses:
                    num_failures += 1
                    continue

                alpha_dict[alpha] = np.asarray(logL_inflated_alpha)

                #alpha_dict[alpha] = (np.mean(logL_inflated_alpha), len(logL_inflated_alpha))

                print(afd_i_idx, x_start_i, alpha/(alpha+betapriorq_all), len(logL_inflated_alpha))


            # skip to next iteration if optimization did not succeed for all priors
            if len(alpha_dict) < len(alfapriorq_list):
                continue

            #weighted_mean_likelihood = sum([k[0] * k[1] for k in alpha_dict.values()]) / sum([k[1] for k in alpha_dict.values()])
            #weighted_mean_p = sum([k[1] * k[2] for k in alpha_dict.values()]) / sum([k[2] for k in alpha_dict.values()])
            # prior_probability

            # same number of iterations for each prior, so we don't have to weight the mean

            mean_logL_inflated = [ k* alpha_dict[k] for k in alpha_dict.keys() ]
            mean_logL_inflated = np.mean(np.concatenate(mean_logL_inflated).ravel())

            print(mean_logL_inflated)

            if mean_logL_inflated > logL_inflated_best:

                # keep maximum log likelihood
                logL_inflated_best = mean_logL_inflated
                #p_inflated_best = weighted_mean_p
                #beta_inflated_best = beta
                #llambda_inflated_best = logL
                #xstart_inflated_best = xstart
                alpha_dict_best = alpha_dict


        # print out likelihood for both models and all alphas for that species if the iteration worked

        if (logL_inflated_best != logL_0) and (logL_best != logL_0):

            record_list = [species[afd_i_idx], str(len(afd_i)), str(len(afd_i[afd_i>0])), str(logL_best), str(logL_inflated_best)]

            for alfapriorq_list_i in alfapriorq_list:

                #record_list.append(str(alpha_dict_best[alfapriorq_list_i][0]))
                record_list.append(str( np.mean( alpha_dict_best[alfapriorq_list_i] ) * alfapriorq_list_i ))

            record_str = ", ".join(record_list)
            record_strs.append(record_str)


    sys.stderr.write("Done with %s species!\n" % carbon_source)
    sys.stderr.write("Writing intermediate file...\n")
    intermediate_filename = "%s/data/%s_gamma_likelihood.csv" % (utils.directory, carbon_source)
    file = open(intermediate_filename,"w")
    record_str = "\n".join(record_strs)
    file.write(record_str)
    file.close()
    sys.stderr.write("Done!\n")
