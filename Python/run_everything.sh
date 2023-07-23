#!/bin/bash


# activate conda environment
source activate macroeco_phylo


# run simulations
python slm_simulation_utils.py

# Figure 1
python plot_taylors_law.py    

# Figure 2
python plot_all_laws.py

# Figure 3
python plot_example_langevin.py

# Figure 4
python plot_afd_with_simulation.py

# Figure 5
python plot_taylors_law_migration.py

# Figure 6
python plot_mean_relative_abundance_ratio.py

# Figure 7
python plot_abundance_ratio_temporal.py



# Supplement

# S1
python plot_occupancy.py

# S2
python plot_stationary_vs_time_dependent_afd.py

# S3 
python plot_paired_gamma_error.py

# S4
python plot_rarefaction.py

# S5
python plot_parent_vs_final_abundance.py

# S6
python plot_n_reads.py

# S7
python plot_mra_parent_comparison.py

# S8
python plot_abundance_hist_parent_vs_descendant.py

# S9 
python plot_simulation_taylors.py

# S10
python plot_global_mean_cv.py

# S11
python plot_cv_ratio_per_replicate.py

