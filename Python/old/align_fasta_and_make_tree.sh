#!/bin/bash


asv=/Users/williamrshoemaker/GitHub/experimental_macroecology/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.fna
asv_muscle=/Users/williamrshoemaker/GitHub/experimental_macroecology/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918_muscle.fna
tree=/Users/williamrshoemaker/GitHub/experimental_macroecology/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.tre


muscle -in ${asv} -out ${asv_muscle}


#~/raxml-ng_v1.1.0_macos_x86_64/raxml-ng --redo  --all --msa ${asv_muscle} --msa-format FASTA --data-type DNA --seed 123456789 --model GTR+G --bs-trees autoMRE


#-nt = nucleotide
# -gtr = general time reversible
#-gamma = distribution

FastTree -nt -gtr -gamma ${asv_muscle} > ${tree}
