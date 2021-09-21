#install.packages("devtools")
#BiocManager::install('phyloseq')
#devtools::install_github("adw96/breakaway")

library(breakaway)
data(toy_otu_table)


drop <- c("X", "ASVs")

Parent_migration.NA.T0 <- read.csv(file='/Users/williamrshoemaker/GitHub/experimental_macroecology/data/Parent_migration.NA.T0.csv', header=TRUE)
row.names(Parent_migration.NA.T0) <- Parent_migration.NA.T0$ASVs
Parent_migration.NA.T0 <- Parent_migration.NA.T0[,!(names(Parent_migration.NA.T0) %in% drop)]
frequencytablelist.Parent_migration.NA.T0 <- build_frequency_count_tables(Parent_migration.NA.T0)

No_migration.4.T18 <- read.csv(file='/Users/williamrshoemaker/GitHub/experimental_macroecology/data/No_migration.4.T18.csv', header=TRUE)
row.names(No_migration.4.T18) <- No_migration.4.T18$ASVs
No_migration.4.T18 <- No_migration.4.T18[,!(names(No_migration.4.T18) %in% drop)]
frequencytablelist.No_migration.4.T18 <- build_frequency_count_tables(No_migration.4.T18)

No_migration.40.T18 <- read.csv(file='/Users/williamrshoemaker/GitHub/experimental_macroecology/data/No_migration.40.T18.csv', header=TRUE)
row.names(No_migration.40.T18) <- No_migration.40.T18$ASVs
No_migration.40.T18 <- No_migration.40.T18[,!(names(No_migration.40.T18) %in% drop)]
frequencytablelist.No_migration.40.T18 <- build_frequency_count_tables(No_migration.40.T18)

Parent_migration.4.T18 <- read.csv(file='/Users/williamrshoemaker/GitHub/experimental_macroecology/data/Parent_migration.4.T18.csv', header=TRUE)
row.names(Parent_migration.4.T18) <- Parent_migration.4.T18$ASVs
Parent_migration.4.T18 <- Parent_migration.4.T18[,!(names(Parent_migration.4.T18) %in% drop)]
frequencytablelist.Parent_migration.4.T18 <- build_frequency_count_tables(Parent_migration.4.T18)

Global_migration.4.T18 <- read.csv(file='/Users/williamrshoemaker/GitHub/experimental_macroecology/data/Global_migration.4.T18.csv', header=TRUE)
row.names(Parent_migration.4.T18) <- Global_migration.4.T18$ASVs
Global_migration.4.T18 <- Global_migration.4.T18[,!(names(Global_migration.4.T18) %in% drop)]
frequencytablelist.Global_migration.4.T18 <- build_frequency_count_tables(Global_migration.4.T18)






breakaway(frequencytablelist.Parent_migration.NA.T0[[1]]) # richness = 1533

breakaway(frequencytablelist.No_migration.4.T18[[1]]) # richness = 29
breakaway(frequencytablelist.No_migration.40.T18[[1]]) # richness = 18
breakaway(frequencytablelist.Parent_migration.4.T18[[1]]) # richness = 30
breakaway(frequencytablelist.Global_migration.4.T18[[1]]) # richness = 36


