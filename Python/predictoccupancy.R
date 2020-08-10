rm(list=ls(all=TRUE))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
require(tidyverse)

##### GGPLOT THEME
mytheme_main <- theme_bw() + theme(
  legend.title  = element_text(family="Helvetica", size=17, color = "#222222"),
  legend.key = element_blank(),
  legend.text  = element_text(family="Helvetica", size=17, color = "#222222"),
  panel.background = element_rect(fill="transparent"),
  #plot.background = element_rect(fill="transparent", colour = NA),
  panel.grid = element_blank(),
  text = element_text( family="Helvetica", size=17, color = "#222222"),
  panel.border = element_blank(),
  axis.title = element_text( family="Helvetica", size=17, color = "#222222"),
  axis.text = element_text( family="Helvetica", size=15, color = "#222222"),
  axis.line = element_line(size = 1., color = "#222222"),
  axis.ticks = element_line(size = 1.,color = "#222222"),
  legend.background = element_rect(fill="transparent", colour = NA)
)
##########

df <- read.csv("mydata.csv") # CHANGE HERE

###IGNORE THIS
# load("./Genomics/Projects/EBI-Taxonomy/alltax_EBI.RData")
# df <- datatax %>% filter( project_id %in% c("ERP012927","ERP015450"), classification %in% c("Lake","GUT") ) %>% select(classification, run_id, otu_id, count, nreads ) %>%
#   rename( condition = classification, sample = run_id, sp = otu_id, count = count, totreads = nreads ) %>%
#   mutate(condition = "Lake+GUT")



### I assume that your data are in this format
# condition | sample | sp | count | totreads
## condition -> is the experimental condition (e.g. glucose)
## sample is the sample
## sp is the taxa id (otu, esv, ...)
## count is the number of reads for that sp
## totreads is the total number of reads in that sample (including unassigned seq)


## ESTIMATE MEAN AND VARIANCE OF THE UNDERLYING REL Abundance
estpars <- df %>% filter(count > 0) %>% # remove entries with zero count (if present)
  group_by( condition, sp ) %>% ## for each species in each condition
  mutate( tf = mean(count/totreads), o = n(), tvpf = mean( (count^2 - count)/totreads^2 ) ) %>% # calculate mean and variance excluding zeros (! for the variance see SI of "laws of diversity")
  ungroup() %>%
  group_by( condition ) %>%  mutate(o = o / n_distinct(sample) ) %>% # calculate occupancy
  ungroup() %>% select(-sample) %>% distinct() %>%
  mutate( f = o*tf, vf = o*tvpf ) %>% mutate(vf = vf - f^2 ) %>%  ungroup() %>% # mean and variance including zero
  select( -tf, -tvpf, -count, -totreads ) %>% distinct()# remove from data
## estpars should look like
# condition | sp | f | vf | o

## PREDICT OCCUPANCY ASSUMING A GAMMA AFD
occpred <- df %>% left_join(estpars %>% mutate( beta = f^2/vf, theta = f/beta ) ) %>%
  group_by( condition, sp, f, vf, o ) %>%  summarize( o_pred = 1- mean((1.+theta*totreads)^(-beta ) )  ) %>%
  ungroup()

occpred %>% filter(vf > 0) %>%  ggplot()  + mytheme_main + ## estimation is based on poisson sampling... sometimes estimates negative variances (eveything can be done with max likelihood)
  aes(
    y = o,
    x = o_pred
  ) + geom_point( color = "gray", alpha = 0.2 ) +
  geom_abline(  ) +
  stat_summary_bin( fun.y = "mean", geom = "point", size = 3, stroke = 1, bins = 13, shape = 1 ) +
  scale_y_continuous(name = "Empirical occupancy") +
  scale_x_continuous( name = "Predicted occupancy" ) +
  facet_wrap( . ~ condition )
