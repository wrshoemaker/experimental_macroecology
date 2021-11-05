

############ LIKELIHOOD ESTIMATE FOR GAMMA

### FIND MAX LIKELIHOOD ESTIMATE OF GAMMA PARS

likelihoodgamma <- function(x, kl, nl ) {
  f <- numeric(2)
  lambda <- exp(x[1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
  if(x[2] > 0){
    f[1] <- mean( (kl - nl*lambda)/(x[2]+nl*lambda)  )
    f[2] <- -mean( (kl - nl*lambda)/(x[2]+nl*lambda)  ) + mean( digamma(kl+x[2]) ) - digamma(x[2]) - mean( log(1+nl*lambda/x[2])  )
  } else{
    f[1] <- 10^5
    f[2] <- 10^5
  }
  
  return(f)
}



maxlikelihoodgamma <- function( kl, nl, nrall , otu_id){
  
  # print(otu_id %>%  unique())
  zeros <- nrall[ which(!nrall %in% nl) ]
  kl <- c(kl, rep(0,length(zeros)))
  nl <- c(nl, zeros)
  
  xstart <- c( log(mean(kl/nl) ), min( mean(kl/nl)^2/mean( (kl^2 - kl)/nl^2 ), 3) )  
  
  try(ans <- nleqslv(xstart,likelihoodgamma, method="Broyden", global="dbldog", kl = kl, nl = nl ), silent = F)  
  
  llambda <- ans$x[1]
  beta <- ans$x[2]
  lambda <- exp(llambda)
  p <- lambda*nl/(beta + lambda*nl)
  
  logL <- sum( -lfactorial(kl) + lgamma(beta+kl) - lgamma(beta) + kl*log(p) + beta*log(1.-p) )
  
  res <- paste(as.character(llambda),as.character(beta),as.character(logL),sep =  "_")
  return(res)
  
}


#### LIKELIHOOD ESTIMATE FOR INFLATED GAMMA
likelihoodinflatedgamma <- function(x, kl, nl, q ) {
  f <- numeric(2)
  lambda <- exp(x[1]) 
  beta <- x[2]
  if(x[2] > 0){
    f[1] <- mean( (kl - nl*lambda)/(beta+nl*lambda)   + (kl==0)*(nl*(q-1)*(beta^(1 + beta)) )/((beta + nl*lambda)*((1 - q) * beta^beta + q*(beta + nl*lambda)^beta) ) )
    f[2] <- -mean( (kl - nl*lambda)/(beta+nl*lambda)  ) + mean( digamma(kl+beta) ) - digamma(beta) - mean( log(1+nl*lambda/beta)    ) + mean( (kl==0)*((q-1)*beta^beta * (nl*lambda + (beta + nl*lambda) * log( beta/(beta + nl*lambda)) )) / ((beta + nl*lambda)*( (q-1)*beta^beta - q*(beta + nl*lambda)^beta)) )
  } else{
    f[1] <- 10^5
    f[2] <- 10^5
  }
  
  return(f)
}



maxlikelihoodinflatedgamma <- function( kl, nl, nrall , otu_id, alfapriorq, betapriorq){
  
  silentopt <- T  
  q <- rbeta(1, alfapriorq, betapriorq)
  
  zeros <- nrall[ which(!nrall %in% nl) ]
  kl <- c(kl, rep(0,length(zeros)))
  nl <- c(nl, zeros)
  
  xstart <- c( log(mean(kl/nl) ), min( mean(kl/nl)^2/mean( (kl^2 - kl)/nl^2 ), 3) )  
  
  try(ans <- nleqslv(xstart,likelihoodinflatedgamma, method="Broyden", global="dbldog", kl = kl, nl = nl , q =  q), silent = silentopt)  
  
  
  llambda <- NA
  beta <- NA
  
  try(llambda <- ans$x[1], silent = silentopt )
  try(beta <- ans$x[2], silent = silentopt)
  lambda <- exp(llambda)
  p <- lambda*nl/(beta + lambda*nl)

  
  logL <- sum( log( q*(kl == 0) + (1.-q) * exp( -lfactorial(kl) + lgamma(beta+kl) - lgamma(beta) + kl*log(p) + beta*log(1.-p) ) ) )
  
  res <- paste(as.character(llambda),as.character(beta),as.character(logL),as.character(q),sep =  "_")
  

  return(res)
  
}


# kl = vector of counts (non zeros)
# Nl = tot number of reads (corresponding to nonzeros)
# nrall = vector of number of redas (including zeros)

## Launch maxlikelihoodinflatedgamma for each otu multiple times (each corresponding to a random value of q)

logLgammainflated <- data.frame()
filename <- paste( "./LogLInflatedGamma_", alfapriorq, "_", betapriorq, ".RData", sep = ""  )
load(filename)

#res <- data.frame()
Nrep <- 50
for( id in unique(proj$idall) ){
  print(id)
  
  pred <- proj %>% filter(idall == id )
  nrall <- ( pred %>% select(run_id, nreads) %>% distinct() )$nreads
  
  for(i in 1:Nrep){
    print(i/Nrep)
    resint <- pred %>%  arrange(-otu_id) %>% group_by(idall, sname, otu_id ) %>% 
      dplyr::summarise(
        est = maxlikelihoodinflatedgamma(count,nreads, nrall = nrall, otu_id, alfapriorq, betapriorq) ,
        ndata = n_distinct(run_id) ) %>% ungroup() %>%
      separate(est, c("llambda", "beta", "logL_gamma", "q"), sep = "_") %>%
      mutate(llambda = as.numeric(llambda), beta = as.numeric(beta), logL_gamma = as.numeric(logL_gamma) , q = as.numeric(q) )
    
    logLgammainflated <- rbind(resint,logLgammainflated)
    rm(resint)
    save(logLgammainflated, file = filename )
  }
}



### USEFUL TO AVERAGE SMALL NUMBERS
mean_exp <- function( x ){
  
  x <- na.omit(x)
  
  xmax <- max(x)
  y <- x - xmax
  m <- xmax  + log(mean(exp(y)))
  return( m )
  
}

saddle_exp <- function( f, x ){
  
  x <- x[!is.na(f)] #remove NAs
  f <- f[!is.na(f)]
  
  fmax <- max(f)
  
  ff <- (f - fmax)
  m <- fmax  + log(mean(x*exp(ff)))
  return( m )
  
}

saddle_prob <- function( f, x ){
  
  x <- x[!is.na(f)] #remove NAs
  f <- f[!is.na(f)]
  
  fmax <- max(f)
  
  ff <- (f - fmax)
  m <- mean(x*exp(ff))/mean(ff) 
  return( m )
  
}