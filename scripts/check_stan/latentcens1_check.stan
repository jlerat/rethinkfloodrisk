data {
  int N; // total number of values
  vector[N] y; // Data 

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - censored data
  int<lower=0> Ncens;

  // Prior parameters
  vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  real censor;
}

parameters {
  // Gumbel parameter vectors
  real ylocn;
  real<lower=logscale_lower, upper=logscale_upper> ylogscale;
}  

transformed parameters {
  real yscale = exp(ylogscale);
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];

  // Transform data to uniform marginals
  for(i in 1:Nobs) {
    int ival = idx_obs[i][1]; 
    y[ival] ~ gumbel(ylocn, yscale);
  }

  // Set censored latent variables
  target += Ncens * gumbel_lcdf(censor | ylocn, yscale);
}

generated quantities {
    real yrnd = gumbel_rng(ylocn, yscale);
    real ucensor = gumbel_cdf(censor | ylocn, yscale);
}

