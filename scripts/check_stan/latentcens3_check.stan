functions {

    #include marginal.stanfunctions

}

data {
  int N; // total number of values
  vector[N] y; // Data 

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - censored data
  int<lower=0> Ncens;
  array[Ncens, 2] int idx_cens;

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
  
  // Transformed latent variable for censored data
  vector<lower=0, upper=1>[Ncens] wlat_cens;
}  

transformed parameters {
  real yscale = exp(ylogscale);

  // Censors in normal space
  real ucensor = gumbel_cdf(censor | ylocn, yscale);

  // Latent variable for censored data
  vector[Ncens] ulat_cens = ucensor * wlat_cens;
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];

  // -- Latent variable matrix ---
  vector[N] yl;
  for(i in 1:Nobs){
    int ival = idx_obs[i][1]; 
    yl[ival] = y[ival];
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    real yval = gumbel_quantile(ulat_cens[i], ylocn, yscale);
    yl[ival] = yval;
    
    // Log-Jacobian of latent variable constraint
    target += log(ucensor) - gumbel_lpdf(yval | ylocn, yscale);
  }

  // --- Likelihood ---
  y ~ gumbel(ylocn, yscale);
}

generated quantities {
    real yrnd = gumbel_rng(ylocn, yscale);
}

