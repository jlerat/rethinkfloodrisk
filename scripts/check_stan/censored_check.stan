data {
  int N; // total number of values
  int P; // total number of variables
  array[N] vector[P] y; // Data 

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - missing data
  int<lower=0> Nmiss;
  array[Nmiss, 2] int idx_miss;

  // Indexing - censored data
  int<lower=0> Ncens;
  array[Ncens, 2] int idx_cens;

  vector[P] censors;

  real<lower=1, upper=10> eta_prior;
  real sigma_prior_latent;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
}

parameters {
  // Correlation
  cholesky_factor_corr[P] L_cor;
  
  // Latent variables for missing data
  vector[Nmiss] zlat_miss;
  
  // Latent variables for censored data
  vector[Ncens] zlat_cens;
}  

model {
  // --- Priors ---
  // Prior for cholesky factor of the correlation matrix
  L_cor ~ lkj_corr_cholesky(eta_prior);

  // Prior for missing latent variables
  zlat_miss ~ normal(0., sigma_prior_latent);

  // -- Latent variable matrix ---
  array[N] vector[P] z;

  // Set obs values
  for(i in 1:Nobs) {
    int ival = idx_obs[i][1]; 
    int ivar = idx_obs[i][2]; 
    z[ival][ivar] = y[ival][ivar];
  }

  // Set missing latent variables
  for(i in 1:Nmiss) {
    int ival = idx_miss[i][1]; 
    int ivar = idx_miss[i][2]; 
    z[ival][ivar] = zlat_miss[i];
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    z[ival][ivar] = zlat_cens[i];
    
    // Prior for latent censored variables
    if(zlat_cens[i] > censors[ivar])
        reject("Censored latent variable should be lower than censor");
    zlat_cens[i] ~ normal(0., sigma_prior_latent) T[, censors[ivar]];
  }

  // --- Likelihood ---
  z ~ multi_normal_cholesky(zero_mean, L_cor);
}

generated quantities {
  vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
}
