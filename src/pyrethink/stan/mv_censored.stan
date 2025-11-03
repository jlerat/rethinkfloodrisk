/**
* Multivariate censored gaussian copula
*
*  Throughout the code:
*   - y variable is multivariate obs
*
**/

functions {

    #include marginal.stanfunctions

}

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

  // Prior parameters
  vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  real<lower=1e-2, upper=2> eta_prior;
  real<lower=0> sigma_prior_latent;

  vector[P] censors;
}

transformed data {
  // Check number of data in each category adds up 
  int<lower=0, upper=0> Ncheck = N * P - Nobs - Ncens - Nmiss;

  row_vector[P] zero_mean = zeros_row_vector(P);
}

parameters {
  // Gumbel parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;

  // Correlation
  cholesky_factor_corr[P] L_cor;
  
  // Latent variables for missing data
  vector<lower=0, upper=1>[Nmiss] wlat_miss;
  
  // Latent variables for censored data
  vector<lower=0, upper=1>[Ncens] wlat_cens;
}  

transformed parameters {
  vector[P] yscale = exp(ylogscale);

  // Compute censors probability
  vector[P] ucensors;
  for(ivar in 1:P) {
    real tau = ylocn[ivar];
    real alpha = yscale[ivar];
    ucensors[ivar] = gumbel_cdf(censors[ivar] | tau, alpha);
  }

  // Latent variable for censored data
  vector[Ncens] ulat_cens;
  for(i in 1:Ncens){
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    ulat_cens[i] = wlat_cens[i] * ucensors[ivar];
  }
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];

  // Prior for cholesky factor of the correlation matrix
  L_cor ~ lkj_corr_cholesky(eta_prior);

  // -- Latent variable matrix ---
  array[N] vector[P] z;

  // Transform data to uniform marginals
  for(i in 1:Nobs) {
    int ival = idx_obs[i][1]; 
    int ivar = idx_obs[i][2]; 

    real tau = ylocn[ivar];
    real alpha = yscale[ivar];

    real obs = y[ival][ivar];
    real zval = inv_Phi(gumbel_cdf(obs | tau, alpha));
    z[ival][ivar] = zval;

    // log-Jacobian of z = inv_Phi(gumbel_cdf(obs))
    // dz/dobs = gumbel_pdf(obs) / phi(z)
    // Hence log(dz/dobs) =
    target += gumbel_lpdf(obs | tau, alpha) - std_normal_lpdf(zval);
  }

  // Set missing latent variables
  for(i in 1:Nmiss) {
    int ival = idx_miss[i][1]; 
    int ivar = idx_miss[i][2]; 
    real zmiss = inv_Phi(wlat_miss[i]);
    z[ival][ivar] = zmiss;

    // Log-jacobian of missing Latent variable
    target += -std_normal_lpdf(zmiss);
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    real zcens = inv_Phi(ulat_cens[i]);
    z[ival][ivar] = zcens;
    
    // log-Jacobian of censored latent variable transform
    target += log(ucensors[ivar]) - std_normal_lpdf(zcens); 
  }

  // --- Likelihood ---
  z ~ multi_normal_cholesky(zero_mean, L_cor);
}

generated quantities {
    vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
    
    vector[P] yrnd;
    for(i in 1:P) {
       yrnd[i] = gumbel_quantile(Phi(zrnd[i]), ylocn[i], yscale[i]);
    }
}

