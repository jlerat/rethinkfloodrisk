/**
* Multivariate censored gaussian copula
*
*  Throughout the code:
*   - y variable is multivariate obs
*
**/

functions {

    #include gev.stanfunctions

}

data {
  int N; // total number of values
  int P; // total number of variables
  matrix[N, P] y; // Data 

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
  
  vector[2] yshape1_prior;
  real shape1_lower;
  real<lower=shape1_lower> shape1_upper;

  real<lower=1, upper=10> eta_prior;
  
  // Censoring thresholds 
  vector[P] censors;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
}

parameters {
  // GEV parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;
  vector<lower=shape1_lower, upper=shape1_upper>[P] yshape1;

  // Correlation
  cholesky_factor_corr[P] L_cor;

  // Latent variables for missing data
  vector[Nmiss] zmiss;
  
  // Latent variables for missing data
  // we do not declare z variables directly here
  // because the they are bounded up and the bounds
  // varies depending on the variable.
  vector[Ncens] wcens;
}  

transformed parameters {
  vector[P] yscale = exp(ylogscale);

  // Threshold for latent censored data
  vector[P] zcensors;
  for(ivar in 1:P) {
    real tau = ylocn[ivar];
    real alpha = yscale[ivar];
    real kappa = yshape1[ivar];
    real yc = censors[ivar];
    zcensors[ivar] = inv_Phi(gev_cdf(yc | tau, alpha, kappa));
  }

  // standard normal
  array[N] row_vector[P] z;

  // Set missing latent
  for(i in 1:Nmiss) {
    array[2] int imiss = idx_miss[i];
    z[imiss[1], imiss[2]] = zmiss[i];
  }

  // Set censored latent using stan upper bound
  // constraint formula
  for(i in 1:Ncens) {
    array[2] int icens = idx_cens[i];
    real zcens = zcensors[icens[2]] - exp(wcens[i]);
    z[icens[1], icens[2]] = zcens;
  }

  // Transform data to uniform marginals
  for(i in 1:Nobs) {
    int ipt = idx_obs[i][1];
    int ivar = idx_obs[i][2];
    real tau = ylocn[ivar];
    real alpha = yscale[ivar];
    real kappa = yshape1[ivar];
    real obs = y[ipt, ivar];
    z[ipt, ivar] = inv_Phi(gev_cdf(obs | tau, alpha, kappa));
  }
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  // Cholesky factor of the correlation matrix
  L_cor ~ lkj_corr_cholesky(eta_prior);

  // --- Likelihood ---
  z ~ multi_normal_cholesky(zero_mean, L_cor);
}

generated quantities {
    vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
    
    vector[P] yrnd;
    for(i in 1:P) {
       yrnd[i] = gev_quantile(Phi(zrnd[i]), ylocn[i], yscale[i], yshape1[i]);
    }
    
}

