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

  // Prior parameters
  vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  real<lower=1, upper=10> eta_prior;
  real<lower=0> sigma_prior_latent;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
}

parameters {
  // Gumbel parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;

  // Correlation
  cholesky_factor_corr[P] L_cor;
  
  // Latent variables for missing data
  vector[Nmiss] zlat_miss;
}  

transformed parameters {
  vector[P] yscale = exp(ylogscale);
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];

  // Cholesky factor of the correlation matrix
  L_cor ~ lkj_corr_cholesky(eta_prior);

  // Latent variable
  zlat_miss ~ normal(0., sigma_prior_latent);

  // --- latent variables ---
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

    // Jacobian z = inv_Phi(gumbel_cdf(obs))
    // dz/dobs = gumbel_pdf(obs) / phi(z)
    // Hence log(dz/dobs) =
    target += gumbel_lpdf(obs | tau, alpha) - std_normal_lpdf(zval);
  }

  for(i in 1:Nmiss) {
    int ival = idx_miss[i][1]; 
    int ivar = idx_miss[i][2]; 
    z[ival][ivar] = zlat_miss[i];
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

