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

  // Prior parameters
  vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  real<lower=1, upper=10> eta_prior;
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
}  

transformed parameters {
  vector[P] yscale = exp(ylogscale);
}

model {
  // standard normal
  array[N] vector[P] z;

  // Transform data to uniform marginals
  for(ivar in 1:P) {
    real tau = ylocn[ivar];
    real alpha = yscale[ivar];

    for(i in 1:N) {
        real obs = y[i][ivar];
        real zval = inv_Phi(gumbel_cdf(obs | tau, alpha));
        z[i][ivar] = zval;

        // log-Jacobian z = inv_Phi(gumbel_cdf(obs))
        // dz/dobs = gumbel_pdf(obs) / phi(z)
        // Hence log(dz/dobs) =
        target += gumbel_lpdf(obs | tau, alpha) - std_normal_lpdf(zval);
    }
  }

  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];

  // Cholesky factor of the correlation matrix
  L_cor ~ lkj_corr_cholesky(eta_prior);

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

