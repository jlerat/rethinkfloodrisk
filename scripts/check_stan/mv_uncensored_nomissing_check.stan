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

  // Gumbel parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;

  // Correlation
  cholesky_factor_corr[P] L_cor;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
  
  vector[P] yscale = exp(ylogscale);
}

generated quantities {
  // standard normal
  array[N] vector[P] z;

  // Transform data to uniform marginals
  for(ivar in 1:P) {
    real tau = ylocn[ivar];
    real alpha = yscale[ivar];

    for(i in 1:N) {
        real obs = y[i][ivar];
        z[i][ivar] = inv_Phi_approx(gumbel_cdf(obs | tau, alpha));
    }
  }   

  vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
  
  vector[P] yrnd;
  for(i in 1:P) {
     yrnd[i] = gumbel_quantile(Phi(zrnd[i]), ylocn[i], yscale[i]);
  }
}

