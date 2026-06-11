/**
* Multivariate censored gaussian copula
*
*  Throughout the code:
*   - y variable is multivariate obs
*
**/

functions {

    #include marginal.stanfunctions
    #include copula.stanfunctions

}

data {
  int N; // total number of values
  int P; // total number of variables
  array[N] vector[P] y; // Data 

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - censored data
  int<lower=0> Ncens;
  array[Ncens, 2] int idx_cens;

  // Choice of marginal (only GEV allowed for now)
  int<lower=0, upper=0> marginal_id; 

  // Copula model
  // 0 : Gaussian, 1: Student
  int<lower=0, upper=1> copula_id;
  real copula_shape; 


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

  real<lower=-1, upper=1> rho_min;
  real<lower=rho_min, upper=1> rho_max;
  
  vector[P] censors;
}

transformed data {
  // Check copula
  real copula_low;
  if (copula_id == 1)
    copula_low = 2.01;
  else
    copula_low = 0;
  real<lower=copula_low, upper=100> copula_test = copula_shape;

  // Check number of data in each category adds up 
  int Ntest = N * P - Nobs - Ncens;
  int<lower=0, upper=0> Ncheck = Ntest; 

  row_vector[P] zero_mean = zeros_row_vector(P);

  // Required for correlation matrix transformation
  real lam = (rho_max - rho_min) / 2;
  matrix[P, P] Id = identity_matrix(P);
  matrix[P, P] corr0 = (1 - rho_max) * Id + (rho_max + rho_min) / 2 * rep_matrix(1., P, P);
}

parameters {
  // GEV parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;
  vector<lower=shape1_lower, upper=shape1_upper>[P] yshape1;

  // Correlation
  cholesky_factor_corr[P] L_IW;
  
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
    real kappa = yshape1[ivar];
    ucensors[ivar] = gev_cdf(censors[ivar] | tau, alpha, kappa);
  }

  // Latent variable for censored data
  vector[Ncens] ulat_cens;
  for(i in 1:Ncens){
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    ulat_cens[i] = wlat_cens[i] * ucensors[ivar];
  }

  // Standard deviations of covariance matrix
  matrix[P, P] Si = diag_matrix(1. / sqrt(rows_dot_self(L_IW)));

  // Computation of shifted correlation matrix 
  matrix[P, P] corr_IW = corr0 + lam * quad_form(multiply_lower_tri_self_transpose(L_IW), Si);
}

model {
  // --- Priors ---
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  // Prior for cholesky factor of the correlation matrix
  L_IW ~ inv_wishart_cholesky(P + 1., Id);

  // -- Latent variable matrix ---
  array[N] vector[P] z;

  // Transform data to uniform marginals
  for(i in 1:Nobs) {
    int ival = idx_obs[i][1]; 
    int ivar = idx_obs[i][2]; 

    real tau = ylocn[ivar];
    real alpha = yscale[ivar];
    real kappa = yshape1[ivar];

    real obs = y[ival][ivar];
    real zval = copula_marginal_quantile(gev_cdf(obs | tau, alpha, kappa), 
                                         copula_id, copula_shape);
    z[ival][ivar] = zval;

    // log-Jacobian of z = inv_Phi(gev_cdf(obs))
    // dz/dobs = gev_pdf(obs) / phi(z)
    // Hence log(dz/dobs) =
    target += gev_lpdf(obs | tau, alpha, kappa) 
        + copula_marginal_quantile_log_jac(zval, copula_id, copula_shape);
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    real zcens = copula_marginal_quantile(ulat_cens[i], 
                                          copula_id,
                                          copula_shape);
    z[ival][ivar] = zcens;
    
    // log-Jacobian of censored latent variable transform
    target += log(ucensors[ivar]) 
        + copula_marginal_quantile_log_jac(zcens, copula_id, copula_shape);
  }

  // --- Likelihood ---
  for(i in 1:N)
    target += copula_log_pdf(z[i], copula_id, copula_shape, corr_IW);
}
