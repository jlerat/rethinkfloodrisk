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
  int F; // Number of factors
  array[N] vector[P] y; // Data 

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - censored data
  int<lower=0> Ncens;
  array[Ncens, 2] int idx_cens;

  // Indexing - missing data
  int<lower=0> Nmiss;
  array[Nmiss, 2] int idx_miss;

  // Choice of marginal (only GEV allowed for now)
  int<lower=0, upper=0> marginal_id; 

  // Copula model
  // 0 : Gaussian, 1: Student
  int<lower=0, upper=1> copula_id;
  real copula_shape; 

  // Prior parameters
  array[P] vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  array[P] vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  array[P] vector[2] yshape1_prior;
  real shape1_lower;
  real<lower=shape1_lower> shape1_upper;

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
  int Ntest = N * P - Nobs - Ncens - Nmiss;
  int<lower=0, upper=0> Ncheck = Ntest; 

  row_vector[P] zero_mean = zeros_row_vector(P);
}

parameters {
  // GEV parameter vectors
  vector[P] ylocn;
  vector<lower=logscale_lower, upper=logscale_upper>[P] ylogscale;
  vector<lower=shape1_lower, upper=shape1_upper>[P] yshape1;

  // Latent variables for missing data
  vector<lower=0, upper=1>[Nmiss] ulat_miss;

  // Latent variables for censored data
  vector<lower=0, upper=1>[Ncens] wlat_cens;

  // Factor parameters -> sample one more factor to obtain 
  // a uniform in the hypersphere of dim F
  // See https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_within_the_n-ball
  array[P] vector[F + 1] zrhos;
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

  // Lengthy code to circumvent stan's datatypes
  // We only store corr here thanks to curly braces
  matrix[P, P] corr;
  {
    matrix[P, F] rhos_matrix;
    for(i in 1:P) 
      rhos_matrix[i] = to_row_vector(zrhos[i][1:F] / sqrt(dot_self(zrhos[i])));
    
    // Could we get the precision instead of corr?
    corr = rhos_matrix * rhos_matrix';
    corr = diag_matrix(1. - diagonal(corr)) + corr;
  }  
}

model {
  // --- Priors ---
  for(i in 1:P) {
    ylocn[i] ~ normal(ylocn_prior[i][1], ylocn_prior[i][2]) T[locn_lower, locn_upper];
    ylogscale[i] ~ normal(ylogscale_prior[i][1], ylogscale_prior[i][2]) T[logscale_lower, logscale_upper];
    yshape1[i] ~ normal(yshape1_prior[i][1], yshape1_prior[i][2]) T[shape1_lower, shape1_upper];
  }

  // Prior for latent factors 
  for(i in 1:P)
    zrhos[i] ~ std_normal();

  // -- Latent variable matrix ---
  // (careful indexes switched compared to other stan code)
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

  // Set missing latent variables
  for(i in 1:Nmiss) {
    int ival = idx_miss[i][1]; 
    int ivar = idx_miss[i][2]; 
    real zmiss = copula_marginal_quantile(ulat_miss[i], 
                                          copula_id,
                                          copula_shape);
    z[ival][ivar] = zmiss;
    
    // log-Jacobian of missing latent variable transform
    target += copula_marginal_quantile_log_jac(zmiss, copula_id, copula_shape);
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
  for(i in 1:N) {
    if(copula_id == 0) 
        z[i] ~ multi_normal(zero_mean, corr);
    else {
        real sig_adj = get_student_std_adjust(copula_shape);
        z[i] ~ multi_student_t(copula_shape, zero_mean, corr * sig_adj);
    }    
  }

}
