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

  // Cluster observed data
  array[N, P, P] int<lower=0, upper=1> clusters;
  array[N] int<lower=0> clusters_counts;

  // Cluster possible data
  int Q; // total number of possible clusters
  array[Q, P, P] int<lower=0, upper=1> clusters_possible;
  array[Q] int<lower=0, upper=P> clusters_possible_counts;

  // Copula model
  // 0 : Gaussian
  // >0 : Student-t df=copula
  real<lower=0, upper=5> copula; 

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
  // Check number of data in each category adds up 
  int Ntest = N * P - Nobs - Ncens;
  int<lower=0, upper=0> Ncheck = Ntest; 

  row_vector[P] zero_mean = zeros_row_vector(P);

  // Required for correlation matrix transformation
  real lam = (rho_max - rho_min) / 2;
  matrix[P, P] Id = identity_matrix(P);
  matrix[P, P] cor0 = (1 - rho_max) * Id + (rho_max + rho_min) / 2 * rep_matrix(1., P, P);

  // Observed cluster data
  array[N, P] int<lower=0> clusters_invidual_counts;
  array[N, P] int<lower=0, upper=P> clusters_invidual_indexes;
  for(i in 1:N) {
    for(icl in 1:clusters_counts[i]) {
        array[P] int cl = clusters[i, icl, :];
        int count = 0;
        for(j in 1:P) {
            if(cl[j] == 1) {
                clusters_individual_indexes[i, count] = j;
                count += 1;
            }
            clusters_individual_counts[i, j] = count;
        }
    }
  }

  // Possible cluster data
  array[Q, P] int<lower=0> clusters_possible_invidual_counts;
  array[Q, P] int<lower=0, upper=P> clusters_possible_invidual_indexes;
  for(i in 1:Q) {
    for(icl in 1:clusters_possible_counts[i]) {
        array[P] int cl = clusters_possible[i, icl, :];
        int count = 0;
        for(j in 1:P) {
            if(cl[j] == 1) {
                clusters_possible_individual_indexes[i, count] = j;
                count += 1;
            }
            clusters_possible_individual_counts[i, j] = count;
        }
    }
  }
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
  matrix[P, P] cor_IW = cor0 + lam * quad_form(multiply_lower_tri_self_transpose(L_IW), Si);
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
    real zval = quantile(gev_cdf(obs | tau, alpha, kappa), copula);
    z[ival][ivar] = zval;

    // log-Jacobian of z = inv_Phi(gev_cdf(obs))
    // dz/dobs = gev_pdf(obs) / phi(z)
    // Hence log(dz/dobs) =
    target += gev_lpdf(obs | tau, alpha, kappa) + quantile_log_jac(zval, copula);
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    real zcens = quantile(ulat_cens[i], copula);
    z[ival][ivar] = zcens;
    
    // log-Jacobian of censored latent variable transform
    target += log(ucensors[ivar]) + quantile_log_jac(zcens, copula);
  }

  // --- Likelihood ---
  for(i in 1:N) {

    // Loop on clusters
    for(icl in 1:clusters_counts[i]) {
        // Get cluster array indexes
        int ncl = clusters_individual_counts[i, icl]; 
        array[ncl] int idx = clusters_individual_indexes[1:ncl];

        // Compute likelihood within cluster
        target += copula_log_pdf(z[i, idx], copula, cor_IW[idx, idx]);
    }

  }
}

generated quantities {
    // Change this for generic copula
    vector[P] zrnd = multi_normal_rng(zero_mean, cor_IW);

    // Add sampling from clusters
    
    vector[P] yrnd;
    for(ivar in 1:P) {
       real tau = ylocn[ivar];
       real alpha = yscale[ivar];
       real kappa = yshape1[ivar];
       yrnd[ivar] = gev_quantile(Phi(zrnd[ivar]), tau, alpha, kappa);
    }
}

