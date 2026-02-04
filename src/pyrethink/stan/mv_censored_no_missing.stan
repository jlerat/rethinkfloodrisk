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
    #include clusters.stanfunctions

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
  array[N] int<lower=0, upper=P> clusters_counts;

  // Partition data
  int Q; // total number of partitions
  array[Q, P, P] int<lower=0, upper=1> partitions;
  array[Q] int<lower=0, upper=P> partitions_counts;
  vector<lower=0, upper=1>[Q] partitions_probabilities;

  // Copula model
  // 0 : Gaussian
  // >2 : Student-t df=copula
  real copula; 

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
  if (copula > 0)
    copula_low = 0;
  else
    copula_low = 2.01;
  real<lower=copula_low, upper=1000> copula_test = copula;

  // Check number of data in each category adds up 
  int Ntest = N * P - Nobs - Ncens;
  int<lower=0, upper=0> Ncheck = Ntest; 

  row_vector[P] zero_mean = zeros_row_vector(P);

  // Required for correlation matrix transformation
  real lam = (rho_max - rho_min) / 2;
  matrix[P, P] Id = identity_matrix(P);
  matrix[P, P] corr0 = (1 - rho_max) * Id + (rho_max + rho_min) / 2 * rep_matrix(1., P, P);

  // Process observed cluster data
  array[N, P, P + 1] int clusters_indexes = 
    cluster_data_processing(clusters, clusters_counts);

  // Process partition data
  array[Q, P, P + 1] int partitions_indexes =
    cluster_data_processing(partitions, partitions_counts);
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
    real zval = copula_marginal_quantile(gev_cdf(obs | tau, alpha, kappa), copula);
    z[ival][ivar] = zval;

    // log-Jacobian of z = inv_Phi(gev_cdf(obs))
    // dz/dobs = gev_pdf(obs) / phi(z)
    // Hence log(dz/dobs) =
    target += gev_lpdf(obs | tau, alpha, kappa) + copula_marginal_quantile_log_jac(zval, copula);
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    real zcens = copula_marginal_quantile(ulat_cens[i], copula);
    z[ival][ivar] = zcens;
    
    // log-Jacobian of censored latent variable transform
    target += log(ucensors[ivar]) + copula_marginal_quantile_log_jac(zcens, copula);
  }

  // --- Likelihood computed separately for each year ---
  // cannot vectorize because of clusters
  int nsta;
  for(i in 1:N) {

    // Loop on clusters for the particular year
    for(icl in 1:clusters_counts[i]) {
        // Number of stations in the particular cluster
        nsta = clusters_indexes[i, icl, 1]; 

        // Indexes of stations belonging to the cluster
        array[nsta] int idxm = clusters_indexes[i, icl, 2:nsta + 1];

        // Compute likelihood within cluster
        target += copula_log_pdf(z[i, idxm], copula, corr_IW[idxm, idxm]);
    }

  }
}

generated quantities {
    // Sample partition
    int ipart = categorical_rng(partitions_probabilities); 

    // Loop through subsets in the selected partition
    vector[P] zrnd, yrnd;

    real tau;
    real alpha;
    real kappa;
    real u;
    int nsta;

    for(isub in 1:partitions_counts[ipart]) {
        // Number of stations in the particular subset
        nsta = partitions_indexes[ipart, isub, 1];

        // Index of stations in the particular subset
        array[nsta] int idxg = partitions_indexes[ipart, isub, 2:nsta + 1];

        // Sample from copula
        zrnd[idxg] = copula_rng(copula, corr_IW[idxg, idxg]);

        // Compute GEV quantile from copula marginal cdf
        for(ivar in idxg) {
            tau = ylocn[ivar];
            alpha = yscale[ivar];
            kappa = yshape1[ivar];
            u = copula_marginal_prob(zrnd[ivar], copula);
            yrnd[ivar] = gev_quantile(u, tau, alpha, kappa);
        }
    }    
}

