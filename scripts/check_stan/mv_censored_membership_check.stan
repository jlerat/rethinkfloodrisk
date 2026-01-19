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

  // Membership using the 
  // compact 2 array representation of symetric matrices
  // See 
  // https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html#scipy.spatial.distance.squareform
  int Q = (P * (P - 1)) // 2;
  array[N] array[Q] int<lower=0, upper=1> membership;

  // Prior parameters
  vector[2] ylocn_prior;
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  
  vector[2] ylogscale_prior;
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  
  real<lower=1, upper=10> eta_prior;
  real<lower=0> sigma_prior_latent;

  vector[P] censors;
}

transformed data {
  // Check number of data in each category adds up 
  int<lower=0, upper=0> Ncheck = N * P - Nobs - Ncens - Nmiss;

  row_vector[P] zero_mean = zeros_row_vector(P);

  // Build membership matrix for each event
  // given membership data in condensed form.
  array[N] matrix<lower=0, upper=1>[P, P] membership_matrices;
  int l;
  real memb;
  for (i in 1:N) {
    for(j in 1:P) for(k in 1:P) {
        l = 1 + (P * (P - 1)) // 2 - ((P - j) * (P - j + 1)) // 2 + k - j - 1;
        memb = membership[i][l];
        membership_matrices[i][j, k] = memb;
        membership_matrices[i][k, j] = memb;      
    }
  }
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
  vector<lower=0,upper=1>[Ncens] wlat_cens;
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

  // Membership probabilities
  memb_prob ~ 

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
    z[ival][ivar] = inv_Phi(wlat_miss[i]);
  }

  // Set censored latent variables
  for(i in 1:Ncens) {
    int ival = idx_cens[i][1]; 
    int ivar = idx_cens[i][2]; 
    real zl = inv_Phi(ulat_cens[i]);
    z[ival][ivar] = zl;
    
    // log=Jacobian of censored latent variable transform
    target += log(ucensors[ivar]) - std_normal_lpdf(zl); 
  }

  // --- Likelihood ---
  for(i in 1:Nobs) {
    int ival = idx_obs[i][1]; 
    
    // Correlation taking into account membership
    Cor = crossprod(L_cor) * membership_matrices[ival];

    // Standard multi normal pdf
    z[ival] ~ multi_normal(zero_mean, Cor);
}

generated quantities {
    vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
    
    real tau = ylocn[i];
    real alpha = yscale[i];
    vector[P] yrnd;
    for(i in 1:P) {
       yrnd[i] = tau - alpha * log(-log(Phi(zrnd[i])));
    }
}

