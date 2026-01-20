functions {

    #include marginal.stanfunctions

}

data {
  int N; // total number of values
  int Q; // total number of membership
  array[N] array[Q] int<lower=0, upper=1> membership;

  vector[2] beta_prior;
}

parameters {
  // Means
  vector[Q] beta;

  // Correlation
  cholesky_factor_corr[P] L_cor;

  // Latent variables
  array[N] array[Q] real<lower=0, upper=1> ulat;
}  

transformed parameters {
  // Standard deviations of covariance matrix
  matrix[Q, Q] Si = diag_matrix(1. / sqrt(rows_dot_self(L_IW)));

  // Computation of shifted correlation matrix 
  matrix[Q, Q] cor_IW = quad_form(multiply_lower_tri_self_transpose(L_IW), Si);
  
  // Latent variables in restricted space
  array[N] array[Q] real<lower=0, upper=1> wlat;
  int mem;
  for (i in 1:N) {
    for(j in 1:Q) {
        mem = membership[i][j];
        wlat[i][j] = mem == 0 ? ulat[i][j] * 0.5 : 0.5 + ulat[i][j] * 0.5;
    }
  }
}

model {
  // --- Priors ---
  ylocn ~ normal(beta_prior[1], beta_prior[2]);

  // Prior for cholesky factor of the correlation matrix
  L_IW ~ inv_wishart_cholesky(Q + 1., Id);

  // Likelihood
  real zval;
  array[N] vector[P] z;

  for (i in 1:N) {
    for(j in 1:Q) {
        zval = inv_Phi(wlat[i][j]);
        z[i][j] = zval;

        // log-Jacobian of z = inv_Phi(w)
        // dz/dw = 1 / phi(z)
        // Hence log(dz/dw) =
        // CHECK THIS
        target -= std_normal_lpdf(zval);
    }
  }
  z ~ multi_normal(beta, cor_IW);
}

