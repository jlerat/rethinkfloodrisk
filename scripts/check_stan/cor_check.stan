functions {
    // See 
    // Barnard, J., McCulloch, R., & Meng, X. L. (2000). 
    // Modeling covariance matrices in terms of standard deviations 
    // and correlations, with application to shrinkage. Statistica Sinica, 10(4), 1281–1311.
    real barnard_lpdf(matrix C) {
        int P = rows(C);
        real logdet = log_determinant(C);
        real lpr = sum(log(diagonal(C))); 
        return P * (P - 1) / 2 * logdet - (P + 1) / 2 * lpr;
    }
}

data {
  int P; // total number of variables
  real<lower=0, upper=1e1> eta_prior;

  real<lower=-1, upper=1> rho_min;
  real<lower=rho_min, upper=1> rho_max;
}

transformed data {
  real lam = (rho_max - rho_min) / 2;

  matrix[P, P] Id = identity_matrix(P);
  vector[P] mu = zeros_vector(P);
  matrix[P, P] cor0 = (1 - rho_max) * Id + (rho_max + rho_min) / 2 * rep_matrix(1., P, P);
}

parameters {
  cholesky_factor_corr[P] L_LKJ;
  cholesky_factor_cov[P] L_IW;
  //cholesky_factor_corr[P] L_BA;
}

transformed parameters {
  matrix[P, P] cor_LKJ = cor0 + lam * multiply_lower_tri_self_transpose(L_LKJ);

  // Standard deviations
  matrix[P, P] Si = diag_matrix(1. / sqrt(rows_dot_self(L_IW)));
  // Computation of shifted correlation matrix 
  matrix[P, P] cor_IW = cor0 + lam * quad_form(multiply_lower_tri_self_transpose(L_IW), Si);
  
  //matrix[P, P] CBA = multiply_lower_tri_self_transpose(L_BA);
  //matrix[P, P] cor_BA = cor0 + lam * CBA;
}

model {
  L_LKJ ~ lkj_corr_cholesky(eta_prior);
  L_IW ~ inv_wishart_cholesky(P + 1., Id);
  
  //target += barnard_lpdf(CBA);
}

generated quantities {
  vector[P] z = multi_normal_rng(mu, cor_IW);
}


