functions {

    #include copula.stanfunctions

}

data {
  int P; // total number of variables

  int Q; // number of samples

  real copula;  // Copula type
  
  cholesky_factor_corr[P] L_cor;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
}

generated quantities {
  array[Q] vector[P] zrnd;
  vector[Q] logprobs;

  for(i in 1:Q){
    zrnd[i] = copula_rng(copula, corr);
    logprobs[i] = copula_log_pdf(zrnd[i], copula, corr)i;
  }  
}

