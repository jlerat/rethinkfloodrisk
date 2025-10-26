data {
  int P; // total number of variables

  int Q; // number of samples
  
  cholesky_factor_corr[P] L_cor;
}

transformed data {
  row_vector[P] zero_mean = zeros_row_vector(P);
}

generated quantities {
  array[Q] vector[P] zrnd;
  for(i in 1:Q){
    zrnd[i] = multi_normal_cholesky_rng(zero_mean, L_cor);
  }  
}

