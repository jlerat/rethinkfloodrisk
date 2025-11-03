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
  int P; // total number of variables
  
  vector[P] ylocn;
  vector[P] ylogscale;
  vector[P] yshape1;

  cholesky_factor_corr[P] L_cor;
}

transformed data {
  vector[P] yscale = exp(ylogscale);
  row_vector[P] zero_mean = zeros_row_vector(P);
}

generated quantities {
    vector[P] zrnd = multi_normal_cholesky_rng(zero_mean, L_cor);
    
    vector[P] yrnd;
    for(i in 1:P) {
       yrnd[i] = gev_quantile(Phi(zrnd[i]), ylocn[i], yscale[i], yshape1[i]);
    }
    
}

