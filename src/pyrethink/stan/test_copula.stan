functions {

    #include copula.stanfunctions

}

data {
  int N; // total number of values
  int P; // number of degrees of freedom
}

transformed data {
  vector[N] p = linspaced_vector(N, 1./N, 1. - 1./N);
  vector[P] df = linspaced_vector(P, 2.1, 5.);

  matrix[3, 3] corr = [[1, 0.9, 0.9], [0.9, 1., 0.9], [0.9, 0.9, 1.]]; 
}

generated quantities {
  vector[N] p0;
  vector[N] zn;
  vector[N] pn;
  vector[N] ljn;
  vector[N] lpdfn;
  matrix[N, 3] znr;
  
  matrix[N, P] zs;
  matrix[N, P] ps;
  matrix[N, P] ljs;
  vector[N] lpdfs;
  matrix[N, 3] zsr;

  real cnorm = 0;
  real dfr = 4;
  vector[3] z;
  
  for(i in 1:N) {
        p0[i] = p[i];
        zn[i] = copula_marginal_quantile(p[i], cnorm);
        pn[i] = copula_marginal_prob(zn[i], cnorm);
        ljn[i] = copula_marginal_quantile_log_jac(zn[i], cnorm);

        z = copula_rng(cnorm, corr);
        znr[i, :] = to_row_vector(z);
        lpdfn[i] = copula_log_pdf(z, cnorm, corr);

        for(j in 1:P) {
          zs[i, j] = copula_marginal_quantile(p[i], df[j]);
          ps[i, j] = copula_marginal_prob(zs[i, j], df[j]);
          ljs[i, j] = copula_marginal_quantile_log_jac(zn[i], df[j]);
        }  

        // Sample from student with df=3
        z = copula_rng(dfr, corr);
        zsr[i, :] = to_row_vector(z);
        lpdfs[i] = copula_log_pdf(z, dfr, corr);
    }
}

