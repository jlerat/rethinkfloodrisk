functions {

    #include copula.stanfunctions

}

data {
  int N; // total number of values
  int P; // number of degrees of freedom
  
}

transformed data {
  vector[N] p = linspaced_vector(N, 1./N, 1. - 1./N);
  vector[P] df = linspaced_vector(P, 0.5, 5.);

  matrix[3, 3] corr = [[1, 0.9, 0.9], [0.9, 1., 0.9], [0.9, 0.9, 1.]]; 
}

generated quantities {
  vector[N] p0;
  vector[N] zn;
  vector[N] pn;
  matrix[N, 3] znr;
  
  matrix[N, P] zs;
  matrix[N, P] ps;
  matrix[N, 3] zsr;
  
  for(i in 1:N) {
        p0[i] = p[i];
        zn[i] = copula_marginal_quantile(p[i], 0.);
        pn[i] = copula_marginal_prob(zn[i], 0.);
        znr[i, :] = to_row_vector(copula_rng(0., corr));

        for(j in 1:P) {
          zs[i, j] = copula_marginal_quantile(p[i], df[j]);
          ps[i, j] = copula_marginal_prob(zs[i, j], df[j]);
        }  
        
        // Sample from student with df=4
        zsr[i, :] = to_row_vector(copula_rng(4., corr));
    }
}

