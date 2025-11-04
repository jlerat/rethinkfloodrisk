functions {

    #include marginal.stanfunctions

}


data {
  int Q; // number of sample
  real tau;
  real<lower=0> alpha;
  real<lower=-1.5, upper=1.5> kappa;
}

generated quantities {
  vector[Q] u = linspaced_vector(Q, 1./Q, 1 - 1./Q);
  vector[Q] uu;
  vector[Q] qq;
  vector[Q] lp;

  for(i in 1:Q) {
    real q = gev_quantile(u[i], tau, alpha, kappa);
    qq[i] = q;
    uu[i] = gev_cdf(q | tau, alpha, kappa);
    lp[i] = gev_lpdf(q | tau, alpha, kappa);
  }  
}

