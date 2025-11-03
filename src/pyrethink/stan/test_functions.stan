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
  vector[Q] z;
  vector[Q] gq;
  for(i in 1:Q) {
    z[i] = inv_Phi(u[i]);
    gq[i] = gev_quantile(u[i], tau, alpha, kappa);
  }  
}

