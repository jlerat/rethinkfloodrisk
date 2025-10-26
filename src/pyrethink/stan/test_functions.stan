functions {

    #include marginal.stanfunctions

}


data {
  int Q; // number of sample
  real tau;
  real<lower=0> alpha;
}

generated quantities {
  vector[Q] u = linspaced_vector(Q, 1./Q, 1 - 1./Q);
  vector[Q] z;
  vector[Q] gq;
  for(i in 1:Q) {
    z[i] = inv_Phi_approx(u[i]);
    gq[i] = gumbel_quantile(u[i], tau, alpha);
  }  
}

