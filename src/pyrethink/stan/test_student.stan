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
}

generated quantities {
  matrix[N, P] z;
  for(i in 1:N)
    for(j in 1:P)
      z[i, j] = student_t_quantile(p[i], df[j]);
}

