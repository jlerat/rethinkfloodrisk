data {
  int P; // Number of variables
  int F; // Number of factors
}

parameters {

  array[P] vector<lower=0.>[F + 1] zrhos;

}

transformed parameters {
  
  matrix[P, P] corr;
  {
    matrix[P, F] rhos_matrix;
    for(i in 1:P) 
      rhos_matrix[i] = to_row_vector(zrhos[i][1:F] / sqrt(dot_self(zrhos[i])));
    
    // Could we get the precision instead of corr?
    corr = rhos_matrix * rhos_matrix';
    corr = diag_matrix(1. - diagonal(corr)) + corr;
  }  

}

model {
    for(i in 1:P)
        zrhos[i] ~ std_normal();

}


