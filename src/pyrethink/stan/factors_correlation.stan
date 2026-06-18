data {
  int P; // Number of variables
  int F; // Number of factors
}

parameters {

  array[P] unit_vector[F + 1] rhos;

}

transformed parameters {
  
  matrix[P, P] corr;
  {
    matrix[P, F] rhos_matrix;
    for(i in 1:P) 
      rhos_matrix[i] = to_row_vector(rhos[i][1:F]);
    
    // Could we get the precision instead of corr?
    corr = rhos_matrix * rhos_matrix';
    corr = diag_matrix(1. - diagonal(corr)) + corr;
  }  

}

