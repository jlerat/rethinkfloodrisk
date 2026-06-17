data {
  int P; // total number of variables
  int F; // Number of factors
}

generated quantities {
    matrix<lower=-1, upper=1>[P, F] rhos;
    matrix<lower=-1, upper=1>[P, F] rhos_unit;
    vector<lower=0, upper=1>[P] rhos_scalings; 
    for(i in 1:P) {
        for(j in 1:F)
            rhos_unit[i, j] = uniform_rng(-1, 1);
        
        rhos_scalings[i] = uniform_rng(0, 1);
        rhos[i] = (rhos_scalings[i] / sqrt(dot_self(rhos_unit[i]))) * rhos_unit[i];
    }
    
    matrix[P, P] corr = add_diag(rhos' * rhos, 1. - rows_dot_self(rhos));
}

