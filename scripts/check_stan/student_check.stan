functions {

    #include student.stanfunctions

}

data {
  int N; // total number of values
  vector[N] u; // Prob data
  
  int P; // Number of degrees of freedom
  vector[P] df; // degrees of freedom values
}

transformed data {
}

generated quantities {
    matrix[N, P] z;

    for (i in  1:N) {
        for(j in 1:P) {
            z[i, j] = student_t_qf(u[i], df[j]); 
        }
    }    
}

