functions {

    #include clusters.stanfunctions

}



data {
    int N; // total number of values
    int P; // total number of variables
    
    // Cluster observed data
    array[N, P, P] int<lower=0, upper=1> clusters;
    array[N] int<lower=0, upper=P> clusters_counts;
}

generated quantities {
    array[N, P + 1, P] int indexes = cluster_data_processing(clusters, clusters_counts);

    //array[N, P] int<lower=0, upper=P> clusters_nbstations = rep_array(0, N, P);
    //array[N, P, P] int<lower=0, upper=P> clusters_indexes = rep_array(0, N, P, P);
    //int nbsta;
    //for(i in 1:N) {
    //  for(icl in 1:clusters_counts[i]) {
    //      nbsta = 1;
    //      for(j in 1:P) {
    //          if(clusters[i, icl, j] == 1) {
    //              clusters_indexes[i, icl, nbsta] = j;
    //              nbsta += 1;
    //          }
    //      }
    //      clusters_nbstations[i, icl] = nbsta - 1;
    //  }
    //}
}
