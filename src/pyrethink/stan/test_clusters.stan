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
    int Ncheck = size(clusters);
    int Pcheck = size(clusters[1, 1, :]);

    array[N, P, P + 1] int indexes = cluster_data_processing(clusters, clusters_counts);
}
