data {
  int N; // total number of values
  int P; // total number of variables

  // Indexing - obs data
  int<lower=10> Nobs;
  array[Nobs, 2] int idx_obs;

  // Indexing - missing data
  int<lower=0> Nmiss;
  array[Nmiss, 2] int idx_miss;

  // Indexing - censored data
  int<lower=0> Ncens;
  array[Ncens, 2] int idx_cens;
}

generated quantities {
  // standard normal
  array[N] row_vector[P] z;

  // Set missing latent
  for(i in 1:Nmiss) {
    array[2] int imiss = idx_miss[i];
    z[imiss[1], imiss[2]] = 3.0;
  }

  // Set censored latent using stan upper bound
  // constraint formula
  for(i in 1:Ncens) {
    array[2] int icens = idx_cens[i];
    z[icens[1], icens[2]] = 2.0;
  }

  // Transform data to uniform marginals
  for(i in 1:Nobs) {
    int ipt = idx_obs[i][1];
    int ivar = idx_obs[i][2];
    z[ipt, ivar] = 1.0;
  }

}

