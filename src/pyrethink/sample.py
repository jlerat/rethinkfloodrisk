from itertools import combinations

import numpy as np

from floodstan.data_processing import univariate2cases
from floodstan.marginals import GEV

from pyrethink import copulas
from pyrethink.partitions import Partitions

RHO_MIN_DEFAULT = 0.
RHO_MAX_DEFAULT = 1.

MARGINAL = GEV()

DELTA_DAYS_MAX_DEFAULT = 10


class StanSamplingMultivariate():
    def __init__(self, data, day_of_year,
                 copula_type,
                 copula_shape,
                 censors=None,
                 skip_clusters=False,
                 rho_min=RHO_MIN_DEFAULT,
                 rho_max=RHO_MAX_DEFAULT,
                 delta_days_max=DELTA_DAYS_MAX_DEFAULT):

        # Check data
        copulas.check_copula(copula_type, copula_shape)

        if rho_min < -1 or rho_min > 1:
            errmsg = f"Expected rho_min in [-1, 1], got {rho_min}."
            raise ValueError(errmsg)

        if rho_max <= rho_min or rho_max > 1:
            errmsg = f"Expected rho_max in ]{rho_min}, 1], got {rho_max}."
            raise ValueError(errmsg)

        self.copula_type = copula_type
        self.copula_shape = copula_shape
        self.rho_min = rho_min
        self.rho_max = rho_max

        self.set_data(data, day_of_year, censors)

        self.delta_days_max = delta_days_max
        self.set_clusters(skip_clusters)
        self.set_initial_parameters()

    @property
    def stan_sample_args(self):
        return {}

    def set_data(self, data, day_of_year, censors):
        data = np.atleast_2d(data)
        day_of_year = np.atleast_2d(day_of_year)

        if data.shape[0] == 1:
            data = data.T
            day_of_year = day_of_year.T

        if data.shape != day_of_year.shape:
            errmsg = "'data' and 'day_of_year' do not have the same shape."
            raise ValueError(errmsg)

        # Eliminates cases where all data are missing
        hasdata = np.any(~np.isnan(data), axis=1)
        data = data[hasdata]

        self.data = data
        self.day_of_year = day_of_year[hasdata]

        # Check censors
        nvars = data.shape[1]
        if censors is None:
            censors = np.zeros(nvars)
        else:
            censors = np.array(censors)

        if censors.shape != (nvars, ):
            errmsg = f"Expected censors as 1D array of length {nvars},"\
                     + f"got an array of shape {censors.shape}."
            raise ValueError(errmsg)

        self.censors = censors

        cases = np.zeros_like(data, dtype=int)
        for ivar, vect in enumerate(data.T):
            icases, vect, censor = univariate2cases(vect, censors[ivar])
            cases[icases.i11, ivar] = 1  # Observed
            cases[icases.i21, ivar] = 2  # Censored
            cases[icases.i31, ivar] = 3  # Missing

        # Need to add 1 because stan array indexes
        # start at 1, not 0.
        self.idx_obs = np.array(np.where(cases == 1)).T + 1
        self.idx_cens = np.array(np.where(cases == 2)).T + 1
        self.idx_miss = np.array(np.where(cases == 3)).T + 1

    def set_clusters(self, skip_clusters):
        dmax = self.delta_days_max
        N, P = self.data.shape

        clusters = np.zeros((N, P, P), dtype=int)
        miss = np.zeros(N, dtype=int)
        counts = np.zeros(N, dtype=int)
        partitions_id = np.zeros(N, dtype=int)
        insame = np.zeros((N, P * (P - 1) // 2), dtype=int)
        day_of_year = self.day_of_year

        parts = Partitions(P)
        parts_psc = parts.pair_in_same_cluster

        for i in range(N):
            doy = day_of_year[i]
            is_valid = np.where(doy >= 0)[0]
            clusters[i, :, doy < 0] = -1

            doy = doy[is_valid]
            if len(doy) == 1:
                clusters[i, 0, is_valid] = 1
            else:
                if skip_clusters:
                    clust = np.zeros(len(doy), dtype=int)
                else:
                    rk = np.argsort(np.argsort(doy))
                    doys = np.sort(doy)
                    clust = np.cumsum(np.insert(np.diff(doys), 0, 0) > dmax)
                    clust = clust[rk]

                for icl, cl in enumerate(np.unique(clust)):
                    isin = is_valid[cl == clust]
                    clusters[i, icl, isin] = 1

            # count missing events
            miss[i] = P - len(is_valid)

            cl = clusters[i]
            counts[i] = np.any(cl[:, is_valid] > 0, axis=1).sum()

            # check if pairs are in the same cluster
            ins = [[int(comb[a]*comb[b] and comb[a] >= 0 and comb[b] >= 0)
                    for ic, (a, b) in enumerate(combinations(range(P), 2))]
                   for comb in cl]
            insame[i] = (np.sum(ins, axis=0) > 0).astype(int)

            if miss[i] == 0:
                delta = parts_psc - insame[[i]]
                pid = np.where(np.abs(delta).sum(axis=1) == 0)[0][0]
            else:
                pid = -1
            partitions_id[i] = pid

        self.clusters = clusters
        self.clusters_counts = counts
        self.clusters_missing = miss
        self.pair_in_same_cluster = insame
        self.partitions_id = partitions_id

    def set_initial_parameters(self):
        # GEV parameters and priors
        P = self.data.shape[1]

        gparams = np.zeros((P, 3))
        for ivar in range(P):
            vect = self.data[:, ivar]
            iok = ~np.isnan(vect)
            vect = vect[iok]
            MARGINAL.fit_lh_moments(vect, eta=2)
            locn, logscale, shape1 = MARGINAL.params
            shape1 = -1e-2  # To avoid boundary problems with GEV
            gparams[ivar] = [locn, logscale, shape1]

        # Inverse Wishart cholesky -> initialised in the middle
        # of rho_min and rho_max
        L_IW = np.eye(P)

        # latent variables
        nmiss = len(self.idx_miss)
        wlat_miss = np.random.uniform(0, 1, size=nmiss)

        ncens = len(self.idx_cens)
        wlat_cens = np.random.uniform(0, 1, size=ncens)

        self.initial_parameters = {
            "ylocn": gparams[:, 0],
            "ylogscale": gparams[:, 1],
            "yshape1": gparams[:, 2],
            "L_IW": L_IW,
            "wlat_cens": wlat_cens,
            "wlat_miss": wlat_miss
        }

    def to_dict(self):
        dd = {
            "N": self.data.shape[0],
            "P": self.data.shape[1],
            "y": self.data,
            "Nobs": len(self.idx_obs),
            "idx_obs": self.idx_obs,
            "Ncens": len(self.idx_cens),
            "idx_cens": self.idx_cens,
            "Nmiss": len(self.idx_miss),
            "idx_miss": self.idx_miss,
            "clusters": self.clusters.astype(int),
            "clusters_counts": self.clusters_counts.astype(int),
            "partitions_id": self.partitions_id,
            "copula": float(self.copula),
            "ylocn_prior": MARGINAL.locn_prior.to_list(),
            "ylogscale_prior": MARGINAL.logscale_prior.to_list(),
            "yshape1_prior": MARGINAL.shape1_prior.to_list(),
            "locn_lower": float(MARGINAL.locn_prior.lower),
            "locn_upper": float(MARGINAL.locn_prior.upper),
            "logscale_lower": float(MARGINAL.logscale_prior.lower),
            "logscale_upper": float(MARGINAL.logscale_prior.upper),
            "shape1_lower": float(MARGINAL.shape1_prior.lower),
            "shape1_upper": float(MARGINAL.shape1_prior.upper),
            "censors": self.censors,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max
        }
        return dd
