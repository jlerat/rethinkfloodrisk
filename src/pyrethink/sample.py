from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

from floodstan.data_processing import univariate2cases
from floodstan.marginals import GEV

STUDENT_DF_MAX = 5.

RHO_MIN_DEFAULT = 0.
RHO_MAX_DEFAULT = 1.

MARGINAL = GEV()

DELTA_DAYS_MAX_DEFAULT = 10


class StanSamplingMultivariate():
    def __init__(self, data, copula,
                 times=None,
                 censors=None,
                 rho_min=RHO_MIN_DEFAULT,
                 rho_max=RHO_MAX_DEFAULT,
                 delta_days_max=DELTA_DAYS_MAX_DEFAULT):

        if copula < 0 or copula > STUDENT_DF_MAX:
            errmsg = f"Expected copula in [0, {STUDENT_DF_MAX}],"\
                     + f" got {copula}."
            raise ValueError(errmsg)

        if rho_min < -1 or rho_min > 1:
            errmsg = f"Expected rho_min in [-1, 1], got {rho_min}."
            raise ValueError(errmsg)

        if rho_max <= rho_min or rho_max > 1:
            errmsg = f"Expected rho_max in ]{rho_min}, 1], got {rho_max}."
            raise ValueError(errmsg)

        self.copula = copula
        self.rho_min = rho_min
        self.rho_max = rho_max

        self.set_data(data, censors)

        self.delta_days_max = delta_days_max
        self.times = times
        self.set_clusters()

        self.set_initial_parameters()

    @property
    def stan_sample_args(self):
        return {}

    def set_data(self, data, censors):
        # Clean data
        if "WATERYEAR" in data:
            data = data.drop("WATERYEAR", axis=1)

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T

        # Eliminates cases where all data are missing
        hasdata = np.any(~np.isnan(data), axis=1)
        data = data[hasdata]
        self.data = data

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

    def set_clusters(self):
        delta_days_max = self.delta_days_max
        N, P = self.data.shape
        clusters = -1 * np.ones((N, P), dtype=int)
        times = self.times
        if times is not None:
            if times.shape != (N, P):
                errmsg = f"Expected times of shape ({N},{P}),"\
                         + f" got {times.shape}."
                raise ValueError(errmsg)

            ordin = times.map(datetime.toordinal)
            for i in range(N):
                ordint = ordin.values[i]
                valid = ordint > 1
                if valid.sum() > 1:
                    cl = fclusterdata(ordint[valid][:, None],
                                      t=delta_days_max,
                                      criterion="distance")
                elif valid.sum() == 1:
                    cl = [1]
                else:
                    continue

                clusters[i, valid] = cl

            import pdb; pdb.set_trace()

            self.clusters = clusters

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
            "clusters": self.clusters,
            "copula": self.copula,
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
