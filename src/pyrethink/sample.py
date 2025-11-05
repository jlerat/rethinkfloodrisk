from itertools import combinations_with_replacement as combsr
import numpy as np
from scipy.stats import norm

from floodstan.data_processing import univariate2cases
from floodstan.marginals import GEV

PCENSOR_DEFAULT = 0.3

ETA_PRIOR_DEFAULT = 0.5

MARGINAL = GEV()


class StanSamplingMultivariate():
    def __init__(self, data, pcensor=PCENSOR_DEFAULT):
        self.pcensor = float(pcensor)
        self.set_data(data)
        self.set_initial_parameters()

    @property
    def stan_sample_args(self):
        return {}

    def set_data(self, data):
        # Clean data
        if "WATERYEAR" in data:
            data = data.drop("WATERYEAR", axis=1)

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T

        hasdata = np.any(~np.isnan(data), axis=1)
        data = data[hasdata]

        self.data = data

        pcensor = self.pcensor
        self.censors = np.zeros(data.shape[1])
        cases = np.zeros_like(data, dtype=int)
        for ivar, vect in enumerate(data.T):
            censor = np.nanpercentile(vect, pcensor * 100) - 1e-10
            icases, vect, censor = univariate2cases(vect, censor)
            cases[icases.i11, ivar] = 1  # Observed
            cases[icases.i21, ivar] = 2  # Censored
            cases[icases.i31, ivar] = 3  # Missing
            self.censors[ivar] = censor

        # Need to add 1 because stan array indexes
        # start at 1, not 0.
        self.idx_obs = np.array(np.where(cases == 1)).T + 1
        self.idx_cens = np.array(np.where(cases == 2)).T + 1
        self.idx_miss = np.array(np.where(cases == 3)).T + 1

    def set_initial_parameters(self):
        # GEV parameters and priors
        P = self.data.shape[1]

        z = np.nan * np.zeros_like(self.data)
        gparams = np.zeros((P, 3))

        for ivar in range(P):
            vect = self.data[:, ivar]
            iok = ~np.isnan(vect)
            vect = vect[iok]
            MARGINAL.fit_lh_moments(vect, eta=2)
            locn, logscale, shape1 = MARGINAL.params
            shape1 = -1e-2  # To avoid boundary problems with GEV
            gparams[ivar] = [locn, logscale, shape1]
            z[iok, ivar] = norm.ppf(MARGINAL.cdf(vect))

        # Compute pairwise covariance matrix
        cov = np.eye(P)
        for i1, i2 in combsr(range(P), 2):
            z12 = z[:, [i1, i2]]
            iok = np.all(~np.isnan(z12), axis=1)
            co = np.cov(z12[iok].T)[0, 1]
            cov[i1, i2] = co
            cov[i2, i1] = co

        # Modify covariance matrix
        # to make sure it's positive definite
        eig, M = np.linalg.eig(cov)
        eig = np.maximum(eig, eig.max() * 1e-3)
        cov = M @ np.diag(eig) @ M.T

        # Compute correlation matrix
        sigs = np.sqrt(np.diag(cov))[:, None]
        cor = (1. / sigs) * cov * (1. / sigs.T)

        # Compute cholesky decomposition
        L_cor = np.linalg.cholesky(cor)

        # latent variables
        nmiss = len(self.idx_miss)
        wlat_miss = np.random.uniform(0, 1, size=nmiss)

        ncens = len(self.idx_cens)
        wlat_cens = np.random.uniform(0, 1, size=ncens)

        self.initial_parameters = {
            "ylocn": gparams[:, 0],
            "ylogscale": gparams[:, 1],
            "yshape1": gparams[:, 2],
            "L_cor": L_cor,
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
            "eta_prior": ETA_PRIOR_DEFAULT
        }
        return dd
