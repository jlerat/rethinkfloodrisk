import numpy as np
from scipy.stats import norm

from floodstan.data_processing import univariate2cases
from floodstan.marginals import Gumbel
from floodstan.sample import STAN_SAMPLE_ARGS

PCENSOR_DEFAULT = 0.3

ETA_PRIOR_DEFAULT = 2

GUMBEL_MARGINAL = Gumbel()


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
        # Gumbel parameters and priors
        P = self.data.shape[1]

        z = np.nan * np.zeros_like(self.data)
        gparams = np.zeros((P, 2))

        for ivar in range(P):
            vect = self.data[:, ivar]
            iok = ~np.isnan(vect)
            vect = vect[iok]
            GUMBEL_MARGINAL.fit_lh_moments(vect, eta=2)
            gparams[ivar] = GUMBEL_MARGINAL.params[:2]
            z[iok, ivar] = norm.ppf(GUMBEL_MARGINAL.cdf(vect))

        iall = np.all(~np.isnan(z), axis=1)
        cor = np.corrcoef(z[iall].T)
        L_cor = np.linalg.cholesky(cor)

        # latent variables
        zmiss = np.random.uniform(-1., 1., len(self.idx_miss))
        wcens = np.random.uniform(-1., 1., len(self.idx_cens))

        self.initial_parameters = {
            "ylocn": gparams[:, 0],
            "ylogscale": gparams[:, 1],
            "L_cor": L_cor,
            "wcens": wcens,
            "zmiss": zmiss
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
            "ylocn_prior": GUMBEL_MARGINAL.locn_prior.to_list(),
            "ylogscale_prior": GUMBEL_MARGINAL.logscale_prior.to_list(),
            "locn_lower": float(GUMBEL_MARGINAL.locn_prior.lower),
            "locn_upper": float(GUMBEL_MARGINAL.locn_prior.upper),
            "logscale_lower": float(GUMBEL_MARGINAL.logscale_prior.lower),
            "logscale_upper": float(GUMBEL_MARGINAL.logscale_prior.upper),
            "censors": self.censors,
            "eta_prior": ETA_PRIOR_DEFAULT
        }
        return dd
