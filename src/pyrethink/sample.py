import numpy as np

from floodstan.data_processing import univariate2cases
from floodstan import marginals

from pyrethink import copulas

RHO_MIN_DEFAULT = 0.
RHO_MAX_DEFAULT = 1.

DELTA_DAYS_MAX_DEFAULT = 10

MARGINALS_ALLOWED = ["GEV"]


class StanSamplingMultivariate():
    def __init__(self, data, day_of_year,
                 copula_name,
                 copula_shape,
                 censors=None,
                 marginal_name="GEV",
                 rho_min=RHO_MIN_DEFAULT,
                 rho_max=RHO_MAX_DEFAULT,
                 delta_days_max=DELTA_DAYS_MAX_DEFAULT):

        # Check data
        nstations = data.shape[1]

        if rho_min < -1 or rho_min > 1:
            errmsg = f"Expected rho_min in [-1, 1], got {rho_min}."
            raise ValueError(errmsg)

        if rho_max <= rho_min or rho_max > 1:
            errmsg = f"Expected rho_max in ]{rho_min}, 1], got {rho_max}."
            raise ValueError(errmsg)

        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)

        # Configure marginal and copula
        self._copula = copulas.factory(copula_name, nstations, copula_shape)
        self.copula_shape = float(copula_shape)

        if marginal_name not in MARGINALS_ALLOWED:
            txt = "/".join(MARGINALS_ALLOWED)
            errmsg = f"Expected marginal in {txt}, got {marginal_name}."
            raise ValueError(errmsg)
        self._marginal = marginals.factory(marginal_name)

        self.set_data(data, day_of_year, censors)
        self.delta_days_max = delta_days_max
        self.set_initial_parameters()

    @property
    def copula_id(self):
        nm = self.copula.name
        return copulas.COPULA_NAMES.index(nm)

    @property
    def copula(self):
        return self._copula

    @property
    def marginal_id(self):
        nm = self.marginal.name
        return MARGINALS_ALLOWED.index(nm)

    @property
    def marginal(self):
        return self._marginal

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

    def set_initial_parameters(self):
        # marginal parameters and priors
        marginal = self.marginal
        P = self.data.shape[1]

        gparams = np.zeros((P, 3))
        for ivar in range(P):
            vect = self.data[:, ivar]
            iok = ~np.isnan(vect)
            vect = vect[iok]
            marginal.fit_lh_moments(vect, eta=2)
            locn, logscale, shape1 = marginal.params
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
        marginal = self.marginal

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
            "marginal_id": int(self.marginal_id),
            "copula_id": int(self.copula_id),
            "copula_shape": float(self.copula_shape),
            "ylocn_prior": marginal.locn_prior.to_list(),
            "ylogscale_prior": marginal.logscale_prior.to_list(),
            "yshape1_prior": marginal.shape1_prior.to_list(),
            "locn_lower": float(marginal.locn_prior.lower),
            "locn_upper": float(marginal.locn_prior.upper),
            "logscale_lower": float(marginal.logscale_prior.lower),
            "logscale_upper": float(marginal.logscale_prior.upper),
            "shape1_lower": float(marginal.shape1_prior.lower),
            "shape1_upper": float(marginal.shape1_prior.upper),
            "censors": self.censors,
            "rho_min": float(self.rho_min),
            "rho_max": float(self.rho_max)
        }
        return dd
