import math
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mvt
from scipy.optimize import minimize_scalar


MARGINAL_EXCEEDANCE_SCORE_KINDS = ["AND", "OR", "KENDALL"]


class KendallFunctionIndependence():
    def __init__(self, nstations):
        self.nstations = nstations

    def cdf(self, t):
        return 1 - gamma.cdf(np.log(1./t), a=self.nstations)

    def ppf(self, p):
        return np.exp(-gamma.ppf(1 - p, a=self.nstations))


class GaussianFactorCopulaCDF():
    """ Compute the CDF for a Gaussian factor copula such that
        Ui = Phi(Zi)
        Zi = sqrt(rho) V + sqrt(1 - rho) Ei
        with V ~ N(0,1)  Ei ~ N(0,1)

        Hence Z ~ N(0, Sigma)
        with Sigma = (1-rho) eye + rho ones

        Computation is done in an arbitrary number of dimensions.
    """
    def __init__(self, nstations, rho, napprox=500):
        self.nstations = nstations
        if rho < 0 or rho >= 1:
            raise ValueError(f"Expected rho in [0, 1[, got {rho}")
        self.rho = rho
        self.napprox = napprox

        # Required for the cdf integration
        mean = np.zeros(nstations)
        cov = (1 - rho) * np.eye(nstations) \
            + rho * np.ones((nstations, nstations))

        self.rv = mvt(mean=mean, cov=cov)
        self.x_vect = np.zeros((1, nstations))

        # Required for the factor integration
        eps = 5e-1 / napprox
        u = np.linspace(eps, 1 - eps, napprox)
        self.v = norm.ppf(u)
        self.du = u[1] - u[0]
        self.buf = np.zeros((1, nstations, napprox))

    @property
    def sqr(self):
        return math.sqrt(self.rho)

    @property
    def csqr(self):
        return math.sqrt(1 - self.rho)

    def cdf(self, x):
        if x.ndim != 2:
            errmsg = "Expected x of dim 2"
            raise ValueError(errmsg)

        if x.shape[0] != 1:
            errmsg = "Expected x.shape[0] to be 1"
            raise ValueError(errmsg)

        if self.napprox > 0:
            # CDF for a factor copula is
            # p(X1<x1, ..., Xn<xn) =
            #      int(prod(Phi(xi - sqrt(rho) v) / sqrt(1 - rho)) dv,
            #          v=-infty,
            #          v=+infty)
            np.add(x[:, :, None], -self.sqr * self.v[None, None, :],
                   out=self.buf)
            np.multiply(self.buf, 1./self.csqr, out=self.buf)
            return norm.cdf(self.buf).prod(axis=1).sum(axis=-1)[0] * self.du
        else:
            return self.rv.cdf(x)

    def cdf_main_diagonal(self, x):
        self.x_vect.fill(x)
        return self.cdf(self.x_vect)

    def sample(self, nsamples):
        v = np.random.normal(size=nsamples)
        eps = np.random.normal(size=(nsamples, self.nstations))
        return norm.cdf(self.sqr * v[:, None] + self.csqr * eps)


class MarginalExceedanceScore():
    def __init__(self, kind, cdf_computer,
                 empirical=False, nsamples=10000,
                 logger=None):
        if kind not in MARGINAL_EXCEEDANCE_SCORE_KINDS:
            txt = "/".join(MARGINAL_EXCEEDANCE_SCORE_KINDS)
            errmsg = f"Expected kind in {txt}, got {kind}."
            raise ValueError(errmsg)
        self.kind = kind
        self.empirical = empirical
        self.cdf_computer = cdf_computer
        self.logger = logger

        self.nsamples = nsamples
        self.u_smp = cdf_computer.sample(nsamples)

        self.kfi = KendallFunctionIndependence(cdf_computer.nstations)

        if kind == "KENDALL":
            self.empirical_kendall = self.compute_empirical_kendall()
        else:
            self.empirical_kendall = None

    def compute_empirical_kendall(self):
        u_smp = self.u_smp
        N, P = u_smp.shape
        probs = np.zeros(N)
        logger = self.logger
        for i in range(N):
            if i % 1000 == 0 and logger is not None:
                logger.info(f"Empirical kendall step {i:6,d} / {N:6,d}")
            probs[i] = np.all(np.greater(u_smp[[i]], u_smp), axis=1).sum() / N

        t = np.arange(1, N + 1) / N
        Kc = np.zeros_like(t)
        for i, tt in enumerate(t):
            Kc[i] = (probs < tt).sum() / N
        return pd.DataFrame({"t": t, "Kc": Kc})

    def objective_function(self, u, lmaep):
        kendall = self.empirical_kendall
        if not self.empirical:
            x = norm.ppf(u)
            match self.kind:
                case "AND":
                    c = self.cdf_computer.cdf_main_diagonal(x)
                case "OR":
                    c = 1 - self.cdf_computer.cdf_main_diagonal(-x)
                case "KENDALL":
                    c0 = self.cdf_computer.cdf_main_diagonal(x)
                    c = 1 - np.interp(c0, kendall.Kc, kendall.t)
        else:
            u_smp = self.u_smp
            nsmp = u_smp.shape[0]
            match self.kind:
                case "AND":
                    c = np.all(u_smp < u, axis=1).sum() / nsmp
                case "OR":
                    c = 1 - np.all(u_smp < 1 - u, axis=1).sum() / nsmp
                case "KENDALL":
                    pass

        err = (math.log(c) - lmaep)**2
        return err

    def compute_bounds(self, maep):
        nsta = self.nstations
        match self.kind:
            case "AND":
                ua = maep
                ub = maep**(1./nsta)
            case "OR":
                ua = 1 - (1 - maep)**(1./nsta)
                ub = maep
            case "KENDALL":
                ua = maep
                ub = self.kfi.ppf(1 - maep)
        return ua, ub

    def compute_score(self, maep):
        bounds = self.compute_bounds(maep)
        lmaep = math.log(maep)
        self.opt = minimize_scalar(self.objective_function, bracket=bounds,
                                   bounds=bounds, args=(lmaep,))
        return self.opt.x
