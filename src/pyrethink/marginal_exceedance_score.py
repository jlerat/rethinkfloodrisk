import math
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mvt
from scipy.optimize import minimize_scalar

from hydrodiy.stat import sutils

COPULAS = ["Independence", "Gaussian",
           "GaussianOneFactor", "Gumbel"]

MARGINAL_EXCEEDANCE_SCORE_KINDS = ["AND", "OR", "KENDALL"]


def copula_factory(name, nstations, *args, **kwargs):
    if name not in COPULAS:
        txt = "/".join(COPULAS)
        errmsg = f"Expected {name} in '{txt}', got {name}."
        raise ValueError(errmsg)

    if name == "Independence":
        return IndependenceCopula(nstations)
    elif name == "Gaussian":
        return GaussianCopula(nstations, *args, **kwargs)
    elif name == "GaussianOneFactor":
        return GaussianOneFactorCopula(nstations, *args, **kwargs)
    elif name == "Gumbel":
        return GumbelCopula(nstations, *args, **kwargs)


class Copula():
    def __init__(self, name, nstations, nsamples=20000, logger=None):
        self.name = name
        self.nstations = nstations
        self.logger = logger

        self.nsamples = nsamples
        self._random_samples = None

        self._kendall_function_data = None
        self._u_data = None

    def __str__(self):
        txt = f"{self.name} copula {self.nstations} dimensions."
        return txt

    @property
    def kendall_function_data(self):
        kdd = self._kendall_function
        if kdd is None:
            logger = self.logger
            if logger is not None:
                logger.info("Computing kendall function")

            u = self.random_samples
            N = len(u)
            probs = sutils.multivariate_dominance(u) / N
            t = np.arange(1, N + 1) / N
            Kc = np.zeros_like(t)
            for i, tt in enumerate(t):
                Kc[i] = (probs < tt).sum() / N

            kdd = pd.DataFrame({"t": t, "Kc": Kc})
            self._kendall_function_data = kdd

        return kdd

    @property
    def random_samples(self):
        if self._random_samples is None:
            self._random_samples = self.sample(self.nsamples)
        return self._random_samples

    def cdf(self, u):
        raise NotImplementedError

    def cdf_main_diagonal(self, u):
        if self._u_data is None:
            u_data = np.repeat(np.atleast_2d(u),
                               self.nstations, 1)
        else:
            u_data = self._u_data
            if isinstance(u, np.ndarray):
                if u_data.shape[0] == len(u):
                    np.copyto(u_data, u[:, None])
                else:
                    u_data = np.repeat(np.atleast_2d(u),
                                       self.nstations, 1)
            else:
                if u_data.shape[0] == 1:
                    u_data.fill(u)
                else:
                    u_data = u * np.ones((1, self.nstations))

        self._u_data = u_data
        return self.cdf(u_data)

    def sample(self, nsamples):
        raise NotImplementedError

    def kendall_function(self, t):
        kdd = self.kendall_function_data
        return np.interp(t, kdd.t, kdd.Kc)

    def inverse_kendall_function(self, p):
        kdd = self.kendall_function_data
        return np.interp(p, kdd.Kc, kdd.t)


class IndependenceCopula(Copula):
    def __init__(self, nstations, nsamples=20000):
        super(IndependenceCopula, self).__init__("Independence",
                                                 nstations,
                                                 nsamples=nsamples)

    def cdf(self, u):
        return np.prod(u, axis=1)

    def sample(self, nsamples):
        return np.random.uniform(0, 1, (nsamples, self.nstations))

    def kendall_function(self, t):
        return 1 - gamma.cdf(np.log(1./t), a=self.nstations)

    def inverse_kendall_function(self, p):
        return np.exp(-gamma.ppf(1 - p, a=self.nstations))


class GaussianCopula(Copula):
    def __init__(self, nstations, cov, nsamples=20000):
        super(GaussianCopula, self).__init__("Gaussian",
                                             nstations,
                                             nsamples)
        if cov.shape != (nstations, nstations):
            errmsg = f"Expected cov of shape ({nstations},{nstations})."\
                     + f" Got {cov.shape}."
            raise ValueError(errmsg)

        mean = np.zeros(nstations)
        self.rv = mvt(mean=mean, cov=cov)

    def cdf(self, u):
        if u.ndim != 2:
            errmsg = "Expected x of dim 2"
            raise ValueError(errmsg)

        return self.rv.cdf(norm.ppf(u))

    def sample(self, nsamples):
        return self.rv.rvs(size=nsamples)


class GaussianOneFactorCopula(Copula):
    """ Compute the CDF for a Gaussian factor copula such that
        Ui = Phi(Zi)
        Zi = sqrt(rho) V + sqrt(1 - rho) Ei
        with V ~ N(0,1)  Ei ~ N(0,1)

        Hence Z ~ N(0, Sigma)
        with Sigma = (1-rho) eye + rho ones

        Computation is done in an arbitrary number of dimensions.
    """
    def __init__(self, nstations, rho, napprox=500, nsamples=20000):
        super(GaussianOneFactorCopula, self).__init__("GaussianOneFactor",
                                                      nstations,
                                                      nsamples)
        if rho < 0 or rho >= 1:
            raise ValueError(f"Expected rho in [0, 1[, got {rho}")
        self.rho = rho
        self.napprox = napprox

        # Required for the cdf integration
        mean = np.zeros(nstations)
        cov = (1 - rho) * np.eye(nstations) \
            + rho * np.ones((nstations, nstations))
        self.rv = mvt(mean=mean, cov=cov)

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

    def cdf(self, u):
        if u.ndim != 2:
            errmsg = "Expected u of dim 2"
            raise ValueError(errmsg)

        if self.buf.shape[0] != len(u):
            self.buf = np.zeros((len(u), self.nstations, self.napprox))

        # CDF for a factor copula is
        # p(X1<x1, ..., Xn<xn) =
        #      int(prod(Phi(xi - sqrt(rho) v) / sqrt(1 - rho)) dv,
        #          v=-infty,
        #          v=+infty)
        np.add(norm.ppf(u)[:, :, None],
               -self.sqr * self.v[None, None, :],
               out=self.buf)
        np.multiply(self.buf, 1./self.csqr, out=self.buf)
        return norm.cdf(self.buf).prod(axis=1).sum(axis=-1) * self.du

    def sample(self, nsamples):
        v = np.random.normal(size=nsamples)
        eps = np.random.normal(size=(nsamples, self.nstations))
        return norm.cdf(self.sqr * v[:, None] + self.csqr * eps)


class GumbelCopula(Copula):
    """ Compute the CDF for an Archimedean copula
        C(u) = phi_inv(sum phi(u_i))

        phi(x) = -log(x)**theta

    """
    def __init__(self, nstations, rho, nsamples):
        super(GumbelCopula, self).__init__("Gumbel",
                                           nstations,
                                           nsamples)
        self.theta = 1 / (1 - rho)
        self.phi = lambda x: (-np.log(x))**self.theta
        self.phi_inv = lambda y: np.exp(-y**(1./self.theta))
        self.x_vect = np.ones((1, nstations))

    def kendall_function(self, t):
        """ See
        Barbe, P., Genest, C., Ghoudi, K., & Rémillard, B. (1996).
        On Kendall’s Process. Journal of Multivariate Analysis,
        58(2), 197–229. https://doi.org/10.1006/jmva.1996.0048
        Page 205
        """
        Kc = t
        for i in range(1, self.nstations):
            f = (-self.phi(t))**i / math.factorial(i)
            # ith derivative of phi
            # TODO!
            g = 0
            Kc += f * g
        return Kc


class MarginalExceedanceScore():
    def __init__(self, kind, copula,
                 empirical=False, nsamples=10000,
                 logger=None):
        if kind not in MARGINAL_EXCEEDANCE_SCORE_KINDS:
            txt = "/".join(MARGINAL_EXCEEDANCE_SCORE_KINDS)
            errmsg = f"Expected kind in {txt}, got {kind}."
            raise ValueError(errmsg)
        self.kind = kind
        self.empirical = empirical
        self.copula = copula
        self.independence_copula = IndependenceCopula(copula.nstations)
        self.logger = logger

        self.nsamples = nsamples
        self.usamples = None

    def objective_function(self, u, lmaep):
        if not self.empirical:
            x = norm.ppf(u)
            match self.kind:
                case "AND":
                    c = self.copula.cdf_main_diagonal(u)
                case "OR":
                    c = 1 - self.copula.cdf_main_diagonal(1 - u)
                case "KENDALL":
                    c0 = self.copula.cdf_main_diagonal(x)
                    c = self.copula.inverse_kendall_function(c0)
        else:
            u_smp = self.u_smp
            nsmp = u_smp.shape[0]
            match self.kind:
                case "AND":
                    c = np.all(u_smp < u, axis=1).sum() / nsmp
                case "OR":
                    c = 1 - np.all(u_smp < 1 - u, axis=1).sum() / nsmp
                case "KENDALL":
                    raise ValueError("Cannot use 'empirical' computation"
                                     + " of KENDALL cdf")

        err = (math.log(c) - lmaep)**2
        return err

    def compute_bounds(self, maep):
        nsta = self.copula.nstations
        match self.kind:
            case "AND":
                ua = maep
                ub = maep**(1./nsta)
            case "OR":
                ua = 1 - (1 - maep)**(1./nsta)
                ub = maep
            case "KENDALL":
                ua = maep
                ub = self.independence_copula.inverse_kendall_function(maep)
        return ua, ub

    def compute_score(self, maep):
        bounds = self.compute_bounds(maep)
        lmaep = math.log(maep)
        self.opt = minimize_scalar(self.objective_function, bracket=bounds,
                                   bounds=bounds, args=(lmaep,))
        return self.opt.x
