import math
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mvt
from scipy.optimize import minimize_scalar

from hydrodiy.stat import sutils

COPULAS = ["Independence", "Comonotone",
           "Gaussian", "GaussianOneFactor",
           "Gumbel"]

SYMETRICAL_COPULAS = ["Independence", "Comonotone",
                      "Gaussian", "GaussianOneFactor"]

MARGINAL_EXCEEDANCE_SCORE_KINDS = ["AND", "OR", "KENDALL"]

NKENDALL_DEFAULT = 20000


def check_aep(aep):
    if aep <= 0 or aep >= 1:
        errmsg = "Expected aep in ]0,1[, got {aep}."
        raise ValueError(errmsg)


def to2d(x, nstations):
    if x.ndim == 1:
        x = x[None, :]

    if x.ndim != 2:
        errmsg = "Expected array of dim 2"
        raise ValueError(errmsg)

    if x.shape[1] != nstations:
        errmsg = "Expected second dimension of array of size {nstations}."\
                 + " Got {x.shape[1]}."
        raise ValueError(errmsg)

    return x


def copula_factory(name, nstations):
    if name not in COPULAS:
        txt = "/".join(COPULAS)
        errmsg = f"Expected {name} in '{txt}', got {name}."
        raise ValueError(errmsg)

    if name == "Independence":
        return IndependenceCopula(nstations)
    if name == "Comonotone":
        return ComonotoneCopula(nstations)
    elif name == "Gaussian":
        return GaussianCopula(nstations)
    elif name == "GaussianOneFactor":
        return GaussianOneFactorCopula(nstations)
    elif name == "Gumbel":
        return GumbelCopula(nstations)


class Copula():
    def __init__(self, name, nstations):
        self.name = name
        self.symetrical = name in SYMETRICAL_COPULAS
        self.nstations = nstations
        self._params = None
        self.logger = None

        # Potential random variable
        self._mean = np.zeros(nstations)
        self._rv = None

        self._nkendall = None
        self._random_samples = None
        self._kendall_function_data = None
        self._u_data = None

    def __str__(self):
        txt = f"{self.name} copula {self.nstations} dimensions."
        return txt

    def compute_kendall_function_data(self, nkendall=NKENDALL_DEFAULT):
        kdd = self._kendall_function_data

        compute = kdd is None
        if kdd is not None:
            compute = self._nkendall != nkendall

        if compute:
            logger = self.logger
            if logger is not None:
                logger.info("Computing kendall function")

            # Generate samples
            self._nkendall = nkendall
            u = self.sample(nkendall)
            self._random_samples = u

            # Compute kendall
            N = len(u)
            printlog = logger is not None
            probs = sutils.multivariate_dominance(u, printlog=printlog) / N
            t = np.arange(1, N + 1) / N
            Kc = np.zeros_like(t)
            for i, tt in enumerate(t):
                Kc[i] = (probs < tt).sum() / N

            kdd = pd.DataFrame({"t": t, "Kc": Kc})
            self._kendall_function_data = kdd

        return kdd

    def pdf(self, u):
        raise NotImplementedError

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

    def survival(self, u):
        u = to2d(u, self.nstations)
        if self.symetrical:
            return self.cdf(1 - u)
        else:
            raise NotImplementedError

    def survival_main_diagonal(self, u):
        if self.symetrical:
            return self.cdf_main_diagonal(1 - u)
        else:
            raise NotImplementedError

    def marginal_transformed_ppf(self, u):
        raise NotImplementedError

    def marginal_transformed_cdf(self, x):
        raise NotImplementedError

    def sample(self, nsamples):
        raise NotImplementedError

    def conditional_sample(self, icond, zcond, itarget):
        raise NotImplementedError

    def kendall_function(self, t, nkendall=NKENDALL_DEFAULT):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(t, kdd.t, kdd.Kc)

    def inverse_kendall_function(self, p, nkendall=NKENDALL_DEFAULT):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(p, kdd.Kc, kdd.t)


class ComonotoneCopula(Copula):
    def __init__(self, nstations):
        super(ComonotoneCopula, self).__init__("Comonotone",
                                               nstations)

    def pdf(self, u):
        u = to2d(u, self.nstations)
        return np.all(u == u[:, [0]], axis=1).astype(float)

    def cdf(self, u):
        u = to2d(u, self.nstations)
        return np.min(u, axis=1)

    def marginal_transformed_ppf(self, u):
        return u

    def marginal_transformed_cdf(self, x):
        return x

    def sample(self, nsamples):
        return np.repeat(np.random.uniform(0, 1, (nsamples, 1)),
                         self.nstations, axis=1)

    def kendall_function(self, t):
        return t

    def inverse_kendall_function(self, p):
        return p


class IndependenceCopula(Copula):
    def __init__(self, nstations):
        super(IndependenceCopula, self).__init__("Independence",
                                                 nstations)

    def pdf(self, u):
        u = to2d(u, self.nstations)
        return np.ones(u.shape[0])

    def cdf(self, u):
        u = to2d(u, self.nstations)
        return np.prod(u, axis=1)

    def marginal_transformed_ppf(self, u):
        return u

    def marginal_transformed_cdf(self, x):
        return x

    def sample(self, nsamples):
        return np.random.uniform(0, 1, (nsamples, self.nstations))

    def kendall_function(self, t):
        return 1 - gamma.cdf(np.log(1./t), a=self.nstations)

    def inverse_kendall_function(self, p):
        return np.exp(-gamma.ppf(1 - p, a=self.nstations))


class GaussianCopula(Copula):
    def __init__(self, nstations):
        super(GaussianCopula, self).__init__("Gaussian",
                                             nstations)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, cor):
        self._params = cor
        self._rv = mvt(self._mean, cor)

    def cdf(self, u):
        u = to2d(u, self.nstations)
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv.cdf(self.marginal_transformed_ppf(u))

    def marginal_transformed_ppf(self, u):
        return norm.ppf(u)

    def marginal_transformed_cdf(self, x):
        return norm.cdf(x)

    def sample(self, nsamples):
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv.rvs(size=nsamples)


class GaussianOneFactorCopula(GaussianCopula):
    """ Compute the CDF for a Gaussian factor copula such that
        Ui = Phi(Zi)
        Zi = sqrt(rho) V + sqrt(1 - rho) Ei
        with V ~ N(0,1)  Ei ~ N(0,1)

        Hence Z ~ N(0, Sigma)
        with Sigma = (1-rho) eye + rho ones

        Computation is done in an arbitrary number of dimensions.
    """
    def __init__(self, nstations):
        super(GaussianOneFactorCopula, self).__init__(nstations)
        self.name = "GaussianOneFactor"
        self._sqr = None
        self._csqr = None

        self.set_approx()

    def set_approx(self, napprox=500):
        self.napprox = napprox
        eps = 5e-1 / napprox
        u = np.linspace(eps, 1 - eps, napprox)
        self.v = norm.ppf(u)
        self.du = u[1] - u[0]
        self.buf = np.zeros((1, self.nstations, napprox))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, rho):
        if rho < 0 or rho >= 1:
            raise ValueError(f"Expected rho in [0, 1[, got {rho}")
        self._params = rho
        self._sqr = math.sqrt(rho)
        self._csqr = math.sqrt(1 - rho)

    @property
    def sqr(self):
        return self._sqr

    @property
    def csqr(self):
        return self._csqr

    def cdf(self, u):
        u = to2d(u, self.nstations)

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
    def __init__(self, nstations):
        super(GumbelCopula, self).__init__("Gumbel",
                                           nstations)
        self._theta = None
        self.phi = lambda x: (-np.log(x))**self.theta
        self.phi_inv = lambda y: np.exp(-y**(1./self.theta))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, rho):
        if rho < 0 or rho >= 1:
            raise ValueError(f"Expected rho in [0, 1[, got {rho}")
        self._params = rho
        self._theta = 1 / (1 - rho)

    @property
    def theta(self):
        return self._theta

    def cdf(self, u):
        u = to2d(u, self.nstations)
        return self.phi_inv(self.phi(u).sum(axis=1))

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
    def __init__(self, kind, copula):
        if kind not in MARGINAL_EXCEEDANCE_SCORE_KINDS:
            txt = "/".join(MARGINAL_EXCEEDANCE_SCORE_KINDS)
            errmsg = f"Expected kind in {txt}, got {kind}."
            raise ValueError(errmsg)
        self.kind = kind

        if not isinstance(copula, Copula):
            errmsg = "Expected copula to be an instance of 'Copula'"
            raise ValueError(errmsg)

        self.copula = copula
        self.independence_copula = IndependenceCopula(copula.nstations)
        self.logger = None
        self._samples = None
        self._u_vect = np.ones((1, copula.nstations))

    def objective_function_set(self, u, lmaep):
        match self.kind:
            case "AND":
                c = self.copula.survival(u)
            case "OR":
                c = 1 - self.copula.cdf(u)
            case "KENDALL":
                c0 = self.copula.cdf(1 - u)
                c = 1 - self.copula.kendall_function(c0)
        if c == 0:
            return np.inf
        err = (math.log(c) - lmaep)**2
        return err

    def objective_function_diagonal(self, u, lmaep):
        u_vect = self._u_vect
        u_vect.fill(u)
        return self.objective_function_set(u_vect, lmaep)

    def compute_bounds(self, maep):
        nsta = self.copula.nstations
        match self.kind:
            case "AND":
                ua = 1 - (1 - maep)**(1./nsta)
                ub = maep
            case "OR":
                ua = maep
                ub = maep**(1./nsta)
            case "KENDALL":
                ub = 1 - maep
                ua = self.independence_copula.inverse_kendall_function(ub)
        return ua, ub

    def compute_score(self, maep):
        check_aep(maep)
        bounds = self.compute_bounds(maep)
        lmaep = math.log(maep)
        opt = minimize_scalar(self.objective_function_diagonal,
                              bracket=bounds, bounds=bounds,
                              args=(lmaep,))
        return opt.x, opt.fun

    def compute_set(self, maep, npoints=50, nitermax=5):
        check_aep(maep)
        if self.copula.nstations != 2:
            errmsg = "Can only compute set for 2 dimensions"
            raise ValueError(errmsg)

        eps = 0.25 / npoints
        u = np.linspace(eps, 1 - eps, 2 * npoints)
        df = []
        niter = 0
        logger = self.logger
        bounds = [0, 1]

        while len(df) < npoints and niter < nitermax:
            if logger is not None:
                logger.info(f"Compute set - iteration {niter}")

            df = pd.DataFrame({"u": u, "v": np.zeros_like(u),
                               "log10_cdf_err": np.zeros_like(u)})
            lmaep = math.log(maep)
            u_vect = self._u_vect

            for iu, uu in enumerate(u):
                u_vect[0, 0] = uu

                def ofun(v, lmaep):
                    u_vect[0, 1] = v
                    return self.objective_function_set(u_vect, lmaep)

                opt = minimize_scalar(ofun, bracket=bounds, bounds=bounds,
                                      args=(lmaep,))
                df.loc[iu, "v"] = opt.x
                df.loc[iu, "log10_cdf_err"] = opt.fun

            df = df.loc[df.log10_cdf_err < 1e-3]
            if len(df) == 0:
                errmsg = "Failed to identify set."
                raise ValueError(errmsg)

            u0 = df.u.iloc[0] - eps
            u1 = df.u.iloc[-1] + eps
            u = np.linspace(u0, u1, npoints)
            niter += 1

        return df, niter


class MarginalExceedanceScoreEmpirical(MarginalExceedanceScore):

    def __init__(self, kind, copula, nsamples):
        super(MarginalExceedanceScoreEmpirical, self).__init__(kind, copula)
        self.samples = self.copula.sample(nsamples)

    def objective_function(self, u, lmaep):
        u_smp = self.samples
        n_smp = len(u_smp)

        match self.kind:
            case "AND":
                c = np.all(u_smp > 1 - u, axis=1).sum() / n_smp
            case "OR":
                c = 1 - np.all(u_smp < u, axis=1).sum() / n_smp
            case "KENDALL":
                raise ValueError("Cannot use 'empirical' computation"
                                 + " of KENDALL cdf")

        err = (math.log(c) - lmaep)**2
        return err
