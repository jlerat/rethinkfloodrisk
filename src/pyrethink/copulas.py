import re
import math

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invwishart
from scipy.stats import uniform_direction
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt

from scipy.optimize import minimize_scalar

from hydrodiy.stat import sutils

COPULA_NAMES = ["Gaussian", "Student",
                "GaussianFactor", "Independence",
                "Comonotone", "Gumbel"]

SYMETRICAL_COPULAS = ["Independence", "Comonotone",
                      "Gaussian", "GaussianFactor",
                      "Student"]

MARGINAL_EXCEEDANCE_SCORE_KINDS = ["AND", "OR",
                                   "KENDALL"]

STUDENT_DF_MIN = 2.01
STUDENT_DF_MAX = 100.


def reference_correlation_matrix(nstations, rho):
    if rho <= -1 or rho >= 1:
        errmsg = "Expected rho in ]-1, 1[."
        raise ValueError(errmsg)

    return (1 - rho) * np.eye(nstations) \
        + rho * np.ones((nstations, nstations))


def cov2corr(cov):
    invsig = 1. / np.sqrt(np.diag(cov))
    return np.einsum("ij,i,j->ij", cov, invsig, invsig)


def random_correlation_matrix(nstations, corr0=None, corr1=None):
    cov = invwishart.rvs(df=nstations+1, scale=np.eye(nstations))
    corr = cov2corr(cov)
    if corr0 is None and corr1 is None:
        return corr
    else:
        return corr0 + (corr1 - corr0) * (corr + 1.) / 2


def check_copula_name(copula_name):
    if copula_name not in COPULA_NAMES:
        txt = "/".join(COPULA_NAMES)
        errmsg = f"Expected 'copula_name' in {txt}, got {copula_name}."
        raise ValueError(errmsg)
    return COPULA_NAMES.index(copula_name)


def check_aep(aep):
    if aep <= 0 or aep >= 1:
        errmsg = f"Expected aep in ]0,1[, got {aep}."
        raise ValueError(errmsg)


def check_mex_kind(mex_kind):
    if mex_kind not in MARGINAL_EXCEEDANCE_SCORE_KINDS:
        txt = "/".join(MARGINAL_EXCEEDANCE_SCORE_KINDS)
        errmsg = f"Expected marginal exceedance score kind in {txt}, got {mex_kind}."
        raise ValueError(errmsg)


def check_semidefpos(x):
    try:
        np.linalg.cholesky(x)
    except Exception:
        errmsg = "Expected a semi-definite positive correlation matrix."
        raise ValueError(errmsg)


def check_cond(nstations, icond, zcond):
    allowed = set(range(nstations))
    scond = set(icond)
    if scond & allowed != scond:
        errmsg = f"Expected all elements of icond to be in [0..{nstations}[."
        raise ValueError(errmsg)

    if len(icond) > 1:
        errmsg = "Only one conditional variable allowed."
        raise ValueError(errmsg)

    if len(icond) != len(zcond):
        errmsg = "Expected icond and zcond of same length."
        raise ValueError(errmsg)

    itarget = np.array(list(allowed - scond))
    return itarget

def to2d(x, nstations):
    if x.ndim == 1:
        x = x[None, :]

    if x.ndim != 2:
        errmsg = "Expected array of dim 2"
        raise ValueError(errmsg)

    if x.shape[1] != nstations:
        errmsg = f"Expected second dimension of array of size {nstations}."\
                 + f" Got {x.shape[1]}."
        raise ValueError(errmsg)

    return x


def normal_cdf_approx(x, out=None):
    """ Approximation from https://www.jiem.org/index.php/jiem/article/view/60
    """
    if out is None:
        out = np.empty(x.shape)

    np.multiply(x, x, out=out)
    np.multiply(x, out, out=out)
    np.exp(-0.07056 * out - 1.5976 * x, out=out)
    np.add(out, 1., out=out)
    np.reciprocal(out, out=out)
    return out


def get_nsamples(nstations):
    return 10000 + 5000 * nstations


def get_copula_spec(name, default_shape=0., default_nfactor=0):
    elems = name.split("_")
    if len(elems) == 1:
        elems = elems + [default_shape, default_nfactor]
    elif len(elems) == 2:
        elems = elems + [default_nfactor]

    if elems[0] not in COPULA_NAMES:
        txt = "/".join(COPULA_NAMES)
        errmsg = f"First part of spec should be in {txt}, got {elems[0]}."
        raise ValueError(errmsg)

    elems[1] = float(elems[1])
    elems[2] = int(elems[2])

    # Constraints
    if not re.search("Factor", elems[0]) and elems[2] > 0:
        errmsg = f"Cannot have factors with copula {elems[0]}."
        raise ValueError(errmsg)

    if re.search("Factor", elems[0]) and elems[2] == 0:
        errmsg = f"Copula {elems[0]} needs at least one factor."
        raise ValueError(errmsg)

    if elems[0] != "Student" and elems[1] != default_shape:
        errmsg = f"Copula {elems[0]} does not accept"\
                 + f" shape parameter != {default_shape}."
        raise ValueError(errmsg)

    return elems


def factory(copula_spec, nstations):
    name, cshape, nfactors = get_copula_spec(copula_spec)

    # Copula id supplied
    if isinstance(name, int):
        name = COPULA_NAMES[name]

    if name == "Independence":
        return IndependenceCopula(nstations)
    if name == "Comonotone":
        return ComonotoneCopula(nstations)
    elif name == "Gaussian":
        return GaussianCopula(nstations)
    elif name == "Student":
        return StudentCopula(nstations, cshape)
    elif name == "GaussianFactor":
        return GaussianFactorCopula(nstations, nfactors)
    elif name == "Gumbel":
        return GumbelCopula(nstations)


class Copula():
    def __init__(self, name, nstations):
        self.copula_id = check_copula_name(name)
        self.name = name
        self.symetrical = name in SYMETRICAL_COPULAS
        self.nstations = nstations
        self._params = None
        self.logger = None
        self.printlog = 0

        # Spec
        self._copula_shape = 0.
        self._copula_nfactors = 0

        # underlying random variable
        self._mean = np.zeros(nstations)
        self._rv = None
        self._nkendall = None
        self._random_samples = None
        self._kendall_function_data = None
        self._u_data = None

    def __str__(self):
        txt = f"{self.name} copula in {self.nstations} dimensions"
        nfact = self.copula_nfactors
        if nfact == 1:
            txt += " using 1 factor"
        elif nfact > 1:
            txt += f" using {nfact} factor(s)"
        return txt

    @property
    def copula_shape(self):
        return self._copula_shape

    @property
    def copula_nfactors(self):
        return self._copula_nfactors

    def compute_kendall_function_data(self, nkendall=None):
        nsta = self.nstations
        nkendall = get_nsamples(nsta) if nkendall is None else nkendall

        kdd = self._kendall_function_data
        compute = kdd is None
        if kdd is not None:
            compute = self._nkendall != nkendall

        if compute:
            logger = self.logger
            printlog = 0
            if logger is not None and self.printlog > 0:
                logger.info("Computing kendall function")
                printlog = self.printlog

            # Generate samples
            self._nkendall = nkendall
            u = self.sample(nkendall)
            self._random_samples = u

            # Compute kendall
            ndoms = sutils.multivariate_dominance(u,
                                                  printlog=printlog)
            cdfs = ndoms / len(u)

            # could also do, but this is way slower
            # cdfs = np.array([self.cdf(uu) for uu in u]).squeeze()

            printlog = logger is not None
            p = np.arange(1, nkendall + 1) / nkendall
            kdd = pd.DataFrame({
                "copula_cdf": np.sort(cdfs),
                "kendall_cdf": p
                })
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

    def aep(self, u, kind):
        check_mex_kind(kind)
        match kind:
            case "AND":
                return self.survival(u)
            case "OR":
                return 1 - self.cdf(u)
            case "KENDALL":
                cdf = self.cdf(u)
                return 1 - self.kendall_function(cdf)

    def survival_main_diagonal(self, u):
        if self.symetrical:
            return self.cdf_main_diagonal(1 - u)
        else:
            raise NotImplementedError

    def marginal_ppf(self, u):
        raise NotImplementedError

    def marginal_cdf(self, x):
        raise NotImplementedError

    def sample(self, nsamples):
        raise NotImplementedError

    def kendall_function(self, cdf, nkendall=None):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(cdf, kdd.copula_cdf, kdd.kendall_cdf)

    def inverse_kendall_function(self, p, nkendall=None):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(p, kdd.kendall_cdf, kdd.copula_cdf)


class ComonotoneCopula(Copula):
    def __init__(self, nstations):
        super(ComonotoneCopula, self).__init__("Comonotone",
                                               nstations)

    def pdf(self, u):
        u = to2d(u, self.nstations)
        return (np.std(u, axis=1) < 1e-10).astype(float)

    def cdf(self, u):
        u = to2d(u, self.nstations)
        return np.min(u, axis=1)

    def marginal_ppf(self, u):
        return u

    def marginal_cdf(self, x):
        return x

    def sample(self, nsamples):
        return np.repeat(np.random.uniform(0, 1, (nsamples, 1)),
                         self.nstations, axis=1)

    def kendall_function(self, cdf):
        return cdf

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

    def marginal_ppf(self, u):
        return u

    def marginal_cdf(self, x):
        return x

    def sample(self, nsamples):
        return np.random.uniform(0, 1, (nsamples, self.nstations))

    def kendall_function(self, cdf):
        return gamma.sf(np.log(1. / cdf), a=self.nstations)

    def inverse_kendall_function(self, p):
        return np.exp(-gamma.isf(p, a=self.nstations))


class GaussianCopula(Copula):
    def __init__(self, nstations):
        super(GaussianCopula, self).__init__("Gaussian",
                                             nstations)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, corr):
        check_semidefpos(corr)
        self._params = corr
        self._rv = mvn(self._mean, corr)

    @property
    def rv(self):
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv

    def pdf(self, u):
        u = to2d(u, self.nstations)
        x = self.marginal_ppf(u)
        p = norm.pdf(x).prod(axis=1)
        return self.rv.pdf(x) / p

    def cdf(self, u):
        u = to2d(u, self.nstations)
        x = self.marginal_ppf(u)
        return self.rv.cdf(x)

    def marginal_ppf(self, u):
        return norm.ppf(u)

    def marginal_cdf(self, x):
        return norm.cdf(x)

    def sample(self, nsamples):
        return self.marginal_cdf(self.sample_z(nsamples))

    def sample_z(self, nsamples):
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv.rvs(size=nsamples)

    def sample_conditional(self, icond, zcond):
        itarget = check_cond(self.nstations, icond, zcond)

        # Conditional covariance matrices
        corr = self.params
        S11 = corr[icond][:, icond]
        S11i = 1./S11  # Because we only allow zcond of length = 1
        S22 = np.ascontiguousarray(corr[itarget][:, itarget])
        S21 = np.ascontiguousarray(corr[itarget][:, icond])

        muc = S21 @ S11i @ zcond
        Sc = S22 - S21 @ S11i @ S21.T
        return mvn.rvs(mean=muc, cov=Sc)


class GaussianFactorCopula(GaussianCopula):
    """ Compute the CDF for a Gaussian factor copula such that
        Ui = Phi(Zi)
        Zi = sum_k sqrt(rho_k) V_k + sqrt(1 - rho) Ei
        with V ~ N(0,1)  Ei ~ N(0,1)

        Hence Z ~ N(0, Sigma)
        with Sigma = (1-rho) x eye + rho x ones

        Computation is done in an arbitrary number of dimensions.
    """
    def __init__(self, nstations, copula_nfactors=1):
        super(GaussianFactorCopula, self).__init__(nstations)
        self.name = "GaussianFactor"
        self._copula_nfactors = int(copula_nfactors)
        if self.copula_nfactors < 1:
            errmsg = "Cannot have a factor copula with less than 1 factor."
            raise ValueError(errmsg)
        self._sqr = None
        self._corr = None
        self.set_approx()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, rhos):
        nsta = self.nstations
        nfact = self.copula_nfactors

        if isinstance(rhos, float):
            # Allows for single rho
            rhos = rhos * np.ones((nsta, nfact))

        rhos = np.array(rhos)
        if rhos.ndim == 1:
            rhos = np.repeat(rhos[:, None], nfact, axis=1)

        if rhos.shape != (nsta, nfact):
            errmsg = f"Expected rhos of shape {nsta}x{nfact}."
            raise ValueError(errmsg)

        sr = (rhos**2).sum(axis=1)
        if np.any(sr > 1):
            raise ValueError("Expected rhos**2 sum <= 1")

        self._params = rhos
        self._sqr = np.sqrt(1 - sr)

        corr = rhos @ rhos.T
        corr = np.eye(nsta) + corr - np.diag(np.diag(corr))
        self._corr = corr
        self._rv = mvn(self._mean, corr)

    @property
    def corr(self):
        return self._corr

    @property
    def sqr(self):
        return self._sqr

    def set_params_via_zrho(self, zrhos):
        nsta = self.nstations
        nfact = self.copula_nfactors
        if zrhos.shape != (nsta, nfact + 1):
            errmsg = f"Expected shape of zrhos to be ({nsta},{nfact+1}),"\
                     + f" got {zrhos.shape}."
            raise ValueError(errmsg)

        # Add a small offset to avoid one the last component being exactly 0
        z2 = zrhos**2
        z2[:, -1] = np.maximum(z2[:, -1], 1e-10)
        sr = np.sqrt(np.sum(z2, axis=1))
        self.params = zrhos[:, :nfact] / sr[:, None]

    def random_params(self, single_value=False):
        nsta = self.nstations
        nfact = self.copula_nfactors
        if single_value:
            sq = math.sqrt(nsta)
            rho = np.random.uniform(-1. / sq, 1. / sq)
            return rho * np.ones((nsta, nfact))
        else:
            return uniform_direction(nfact + 1).rvs(size=nsta)[:, :-1]

    def set_approx(self, napprox=100):
        self.napprox = napprox
        eps = 5e-1 / napprox
        u = np.linspace(eps, 1 - eps, napprox)
        v = norm.ppf(u)
        self.v = [m[None, None, :] for m in np.meshgrid(*[v] * self.copula_nfactors)]
        self.du = u[1] - u[0]

        dims = [1, self.nstations] + [napprox] * self.copula_nfactors
        self.buf = np.empty(dims)
        self.buf_cdf = np.empty(dims)


    def cdf(self, u, approx=True):
        """ Fast computation of CDF useful for high dimensions.
        Note that pdf is not approximated.
        """
        u = to2d(u, self.nstations)
        if self.buf.shape[0] != len(u):
            dim = [len(u), self.nstations] + [self.napprox] * self.copula_nfactors
            self.buf = np.empty(dim)
            self.buf_cdf = np.empty(dim)

        self.buf.fill(0.)

        # CDF for a factor copula is
        # p(X1<x1, ..., Xn<xn) =
        #      int(prod(Phi(xi - rho_i1 v1 - rhoi2 v2 ...) / sqrt(1 - sum rho_ik^2)) dv1 dv2.. dvp,
        #          vk=-infty,
        #          vk=+infty)

        nfact = self.copula_nfactors
        dims = np.arange(2, 2 + nfact).tolist()
        # divide by nfact because we add this nfact times in the loop below
        dx = np.expand_dims(self.marginal_ppf(u) / nfact, dims)
        rhos = self.params
        for i in range(nfact):
            r = np.expand_dims(rhos[:, i][None, :], dims)
            np.add(self.buf, dx - r * self.v[i], out=self.buf)

        sqr = np.expand_dims(self.sqr[None, :], dims)
        np.multiply(self.buf, 1. / sqr, out=self.buf)

        dims = np.arange(1, 1 + nfact).tolist()
        if approx:
            normal_cdf_approx(self.buf, out=self.buf_cdf)
        else:
            self.buf_cdf[:] = norm.cdf(self.buf)

        out = np.apply_over_axes(np.sum, self.buf_cdf.prod(axis=1), dims) * self.du**nfact
        return out.squeeze()

    def sample_z(self, nsamples):
        nfact = self.copula_nfactors
        v = np.random.normal(size=(nsamples, nfact))
        eps = np.random.normal(size=(nsamples, self.nstations))
        return v @ self.params.T + eps * self.sqr[None, :]

    def sample(self, nsamples):
        z = self.sample_z(nsamples)
        return self.marginal_cdf(z)

    def kendall_function(self, cdf, nkendall=None):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(cdf, kdd.copula_cdf, kdd.kendall_cdf)

    def inverse_kendall_function(self, p, nkendall=None):
        kdd = self.compute_kendall_function_data(nkendall)
        return np.interp(p, kdd.kendall_cdf, kdd.copula_cdf)



class StudentCopula(Copula):
    def __init__(self, nstations, df=4.):
        super(StudentCopula, self).__init__("Student",
                                             nstations)
        smi = STUDENT_DF_MIN
        sma = STUDENT_DF_MAX
        if df < smi or df > sma:
            errmsg = f"Expected df in [{smi}, {sma}], got {df}."
            raise ValueError(errmsg)
        self._copula_shape = df

    @property
    def scale(self):
        df = self.copula_shape
        return math.sqrt((df - 2) / df)

    @property
    def params(self):
        return self._params

    @property
    def corr_scaled(self):
        return self.params * self.scale**2

    @params.setter
    def params(self, corr):
        check_semidefpos(corr)
        self._params = corr
        df = self.copula_shape
        scorr = self.corr_scaled
        self._rv = mvt(loc=self._mean, shape=scorr, df=df)

    @property
    def rv(self):
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv

    def pdf(self, u):
        u = to2d(u, self.nstations)
        x = self.marginal_ppf(u)
        p = norm.pdf(x).prod(axis=1)
        return self.rv.pdf(x) / p

    def cdf(self, u):
        u = to2d(u, self.nstations)
        x = self.marginal_ppf(u)
        return self.rv.cdf(x)

    def marginal_ppf(self, u):
        return student_t.ppf(u, scale=self.scale,
                             df=self.copula_shape)

    def marginal_cdf(self, x):
        return student_t.cdf(x, scale=self.scale,
                             df=self.copula_shape)

    def sample(self, nsamples):
        return self.marginal_cdf(self.sample_z(nsamples))

    def sample_z(self, nsamples):
        if self._rv is None:
            errmsg = "rv is None, set parameters."
            raise ValueError(errmsg)
        return self._rv.rvs(size=nsamples)

    def sample_conditional(self, icond, zcond):
        itarget = check_cond(self.nstations, icond, zcond)

        # Conditional covariance matrices
        scorr = self.corr_scaled
        S11 = scorr[icond][:, icond]
        S11i = 1./S11  # Because we only allow zcond of length = 1
        S22 = np.ascontiguousarray(scorr[itarget][:, itarget])
        S21 = np.ascontiguousarray(scorr[itarget][:, icond])

        muc = S21 @ S11i @ zcond
        Sc = S22 - S21 @ S11i @ S21.T

        df0 = self.copula_shape
        p1 = 1  # Because we only allow zcond of length = 1
        d1 = zcond.T @ S11i @ zcond

        a = (df0 + d1) / (df0 + p1)
        df = df0 + p1
        return mvt.rvs(loc=muc, shape=a*Sc, df=df)


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
        raise NotImplementedError

        Kc = t
        for i in range(1, self.nstations):
            f = (-self.phi(t))**i / math.factorial(i)
            # ith derivative of phi
            # TODO!
            g = 0
            Kc += f * g
        return Kc


class MarginalExceedanceScore():
    def __init__(self, mex_kind, copula):
        check_mex_kind(mex_kind)
        self.mex_kind = mex_kind

        if not isinstance(copula, Copula):
            errmsg = "Expected copula to be an instance of 'Copula'"
            raise ValueError(errmsg)

        self.copula = copula
        self.independence_copula = IndependenceCopula(copula.nstations)
        self.comonotone_copula = ComonotoneCopula(copula.nstations)
        self.logger = None
        self._samples = None
        self._u_vect = np.ones((1, copula.nstations))

    def objective_function(self, u, lmaep):
        c = self.copula.aep(u, self.mex_kind).squeeze()
        if c == 0:
            return np.inf
        err = (math.log(c) - lmaep)**2
        return err

    def objective_function_1d(self, u, lmaep):
        u_vect = self._u_vect
        u_vect.fill(u)
        return self.objective_function(u_vect, lmaep)

    def common_marginal_exceedance_score_bounds(self, maep):
        nsta = self.copula.nstations

        match self.mex_kind:
            case "AND":
                ua = 1 - maep ** (1. / nsta)
                ub = 1 - maep
            case "OR":
                ua = 1 - maep
                ub = (1 - maep) ** (1. / nsta)
            case "KENDALL":
                copi = self.independence_copula
                cdf = copi.inverse_kendall_function(1 - maep)
                ua = 1 - cdf ** (1. / nsta)

                copc = self.comonotone_copula
                cdf = copc.inverse_kendall_function(1 - maep)
                ub = cdf

        return ua, ub

    def common_marginal_exceedance_score(self, maep):
        """ Marginal exceedance score, i.e. the value alpha such that
        Pr(ex(u, u0)) = maep
        with ex the exceedance operator defined on Rn x Rn -> {0, 1}
        and  u0, a reference vector such that
        for all i in 1..n,  u0[i] = alpha
        """
        check_aep(maep)
        bounds = self.common_marginal_exceedance_score_bounds(maep)
        lmaep = math.log(maep)
        opt = minimize_scalar(self.objective_function_1d,
                              bracket=bounds, bounds=bounds,
                              args=(lmaep,))
        return opt.x, opt.fun

    def marginal_exceedance_set(self, maep, npoints=200, u0=1e-5, nitermax=5,
                                logerrmax=1e-3):
        """ Marginal exceedance set in 2 dimensions,
        i.e.  the set of values (a1, a2) such that
        Pr(ex(u, [a1, a2])) = maep
        with ex the exceedance operator defined by the 'kind' parameter.
        """
        check_aep(maep)
        if self.copula.nstations != 2:
            errmsg = "Can only compute set for 2 dimensions"
            raise ValueError(errmsg)

        u = np.linspace(u0, 1 - u0, 2 * npoints)
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
                    return self.objective_function(u_vect, lmaep)

                opt = minimize_scalar(ofun, bracket=bounds, bounds=bounds,
                                      args=(lmaep,))
                df.loc[iu, "v"] = opt.x
                df.loc[iu, "log10_cdf_err"] = opt.fun

            iok = np.where(df.log10_cdf_err < logerrmax)[0]
            ia = max(0, iok.min() - 1)
            ib = min(len(df) - 1, iok.max())
            if ia == ib:
                errmsg = "Failed to identify set."
                raise ValueError(errmsg)

            ua = max(df.u.iloc[ia] - u0, 0)
            ub = min(df.u.iloc[ib] + u0, 1)
            u = np.linspace(ua, ub, npoints)

            niter += 1

        iok = df.log10_cdf_err < logerrmax
        df = df.loc[iok]

        return df, niter


class MarginalExceedanceScoreEmpirical(MarginalExceedanceScore):

    def __init__(self, kind, copula, nsamples):
        super(MarginalExceedanceScoreEmpirical, self).__init__(kind, copula)
        self.samples = self.copula.sample(nsamples)

    def objective_function(self, u, lmaep):
        u = to2d(u, self.copula.nstations)
        u_smp = self.samples
        n_smp = len(u_smp)

        match self.mex_kind:
            case "AND":
                c = np.all(u_smp > u, axis=1).sum() / n_smp
            case "OR":
                c = 1 - np.all(u_smp < u, axis=1).sum() / n_smp
            case "KENDALL":
                raise ValueError("Cannot use 'empirical' computation"
                                 + " of KENDALL cdf")

        err = (math.log(c) - lmaep)**2
        return err
