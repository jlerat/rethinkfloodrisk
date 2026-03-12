import math
from itertools import combinations

import numpy as np

from scipy.stats import norm
from scipy.stats import invwishart
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt

from pyrethink.partitions import Partitions

COPULA_TYPES = {
        0: "gaussian",
        1: "student"
        }

STUDENT_DF_MIN = 2.01
STUDENT_DF_MAX = 100.


def corr_ref(nstations, rho):
    if rho <= -1 or rho >= 1:
        errmsg = "Expected rho in ]-1, 1[."
        raise ValueError(errmsg)

    return (1 - rho) * np.eye(nstations) \
        + rho * np.ones((nstations, nstations))


def cov2corr(cov):
    invsig = 1. / np.sqrt(np.diag(cov))
    return np.einsum("ij,i,j->ij", cov, invsig, invsig)


def random_corr(nstations, corr0=None, corr1=None):
    cov = invwishart.rvs(df=nstations+1, scale=np.eye(nstations))
    corr = cov2corr(cov)
    if corr0 is None and corr1 is None:
        return corr
    else:
        return corr0 + (corr1 - corr0) * (corr + 1.) / 2


def copula_marginal_ppf(copula_type, copula_shape, u):
    if copula_type == 1:
        scale = math.sqrt((copula_shape - 2) / copula_shape)
        return student_t.ppf(u, scale=scale, df=copula_shape)
    elif copula_type == 0:
        return norm.ppf(u)
    else:
        errmsg = "copula_marginal_ppf not handling copula type {copula_type}."
        raise ValueError(errmsg)


def copula_marginal_cdf(copula_type, copula_shape, z):
    if copula_type == 1:
        scale = math.sqrt((copula_shape - 2) / copula_shape)
        return student_t.cdf(z, scale=scale, df=copula_shape)
    elif copula_type == 0:
        return norm.cdf(z)
    else:
        errmsg = "copula_marginal_cdf not handling copula type {copula_type}."
        raise ValueError(errmsg)


def check_copula(copula_type, copula_shape):
    if copula_type not in COPULA_TYPES:
        txt = "/".join([str(t) for t in COPULA_TYPES])
        errmsg = f"Expected 'copula_type' in {txt}, got {copula_type}."
        raise ValueError(errmsg)

    if copula_shape > 0:
        mini = STUDENT_DF_MIN
        maxi = STUDENT_DF_MAX
        if copula_shape < mini or copula_shape > maxi:
            errmsg = f"Expected 'copula_shape' in [{mini}, {maxi}]."
            raise ValueError(errmsg)

    elif copula_shape < 0:
        errmsg = "Expected 'copula_shape' >= 0."
        raise ValueError(errmsg)


class Copula():
    def __init__(self, copula_type, copula_shape, nstations):
        check_copula(copula_type, copula_shape)
        self._copula_type = int(copula_type)
        self._copula_shape = float(copula_shape)
        self._partitions = Partitions(nstations)
        self._mean = np.zeros(nstations)

    @property
    def copula_type(self):
        return self._copula_type

    @property
    def copula_shape(self):
        return self._copula_shape

    @property
    def nstations(self):
        return self._partitions.nelements

    @property
    def partitions(self):
        return self._partitions

    @property
    def corr(self):
        return self._corr

    @corr.setter
    def corr(self, val):
        nsta = self.nstations
        if val.shape != (nsta, nsta):
            errmsg = f"Expected a square matrix of shape ({nsta},{nsta})."
            raise ValueError(errmsg)

        try:
            np.linalg.cholesky(val)
        except Exception:
            errmsg = "Expected a semi-definite positive matrix."
            raise ValueError(errmsg)

        if not np.allclose(np.diag(val), 1):
            errmsg = "Expected a matrix with ones on the diagonal."
            raise ValueError(errmsg)
        self._corr = val

    @property
    def corr_rescaled(self):
        copula_type = self.copula_type
        if copula_type == 0:
            return self.corr
        elif copula_type == 1:
            df = self.copula_shape
            return self.corr * (df - 2) / df
        else:
            errmsg = "corr_rescaled not handling copula type {copula_type}."
            raise ValueError(errmsg)

    @property
    def mean(self):
        return self._mean

    def conditional_sample_given_partition(self, ipart, icond, zcond, itarget):
        """ Sample copula with given zcond """

        # Check inputs
        nsta = self.nstations
        allowed = set(range(nsta))
        scond = set(icond)
        if scond & allowed != scond:
            errmsg = f"Expected all elements of icond to be in [0..{nsta}[."
            raise ValueError(errmsg)

        starget = set(itarget)
        if starget & allowed != starget:
            errmsg = f"Expected all elements of itarget to be in [0..{nsta}[."
            raise ValueError(errmsg)

        if len(icond) > 1:
            errmsg = "Only one conditional variable allowed."
            raise ValueError(errmsg)

        if len(scond & starget) > 0:
            errmsg = "Expected no common elements in icond and itarget."
            raise ValueError(errmsg)

        if len(icond) != len(zcond):
            errmsg = "Expected icond and zcond of same length."
            raise ValueError(errmsg)

        # Get data
        sets = self.partitions.ipart2sets(ipart)
        sets_cond = sets[icond]
        sets_target = sets[itarget]

        corr_rescaled = self.corr_rescaled

        # Sample targets that are in same set than cond
        idx_join = sets_cond == sets_target
        ijoin = itarget[idx_join]
        z = np.zeros(len(itarget))
        copula_type = self.copula_type

        if len(ijoin) > 0:
            # Conditional covariance matrices
            S11 = corr_rescaled[icond][:, icond]
            S11i = 1./S11  # Because we only allow zcond of length = 1
            S22 = np.ascontiguousarray(corr_rescaled[ijoin][:, ijoin])
            S21 = np.ascontiguousarray(corr_rescaled[ijoin][:, icond])

            muc = S21 @ S11i @ zcond
            Sc = S22 - S21 @ S11i @ S21.T

            if copula_type == 1:
                # MV student conditional
                # See https://en.wikipedia.org/wiki/Multivariate_t-distribution
                #    #Conditional_Distribution
                nu = self.copula_shape

                p1 = 1  # Because we only allow zcond of length = 1
                d1 = zcond.T @ S11i @ zcond

                a = (nu + d1) / (nu + p1)
                df = nu + p1
                z[idx_join] = mvt.rvs(loc=muc, shape=a*Sc, df=df)

            elif copula_type == 0:
                # MV normal conditional
                z[idx_join] = mvn.rvs(mean=muc, cov=Sc)

            else:
                errmsg = "Sampling not handling copula type {copula_type}."
                raise ValueError(errmsg)

        # Sample targets that are not in same set than cond
        idx_sep = sets_cond != sets_target
        isep = itarget[idx_sep]

        if len(isep) > 0:
            S22 = np.ascontiguousarray(corr_rescaled[isep][:, isep])
            mu22 = self.mean[isep]
            if copula_type == 1:
                z[idx_sep] = mvt.rvs(loc=mu22, shape=S22, df=self.copula_shape)
            elif copula_type == 0:
                z[idx_sep] = mvn.rvs(mean=mu22, cov=S22)
            else:
                errmsg = "Sampling not handling copula type {copula_type}."
                raise ValueError(errmsg)

        return z

    def get_rv(self, iselected):
        copula_type = self.copula_type
        scorr = self.corr_rescaled[iselected][:, iselected]
        scorr = np.ascontiguousarray(scorr)
        mu = np.zeros(self.nstations)

        if copula_type == 1:
            df = self.copula_shape
            return mvt(loc=mu[iselected], shape=scorr, df=df)
        elif copula_type == 0:
            return mvn(mean=mu[iselected], cov=scorr)
        else:
            errmsg = "Cannot get rv for copula type {copula_type}."
            raise ValueError(errmsg)

    def get_sets_iterator(self, ipart, iselect):
        nsta = self.nstations
        iselect = np.arange(nsta) if iselect is None \
            else iselect
        isin = np.zeros(nsta).astype(bool)
        isin[iselect] = True

        sets = self.partitions.ipart2sets(ipart)
        sets_selected = sets[iselect]

        for iset in np.unique(sets_selected):
            idx = (iset == sets) & isin
            yield idx

    def pdf_given_partition(self, ipart, z, iselect=None):
        nsta = self.nstations
        if len(z) != nsta:
            errmsg = f"Expected z of length {nsta}."
            raise ValueError(errmsg)

        pdf = 1.
        for idx in self.get_sets_iterator(ipart, iselect):
            rv = self.get_rv(idx)
            pdf *= rv.pdf(z[idx])

        return pdf

    def cdf_given_partition(self, ipart, z, iselect=None):
        nsta = self.nstations
        if len(z) != nsta:
            errmsg = f"Expected z of length {nsta}."
            raise ValueError(errmsg)

        cdf = 1.
        for idx in self.get_sets_iterator(ipart, iselect):
            rv = self.get_rv(idx)
            cdf *= rv.cdf(z[idx])

        return cdf

    def survival_given_partition(self, ipart, z):
        nsta = self.nstations
        if len(z) != nsta:
            errmsg = f"Expected z of length {nsta}."
            raise ValueError(errmsg)

        surv = 1. - norm.cdf(z).sum()
        sets = self.partitions.ipart2sets(ipart)

        for nv in range(2, nsta + 1):
            f = (-1)**nv
            for ix in combinations(np.arange(nsta), nv):
                # Compute cdf for each combination
                ix = np.array(ix)
                cdf = 1.
                for s in np.unique(sets[ix]):
                    ixs = ix[sets[ix] == s]
                    rvs = self.get_rv(ixs)
                    cdf *= rvs.cdf(z[ixs])

                surv += f * cdf

        return surv

    def sample_z_given_partition(self, ipart, nsamples):
        sets = self.partitions.ipart2sets(ipart)
        copula_type = self.copula_type
        copula_shape = self.copula_shape
        mu = self.mean
        corr_rescaled = self.corr_rescaled

        nsta = self.nstations
        z = np.empty((nsamples, nsta))

        for iset in np.unique(sets):
            idx = iset == sets
            nval = idx.sum()
            scorr = np.ascontiguousarray(corr_rescaled[idx][:, idx])

            if copula_type == 1:
                rv = mvt(loc=mu[idx], shape=scorr, df=copula_shape)
            elif copula_type == 0:
                rv = mvn(mean=mu[idx], cov=scorr)
            else:
                errmsg = "Sampling not handling copula type {copula_type}."
                raise ValueError(errmsg)

            z[:, idx] = rv.rvs(size=nsamples).reshape((nsamples, nval))

        return z

    def sample_u_given_partition(self, ipart, nsamples):
        z = self.sample_z_given_partition(ipart, nsamples)
        copula_type = self.copula_type
        copula_shape = self.copula_shape
        if copula_type == 1:
            adj = math.sqrt(self.corr_rescaled[0, 0])
            u = student_t.cdf(z, df=copula_shape, loc=0, scale=adj)
        elif copula_type == 0:
            u = norm.cdf(z)
        else:
            errmsg = "Sampling not handling copula type {copula_type}."
            raise ValueError(errmsg)

        return u

    def sample_z(self, probabilities, nsamples):
        nsta = self.nstations
        z = np.empty((nsamples, nsta))
        iparts = self.partitions.sample(probabilities, nsamples)

        for ipart in np.unique(iparts):
            idx = ipart == iparts
            z[idx, :] = self.sample_z_given_partition(ipart, idx.sum())

        return z, iparts

    def sample_u(self, probabilities, nsamples):
        z, iparts = self.sample_z(probabilities, nsamples)
        copula_type = self.copula_type
        copula_shape = self.copula_shape
        if copula_type == 1:
            adj = math.sqrt(self.corr_rescaled[0, 0])
            u = student_t.cdf(z, df=copula_shape, loc=0, scale=adj)
        elif copula_type == 0:
            u = norm.cdf(z)
        else:
            errmsg = "Sampling not handling copula type {copula_type}."
            raise ValueError(errmsg)

        return u, iparts
