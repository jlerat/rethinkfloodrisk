import math
from itertools import combinations

import numpy as np
from scipy.stats import norm
from scipy.stats import invwishart
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt

from floodstan.data_processing import univariate2cases
from floodstan.marginals import GEV

STUDENT_DF_MIN = 2.01
STUDENT_DF_MAX = 1000.

RHO_MIN_DEFAULT = 0.
RHO_MAX_DEFAULT = 1.

MARGINAL = GEV()

DELTA_DAYS_MAX_DEFAULT = 10


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


def copula_marginal_ppf(copula, u):
    check_copula(copula)
    if copula > 0:
        scale = math.sqrt((copula - 2) / copula)
        return student_t.ppf(u, scale=scale, df=copula)
    else:
        return norm.ppf(u)


def copula_marginal_cdf(copula, z):
    check_copula(copula)
    if copula > 0:
        scale = math.sqrt((copula - 2) / copula)
        return student_t.cdf(z, scale=scale, df=copula)
    else:
        return norm.cdf(z)


class Partitions():
    def __init__(self, nelements):
        if nelements > 10:
            errmsg = "Expected nelements <= 10."
            raise ValueError(errmsg)

        # Initialise
        self.nelements = nelements
        self.data = list(range(nelements))
        self.subsets = []
        self.pair_in_same_cluster = []
        self.subsets_counts = []
        self.nsubsets = 0

        # Populate
        self.add_subsets(0, [])

        # Convert to arrays
        self.subsets = np.array(self.subsets)
        self.subsets_counts = np.array(self.subsets_counts)
        self.pair_in_same_cluster = np.array(self.pair_in_same_cluster)

    @property
    def npairs(self):
        nel = self.nelements
        return nel * (nel - 1) // 2

    def add_subsets(self, index, ans):
        data = self.data
        nel = self.nelements
        npairs = self.npairs

        if index == len(data):
            combs = np.zeros((nel, nel))
            insame = [0] * npairs
            for ipart, parts in enumerate(ans):
                for d in parts:
                    combs[ipart, d] = 1

                for ic, (a, b) in enumerate(combinations(range(nel), 2)):
                    v = combs[ipart, a] * combs[ipart, b] or insame[ic]
                    insame[ic] = int(v)

            self.pair_in_same_cluster.append(insame)
            self.subsets.append(combs)
            self.subsets_counts.append(len(ans))
            self.nsubsets += 1

            return

        elem = data[index]

        for i in range(len(ans)):
            ans[i].append(elem)
            self.add_subsets(index + 1, ans)
            ans[i].pop()

        ans.append([elem])
        self.add_subsets(index + 1, ans)
        ans.pop()

    def find_subset_index(self, pair_in_same):
        found = []
        for i in range(self.nsubsets):
            same = self.pair_in_same_cluster[i]
            if all(s == p for s, p in zip(same, pair_in_same)):
                found.append(i)

        return found

    def extract_index(self, index):
        return [self.subsets[i] for i in index]

    def find_subset(self, pair_in_same):
        idx = self.find_subset_index(pair_in_same)
        return self.extract_index(idx)

    def ipart2sets(self, ipart):
        part = self.subsets[ipart]
        cnt = self.subsets_counts[ipart]
        part = part[:cnt]
        i1, i2 = np.where(part == 1)
        return i1[np.argsort(i2)]


def check_copula(copula):
    if copula > 0:
        mini = STUDENT_DF_MIN
        maxi = STUDENT_DF_MAX
        if copula < mini or copula > maxi:
            errmsg = f"Expected copula in [{mini}, {maxi}]."
            raise ValueError(errmsg)

    elif copula < 0:
        errmsg = "Expected copula >= 0."
        raise ValueError(errmsg)


class StanSamplingMultivariate():
    def __init__(self, data, day_of_year, copula,
                 censors=None,
                 dirichlet_alpha=2,
                 rho_min=RHO_MIN_DEFAULT,
                 rho_max=RHO_MAX_DEFAULT,
                 delta_days_max=DELTA_DAYS_MAX_DEFAULT):

        # Check data
        check_copula(copula)

        if rho_min < -1 or rho_min > 1:
            errmsg = f"Expected rho_min in [-1, 1], got {rho_min}."
            raise ValueError(errmsg)

        if rho_max <= rho_min or rho_max > 1:
            errmsg = f"Expected rho_max in ]{rho_min}, 1], got {rho_max}."
            raise ValueError(errmsg)

        self.copula = copula
        self.rho_min = rho_min
        self.rho_max = rho_max

        # No clustering accounted if dirichlet prior < 1
        # (i.e. no prior)
        skip_clusters = dirichlet_alpha < 1
        self.dirichlet_alpha = max(1, dirichlet_alpha)

        self.set_data(data, day_of_year, censors)

        self.delta_days_max = delta_days_max
        self.set_clusters(skip_clusters)

        self.set_partitions()

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
        insame = np.zeros((N, P * (P - 1) // 2), dtype=int)
        day_of_year = self.day_of_year

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

        self.clusters = clusters
        self.clusters_counts = counts
        self.clusters_missing = miss
        self.pair_in_same_cluster = insame

    def set_partitions(self):
        # Dirichlet prior parameter
        alpha = self.dirichlet_alpha

        # Initialise
        N, P = self.data.shape
        parts = Partitions(P)
        self.partition_object = parts
        self.partitions = parts.subsets
        self.partitions_counts = parts.subsets_counts

        # Get pairs from observed data
        pair_in_same_data = self.pair_in_same_cluster

        # Exclude missing data
        miss = self.clusters_missing
        pair_in_same_data = pair_in_same_data[miss == 0]

        # Compute probabilities from max posterior of
        # categorical dist with Dirichlet prior
        # see https://en.wikipedia.org/wiki/Categorical_distribution
        Q = parts.nsubsets
        probs = np.zeros(Q)

        for i in range(Q):
            # Find all observed clusters matching the given partition
            pair_in_same = parts.pair_in_same_cluster[i]
            diff = np.abs(pair_in_same_data - pair_in_same[None, :])
            nmatch = (diff.sum(axis=1) == 0).sum()

            # maximum posterior of categorical prob
            probs[i] = (alpha + nmatch - 1)

        probs = probs / probs.sum()
        self.partitions_probabilities = probs

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
            "Q": self.partitions.shape[0],
            "partitions": self.partitions.astype(int),
            "partitions_counts": self.partitions_counts.astype(int),
            "partitions_probabilities":
                self.partitions_probabilities,
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


class CopulaSampling():
    def __init__(self, copula, nstations):
        check_copula(copula)
        self._copula = copula
        self._nstations = nstations
        self._partitions = Partitions(nstations)
        self._mean = np.zeros(nstations)

    @property
    def copula(self):
        return self._copula

    @property
    def nstations(self):
        return self._nstations

    @property
    def partitions(self):
        return self._partitions

    @property
    def corr(self):
        return self._corr

    @property
    def corr_rescaled(self):
        copula = self.copula
        cov_adjust = (copula - 2) / copula if copula > 0 else 1.
        return self.corr * cov_adjust

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
    def mean(self):
        return self._mean

    def conditional_sample(self, ipart, icond, zcond, itarget):
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

        copula = self.copula
        corr_rescaled = self.corr_rescaled

        # Sample targets that are in same set than cond
        idx_join = sets_cond == sets_target
        ijoin = itarget[idx_join]
        z = np.zeros(len(itarget))

        if len(ijoin) > 0:
            # Conditional covariance matrices
            S22 = np.ascontiguousarray(corr_rescaled[ijoin][:, ijoin])
            S21 = np.ascontiguousarray(corr_rescaled[ijoin][:, icond])

            # Full matrices formula:
            # muc = S21 @ S11i @ zcond
            # Sc = S22 - S21 @ S11i @ S21.T

            # .. because S11 is a single number:
            muc = S21 @ zcond
            Sc = S22 - S21 @ S21.T

            if copula > 0:
                # See https://en.wikipedia.org/wiki/Multivariate_t-distribution
                #    #Conditional_Distribution
                nu = copula

                # Full matrices
                # p1 = len(zcond)
                # d1 = zcond.T @ S11i @ zcond

                # .. because S11i and zcond are single numbers
                # ..
                p1 = 1
                d1 = zcond**2

                a = (nu + d1) / (nu + p1)
                df = nu + p1
                z[idx_join] = mvt.rvs(loc=muc, shape=a*Sc, df=df)

            else:
                z[idx_join] = mvn.rvs(mean=muc, cov=Sc)

        # Sample targets that are not in same set than cond
        idx_sep = sets_cond != sets_target
        isep = itarget[idx_sep]

        if len(isep) > 0:
            S22 = np.ascontiguousarray(corr_rescaled[isep][:, isep])
            mu22 = self.mean[isep]
            if copula > 0:
                z[idx_sep] = mvt.rvs(loc=mu22, shape=S22, df=copula)
            else:
                z[idx_sep] = mvn.rvs(mean=mu22, cov=S22)

        return z

    def cdf(self, ipart, z):
        sets = self.partitions.ipart2sets(ipart)
        copula = self.copula
        mu = self.mean
        corr_rescaled = self.corr_rescaled

        value = 1.
        for iset in np.unique(sets):
            idx = iset == sets
            scorr = np.ascontiguousarray(corr_rescaled[idx][:, idx])

            if copula > 0:
                rv = mvt(loc=mu[idx], shape=scorr, df=copula)
            else:
                rv = mvn(mean=mu[idx], cov=scorr)

            value *= rv.cdf(z[idx])

        return value

    def sample(self, ipart, nsamples):
        sets = self.partitions.ipart2sets(ipart)
        copula = self.copula
        mu = self.mean
        corr_rescaled = self.corr_rescaled

        nsta = self.nstations
        z = np.empty((nsamples, nsta))

        for iset in np.unique(sets):
            idx = iset == sets
            nval = idx.sum()
            scorr = np.ascontiguousarray(corr_rescaled[idx][:, idx])

            if copula > 0:
                rv = mvt(loc=mu[idx], shape=scorr, df=copula)
            else:
                rv = mvn(mean=mu[idx], cov=scorr)

            z[:, idx] = rv.rvs(size=nsamples).reshape((nsamples, nval))

        return z
