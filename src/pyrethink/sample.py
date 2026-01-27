from itertools import product as prod
import numpy as np
import pandas as pd

from floodstan.data_processing import univariate2cases
from floodstan.marginals import GEV

STUDENT_DF_MAX = 5.

RHO_MIN_DEFAULT = 0.
RHO_MAX_DEFAULT = 1.

MARGINAL = GEV()

DELTA_DAYS_MAX_DEFAULT = 10


class Partitions():
    def __init__(self, nelements):
        if nelements > 10:
            errmsg = "Expected nelements <= 10."
            raise ValueError(errmsg)

        self.nelements = nelements
        self.data = list(range(nelements))
        self.subsets = []
        self.counts = []
        self.nsubsets = 0
        self.add_subsets(0, [])

    def add_subsets(self, index, ans):
        data = self.data
        nel = self.nelements

        if index == len(data):
            combs = []
            nmax = 0
            for ipart, parts in enumerate(ans):
                comb = [0] * nel
                for d in parts:
                    comb[d] = 1
                combs.append(comb)
                nmax = max(nmax, len(parts))

            self.counts.append([len(combs), nmax])
            self.subsets.append(combs)
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

    def select(self, nevents, nelem=None):
        selected = []
        for i in range(self.nsubsets):
            nev, nel = self.counts[i]
            cel = True if nelem is None else nel == nelem
            if nev == nevents and cel:
                selected.append(self.subsets[i])

        return selected


class StanSamplingMultivariate():
    def __init__(self, data, day_of_year, copula,
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

        self.set_data(data, day_of_year, censors)

        self.delta_days_max = delta_days_max
        self.set_clusters()

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

    def set_clusters(self):
        dmax = self.delta_days_max
        N, P = self.data.shape
        clusters = np.zeros((N, P, P), dtype=int)
        counts = np.zeros((N, 3), dtype=int)
        day_of_year = self.day_of_year

        for i in range(N):
            doy = day_of_year[i]
            is_valid = np.where(doy >= 0)[0]
            clusters[i, :, doy < 0] = -1

            doy = doy[is_valid]
            if len(doy) == 1:
                clusters[i, 0, is_valid] = 1
            else:
                rk = np.argsort(np.argsort(doy))
                doys = np.sort(doy)
                clust = np.cumsum(np.insert(np.diff(doys), 0, 0) > dmax)
                clust = clust[rk]
                for icl, cl in enumerate(np.unique(clust)):
                    isin = is_valid[cl == clust]
                    clusters[i, icl, isin] = 1

            # count number of events and max number of
            # stations per event
            sm = clusters[i][:, is_valid].sum(axis=1)
            nev = (sm > 0).sum()
            nstamax = sm.max()
            nmiss = P - len(is_valid)
            counts[i, :] = nmiss, nev, nstamax

        self.clusters = clusters
        self.clusters_counts = counts

        nevs = np.unique(counts[:, 1])
        nsta = np.unique(counts[:, 2])

        cases = pd.DataFrame(0, index=nevs, columns=nsta)
        cases.index.name = "nevents"
        cases.columns.name = "nstations_per_cluster_max"

        for nev, nsta in prod(nevs, nsta):
            idx = (counts[:, 0] == 0) & (counts[:, 1] == nev) \
                & (counts[:, 2] == nsta)
            ncases = idx.sum()
            cases.loc[nev, nsta] = ncases

        self.clusters_counts_cases = cases

    def set_partitions(self):
        P = self.data.shape[1]
        parts = Partitions(P)
        nsubsets = parts.nsubsets

        cases = self.clusters_counts_cases
        nevs = cases.index
        nsta = cases.columns
        ntot = cases.sum().sum()

        probs = []
        submats = []
        for nev, nsta in prod(nevs, nsta):
            nc = cases.loc[nev, nsta]
            if nc > 0:
                subs = parts.select(nev, nsta)

                # Probability
                nsubs = len(subs)
                # WRONG !!! should not use nsubsets
                # only the number of selected subs
                prob = nc / ntot * nsubs / nsubsets
                probs.extend([prob] * nsubs)

                for isub, sub in enumerate(subs):
                    # Convert subset to matrix
                    smat = np.zeros((P, P))
                    for i, s in enumerate(sub):
                        smat[i, :] = s

                    submats.append(smat)

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
            "clusters_counts": self.clusters_counts,
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
