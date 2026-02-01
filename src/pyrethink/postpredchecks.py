from itertools import combinations as combs
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt
from scipy.stats import kendalltau

from floodstan.marginals import lh_moments
from floodstan.marginals import GEV

from pyrethink.sample import STUDENT_DF_MAX


def univariate_statistics(data, eta=2, qtails=[50, 75, 90, 95]):
    stats = np.zeros(3 + len(qtails))

    lmom = lh_moments(data, eta, True)
    stats[0] = lmom[1] / lmom[0]
    stats[1] = lmom[2] / lmom[1]
    stats[2] = lmom[3] / lmom[1]

    stats[3:] = np.nanpercentile(data, qtails)

    idx = [f"l{m}{eta}"
           for m in ["coeffvar", "skewness", "kurtosis"]]\
        + [f"percentile_q{q}" for q in qtails]

    return pd.Series(stats, index=idx)


def bivariate_dependence_statistics(data,
                                    qtails=[50, 75, 90, 95]):
    data = np.array(data)
    data = data[np.all(~np.isnan(data), axis=1)]
    N, P = data.shape
    x = data[:, 0]
    y = data[:, 1]

    if P != 2:
        errmsg = "Expected 2 columns in data, got P={P}."
        raise ValueError(errmsg)

    names = ["kendalltau", "kendalltau_high"]\
        + [f"taildep_q{q}" for q in qtails]
    stats = pd.Series(np.nan, index=names)

    # Kendall tau
    stats["kendalltau"] = kendalltau(x, y).statistic

    # Kendall tau above medians
    medians = np.nanmedian(data, axis=0)
    ihigh = (x >= medians[0]) & (y >= medians[1])
    n = "kendalltau_high"
    stats[n] = kendalltau(x[ihigh], y[ihigh]).statistic

    # Tail dependence
    qt = np.nanpercentile(data, qtails, axis=0).T
    is_greater = data[:, :, None] - qt[None, :, :] >= 0
    cnt = is_greater.all(axis=1).sum(axis=0)
    for iq, q in enumerate(qtails):
        stats[f"taildep_q{q}"] = cnt[iq] / N

    return stats


def generate_samples(params, copula, nsamples):
    if copula > STUDENT_DF_MAX or copula < 0:
        errmsg = f"Expected df in [0, {STUDENT_DF_MAX}], got {df}."
        raise ValueError(errmsg)

    ylocn = params.filter(regex="ylocn").values
    ylogscale = params.filter(regex="ylogscale").values
    yshape1 = params.filter(regex="yshape1").values

    P = len(ylocn)
    corr = params.filter(regex="corr_IW").values.reshape((P, P)).T

    if copula > 0:
        z = mvt.rvs(loc=np.zeros(P), shape=corr, df=copula, size=nsamples)
    else:
        z = mvn.rvs(mean=np.zeros(P), cov=corr, size=nsamples)

    ge = GEV()
    y = np.zeros_like(z)
    for i in range(P):
        ge.params = [ylocn[i], ylogscale[i], yshape1[i]]
        if copula > 0:
            u = student_t.cdf(z[:, i], df=copula)
        else:
            u = norm.cdf(z[:, i])

        y[:, i] = ge.ppf(u)

    return y


def compute_predictive_checks(metric_obs, metric_sim):
    nparams = len(metric_sim)
    sim_mean = metric_sim.mean()
    sim_std = metric_sim.std()

    pvalue = (metric_sim - metric_obs >= 0).sum(axis=0) / (nparams + 1)

    diff = (metric_sim < metric_obs) | (metric_sim > metric_obs)
    ndiff = diff.sum()
    pvalue_discr = (metric_sim - metric_obs > 0).sum(axis=0) / (ndiff + 1)

    pchecks = {
        "obs": metric_obs,
        "simmean": sim_mean,
        "pvalue": pvalue,
        "pvaluediscr": pvalue_discr,
        "simdiff": sim_mean - metric_obs,
        "simstd": sim_std,
        "zscore": (sim_mean - metric_obs) / sim_std
        }
    return pd.DataFrame(pchecks)


def posterior_predictive_checks(yobs, params,
                                copula,
                                logger=None,
                                iterlog=500):
    yobs = np.array(yobs)
    nsamples = len(params)

    # Compute obs
    univ_obs = pd.DataFrame([univariate_statistics(v)
                             for v in yobs.T]).T
    nvar = yobs.shape[1]
    univ_obs.columns = [f"univ[{ivar + 1}]" for ivar in range(nvar)]

    biv_obs = []
    for i1, i2 in combs(range(nvar), 2):
        m = bivariate_dependence_statistics(yobs[:, [i1, i2]])
        m.name = f"bivariate[{i1 + 1},{i2 + 1}]"
        biv_obs.append(m)

    biv_obs = pd.DataFrame(biv_obs).T

    # Loop over params
    nval = len(yobs)
    univ_sim = {ivar: [] for ivar in range(nvar)}
    biv_sim = {(i1, i2): [] for i1, i2 in combs(range(nvar), 2)}

    for isample, param in params.iterrows():
        if logger is not None and isample % iterlog == 0:
            msg = f"[postpred] processing param {isample + 1:5d} / {nsamples}."
            logger.info(msg)

        ysim = generate_samples(param, copula, nval)

        for ivar in range(nvar):
            un = univariate_statistics(ysim[:, ivar])
            univ_sim[ivar].append(un)

        for i1, i2 in combs(range(nvar), 2):
            bi = bivariate_dependence_statistics(ysim[:, [i1, i2]])
            biv_sim[(i1, i2)].append(bi)

    # Compute predictiive checks
    # .. univariate
    pcheck_univ = []
    for ivar in range(nvar):
        # Reformat univ sim stats
        usim = pd.concat(univ_sim[ivar], axis=1).T
        univ_sim[ivar] = usim

        # Get univ obs stats
        uobs = univ_obs.loc[:, f"univ[{ivar + 1}]"]

        # Compute pred check
        pc = compute_predictive_checks(uobs, usim)
        pc.columns = [f"{cn}[{ivar + 1}]" for cn in pc.columns]
        pcheck_univ.append(pc)

    pcheck_univ = pd.concat(pcheck_univ, axis=1)

    # .. bivariate
    pcheck_biv = []
    for i1, i2 in combs(range(nvar), 2):
        # Reformat biv sim stats
        bsim = pd.concat(biv_sim[(i1, i2)], axis=1).T
        biv_sim[(i1, i2)] = bsim

        # Get biv obs stats
        bobs = biv_obs.loc[:, f"bivariate[{i1 + 1},{i2 + 1}]"]

        # Compute pred check
        pc = compute_predictive_checks(bobs, bsim)
        pc.columns = [f"{cn}[{i1 + 1},{i2 + 1}]" for cn in pc.columns]
        pcheck_biv.append(pc)

    pcheck_biv = pd.concat(pcheck_biv, axis=1)
    data = {
        "univ_obs": univ_obs,
        "univ_sim": univ_sim,
        "biv_obs": biv_obs,
        "biv_sim": biv_sim
        }
    return pcheck_univ, pcheck_biv, data
