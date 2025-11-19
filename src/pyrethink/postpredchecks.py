from itertools import combinations as combs
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import kendalltau

from floodstan.marginals import lh_moments
from floodstan.marginals import GEV


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


def bivariate_dependence_statistics(data):
    stats = np.zeros(2)
    data = np.array(data)
    data = data[np.all(~np.isnan(data), axis=1)]

    # Kendall tau
    stats[0] = kendalltau(data[:, 0], data[:, 1]).statistic

    medians = np.nanmedian(data, axis=0)
    ihigh = (data[:, 0] >= medians[0]) & (data[:, 1] >= medians[1])
    stats[1] = kendalltau(data[ihigh, 0], data[ihigh, 1]).statistic

    idx = ["kendalltau", "kendalltauhigh"]
    return pd.Series(stats, index=idx)


def generate_samples(params, nval):
    ylocn = params.filter(regex="ylocn").values
    ylogscale = params.filter(regex="ylogscale").values
    yshape1 = params.filter(regex="yshape1").values

    P = len(ylocn)
    cor = params.filter(regex="cor_IW").values.reshape((P, P)).T

    z = np.random.multivariate_normal(mean=np.zeros(P), cov=cor,
                                      size=nval)
    ge = GEV()
    y = np.zeros_like(z)
    for i in range(P):
        ge.params = [ylocn[i], ylogscale[i], yshape1[i]]
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

        ysim = generate_samples(param, nval)

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
