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
    L_cor = params.filter(regex="L_cor").values.reshape((P, P)).T
    cor = L_cor @ L_cor.T

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


def posterior_predictive_checks(yobs, params):
    yobs = np.array(yobs)

    # Compute obs
    univ_obs = pd.DataFrame([univariate_statistics(v)
                             for v in yobs.T])
    P = yobs.shape[1]
    univ_obs.columns = [f"univ_{i}" for i in range(P)]

    biv_obs = []
    for i1, i2 in combs(range(P), 2):
        m = bivariate_dependence_statistics(yobs[:, [i1, i2]])
        m.name = "bivariate_{i1}_{i2}"
        biv_obs.append(m)
    biv_obs = pd.DataFrame(biv_obs)

    # Loop over params
    nval = len(yobs)
    univ_sim = []
    biv_sim = []
    for iparam, param in params.iterrows():
        ysim = generate_samples(param, nval)

        univ = pd.DataFrame([univariate_statistics(v)
                             for v in ysim.T])
        univ_sim.append(univ)

        biv = []
        for i1, i2 in combs(range(P), 2):
            m = bivariate_dependence_statistics(yobs[:, [i1, i2]])
            m.name = "bivariate_{i1}_{i2}"
            biv.append(m)
        biv = pd.DataFrame(biv)
        biv_sim.append(biv)

    # pcheck_y = compute_predictive_checks(univy_obs, univy_sim)
    # pcheck_z = compute_predictive_checks(univz_obs, univz_sim)
    # pcheck_dep = compute_predictive_checks(dep_obs, dep_sim)

    data = {
        "univ_obs": univ_obs,
        "univ_sim": univ_sim,
        "biv_obs": biv_obs,
        "biv_sim": biv_sim
        }

    return data
