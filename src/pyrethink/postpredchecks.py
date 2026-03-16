from itertools import combinations as combs
import numpy as np
import pandas as pd

from scipy.stats import kendalltau
from scipy.interpolate import RBFInterpolator

from floodstan.marginals import lh_moments
from floodstan.marginals import GEV

from pyrethink.copulas import Copula


PERC_TAILS_DEFAULT = np.arange(50, 95, 5)


def univariate_statistics(data, eta=2, perc_tails=np.arange(50, 110, 10)):
    stats = np.zeros(3 + len(perc_tails))

    lmom = lh_moments(data, eta, True)
    stats[0] = lmom[1] / lmom[0]
    stats[1] = lmom[2] / lmom[1]
    stats[2] = lmom[3] / lmom[1]

    stats[3:] = np.nanpercentile(data, perc_tails)

    idx = [f"l{m}{eta}"
           for m in ["coeffvar", "skewness", "kurtosis"]]\
        + [f"percentile_q{q}" for q in perc_tails]

    return pd.Series(stats, index=idx)


def joint_exceedance_probabilities(data, perc_tails, alpha=0.):
    data = np.array(data)
    N, P = data.shape
    rk = np.argsort(np.argsort(data, axis=0), axis=0)
    denom = N + 1 - 2 * alpha
    marginal = (rk + 1 - alpha) / denom

    diff = rk[:, None, :] - rk[None, :, :]
    cdf = (np.all(diff >= 0, axis=-1).sum(axis=1) - alpha) / denom
    surv = (np.all(diff <= 0, axis=-1).sum(axis=1) - alpha) / denom

    prob = np.array(perc_tails) * 1e-2
    if P > 1:
        x = np.repeat(prob[:, None], P, axis=1)
        plow_joint = RBFInterpolator(marginal, cdf)(x)
        phigh_joint = RBFInterpolator(marginal, surv)(x)
    else:
        plow_joint = prob
        phigh_joint = 1 - prob

    return plow_joint, phigh_joint


def xi_bounds(prob):
    xi0 = 2 - np.log(2 * prob - 1) / np.log(prob)
    xi1 = np.ones_like(prob)
    return xi0, xi1


def multivariate_dependence_statistics(data,
                                       perc_tails=PERC_TAILS_DEFAULT):
    """
    See Coles, S., Heffernan, J., & Tawn, J. (1999).
    Dependence Measures for Extreme Value Analyses.
    Extremes, 2(4), 339–365.
    https://doi.org/10.1023/A:1009963131610

    Joe, H. (2014). Dependence Modeling with Copulas (0 ed.).
    Chapman and Hall/CRC.
    https://doi.org/10.1201/b17116
    Section 2.13
    """
    data = np.array(data)
    data = data[np.all(~np.isnan(data), axis=1)]
    N, P = data.shape

    p0, p1 = joint_exceedance_probabilities(data, perc_tails)
    u = perc_tails / 100

    tau = p1 / (1 - u)
    xi = 2 - np.log(p0) / np.log(u)
    xibar = 2 * np.log(1 - u) / np.log(p1) - 1

    df = pd.DataFrame({"tau": tau, "xi": xi, "xibar": xibar},
                      index=perc_tails)
    df.index.name = "percentile"
    return df


def dependence2series(dep):
    se = pd.melt(dep.reset_index(), id_vars="percentile")
    se.loc[:, "stat"] = se.variable + "_q" + se.percentile.astype(str)
    return se.set_index("stat").loc[:, "value"]


def bivariate_dependence_statistics(data,
                                    perc_tails=PERC_TAILS_DEFAULT):
    data = np.array(data)
    data = data[np.all(~np.isnan(data), axis=1)]
    N, P = data.shape
    x = data[:, 0]
    y = data[:, 1]

    if P != 2:
        errmsg = "Expected 2 columns in data, got P={P}."
        raise ValueError(errmsg)

    # Kendall tau
    stats = {}
    stats["kendalltau"] = kendalltau(x, y).statistic

    # Kendall tau above medians
    medians = np.nanmedian(data, axis=0)
    ihigh = (x >= medians[0]) & (y >= medians[1])
    n = "kendalltau_high"
    stats[n] = kendalltau(x[ihigh], y[ihigh]).statistic

    # Tail dependence
    stats["dependence"] = multivariate_dependence_statistics(data, perc_tails)

    return stats


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
                                copula_type,
                                copula_shape,
                                partitions_id,
                                dirichlet_alpha,
                                logger=None,
                                iterlog=500):
    yobs = np.array(yobs)

    # Dimensions
    nval, nsta = yobs.shape
    nsamples = len(params)

    # copula sampling tools
    copsamp = Copula(copula_type, copula_shape, nsta)
    probs = copsamp.partitions.compute_probabilities(partitions_id,
                                                     dirichlet_alpha)
    # Compute obs
    univ_obs = pd.DataFrame([univariate_statistics(v)
                             for v in yobs.T]).T
    univ_obs.columns = [f"univ[{ivar + 1}]" for ivar in range(nsta)]

    biv_obs = []
    for i1, i2 in combs(range(nsta), 2):
        m = bivariate_dependence_statistics(yobs[:, [i1, i2]])
        se = dependence2series(m["dependence"])
        for cn in ["kendalltau", "kendalltau_high"]:
            se[cn] = m[cn]
        se.name = f"bivariate[{i1 + 1},{i2 + 1}]"
        biv_obs.append(se)

    biv_obs = pd.DataFrame(biv_obs).T

    multi_obs = multivariate_dependence_statistics(yobs)
    multi_obs = dependence2series(multi_obs)

    # Marginal
    gev = GEV()

    # Loop over params
    univ_sim = {ivar: [] for ivar in range(nsta)}
    biv_sim = {(i1, i2): [] for i1, i2 in combs(range(nsta), 2)}
    multi_sim = []

    for isample, param in params.iterrows():
        if logger is not None and isample % iterlog == 0:
            msg = f"[postpred] processing param {isample + 1:5d} / {nsamples}."
            logger.info(msg)

        # Sample data with same size as obs
        corr = param.filter(regex="corr_IW").values.reshape((nsta, nsta))
        copsamp.corr = corr
        usim, iparts = copsamp.sample_u(probs, nval)
        ysim = np.empty((nval, nsta))
        for ista in range(nsta):
            cc = [f"ylocn[{ista + 1}]", f"ylogscale[{ista + 1}]",
                  f"yshape1[{ista + 1}]"]
            gev.params = param.loc[cc]
            ysim[:, ista] = gev.ppf(usim[:, ista])

        # Compute
        for ivar in range(nsta):
            un = univariate_statistics(ysim[:, ivar])
            univ_sim[ivar].append(un)

        for i1, i2 in combs(range(nsta), 2):
            bi = bivariate_dependence_statistics(ysim[:, [i1, i2]])
            se = dependence2series(bi["dependence"])
            for cn in ["kendalltau", "kendalltau_high"]:
                se[cn] = bi[cn]
            se.name = f"bivariate[{i1 + 1},{i2 + 1}]"
            biv_sim[(i1, i2)].append(se)

        mult = multivariate_dependence_statistics(ysim)
        mult = dependence2series(mult)
        multi_sim.append(pd.DataFrame(mult).T)

    # Compute predictiive checks
    # .. univariate
    pcheck_univ = []
    for ivar in range(nsta):
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
    for i1, i2 in combs(range(nsta), 2):
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

    # ... multivariate
    multi_sim = pd.concat(multi_sim, ignore_index=True)
    pcheck_multi = compute_predictive_checks(multi_obs, multi_sim)

    data = {
        "univ_obs": univ_obs,
        "univ_sim": univ_sim,
        "biv_obs": biv_obs,
        "biv_sim": biv_sim,
        "multi_obs": multi_obs,
        "multi_sim": multi_sim
        }

    return pcheck_univ, pcheck_biv, pcheck_multi, data
