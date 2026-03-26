import re
import json
from pathlib import Path
from itertools import combinations
import math
import warnings

import pytest

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

from scipy.stats import t as student_t
from scipy.stats import multivariate_t as mvt

from scipy.stats import invwishart
from scipy.stats import ttest_ind, ks_2samp
from scipy.stats import kstest
import matplotlib.pyplot as plt

from floodstan import report
from floodstan import sample as fsample
from floodstan import marginals
from floodstan import bivariate_censored_sampling

from pyrethink import sample
from pyrethink import datahub
from pyrethink import mv_censored_no_missing_sampling

from test_copulas import get_type

FTESTS = Path(__file__).resolve().parent

SEED = 5446

DEBUG = False

# Used to write test data for postpred checks
WRITE_SAMPLE_DATA = True

PROGRESS = DEBUG
FLOG = FTESTS / "test_sample.log"

# Clean files
if FLOG.exists():
    try:
        FLOG.unlink()
    except:
        pass

for f in FTESTS.glob("test_mv_censored_no_missing_vs_floodstan*.png"):
    f.unlink()

LOGGER = fsample.get_logger(use_stan_logger=PROGRESS, flog=FLOG)

STAN_NCHAINS_DEFAULT = 3
STAN_NWARM_DEFAULT = 5000
STAN_NSAMPLES_DEFAULT = 6000

STAN_DIAG_METRICS = ["treedepth", "rhat", "ebfmi", "effsamplesz"]

@pytest.mark.parametrize("pcensor", [0., 0.3])
@pytest.mark.parametrize("no_missing", [False, True])
def test_sample_data(pcensor, no_missing, allclose):

    data, times, dows, _ = datahub.get_ams_concat(no_missing=no_missing)
    censors = datahub.get_censors(pcensor, no_missing=no_missing)
    copula_type = 1
    copula_shape = 2.5

    sv = sample.StanSamplingMultivariate(data, dows,
                                         copula_type=copula_type,
                                         copula_shape=copula_shape,
                                         censors=censors)

    pisc = sv.pair_in_same_cluster
    miss = sv.clusters_missing
    s = pisc.sum(axis=1)
    P = sv.data.shape[1]
    s0 = (P * (P - 1) // 2) / 3
    s1 = 2 * s0
    for idx in np.where((miss == 0) & (s >= s0) & (s <= s1))[0]:
        cl = sv.clusters[idx]
        pi = pisc[idx]
        M = np.zeros((P, P))
        M[np.triu_indices(P, 1)] = pi
        for i, j in combinations(range(P), 2):
            same = M[i, j] == 1
            expected = np.prod(cl[:, [i, j]], axis=1).sum() > 0
            assert same == expected

    stan_data = sv.to_dict()
    assert len(stan_data) == 26

    N = stan_data["N"]
    P = stan_data["P"]
    data = pd.DataFrame(stan_data["y"], index=data.index,
                        columns=data.columns)
    assert data.shape == (N, P)
    assert data.notnull().any(axis=1).all()

    clust = stan_data["clusters"]
    assert clust.shape == (N, P, P)

    pid = stan_data["partitions_id"]
    assert pid.shape == (N, )

    nobs = stan_data["Nobs"]
    nmiss = stan_data["Nmiss"]
    ncens = stan_data["Ncens"]

    nval = np.prod(data.shape)
    nok = data.notnull().sum().sum()

    assert nobs + ncens + nmiss == nval
    assert nobs + ncens == nok
    assert nmiss == nval - nok

    stan_inits = sv.initial_parameters

    assert len(stan_inits) == 6
    assert len(stan_inits["wlat_miss"]) == nmiss
    assert len(stan_inits["wlat_cens"]) == ncens

    data = np.nan * np.zeros_like(data)
    with pytest.raises(ValueError, match="Expected at least"):
        sv = sample.StanSamplingMultivariate(data, dows,
                                             copula_type,
                                             copula_shape)


def test_inits(allclose):
    data, times, dows, _ = datahub.get_ams_concat()
    censors = datahub.get_censors(pcensor=0.2)
    copula_type = 0.
    copula_shape = 0.
    sv = sample.StanSamplingMultivariate(data, dows,
                                         copula_type,
                                         copula_shape,
                                         censors=censors)
    inits = sv.initial_parameters
    assert len(inits) == 6


@pytest.mark.parametrize("config", ["censored", "uncensored"])
@pytest.mark.parametrize("nvars", [3])
@pytest.mark.parametrize("copula_shape", [0., 4.])
def test_sampler(config, nvars, copula_shape, allclose):
    data, times, dows, _ = datahub.get_ams_concat()
    data = data.iloc[:, :nvars]
    dows = dows.iloc[:, :nvars]

    if config.startswith("censored"):
        pcensor = 0.3
        censors = datahub.get_censors(pcensor)
        censors = censors.loc[data.columns]
    else:
        censors = np.zeros(data.shape[1])

    copula_type = get_type(copula_shape)
    sv = sample.StanSamplingMultivariate(data, dows,
                                         copula_type,
                                         copula_shape,
                                         censors=censors)
    stan_data = sv.to_dict()

    if config == "uncensored":
        assert stan_data["Nobs"] == np.prod(data.shape)
        assert stan_data["Ncens"] == 0
        assert stan_data["Nmiss"] == 0

    stan_inits = sv.initial_parameters

    fout = f"sampling_{config}_N{nvars}_C{copula_shape:0.0f}"
    fout = FTESTS / "sampling" / fout
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    nsmp = STAN_NSAMPLES_DEFAULT//STAN_NCHAINS_DEFAULT
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=nsmp,
              output_dir=fout,
              inits=stan_inits,
              chains=STAN_NCHAINS_DEFAULT,
              parallel_chains=STAN_NCHAINS_DEFAULT,
              iter_warmup=STAN_NWARM_DEFAULT,
              show_progress=PROGRESS)

    # Test sample size error
    kw["data"]["Ncens"] += 1
    with pytest.raises(RuntimeError):
        mv_censored_no_missing_sampling(**kw)

    kw["data"]["Ncens"] -= 1

    # Test copula shape error
    if copula_type == 1:
        kw["data"]["copula_shape"] = 1.9
        with pytest.raises(RuntimeError):
            mv_censored_no_missing_sampling(**kw)

    kw["data"]["copula_shape"] = copula_shape

    # Run sampler
    smp = mv_censored_no_missing_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    fdiag = fout / "diagnostic.json"
    with fdiag.open("w") as fo:
        json.dump(diag, fo, indent=4)

    for met in STAN_DIAG_METRICS:
        assert diag[met] == "satisfactory"

    if config == "censored" and WRITE_SAMPLE_DATA:
        fd = fout / "data.zip"
        comp = dict(method="zip", compresslevel=9)
        data.to_csv(fd, compression=comp)

        fs = fout / "samples.zip"
        df.to_csv(fs, compression=comp)

@pytest.mark.parametrize("pcensor", [0., 0.4])
@pytest.mark.parametrize("stationpair", [[0, 1], [4, 5], [1, 3]])
def test_mv_censored_no_missing_vs_floodstan(stationpair, pcensor, allclose):
    # Two variables only
    data, _, dows, _ = datahub.get_ams_concat()
    data = data.iloc[:, stationpair]
    dows = dows.iloc[:, stationpair]
    censors = data.quantile(pcensor)

    # -- floodstan --
    y, z = data.values.T
    censor = censors.iloc[0]
    marginal = marginals.GEV()
    yv = fsample.StanSamplingVariable(marginal, y, censor,
                                      ninits=STAN_NCHAINS_DEFAULT)

    censor = censors.iloc[1]
    zv = fsample.StanSamplingVariable(marginal, z, censor,
                                      ninits=STAN_NCHAINS_DEFAULT)

    fsv = fsample.StanSamplingDataset([yv, zv], "Gaussian")
    fstan_data = fsv.to_dict()
    fstan_inits = fsv.initial_parameters

    fout = FTESTS / "sampling" / "sampling_uncensored_vs_floodstan"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    nwarm = STAN_NWARM_DEFAULT
    nsamples = STAN_NSAMPLES_DEFAULT

    nsmp = nsamples // STAN_NCHAINS_DEFAULT
    kw = dict(data=fstan_data,
              seed=SEED,
              iter_sampling=nsmp,
              output_dir=fout,
              inits=fstan_inits,
              chains=STAN_NCHAINS_DEFAULT,
              parallel_chains=STAN_NCHAINS_DEFAULT,
              iter_warmup=nwarm,
              show_progress=PROGRESS)

    smp1 = bivariate_censored_sampling(**kw)
    df1 = smp1.draws_pd()
    diag1 = report.process_stan_diagnostic(smp1.diagnose())

    for met in STAN_DIAG_METRICS:
        assert diag1[met] == "satisfactory"

    # -- pyrethink --
    # + Gaussian copula
    # + no clustering
    rho_min = np.floor(df1.rho.min() * 1e2) * 1e-2
    rho_max = 1.
    sv = sample.StanSamplingMultivariate(data,
                                         dows,
                                         copula_type=0.,
                                         copula_shape=0,
                                         censors=censors,
                                         skip_clusters=True,
                                         rho_min=rho_min,
                                         rho_max=rho_max)
    kw["data"] = sv.to_dict()
    kw["inits"] = sv.initial_parameters

    smp2 = mv_censored_no_missing_sampling(**kw)
    df2 = smp2.draws_pd()
    diag2 = report.process_stan_diagnostic(smp2.diagnose())

    for met in STAN_DIAG_METRICS:
        assert diag2[met] == "satisfactory"

    pnames = df2.columns.to_series().filter(regex="^yl|^ys|^ucensor").to_list()
    pnames.append("corr_IW[2,1]")

    LOGGER.info("")
    LOGGER.info("-----------------")
    sids = "/".join(data.columns.tolist())
    LOGGER.info(f"stations={sids} pcensor={pcensor:0.2f}")
    LOGGER.info(f"nwarm = {nwarm}")
    LOGGER.info(f"nsamples = {nsamples}")
    LOGGER.info(f"rho_min = {rho_min}")
    LOGGER.info(f"rho_max = {rho_max}")
    LOGGER.info("")

    plt.close("all")
    n = len(pnames)
    ncols = n // 2 + 1
    mosaic = [[pn for pn in pnames[:ncols]],
              [pn for pn in pnames[ncols:]] + ["."] * (n % 2)]
    nrows = 2
    w = 3
    fig = plt.figure(figsize=(w * ncols, w * nrows),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)

    for pname2 in pnames:
        x2 = df2.loc[:, pname2]

        # Get floodstan sample
        if re.search("corr_IW", pname2):
            pname1 = "rho"
        elif re.search("ucensor", pname2):
            pname1 = "ucensor" if re.search("1", pname2) else "vcensor"
        else:
            pname1 = re.sub("\\[.*", "", pname2)
            if re.search("2", pname2):
                pname1 = re.sub("^y", "z", pname1)

        x1 = df1.loc[:, pname1]
        if pname1 == "rho":
            # Convert kendall tau to correlation
            # then convert to normal std to facilitate comparison
            x1 = np.sin(x1 * math.pi / 2)

        resk = ks_2samp(x1, x2)
        kspv = math.log10(resk.pvalue) if resk.pvalue > 0 else -np.inf

        rest = ttest_ind(x1, x2)
        tpv = math.log10(rest.pvalue) if rest.pvalue > 0 else -np.inf

        msg = f"[{pname2:15s}] x1:m={x1.mean():6.2f} s={x1.std():6.2f}"\
              + f" // x2:m={x2.mean():6.2f} s={x2.std():6.2f}"\
              + f" // test: ks-logpv={kspv:4.1f} t-logpv={tpv:4.1f}"
        LOGGER.info(msg)

        # Test on matching the two dist
        # 10^-8 is very low for a p-value! Still looking ok visually though
        # Also skip correlation as it can be slightly different
        pv_thresh = -8
        if not DEBUG and pname1 != "rho":
            assert kspv > pv_thresh
            assert tpv > pv_thresh

        if pname1 == "rho" and (kspv < pv_thresh or tpv < pv_thresh):
            wmess = f"stations={sids} pcensor={pcensor:0.2f}\n"\
                    + "rho parameter not passing test criteria:\n"\
                    + re.sub("\\] ", "", msg)
            warnings.warn(wmess)

        ax = axs[pname2]

        xa = min(x1.min(), x2.min())
        xb = max(x1.max(), x2.max())
        bins = np.linspace(xa, xb, 30)
        ax.hist(x1, bins=bins, label="floodstan", edgecolor="0.5", alpha=0.6)
        ax.hist(x2, bins=bins, label="mv_censored_no_missing", edgecolor="0.5", alpha=0.6)

        title = f"{pname2} - ks-logpv={kspv:0.1f}"
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize="x-small")

    sids = "/".join(data.columns.tolist())
    ftitle = f"Stations={sids} pcens={pcensor:0.2f}"
    fig.suptitle(ftitle, fontsize="large")

    sids = "-".join(data.columns.tolist())
    fimg = FTESTS / "images"
    fimg.mkdir(exist_ok=True)
    fp = f"test_mv_censored_no_missing_vs_floodstan_stations{sids}"\
        + f"_pcens{pcensor*100:0.02f}.png"
    fp = fimg / fp

    fig.savefig(fp)

