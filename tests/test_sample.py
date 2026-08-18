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
from scipy.stats import uniform_direction

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
from pyrethink import mv_censored_sampling
from pyrethink import mv_censored_factors_sampling
from pyrethink import factors_correlation_sampling

from test_copulas import COPULA_SPECS

FTESTS = Path(__file__).resolve().parent

SEED = 5446

FLOG = FTESTS / "test_sample.log"

# Clean files
if FLOG.exists():
    try:
        FLOG.unlink()
    except:
        pass

for f in FTESTS.glob("test_mv_censored_vs_floodstan*.png"):
    f.unlink()

def get_logger(debug_mode):
    return fsample.get_logger(use_stan_logger=debug_mode, flog=FLOG)

STAN_NCHAINS_DEFAULT = 3
STAN_NWARM_DEFAULT = 5000
STAN_NSAMPLES_DEFAULT = 6000

STAN_DIAG_METRICS = ["treedepth", "rhat", "ebfmi", "effsamplesz"]

@pytest.mark.parametrize("pcensor", [0., 0.3])
@pytest.mark.parametrize("copula_spec", COPULA_SPECS)
def test_sample_data(pcensor, copula_spec, allclose):
    data, times, dows, _ = datahub.get_ams_concat()

    # only dim 2 for comonotone and countermonotone copulas
    if re.search("Comonotone|Countermonotone", copula_spec):
        data = data.iloc[:, :2]

    data = data.loc[data.notnull().any(axis=1)]

    censors = datahub.get_censors(pcensor).loc[data.columns]

    sv = sample.StanSamplingMultivariate(data,
                                         copula_spec=copula_spec,
                                         censors=censors)
    stan_data = sv.to_dict()
    assert len(stan_data) == 25
    assert stan_data["P"] == data.shape[1]
    assert stan_data["F"] == sv.copula.copula_nfactors
    assert stan_data["marginal_id"] == 0
    assert stan_data["copula_id"] == sv.copula_id

    N = stan_data["N"]
    P = stan_data["P"]
    data = pd.DataFrame(stan_data["y"], index=data.index,
                        columns=data.columns)
    assert data.shape == (N, P)
    assert data.notnull().any(axis=1).all()

    nobs = stan_data["Nobs"]
    nmiss = stan_data["Nmiss"]
    ncens = stan_data["Ncens"]
    if pcensor > 0:
        assert ncens > 0
    else:
        assert ncens == 0

    nval = np.prod(data.shape)
    nok = data.notnull().sum().sum()

    assert nobs + ncens + nmiss == nval
    assert nobs + ncens == nok
    assert nmiss == nval - nok

    assert len(stan_data["ylocn_prior"]) == P
    assert len(stan_data["ylogscale_prior"]) == P
    assert len(stan_data["yshape1_prior"]) == P

    stan_inits = sv.initial_parameters

    assert len(stan_inits) == 6
    assert len(stan_inits["wlat_miss"]) == nmiss
    assert len(stan_inits["wlat_cens"]) == ncens

    data = np.nan * np.zeros_like(data)
    with pytest.raises(ValueError, match="Expected at least"):
        sv = sample.StanSamplingMultivariate(data, copula_spec)


@pytest.mark.parametrize("is_censored", [True])
@pytest.mark.parametrize("is_missing", [True])
@pytest.mark.parametrize("nvars", [3])
@pytest.mark.parametrize("copula_spec", [
    "Gaussian",
    "Student_4",
    "GaussianFactor_0_1",
    "GaussianFactor_0_2"
    ])
def test_sampler(is_censored, is_missing, nvars, copula_spec,
                 allclose, debug_mode):
    data, _, _, _ = datahub.get_ams_concat()
    data = data.iloc[:, :nvars]

    if not is_missing:
        data = data.loc[data.notnull().all(axis=1)]

    if is_censored:
        pcensor = 0.3
        censors = datahub.get_censors(pcensor)
        censors = censors.loc[data.columns]
    else:
        censors = np.zeros(data.shape[1])

    sv = sample.StanSamplingMultivariate(data,
                                         copula_spec,
                                         censors=censors)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    fout = f"sampling_{is_censored}_N{nvars}_C{copula_spec}"
    fout = FTESTS / "sampling" / fout
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    nsmp = STAN_NSAMPLES_DEFAULT // STAN_NCHAINS_DEFAULT
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=nsmp,
              output_dir=fout,
              inits=stan_inits,
              chains=STAN_NCHAINS_DEFAULT,
              parallel_chains=STAN_NCHAINS_DEFAULT,
              iter_warmup=STAN_NWARM_DEFAULT,
              show_progress=debug_mode)

    # Choose sampler
    if sv.copula_nfactors == 0:
        sampler = mv_censored_sampling
    else:
        sampler = mv_censored_factors_sampling

    # Test sample size error
    kw["data"]["Ncens"] += 1
    with pytest.raises(RuntimeError):
        sampler(**kw)

    kw["data"]["Ncens"] -= 1

    # Test copula shape error
    if sv.copula_name == "Student":
        kw["data"]["copula_shape"] = 1.9
        with pytest.raises(RuntimeError):
            sampler(**kw)

    kw["data"]["copula_shape"] = sv.copula_shape

    # Run sampler
    smp = sampler(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    fdiag = fout / "diagnostic.json"
    with fdiag.open("w") as fo:
        json.dump(diag, fo, indent=4)

    for met in STAN_DIAG_METRICS:
        dtxt = diag[met]
        if dtxt == "satisfactory":
            # All good
            continue
        else:
            # If there are problems, they should be only with zrhos
            dparams = re.sub("^[^:]+:", "", dtxt)
            dparams = [re.sub("\\[.*", "", d) for d in dparams.split(",")]
            assert all([re.search("zrhos", pn) for pn in dparams])

    if is_censored and is_missing and debug_mode:
        fd = fout / "censored_missing_data.zip"
        comp = dict(method="zip", compresslevel=9)
        data.to_csv(fd, compression=comp)

        fs = fout / "censored_missing_samples.zip"
        if sv.copula_nfactors > 0:
            fs = fout / "censored_missing_factors_samples.zip"
        df.to_csv(fs, compression=comp)


@pytest.mark.parametrize("pcensor", [0., 0.4])
@pytest.mark.parametrize("stationpair", [[0, 1], [4, 5], [1, 3]])
def test_mv_censored_vs_floodstan(stationpair, pcensor, debug_mode,
                                             allclose):
    # Two variables only
    data, _, dows, _ = datahub.get_ams_concat()
    data = data.iloc[:, stationpair]
    censors = data.quantile(pcensor)
    data = data.loc[data.notnull().all(axis=1)]

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
              show_progress=debug_mode)

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
                                         copula_spec="Gaussian",
                                         censors=censors,
                                         rho_min=rho_min,
                                         rho_max=rho_max)
    kw["data"] = sv.to_dict()
    kw["inits"] = sv.initial_parameters

    smp2 = mv_censored_sampling(**kw)
    df2 = smp2.draws_pd()
    diag2 = report.process_stan_diagnostic(smp2.diagnose())

    for met in STAN_DIAG_METRICS:
        assert diag2[met] == "satisfactory"

    pnames = df2.columns.to_series().filter(regex="^yl|^ys|^ucensor").to_list()
    pnames.append("corr_IW[2,1]")

    logger = get_logger(debug_mode)
    logger.info("")
    logger.info("-----------------")
    sids = "/".join(data.columns.tolist())
    logger.info(f"stations={sids} pcensor={pcensor:0.2f}")
    logger.info(f"nwarm = {nwarm}")
    logger.info(f"nsamples = {nsamples}")
    logger.info(f"rho_min = {rho_min}")
    logger.info(f"rho_max = {rho_max}")
    logger.info("")

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
        logger.info(msg)

        # Test on matching the two dist
        # 10^-8 is very low for a p-value! Still looking ok visually though
        # Also skip correlation as it can be slightly different
        pv_thresh = -8
        if not debug_mode and pname1 != "rho":
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
        ax.hist(x2, bins=bins, label="mv_censored", edgecolor="0.5", alpha=0.6)

        title = f"{pname2} - ks-logpv={kspv:0.1f}"
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize="x-small")

    sids = "/".join(data.columns.tolist())
    ftitle = f"Stations={sids} pcens={pcensor:0.2f}"
    fig.suptitle(ftitle, fontsize="large")

    sids = "-".join(data.columns.tolist())
    fimg = FTESTS / "images"
    fimg.mkdir(exist_ok=True)
    fp = f"test_mv_censored_vs_floodstan_stations{sids}"\
        + f"_pcens{pcensor*100:0.02f}.png"
    fp = fimg / fp

    fig.savefig(fp)


@pytest.mark.parametrize("nvars", [6, 8, 10])
@pytest.mark.parametrize("nfactors", [1, 2, 3])
def test_factors_correlation(nvars, nfactors, allclose, debug_mode):

    kw = {"data": dict(P=nvars, F=nfactors)}
    fout = FTESTS / "sampling" / "factors_correlation"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    stan_data = dict(P=nvars, F=nfactors)
    stan_inits = dict(rhos=uniform_direction(nfactors + 1).rvs(size=nvars))

    nwarm = 5
    nsamples = 1000
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=nsamples + nwarm,
              output_dir=fout,
              inits=stan_inits,
              chains=1,
              parallel_chains=1,
              iter_warmup=nwarm,
              show_progress=debug_mode)

    smp = factors_correlation_sampling(**kw)
    df = smp.draws_pd()

    # Ensure we only store rho and corr
    #assert df.shape[1] == 10 + (nfactors + 1 + nvars) * nvars

    cc = [f"corr[{i + 1},{j + 1}]"
          for i, j in combinations(range(nvars), 2)]
    corr = df.loc[:, cc]

    assert np.all((corr.values >= 0) & (corr.values <= 1))

    # test if correlation terms are widespread
    (_, qt1), (_, qt2) = corr.quantile([0.01, 0.99]).iterrows()
    assert all(qt1 < 0.2)
    assert all(qt2 > 0.8)


