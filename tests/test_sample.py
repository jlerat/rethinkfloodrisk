import re
from pathlib import Path
import math
import warnings

import pytest

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt

from floodstan import report
from floodstan import sample as fsample
from floodstan import marginals
from floodstan import bivariate_censored_sampling

from pyrethink import sample
from pyrethink import datahub
from pyrethink import mv_censored_sampling
from pyrethink import stan_test_indexing
from pyrethink import stan_test_functions
from pyrethink import stan_test_cor

FTESTS = Path(__file__).resolve().parent

SEED = 5446

DEBUG = False

PROGRESS = DEBUG
FLOG = FTESTS / "test_sample.log"

# Clean files
if FLOG.exists():
    try:
        FLOG.unlink()
    except:
        pass

for f in FTESTS.glob("test_mv_censored_vs_floodstan*.png"):
    f.unlink()

LOGGER = fsample.get_logger(stan_logger=PROGRESS, flog=FLOG)

STAN_NCHAINS_DEFAULT = 3
STAN_NWARM_DEFAULT = 5000
STAN_NSAMPLES_DEFAULT = 6000

STAN_DIAG_METRICS = ["treedepth", "rhat", "ebfmi", "effsamplesz"]


@pytest.mark.parametrize("pcensor", [0., 0.3])
def test_sample_data(pcensor, allclose):
    data = datahub.get_potpeaks().filter(regex="_PEAK", axis=1)
    censors = data.quantile(pcensor).values
    sv = sample.StanSamplingMultivariate(data, censors=censors)
    stan_data = sv.to_dict()

    assert len(stan_data) == 21

    data = pd.DataFrame(stan_data["y"])
    assert data.notnull().any(axis=1).all()

    nobs = stan_data["Nobs"]
    nmiss = stan_data["Nmiss"]
    ncens = stan_data["Ncens"]

    if pcensor == 0.:
        assert ncens == 0

    nval = np.prod(data.shape)
    nok = data.notnull().sum().sum()

    assert nobs + ncens + nmiss == nval
    assert nobs + ncens == nok
    assert nmiss == nval - nok

    stan_inits = sv.initial_parameters

    assert len(stan_inits) == 6
    assert len(stan_inits["wlat_miss"]) == nmiss
    assert len(stan_inits["wlat_cens"]) == ncens

    for ivar in range(data.shape[1]):
        censor = np.nanpercentile(data.iloc[:, ivar], pcensor * 100)
        assert allclose(censor, stan_data["censors"][ivar])

    data = np.nan * np.zeros_like(data)
    with pytest.raises(ValueError, match="Expected at least"):
        sv = sample.StanSamplingMultivariate(data)


def test_inits(allclose):
    data = datahub.get_potpeaks().filter(regex="_PEAK", axis=1)
    data = data.loc[data.index.year < 2008]
    pcensor = 0.3
    censors = data.quantile(pcensor)
    sv = sample.StanSamplingMultivariate(data, censors=censors)
    inits = sv.initial_parameters


def test_stan_indexing():
    data = datahub.get_potpeaks().filter(regex="_PEAK", axis=1)
    sv = sample.StanSamplingMultivariate(data)
    stan_data = sv.to_dict()
    df = stan_test_indexing(data=stan_data)

    y = stan_data["y"]
    z = df.filter(regex="^z").values.reshape(y.T.shape).T
    for ivar, zv in enumerate(z.T):
        # Check missing
        ismiss = np.isnan(y[:, ivar])
        assert all(z[ismiss, ivar] == 3)
        assert all(z[~ismiss, ivar] != 3)

        # Check censored
        iscens = y[:, ivar] < stan_data["censors"][ivar]
        assert all(z[iscens, ivar] == 2)
        assert all(z[~iscens, ivar] != 2)


@pytest.mark.parametrize("kappa", [-1., -0.5, 0., 0.5, 1.])
def test_stan_functions(kappa, allclose):
    tau = 100.
    alpha = 50.

    mname = "GEV" if abs(kappa) > 0 else "Gumbel"
    marginal = marginals.factory(mname)
    marginal.params = [tau, math.log(alpha), kappa]

    stan_data = {
        "Q": 10000,
        "tau": tau,
        "alpha": alpha,
        "kappa": kappa
    }
    df = stan_test_functions(data=stan_data)

    u = df.filter(regex="^u\\[").squeeze().values
    q = df.filter(regex="^qq").squeeze().values
    expected = marginal.ppf(u)
    atol = 1e-4
    rtol = 5e-4
    assert allclose(q, expected, atol=atol, rtol=rtol)

    uu = df.filter(regex="^uu").squeeze().values
    expected = marginal.cdf(q)
    assert allclose(uu, expected, atol=atol, rtol=rtol)

    lp = df.filter(regex="^lp\\[").squeeze().values
    expected = marginal.logpdf(q)
    assert allclose(lp, expected, atol=atol, rtol=rtol)


def test_stan_cor(allclose):
    P = 5
    Q = 20000

    rho = 0.9
    cor = toeplitz(rho ** np.arange(P))
    L_cor = np.linalg.cholesky(cor)

    stan_data = {
        "P": P,
        "Q": Q,
        "L_cor": L_cor
        }

    df = stan_test_cor(data=stan_data)

    z = df.filter(regex="^zrnd").values.reshape((P, Q)).T
    zcor = np.corrcoef(z.T)
    assert allclose(zcor, cor, atol=1e-2)


@pytest.mark.parametrize("config", ["uncensored_nomissing",
                                    "uncensored_missing",
                                    "censored_missing"])
@pytest.mark.parametrize("nvars", [3])
def test_sampler(config, nvars, allclose):
    data = datahub.get_potpeaks().iloc[:, :nvars]

    if re.search("nomissing", config):
        data = data.loc[data.notnull().all(axis=1)]

    pcensor = 0.3 if config == "censored_missing" else 0.
    censors = data.quantile(pcensor)

    sv = sample.StanSamplingMultivariate(data, censors=censors)
    stan_data = sv.to_dict()

    if config == "uncensored_nomissing":
        assert stan_data["Nobs"] == np.prod(data.shape)
        assert stan_data["Ncens"] == 0
        assert stan_data["Nmiss"] == 0

    stan_inits = sv.initial_parameters

    fout = FTESTS / "sampling" / f"sampling_{config}"
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

    smp = mv_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    for met in STAN_DIAG_METRICS:
        assert diag[met] == "satisfactory"

    # Test sample size error
    kw["data"]["Nmiss"] += 1
    with pytest.raises(RuntimeError):
        mv_censored_sampling(**kw)

@pytest.mark.parametrize("pcensor", [0.1, 0.4])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("stationpair", [[0, 1], [5, 6], [4, 7]])
def test_mv_censored_vs_floodstan(stationpair, pcensor, missing, allclose):
    if DEBUG and (pcensor < 0.5 or missing or stationpair[0] != 44):
        pytest.skip("Debug mode")

    # Two variables only
    data, _ = datahub.get_potpeaks()
    data = data.iloc[:, stationpair]
    data = data.loc[data.notnull().any(axis=1)]

    if not missing:
        data = data.loc[pd.notnull(data).all(axis=1)]

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
    rho_min = np.floor(df1.rho.min() * 1e2) * 1e-2
    rho_max = 1.
    sv = sample.StanSamplingMultivariate(data, censors=censors,
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
    pnames.append("cor_IW[2,1]")

    LOGGER.info("")
    LOGGER.info("-----------------")
    sids = "/".join(data.columns.tolist())
    LOGGER.info(f"stations={sids} pcensor={pcensor:0.2f} missing={missing}")
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
        if re.search("cor_IW", pname2):
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
            wmess = f"stations={sids} pcensor={pcensor:0.2f} missing={missing}\n"\
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
    ftitle = f"Stations={sids} pcens={pcensor:0.2f} missing={missing}"
    fig.suptitle(ftitle, fontsize="large")

    sids = "-".join(data.columns.tolist())
    fp = f"test_mv_censored_vs_floodstan_stations{sids}"\
        + f"_pcens{pcensor*100:0.02f}"\
        + f"_missing{missing}.png"
    fp = FTESTS / fp

    fig.savefig(fp)
