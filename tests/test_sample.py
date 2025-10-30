import re
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, gumbel_r
from scipy.stats import ttest_ind, ks_2samp
import pytest

from floodstan import report
from floodstan import sample as fsample
from floodstan import marginals
from floodstan import bivariate_censored_sampling

from pyrethink import sample
from pyrethink import datahub
from pyrethink import mv_censored_sampling
from pyrethink import mv_uncensored_sampling
from pyrethink import mv_uncensored_nomissing_sampling
from pyrethink import stan_test_indexing
from pyrethink import stan_test_functions
from pyrethink import stan_test_cor

FTESTS = Path(__file__).resolve().parent

SEED = 5446

PROGRESS = False
LOGGER = fsample.get_logger(stan_logger=PROGRESS)

STAN_NCHAINS_DEFAULT = 5
STAN_NWARM_DEFAULT = 5000
STAN_NSAMPLES_DEFAULT = 5000

@pytest.mark.parametrize("pcensor", [0., 0.3])
def test_sample_data(pcensor, allclose):
    data = datahub.get_truepeaks().drop("WATERYEAR", axis=1)
    sv = sample.StanSamplingMultivariate(data, pcensor=pcensor)
    stan_data = sv.to_dict()

    assert len(stan_data) == 18

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
    assert len(stan_inits) == 5
    assert len(stan_inits["zlat_miss"]) == nmiss
    assert len(stan_inits["wlat_cens"]) == ncens

    for ivar in range(data.shape[1]):
        censor = np.nanpercentile(data.iloc[:, ivar], pcensor * 100)
        assert allclose(censor, stan_data["censors"][ivar])


def test_stan_indexing():
    data = datahub.get_truepeaks()
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


def test_stan_functions(allclose):
    tau = 0.5
    alpha = 3.

    stan_data = {
        "Q": 10000,
        "tau": tau,
        "alpha": alpha
    }
    df = stan_test_functions(data=stan_data)

    u = df.filter(regex="^u").squeeze().values
    z = df.filter(regex="^z").squeeze().values
    expected = norm.ppf(u)
    assert allclose(z, expected, atol=1.5e-3)

    gq = df.filter(regex="^gq").squeeze().values
    rv = gumbel_r(loc=tau, scale=alpha)
    expected = rv.ppf(u)
    assert allclose(gq, expected, atol=5e-4)


def test_stan_cor(allclose):
    data = datahub.get_truepeaks().iloc[:, :3]
    sv = sample.StanSamplingMultivariate(data)
    stan_inits = sv.initial_parameters
    L_cor = stan_inits["L_cor"]
    stan_data = {
        "P": len(L_cor),
        "Q": 2000,
        "L_cor": L_cor
        }

    df = stan_test_cor(data=stan_data)
    P = stan_data["P"]
    Q = stan_data["Q"]
    z = df.filter(regex="^zrnd").values.reshape((P, Q)).T
    cor = np.corrcoef(z.T)
    expected = L_cor @ L_cor.T
    assert allclose(cor, expected, atol=3e-2)


@pytest.mark.parametrize("nvars", [3])
@pytest.mark.parametrize("config", ["uncensored_nomissing",
                                    "uncensored_missing",
                                    "censored_missing"])
def test_sampler(config, nvars, allclose):
    data = datahub.get_truepeaks().iloc[:, :nvars]

    if re.search("nomissing", config):
        data = data.loc[data.notnull().all(axis=1)]

    pcensor = 0.3 if config == "censored_missing" else 0.

    sv = sample.StanSamplingMultivariate(data, pcensor=pcensor)
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

    if config == "uncensored_nomissing":
        smp = mv_uncensored_nomissing_sampling(**kw)
    elif config == "uncensored_missing":
        smp = mv_uncensored_sampling(**kw)
    elif config == "censored_missing":
        smp = mv_censored_sampling(**kw)

    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    assert diag["treedepth"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["effsamplesz"] == "satisfactory"


@pytest.mark.parametrize("pcensor", [0., 0.1, 0.5])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("station", [0, 5])
def test_censored_vs_floodstan(station, pcensor, missing, allclose):
    # Two variables only
    data = datahub.get_truepeaks().iloc[:, station: station + 2]
    data = data.loc[data.notnull().any(axis=1)]

    if not missing:
        data = data.loc[pd.notnull(data).all(axis=1)]

    # -- floodstan --
    y, z = data.values.T
    censor = np.nanpercentile(y, pcensor * 100)
    marginal = marginals.Gumbel()
    yv = fsample.StanSamplingVariable(marginal, y, censor,
                                      ninits=STAN_NCHAINS_DEFAULT)

    censor = np.nanpercentile(z, pcensor * 100)
    zv = fsample.StanSamplingVariable(marginal, z, censor,
                                      ninits=STAN_NCHAINS_DEFAULT)

    fsv = fsample.StanSamplingDataset([yv, zv], "Gaussian")
    fstan_data = fsv.to_dict()
    fstan_inits = fsv.initial_parameters

    fout = FTESTS / "sampling" / "sampling_uncensored_vs_floodstan"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    #nwarm = 100
    #nsamples = 100
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

    # -- pyrethink --
    sv = sample.StanSamplingMultivariate(data, pcensor=pcensor)
    kw["data"] = sv.to_dict()
    kw["inits"] = sv.initial_parameters

    smp2 = mv_censored_sampling(**kw)
    df2 = smp2.draws_pd()
    diag2 = report.process_stan_diagnostic(smp2.diagnose())

    pnames = df2.columns.to_series().filter(regex="^yl|^zcensor").to_list()
    pnames.append("L_cor[2,1]")

    print("")
    print(f"---- pcensor={pcensor:0.2f} missing={missing} ----")
    for pname2 in pnames:
        x2 = df2.loc[:, pname2]

        # Get floodstan sample
        if re.search("L_cor", pname2):
            pname1 = "rho"
        elif re.search("zcensor", pname2):
            pname1 = "ucensor" if re.search("1", pname2) else "vcensor"
        else:
            pname1 = re.sub("\\[.*", "", pname2)
            if re.search("2", pname2):
                pname1 = re.sub("^y", "z", pname1)

        x1 = df1.loc[:, pname1]
        if pname1 == "rho":
            # Convert kendall tau to correlation
            x1 = np.sin(x1 * math.pi / 2)
        elif re.search("zcensor", pname2):
            # Convert probability to normal cdf
            x1 = norm.ppf(x1)

        resk = ks_2samp(x1, x2)
        kspv = math.log10(resk.pvalue) if resk.pvalue > 0 else -np.inf

        rest = ttest_ind(x1, x2)
        tpv = math.log10(rest.pvalue) if rest.pvalue > 0 else -np.inf

        msg = f"[{pname2:15s}] mean(x1)={x1.mean():6.2f}"\
              + f" mean(x2)={x2.mean():6.2f}"\
              + f" ks-logpv={kspv:4.2f} t-logpv={tpv:4.2f}"
        print(msg)

        #if pcensor == 0:
        #    assert kspv > -3
        #    assert tpv > -3

    print("")
