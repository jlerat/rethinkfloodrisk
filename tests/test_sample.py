from pathlib import Path
import numpy as np
from scipy.stats import norm, gumbel_r
import pytest

from floodstan import report
from floodstan import sample as fsample
from floodstan import marginals

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

PROGRESS = True
LOGGER = fsample.get_logger(stan_logger=PROGRESS)

def test_sample_data(allclose):
    data = datahub.get_truepeaks()
    sv = sample.StanSamplingMultivariate(data)

    stan_data = sv.to_dict()
    assert len(stan_data) == 17
    assert len(stan_data["idx_obs"]) == 515
    assert len(stan_data["idx_cens"]) == 222
    assert len(stan_data["idx_miss"]) == data.isnull().sum().sum()

    stan_inits = sv.initial_parameters
    assert len(stan_inits) == 5


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


def test_stan_functions():
    tau = 0.5
    alpha = 3.

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
    assert np.allclose(z, expected, atol=1.5e-3)

    gq = df.filter(regex="^gq").squeeze().values
    rv = gumbel_r(loc=tau, scale=alpha)
    expected = rv.ppf(u)
    assert np.allclose(gq, expected, atol=5e-4)


@pytest.mark.parametrize("nvars", [3])
def test_uncensored_nomissing(nvars, allclose):
    data = datahub.get_truepeaks().iloc[:, :nvars]
    data = data.loc[data.notnull().all(axis=1)]
    sv = sample.StanSamplingMultivariate(data, pcensor=0.0)

    stan_data = sv.to_dict()
    assert stan_data["Nobs"] == np.prod(data.shape)
    assert stan_data["Ncens"] == 0
    assert stan_data["Nmiss"] == 0

    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args
    stan_nchains = 5
    stan_nwarm = 5000
    stan_nsamples = 5000

    fout = FTESTS / "sampling_uncensored_nomissing"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm,
              show_progress=PROGRESS)
    kw.update(stan_args)

    smp = mv_uncensored_nomissing_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    assert diag["treedepth"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["effsamplesz"] == "satisfactory"


@pytest.mark.parametrize("nvars", [3])
def test_uncensored(nvars, allclose):
    data = datahub.get_truepeaks().iloc[:, :nvars]
    sv = sample.StanSamplingMultivariate(data, pcensor=0.0)
    stan_data = sv.to_dict()

    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args
    stan_nchains = 5
    stan_nwarm = 5000
    stan_nsamples = 5000

    fout = FTESTS / "sampling_uncensored"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm,
              show_progress=PROGRESS)
    kw.update(stan_args)

    smp = mv_uncensored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    assert diag["treedepth"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["effsamplesz"] == "satisfactory"


def test_uncensored_vs_floodstan(nvars, allclose):
    pytest.skip("WIP")
    data = datahub.get_truepeaks().iloc[:, :2]
    y, z = data.values.T
    censor = y.median()
    marginal = marginals.Gumbel()
    yv = fsample.StanSamplingVariable(marginal, y, censor,
                                      ninits=stan_nchains)

    censor = z.median()
    zv = fsample.StanSamplingVariable(marginal, z, censor,
                                      ninits=stan_nchains)

    fsv = sample.StanSamplingDataset([yv, zv], "Gaussian")
    fstan_data = fsv.to_dict()
    fstan_inits = fsv.initial_parameters

    fout = FTESTS / "sampling_uncensored_vs_floodstan"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    msg = "Error during sampling"
    smp = bivariate_censored_sampling(data=stan_data,
                                  chains=stan_nchains,
                                  seed=SEED,
                                  iter_warmup=stan_nwarm,
                                  iter_sampling=
                                  stan_nsamples//stan_nchains,
                                  parallel_chains=stan_nchains,
                                  inits=stan_inits,
                                  show_progress=False)



@pytest.mark.parametrize("nvars", [3])
def test_censored(nvars, allclose):
    pytest.skip("WIP")
    data = datahub.get_truepeaks().iloc[:, :nvars]
    sv = sample.StanSamplingMultivariate(data)
    stan_data = sv.to_dict()

    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args
    stan_nchains = 5
    stan_nwarm = 5000
    stan_nsamples = 5000

    fout = FTESTS / "sampling_censored"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm,
              show_progress=PROGRESS)
    kw.update(stan_args)

    smp = mv_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    import pdb; pdb.set_trace()

    assert diag["treedepth"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["effsamplesz"] == "satisfactory"


