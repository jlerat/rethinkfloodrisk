from pathlib import Path
from scipy.stats import norm
import pytest

from floodstan import report
from floodstan.sample import get_logger

from pyrethink import sample
from pyrethink import datahub
from pyrethink import multivariate_censored_sampling

FTESTS = Path(__file__).resolve().parent

SEED = 5446

def test_sample_data(allclose):
    data = datahub.get_truepeaks()
    sv = sample.StanSamplingMultivariate(data)

    stan_data = sv.to_dict()
    assert len(stan_data) == 20

    stan_inits = sv.initial_parameters
    assert len(stan_inits) == 6
    zcensor = norm.ppf(sv.pcensor)
    assert all(stan_inits["zcens"] < zcensor)


@pytest.mark.parametrize("nvars", [2])
def test_sampling(nvars, allclose):
    data = datahub.get_truepeaks().iloc[:, :nvars]
    sv = sample.StanSamplingMultivariate(data)

    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args
    stan_nsamples = 1000
    stan_nchains = 5
    stan_nwarm = 1000

    fout = FTESTS / "sampling"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    progress = True
    LOGGER = get_logger(stan_logger=progress)

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm,
              show_progress=progress)
    kw.update(stan_args)

    # Sample
    smp = multivariate_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
