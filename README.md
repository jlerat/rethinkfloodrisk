This package supports the paper "Fragility and ambiguity in design floods".

# Installation
- Git clone this repository.
- Create a suitable python environment. We recommend using [uv](https://docs.astral.sh/uv) combined with the package definition provided in the [pyproject.toml](pyproject.toml) file in this repository.
- Install via `uv run pip install -e .`

# Basic use

```python
from pathlib import Path
import numpy as np

# .. requires functions from the floodstan package
from floodstan import report as fs_report

from pyrethink import sample
from pyrethink import report as pr_report
from pyrethink import mv_censored_sampling

# Random data to be replaced by actual ones
nstations = 3 # assumes 3 stations
nval = 50 # assumes 50 years of data

rho = 0.8 # spatial correlation between sites
mean = -0.2 * np.ones(nstations)
cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))
obs = np.exp(np.random.multivariate_normal(mean=mean,cov=cov,size=nval))

# Censoring threshold (30%)
pcensor = 0.3
censors = np.nanpercentile(obs, pcensor * 100, axis=0)

# Configure sampling data, including copula specifications
copula_spec = "Gaussian"
sv = sample.StanSamplingMultivariate(obs,
                                     copula_spec,
                                     censors=censors)

# Create output directory for stan computations
fout = Path("outputs") / "stan"
fout.mkdir(exist_ok=True)

# Run stan sampler (can take time)
nwarm = 500 # to be set to 10,000 for proper computations
nsamples = 1000 # to be set to 20,000 for proper computations
nchains = 3

kw = dict(data=sv.to_dict(),
          seed=5446,
          iter_sampling=nsamples // nchains,
          output_dir=fout,
          inits=sv.initial_parameters,
          chains=nchains,
          parallel_chains=nchains,
          iter_warmup=nwarm,
          show_progress=True)
samples = mv_censored_sampling(**kw)

# diagnostic
diag = fs_report.process_stan_diagnostic(samples.diagnose())
print("\nMCMC computation diagnostic:")
for prop in ["treedepth", "ebfmi", "rhat"]:
    print(f"{prop:10s} : {diag[prop]}")

# Process quantiles information (can take time)
df = samples.draws_pd()
stat, _ = pr_report.ffa_report(df)

# Report quantiles for each station
print("\nQuantile estimation:")
for station in range(1, nstations + 1):
    print(f"Station {station}")
    for aep in [10, 100, 500]:
        idx = f"DESIGN_ERI{aep}[{station}]"
        mq = stat.loc[idx, "POSTERIOR_PREDICTIVE"]
        m1 = stat.loc[idx, "5%"]
        m2 = stat.loc[idx, "95%"]
        txt = f"\t1:{aep:<3d} flood : post predictive={mq:5.1f} [{m1:5.1f}, {m2:5.1f}]"
        print(txt)
```

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.


