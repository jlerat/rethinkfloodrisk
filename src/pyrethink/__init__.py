from pathlib import Path

from floodstan import load_stan_model

__version__ = "0.1"

STAN_FILES_FOLDER = Path(__file__).parent / "stan"

# Stan sampler
MVN_NAME = "multivariate_censored_sampling"
multivariate_censored_sampling = load_stan_model(MVN_NAME,
                                                 sf_folder=STAN_FILES_FOLDER)
