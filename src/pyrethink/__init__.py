from pathlib import Path

from floodstan import load_stan_model

__version__ = "0.1"

STAN_FILES_FOLDER = Path(__file__).parent / "stan"

# Test
stan_test_mv = load_stan_model("stan_test_mv",
                               sf_folder=STAN_FILES_FOLDER)
stan_test_functions = load_stan_model("stan_test_functions",
                                      sf_folder=STAN_FILES_FOLDER)
stan_test_indexing = load_stan_model("stan_test_indexing",
                                     sf_folder=STAN_FILES_FOLDER)
stan_test_cor = load_stan_model("stan_test_cor",
                                sf_folder=STAN_FILES_FOLDER)

# Stan sampler
name = "mv_censored_sampling"
mv_censored_sampling = load_stan_model(name,
                                       sf_folder=STAN_FILES_FOLDER)
