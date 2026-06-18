from pathlib import Path

from floodstan import load_stan_model

STAN_FILES_FOLDER = Path(__file__).parent / "stan"

# Tests
FO = STAN_FILES_FOLDER
stan_test_mv = load_stan_model("stan_test_mv", sf_folder=FO)

stan_test_functions = load_stan_model("stan_test_functions",
                                      sf_folder=FO)

stan_test_indexing = load_stan_model("stan_test_indexing",
                                     sf_folder=FO)

stan_test_cor = load_stan_model("stan_test_cor", sf_folder=FO)

stan_test_copula = load_stan_model("stan_test_copula", sf_folder=FO)


# Stan sampler
name = "mv_censored_no_missing_sampling"
mv_censored_no_missing_sampling = load_stan_model(name, sf_folder=FO)

name = "mv_censored_sampling"
mv_censored_sampling = load_stan_model(name, sf_folder=FO)

name = "mv_censored_factors_sampling"
mv_censored_factors_sampling = load_stan_model(name, sf_folder=FO)

factors_correlation_sampling = load_stan_model("factors_correlation",
                                               sf_folder=FO)
