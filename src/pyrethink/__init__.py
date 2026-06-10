from pathlib import Path

from floodstan import load_stan_model

STAN_FILES_FOLDER = Path(__file__).parent / "stan"

# Tests
stan_test_mv = load_stan_model("stan_test_mv",
                               sf_folder=STAN_FILES_FOLDER)

stan_test_functions = load_stan_model("stan_test_functions",
                                      sf_folder=STAN_FILES_FOLDER)

stan_test_indexing = load_stan_model("stan_test_indexing",
                                     sf_folder=STAN_FILES_FOLDER)

stan_test_cor = load_stan_model("stan_test_cor",
                                sf_folder=STAN_FILES_FOLDER)

stan_test_copula = load_stan_model("stan_test_copula",
                                   sf_folder=STAN_FILES_FOLDER)

stan_test_clusters = load_stan_model("stan_test_clusters",
                                     sf_folder=STAN_FILES_FOLDER)

# Stan sampler
name = "mv_censored_no_missing_sampling"
FO = STAN_FILES_FOLDER
mv_censored_no_missing_sampling = load_stan_model(name, sf_folder=FO)

