import re
import numpy as np

from floodstan.marginals import PARAMETERS, GEV
from floodstan import quadapprox

from floodstan.report import QUANTILES
from floodstan.report import DESIGN_ARIS
from floodstan.report import _prepare_design_aris

MARGINAL = GEV()


def ffa_report(params,
               design_eris=DESIGN_ARIS):
    """ Generate report variables.

    Parameters
    ----------
    params : pandas.DataFrame
        List of parameter sets
    design_eris : list
        List of design flood event return interval (eri) to be computed.

    Returns
    -------
    report_df : pandas.DataFrame
        Reported variables for all parameters

    report_stat : pandas.DataFrame
        Statistics for all reported variables.
        See floodstan.report.QUANTILES.
    """
    nsamples = len(params)
    params = params.reset_index(drop=True)

    design_eris, design_cdf, design_columns, post_pred_cdf = \
        _prepare_design_aris(design_eris, 0.)
    design_columns = [re.sub("ARI", "ERI", cn) for cn in design_columns]

    # Extract parameters
    params_df = {}
    cols = []
    for pname in PARAMETERS:
        df = params.filter(regex="y"+pname, axis=1)
        params_df[pname] = df
        cols.extend(df.columns.tolist())

    report_df = params.loc[:, cols]

    # Number of variables
    nvar = params_df[PARAMETERS[0]].shape[1]

    # Initialise report data
    cols = [f"{cn}[{ivar + 1}]" for cn in design_columns
            for ivar in range(nvar)]
    report_df.loc[:, cols] = np.nan

    # prepare data for posterior predictive distribution
    # computation
    nxi = len(post_pred_cdf)
    xi = np.nan * np.zeros((nxi, nvar))

    # .. compute design values using expected parameter
    design_meanp = np.zeros((len(design_cdf), nvar))
    for ivar in range(nvar):
        for pname in PARAMETERS:
            cn = f"y{pname}[{ivar + 1}]"
            MARGINAL[pname] = params_df[pname].loc[:, cn].mean()

        xu = np.unique(MARGINAL.ppf(post_pred_cdf))
        xi[:len(xu), ivar] = xu

        # .. compute quantile distribution using mean params
        #    of design floods
        design_meanp[:, ivar] = MARGINAL.ppf(design_cdf)

    xi = xi[np.any(~np.isnan(xi), axis=1)]
    xm = (xi[:-1] + xi[1:]) / 2

    # .. initialise parameter vectors
    nxi = len(xi) + 1
    a_coefs = np.zeros((nxi, nvar))
    b_coefs = np.zeros((nxi, nvar))
    c_coefs = np.zeros((nxi, nvar))
    nparams_ok = np.zeros(nvar)

    # Loop through parameters
    for ivar in range(nvar):
        for isample in range(nsamples):
            # .. set parameters
            try:
                for pname in PARAMETERS:
                    cn = f"y{pname}[{ivar + 1}]"
                    MARGINAL[pname] = params_df[pname].loc[isample, cn]
            except ValueError:
                continue

            # .. get quadratic aprox coefficient to compute predictive
            #    posterior distribution
            xxi = xi[:, ivar]
            xxi = xxi[~np.isnan(xxi)]
            fi = MARGINAL.cdf(xxi)
            x0, x1 = MARGINAL.support
            fi[xxi < x0] = 0.
            fi[xxi > x1] = 1.

            xxm = xm[:, ivar]
            xxm = xxm[~np.isnan(xxm)]
            fm = MARGINAL.cdf(xxm)
            fm[xxm < x0] = 0.
            fm[xxm > x1] = 1.

            a, b, c = quadapprox.get_coefficients(xxi, fi, fm, True)
            # .. potential pb if len(xxi) != len(a_coefs)
            a_coefs[:, ivar] += a
            b_coefs[:, ivar] += b
            c_coefs[:, ivar] += c

            nparams_ok[ivar] += 1

            # .. compute design streamflow
            cols = [f"{cn}[{ivar + 1}]" for cn in design_columns]
            q = MARGINAL.ppf(design_cdf)
            report_df.loc[isample, cols] = q

    # Standardize coefs of posterior predictive
    npo = nparams_ok[None, :]
    a_coefs = a_coefs / npo
    b_coefs = b_coefs / npo
    c_coefs = c_coefs / npo

    # Build stat report
    # .. compute stat
    report_stat = report_df.describe(percentiles=QUANTILES)
    report_stat = report_stat.drop(["count"], axis=0)
    report_stat.loc["SKEW", :] = report_df.skew()

    # .. compute confidence interval
    ridx = report_stat.index
    if "5%" in ridx and "95%" in ridx:
        report_stat.loc["CI90", :] = report_stat.loc["95%"]\
            - report_stat.loc["5%"]

    # .. compute finite value proportion
    cc = report_stat.columns
    report_stat.loc["ISFINITE[%]", cc] = report_df.loc[:, cc]\
        .apply(lambda x: np.isfinite(x).sum() / len(x) * 100)
    report_stat.loc["ISZERO[%]", cc] = report_df.loc[:, cc]\
        .apply(lambda x: (np.abs(x) < 1e-10).sum() / len(x) * 100)

    # .. final formatting
    report_stat = report_stat.T
    report_stat.columns = [cn.upper() for cn in report_stat.columns]

    # .. compute posterior predictive distribution
    design_post = np.zeros((len(design_cdf), nvar))
    for ivar in range(nvar):
        xxi = xi[:, ivar]
        aa = a_coefs[:, ivar]
        bb = b_coefs[:, ivar]
        cc = c_coefs[:, ivar]
        design_post[:, ivar] = quadapprox.inverse(design_cdf, xxi, aa, bb, cc)

    cnp = "POSTERIOR_PREDICTIVE"
    cnm = "EXPECTED_PARAMETERS"
    for cn in [cnp, cnm]:
        report_stat.loc[:, cn] = np.nan

    for eri, qp, qm in zip(design_eris, design_post, design_meanp):
        idx = [re.sub("\\.0", "", f"DESIGN_ERI{eri}[{ivar + 1}]")
               for ivar in range(nvar)]
        report_stat.loc[idx, cnp] = qp
        report_stat.loc[idx, cnm] = qm

    # .. add expected parameters
    for pname in PARAMETERS:
        report_stat.loc[idx, cnm] = report_stat.loc[idx, "MEAN"]

    return report_stat, report_df
