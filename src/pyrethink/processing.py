import numpy as np

def eep2aep(nu, eep):
    return 1 - np.exp(-nu * eep)


def linear_interpolation(xx, x, y):
    """ Linear interpolation """
    # Sort values
    isort = np.argsort(x)
    x = np.array(x)[isort]
    if np.any(np.diff(x) <= 0):
        errmsg = "Cannot process duplicates in x."
        raise ValueError(errmsg)

    if len(y) != len(x):
        errmsg = "Expected x and y of same length."
        raise ValueError(errmsg)

    y = np.array(y)[isort]
    xx = np.atleast_1d(xx)

    # interpolation coefficients
    D = np.abs(x[:, None] - x[None, 1:-1])
    D = np.column_stack([D, np.ones(len(x)), x])
    coefs = np.linalg.solve(D, y)

    # Run interpolation
    D = np.abs(xx[:, None] - x[None, 1:-1])
    D = np.column_stack([D, np.ones(len(xx)), xx])
    return (D @ coefs).squeeze()
