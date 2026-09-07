import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize


def binary_agreement(params, tref, xref, tsig, xsig):
    a, b = params

    tsig_mapped = a * tsig + b

    order = np.argsort(tsig_mapped)
    ts = tsig_mapped[order]
    xs = xsig[order]

    f = interp1d(ts, xs, kind="nearest", bounds_error=False, fill_value=np.nan)

    xs_on_ref = f(tref)

    valid = (~np.isnan(xs_on_ref)) & (~np.isnan(xref))
    if valid.sum() == 0:
        return 1e9

    agreement = np.mean(xs_on_ref[valid] == xref[valid])
    return -agreement


def fit_clock_mapping(tref, xref, tsig, xsig, drift_ppm=5000, offset_range=None):
    tref = np.asarray(tref)
    xref = np.asarray(xref).astype(int)
    tsig = np.asarray(tsig)
    xsig = np.asarray(xsig).astype(int)

    if offset_range is None:
        offset_range = (tref.min() - tsig.max(), tref.max() - tsig.min())

    a_min = 1 - drift_ppm * 1e-6
    a_max = 1 + drift_ppm * 1e-6

    bounds = [(a_min, a_max), offset_range]

    result_global = differential_evolution(
        binary_agreement,
        bounds=bounds,
        args=(tref, xref, tsig, xsig),
        tol=1e-6,
        polish=False
    )

    result_local = minimize(
        binary_agreement,
        result_global.x,
        args=(tref, xref, tsig, xsig),
        method="Nelder-Mead"
    )

    a, b = result_local.x
    score = -result_local.fun

    return a, b, score
