"""Monte Carlo simulation for equities and FX."""

import numpy as np

from .config import FOREIGN_REGIONS
from .correlation import factor_list_for_selection, build_factor_corr_matrix


def simulate_equities_and_fx(basket_composition, params, T, n_steps, n_paths, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    factors = factor_list_for_selection(basket_composition)
    C, adjusted = build_factor_corr_matrix(factors, params)
    L = np.linalg.cholesky(C)

    d = len(factors)
    Z = rng.standard_normal(size=(n_steps, n_paths, d))
    Zc = Z @ L.T

    time = np.linspace(0.0, T, n_steps + 1)

    # Store each region path separately (clean + easy)
    S_paths = {r: np.zeros((n_steps + 1, n_paths), dtype=float) for r in basket_composition}
    X_paths = {r: np.ones((n_steps + 1, n_paths), dtype=float) for r in basket_composition}

    # init
    for r in basket_composition:
        S_paths[r][0, :] = float(params[r]["S0"])
        X_paths[r][0, :] = float(params[r]["fx0"]) if r in FOREIGN_REGIONS else 1.0

    f_idx = {f: k for k, f in enumerate(factors)}
    r_d = float(params["Local"]["r"])

    for t in range(1, n_steps + 1):
        # Equities under P: (mu - q)
        for r in basket_composition:
            mu = float(params[r]["mu"])
            q = float(params[r].get("q", 0.0))
            sig = float(params[r]["sigma"])
            z = Zc[t - 1, :, f_idx[("S", r)]]
            S_prev = S_paths[r][t - 1, :]
            S_paths[r][t, :] = S_prev * np.exp(((mu - q) - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * z)

        # FX under (r_d - r_f)
        for r in basket_composition:
            if r not in FOREIGN_REGIONS:
                X_paths[r][t, :] = 1.0
                continue

            r_f = float(params[r]["r"])
            sigx = float(params[r]["sigma_fx"])
            z = Zc[t - 1, :, f_idx[("X", r)]]
            X_prev = X_paths[r][t - 1, :]
            X_paths[r][t, :] = X_prev * np.exp(((r_d - r_f) - 0.5 * sigx * sigx) * dt + sigx * np.sqrt(dt) * z)

    return time, S_paths, X_paths, factors, C, adjusted
