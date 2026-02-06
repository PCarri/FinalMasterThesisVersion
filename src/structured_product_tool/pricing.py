"""Pricing utilities for basket options."""

import numpy as np
from scipy.stats import norm

from .config import FOREIGN_REGIONS
from .correlation import factor_label


def bs_from_forward(F, K, df, sigma, tau, is_call=True):
    F = float(F)
    K = float(K)
    df = float(df)
    sigma = float(sigma)
    tau = float(tau)
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic
    vol_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ------------------------------------------------------------
# Basket option pricing via moment matching (M1, M2)
# ------------------------------------------------------------

def basket_domestic_forwards_and_loadings(
    basket_composition, params, factors, C_factors, fx_mode_sp, T
):
    """
    Returns:
      F_i      : domestic forward of P_T^{(i)}
      a_i      : factor loading vector for domestic log-return of P^{(i)}
                (so that Var(log P_i) = a_i^T C a_i, using factor vols as loadings)
      labels   : factor labels in same order as factors
    """
    labels = [factor_label(f) for f in factors]
    idx = {factors[k]: k for k in range(len(factors))}
    r_d = float(params["Local"]["r"])

    n = len(basket_composition)
    F = np.zeros(n, dtype=float)
    loadings = np.zeros((n, len(factors)), dtype=float)

    for i, r in enumerate(basket_composition):
        S0 = float(params[r]["S0"])
        q = float(params[r].get("q", 0.0))
        sigS = float(params[r]["sigma"])
        X0 = float(params[r]["fx0"]) if r in FOREIGN_REGIONS else 1.0

        # equity factor always loads with sigS
        loadings[i, idx[("S", r)]] += sigS

        # Domestic asset (or local-only basket)
        if r not in FOREIGN_REGIONS or fx_mode_sp == "Plain (Local only)":
            F[i] = (S0 * X0) * np.exp((r_d - q) * T)
            continue

        # Foreign assets
        if fx_mode_sp == "Take FX risk (Compo)":
            sigX = float(params[r]["sigma_fx"])
            loadings[i, idx[("X", r)]] += sigX
            F[i] = (S0 * X0) * np.exp((r_d - q) * T)

        else:
            # Quanto
            r_f = float(params[r]["r"])
            sigX = float(params[r]["sigma_fx"])

            # use the factor correlation corr(S:r, X:r)
            rho_SX = float(C_factors[idx[("S", r)], idx[("X", r)]])
            adj = rho_SX * sigS * sigX

            F[i] = (X0 * S0) * np.exp((r_f - q - adj) * T)
            # Note: no FX loading in quanto payoff (fixed conversion), only in forward adjustment.

    return F, loadings


def price_basket_option_moment_matching(
    N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T,
    K, is_call=True
):
    r_d = float(params["Local"]["r"])
    df = np.exp(-r_d * T)

    F_i, loadings = basket_domestic_forwards_and_loadings(
        basket_composition=basket_composition,
        params=params,
        factors=factors,
        C_factors=C_factors,
        fx_mode_sp=fx_mode_sp,
        T=T,
    )

    # cov(log P_i, log P_j) per year = a_i^T C a_j
    cov = loadings @ C_factors @ loadings.T
    var = np.diag(cov).copy()
    sigmaP = np.sqrt(np.maximum(var, 1e-16))
    denom = np.outer(sigmaP, sigmaP)
    rhoP = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(rhoP, 1.0)

    # Moments: u_i = N_i F_i
    u = N_shares * F_i
    M1 = float(np.sum(u))

    A = np.exp(rhoP * np.outer(sigmaP, sigmaP) * T)
    M2 = float(np.sum(np.outer(u, u) * A))

    if M1 <= 0 or M2 <= 0:
        return 0.0, {"M1": M1, "M2": M2, "sigmaB": 0.0, "FB": M1}

    sigmaB2 = (1.0 / T) * np.log(max(M2 / (M1 * M1), 1e-16))
    sigmaB = float(np.sqrt(max(sigmaB2, 0.0)))

    FB = M1
    pv = bs_from_forward(F=FB, K=K, df=df, sigma=sigmaB, tau=T, is_call=is_call)

    details = {"M1": M1, "M2": M2, "sigmaB": sigmaB, "FB": FB}
    return float(pv), details
