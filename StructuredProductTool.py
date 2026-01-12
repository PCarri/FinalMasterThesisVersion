import re
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib as mpl
from scipy.stats import norm

mpl.rcParams["font.family"] = "serif"
st.set_page_config(layout="wide")


# ------------------------------------------------------------
# Universe
# ------------------------------------------------------------
REGIONS = ["Local", "Foreign 1", "Foreign 2"]
FOREIGN_REGIONS = ["Foreign 1", "Foreign 2"]


STRATEGIES = [
    "ZCB + Basket Call",
    "ZCB + Basket Call Spread",
    "ZCB + Basket Risk Reversal (Long Call + Short Put)",
]


# ------------------------------------------------------------
# Helpers: correlation dict keys and Streamlit keys
# ------------------------------------------------------------
def factor_label(f):
    # f = ("S"/"X", "Region")
    return f"{f[0]}:{f[1]}"

def corr_key(a, b):
    return tuple(sorted([a, b]))

def safe_widget_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
def make_default_corr_params():
    """
    Store correlations as a dict keyed by tuple(sorted([label_a, label_b])).
    Example key: ("S:Local","X:Foreign 1") -> 0.10
    """
    default = {}

    def set_pair(a, b, v):
        if a == b:
            return
        default[corr_key(a, b)] = float(v)

    # Equity-equity (3)
    set_pair("S:Local", "S:Foreign 1", 0.30)
    set_pair("S:Local", "S:Foreign 2", 0.25)
    set_pair("S:Foreign 1", "S:Foreign 2", 0.35)

    # FX-FX (1)
    set_pair("X:Foreign 1", "X:Foreign 2", 0.20)

    # Equity-FX (6 possible if 3 equities × 2 FX)
    # Own pairs
    set_pair("S:Foreign 1", "X:Foreign 1", 0.65)
    set_pair("S:Foreign 2", "X:Foreign 2", -0.30)

    # Cross pairs (defaults 0)
    set_pair("S:Local", "X:Foreign 1", 0.00)
    set_pair("S:Local", "X:Foreign 2", 0.00)
    set_pair("S:Foreign 1", "X:Foreign 2", 0.00)
    set_pair("S:Foreign 2", "X:Foreign 1", 0.00)

    return default


def make_default_params():
    """
    Conventions:
    - FX X_t = domestic currency per 1 unit of foreign currency.
    - Local FX is identically 1.
    - Equity simulations use mu (expected return) under P: dS/S = (mu-q)dt + sigma dW
    - FX simulations use (r_d - r_f) drift: dX/X = (r_d-r_f)dt + sigma_X dW
    - Correlations defined over factor shocks:
      factors = [S:Local, S:F1, S:F2, X:F1, X:F2] (subset depending on selection)
    """
    return {
        "Local": {
            "S0": 100.0,
            "mu": 0.07,
            "r": 0.03,      # domestic risk-free
            "sigma": 0.20,
            "q": 0.00,
            "fx0": 1.0,
        },
        "Foreign 1": {
            "S0": 120.0,
            "mu": 0.08,
            "r": 0.04,      # foreign risk-free (currency 1)
            "sigma": 0.22,
            "q": 0.00,
            "fx0": 0.92,
            "sigma_fx": 0.12,
        },
        "Foreign 2": {
            "S0": 80.0,
            "mu": 0.09,
            "r": 0.01,      # foreign risk-free (currency 2)
            "sigma": 0.24,
            "q": 0.00,
            "fx0": 0.0062,
            "sigma_fx": 0.15,
        },
        "CORR": make_default_corr_params(),
    }


# ------------------------------------------------------------
# Investor UI
# ------------------------------------------------------------
def investor_firstSelections():
    st.sidebar.markdown("## Investor inputs")

    investment_amount = float(st.sidebar.slider("Investment amount", 1_000, 100_000, 10_000, 1_000))
    T = float(st.sidebar.slider("Maturity T (years)", 1, 5, 1, 1))

    basket_composition = st.sidebar.multiselect(
        "Select regions:",
        options=REGIONS,
        default=["Local"],
        max_selections=3,
    )
    if len(basket_composition) == 0:
        st.warning("Select at least one region.")
        basket_composition = ["Local"]

    st.sidebar.markdown("### Basket weights")
    weights_pct = {r: 0.0 for r in basket_composition}

    if len(basket_composition) == 1:
        weights_pct[basket_composition[0]] = 100.0
        st.sidebar.info(f"100% in {basket_composition[0]}")
    elif len(basket_composition) == 2:
        r0, r1 = basket_composition
        w0 = st.sidebar.slider(f"Weight {r0} (%)", 0.0, 100.0, 50.0, 1.0)
        weights_pct[r0] = w0
        weights_pct[r1] = 100.0 - w0
    else:
        r0, r1, r2 = basket_composition
        w0 = st.sidebar.slider(f"Weight {r0} (%)", 0.0, 100.0, 34.0, 1.0)
        rem = 100.0 - w0
        w1 = st.sidebar.slider(f"Weight {r1} (%)", 0.0, rem, min(33.0, rem), 1.0)
        weights_pct[r0] = w0
        weights_pct[r1] = w1
        weights_pct[r2] = 100.0 - w0 - w1

    weights_vector = np.array([weights_pct[r] for r in basket_composition], dtype=float) / 100.0

    has_foreign = any(r in basket_composition for r in FOREIGN_REGIONS)
    if has_foreign:
        fx_mode_sp = st.sidebar.radio(
            "Structured products FX treatment (only)",
            options=["Take FX risk (Compo)", "Hedge FX risk (Quanto)"],
            index=0,
        )
    else:
        fx_mode_sp = "Plain (Local only)"

    return investment_amount, T, basket_composition, weights_vector, fx_mode_sp


# ------------------------------------------------------------
# Practitioner Panel: market inputs (NO correlations here)
# ------------------------------------------------------------
def practitioner_panel(params):
    with st.sidebar.expander("Practitioner control panel (advanced)", expanded=False):
        st.caption("Edit market inputs (defaults are used if not changed).")

        for region in REGIONS:
            st.markdown(f"**{region}**")
            params[region]["S0"] = st.number_input(
                f"{region} S0 (asset ccy)",
                value=float(params[region]["S0"]),
                step=1.0,
            )
            params[region]["mu"] = st.number_input(
                f"{region} mu (expected return, P)",
                value=float(params[region]["mu"]),
                step=0.001,
                format="%.4f",
            )
            params[region]["r"] = st.number_input(
                f"{region} r (cont.)",
                value=float(params[region]["r"]),
                step=0.001,
                format="%.4f",
            )
            params[region]["sigma"] = st.number_input(
                f"{region} sigma (equity)",
                value=float(params[region]["sigma"]),
                step=0.01,
                format="%.4f",
            )
            params[region]["q"] = st.number_input(
                f"{region} q (div yield)",
                value=float(params[region]["q"]),
                step=0.001,
                format="%.4f",
            )

            # FX inputs
            if region == "Local":
                params[region]["fx0"] = 1.0
                st.caption("Local FX fixed at 1.0")
            else:
                params[region]["fx0"] = st.number_input(
                    f"{region} FX X0 (dom per 1 ccy)",
                    value=float(params[region]["fx0"]),
                    step=0.0001,
                    format="%.6f",
                )
                params[region]["sigma_fx"] = st.number_input(
                    f"{region} sigma_fx",
                    value=float(params[region]["sigma_fx"]),
                    step=0.01,
                    format="%.4f",
                )

    return params


# ------------------------------------------------------------
# Correlation UI: unified factor correlation matrix sliders
# ------------------------------------------------------------
def factor_list_for_selection(basket_composition):
    """
    Factor order (stable):
      1) Equities for selected regions: ("S", region)
      2) FX for selected foreign regions: ("X", region)
    """
    factors = []
    for r in basket_composition:
        factors.append(("S", r))
    for r in basket_composition:
        if r in FOREIGN_REGIONS:
            factors.append(("X", r))
    return factors


def correlation_panel(params, basket_composition):
    factors = factor_list_for_selection(basket_composition)
    labels = [factor_label(f) for f in factors]

    with st.sidebar.expander("Correlation matrix (equities + FX factors)", expanded=True):
        st.caption(
            "Define correlations between all simulated risk-factor shocks. "
            "For 3 assets with 2 foreign currencies, this is a 5×5 factor matrix "
            "(3 equities + 2 FX). If the matrix is not PSD, the tool will regularize it for Cholesky."
        )

        # sliders for all pairwise correlations among currently active factors
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                k = corr_key(a, b)
                v0 = float(params["CORR"].get(k, 0.0))
                v = st.slider(
                    f"corr({a}, {b})",
                    -0.95, 0.95, v0, 0.01,
                    key=safe_widget_key(f"corr_{a}_{b}")
                )
                params["CORR"][k] = float(v)

    return params


# ------------------------------------------------------------
# PSD / Cholesky safety
# ------------------------------------------------------------
def nearest_psd_corr(C, eps=1e-10):
    # eigenvalue clipping + renormalization to diag=1
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    C_psd = (V * w) @ V.T
    d = np.sqrt(np.diag(C_psd))
    C_corr = C_psd / np.outer(d, d)
    np.fill_diagonal(C_corr, 1.0)
    return C_corr


def build_factor_corr_matrix(factors, params):
    labels = [factor_label(f) for f in factors]
    d = len(labels)
    C = np.eye(d, dtype=float)

    for i in range(d):
        for j in range(i + 1, d):
            k = corr_key(labels[i], labels[j])
            rho = float(params["CORR"].get(k, 0.0))
            C[i, j] = rho
            C[j, i] = rho

    # sym + diag
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    # try cholesky; if fails -> regularize
    adjusted = False
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        C = nearest_psd_corr(C)
        adjusted = True

    return C, adjusted


# ------------------------------------------------------------
# Correlated simulator: equities + FX
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Buy & Hold basket (ALWAYS stochastic FX)
# ------------------------------------------------------------
def buy_and_hold_basket(investment_amount, basket_composition, weights_vector, params, S_paths, X_paths):
    S0 = np.array([params[r]["S0"] for r in basket_composition], dtype=float)
    X0 = np.array([params[r]["fx0"] if r in FOREIGN_REGIONS else 1.0 for r in basket_composition], dtype=float)
    P0 = S0 * X0

    budget_i = investment_amount * weights_vector
    N_shares = budget_i / P0  # thesis eq: N_i = w_i V0 / (S0 X0)

    V = np.zeros_like(next(iter(S_paths.values())))
    for j, r in enumerate(basket_composition):
        V += N_shares[j] * S_paths[r] * X_paths[r]

    logret_T = np.log(V[-1, :] / V[0, :])
    return N_shares, P0, V, logret_T


# ------------------------------------------------------------
# Black forward pricing helper
# ------------------------------------------------------------
def bs_from_forward(F, K, df, sigma, tau, is_call=True):
    F = float(F); K = float(K); df = float(df); sigma = float(sigma); tau = float(tau)
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic
    vol_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ------------------------------------------------------------
# Basket option pricing via moment matching (M1,M2)
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
        T=T
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


# ------------------------------------------------------------
# Structured products
# ------------------------------------------------------------
def zcb_pv(face_value, r_cont, T):
    return float(face_value) * np.exp(-float(r_cont) * float(T))


def basket_terminal_value_for_payoff(N_shares, basket_composition, params, S_T, X_T, fx_mode_sp):
    """
    Structured-product basket B_T:
      - Compo: P_T^{(i)} = S_T^{(i)} X_T^{(i)}
      - Quanto: P_T^{(i)} = S_T^{(i)} X0^{(i)}
      - Domestic: P_T^{(i)} = S_T^{(i)}
    """
    B = np.zeros(S_T.shape[0], dtype=float)
    for j, r in enumerate(basket_composition):
        if r not in FOREIGN_REGIONS or fx_mode_sp == "Plain (Local only)":
            P = S_T[:, j] * 1.0
        else:
            if fx_mode_sp == "Take FX risk (Compo)":
                P = S_T[:, j] * X_T[:, j]
            else:
                X0 = float(params[r]["fx0"])
                P = S_T[:, j] * X0
        B += N_shares[j] * P
    return B


def structured_product_terminal_values(
    investment_amount,
    protection_pct,
    basket_composition,
    weights_vector,
    params,
    fx_mode_sp,
    T,
    strategy_name,
    call_up=1.10,
    spread_up=1.10,
    rr_call=1.05,
    rr_put=0.95,
    S_paths=None,
    X_paths=None,
    factors=None,
    C_factors=None
):
    r_d = float(params["Local"]["r"])

    # Shares N_i fixed at inception using domestic initial prices P0 = S0*X0
    S0 = np.array([params[r]["S0"] for r in basket_composition], dtype=float)
    X0 = np.array([params[r]["fx0"] if r in FOREIGN_REGIONS else 1.0 for r in basket_composition], dtype=float)
    P0 = S0 * X0
    budget_i = investment_amount * weights_vector
    N_shares = budget_i / P0

    B0 = float(np.sum(N_shares * P0))  # ~ investment_amount by construction

    # ZCB + option budget
    face = investment_amount * (protection_pct / 100.0)
    pv_zcb = zcb_pv(face, r_d, T)
    opt_budget = investment_amount - pv_zcb

    # --- Define basket strikes and basket option premium (moment-matching) ---
    prem_det = None

    if strategy_name == "ZCB + Basket Call":
        K = call_up * B0
        prem, prem_det = price_basket_option_moment_matching(
            N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T, K, is_call=True
        )
        net_prem = prem
        alpha = 0.0 if net_prem <= 0 else min(1.0, opt_budget / net_prem)
        leftover0 = opt_budget - alpha * net_prem

        def opt_payoff(BT):
            return np.maximum(BT - K, 0.0)

        strikes = {"K": K}

    elif strategy_name == "ZCB + Basket Call Spread":
        K1 = 1.00 * B0
        K2 = spread_up * B0
        prem1, det1 = price_basket_option_moment_matching(
            N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T, K1, is_call=True
        )
        prem2, det2 = price_basket_option_moment_matching(
            N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T, K2, is_call=True
        )
        net_prem = max(prem1 - prem2, 0.0)
        alpha = 0.0 if net_prem <= 0 else min(1.0, opt_budget / net_prem)
        leftover0 = opt_budget - alpha * net_prem

        prem_det = {"leg1": det1, "leg2": det2}

        def opt_payoff(BT):
            return np.maximum(BT - K1, 0.0) - np.maximum(BT - K2, 0.0)

        strikes = {"K1": K1, "K2": K2}

    elif strategy_name == "ZCB + Basket Risk Reversal (Long Call + Short Put)":
        Kc = rr_call * B0
        Kp = rr_put * B0

        premC, detC = price_basket_option_moment_matching(
            N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T, Kc, is_call=True
        )
        premP, detP = price_basket_option_moment_matching(
            N_shares, basket_composition, params, factors, C_factors, fx_mode_sp, T, Kp, is_call=False
        )
        net_prem = premC - premP  # can be negative (credit)
        if net_prem <= 0:
            alpha = 1.0
            leftover0 = opt_budget - alpha * net_prem  # increases if credit
        else:
            alpha = min(1.0, opt_budget / net_prem)
            leftover0 = opt_budget - alpha * net_prem

        prem_det = {"call": detC, "put": detP}

        def opt_payoff(BT):
            return np.maximum(BT - Kc, 0.0) - np.maximum(Kp - BT, 0.0)

        strikes = {"Kc": Kc, "Kp": Kp}

    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")

    # leftover cash grows at r_d
    cash_T = max(0.0, float(leftover0)) * np.exp(r_d * T)

    # Terminal basket distribution from simulated S and X
    S_T_mat = np.column_stack([S_paths[r][-1, :] for r in basket_composition])
    X_T_mat = np.column_stack([X_paths[r][-1, :] for r in basket_composition])

    B_T = basket_terminal_value_for_payoff(N_shares, basket_composition, params, S_T_mat, X_T_mat, fx_mode_sp)
    V_T = face + cash_T + alpha * opt_payoff(B_T)

    details = {
        "B0": B0,
        "face": face,
        "pv_zcb": pv_zcb,
        "opt_budget": opt_budget,
        "alpha": alpha,
        "leftover0": float(leftover0),
        "cash_T": cash_T,
        "net_premium": float(net_prem),
        "strikes": strikes,
        "premium_details": prem_det,
    }
    return V_T, details


def payoff_curve_vs_basket(details, strategy_name, grid_min=0.5, grid_max=1.8, n_points=220):
    B0 = float(details["B0"])
    B_grid = np.linspace(grid_min * B0, grid_max * B0, n_points)

    strikes = details["strikes"]
    if strategy_name == "ZCB + Basket Call":
        K = strikes["K"]
        opt = np.maximum(B_grid - K, 0.0)
    elif strategy_name == "ZCB + Basket Call Spread":
        K1, K2 = strikes["K1"], strikes["K2"]
        opt = np.maximum(B_grid - K1, 0.0) - np.maximum(B_grid - K2, 0.0)
    else:
        Kc, Kp = strikes["Kc"], strikes["Kp"]
        opt = np.maximum(B_grid - Kc, 0.0) - np.maximum(Kp - B_grid, 0.0)

    V_grid = float(details["face"]) + float(details["cash_T"]) + float(details["alpha"]) * opt
    return B_grid, V_grid


# =========================
# App starts here
# =========================
st.title("Equity-Linked Product Playground")
st.write(
    "Unified factor correlation matrix (equities + FX), correlated simulation, and structured products priced as an option on the basket "
    "via moment matching."
)

# 1) Investor selections
investment_amount, T, basket_composition, weights_vector, fx_mode_sp = investor_firstSelections()

# 2) Defaults + practitioner overrides
params = deepcopy(make_default_params())
params = practitioner_panel(params)

# 3) Unified correlation section (UPDATED)
params = correlation_panel(params, basket_composition)

# 4) Simulation settings
dt = 1 / 252
n_steps = int(T / dt)
n_paths = 500
rng = np.random.default_rng(7)

# 5) Simulate correlated equities + FX
time, S_paths, X_paths, factors, C_used, corr_adjusted = simulate_equities_and_fx(
    basket_composition=basket_composition,
    params=params,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths,
    rng=rng
)

# 6) Buy & hold basket (always stochastic FX)
st.subheader("Buy & Hold basket (domestic value uses stochastic FX for foreign assets)")
N_shares, P0, V_bh, logret_bh = buy_and_hold_basket(
    investment_amount=investment_amount,
    basket_composition=basket_composition,
    weights_vector=weights_vector,
    params=params,
    S_paths=S_paths,
    X_paths=X_paths
)

colA, colB = st.columns([1.3, 1.0])
with colA:
    shares_df = pd.DataFrame({
        "Region": basket_composition,
        "Weight": weights_vector,
        "S0 (asset)": [params[r]["S0"] for r in basket_composition],
        "X0 (FX)": [params[r]["fx0"] if r in FOREIGN_REGIONS else 1.0 for r in basket_composition],
        "P0 (domestic)": P0,
        "Budget (domestic)": investment_amount * weights_vector,
        "Shares N": N_shares,
    })
    st.dataframe(
        shares_df.style.format({
            "Weight": "{:.2%}",
            "S0 (asset)": "{:.2f}",
            "X0 (FX)": "{:.6f}",
            "P0 (domestic)": "{:.2f}",
            "Budget (domestic)": "{:,.2f}",
            "Shares N": "{:.6f}",
        }),
        use_container_width=True
    )

with colB:
    factor_names = [factor_label(f) for f in factors]
    corr_df = pd.DataFrame(C_used, index=factor_names, columns=factor_names)
    if corr_adjusted:
        st.warning("Correlation matrix was not PSD and was regularized to allow Cholesky.")
    st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True)

col1, col2 = st.columns([1.6, 1.0])
with col1:
    max_show = min(100, V_bh.shape[1])
    st.line_chart(pd.DataFrame(V_bh[:, :max_show], index=time), height=420)

with col2:
    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=140)
    ax.hist(logret_bh, bins=40)
    ax.set_xlabel("log(V_T / V_0)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Buy & Hold terminal log-return", fontsize=9)
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Structured product controls
# ------------------------------------------------------------
st.sidebar.markdown("## Structured products inputs")
protection_pct = st.sidebar.slider("Capital protection at maturity (%)", 0, 100, 100, 5)

call_up = st.sidebar.slider("Basket call strike moneyness (K/B0)", 1.00, 1.30, 1.10, 0.01)
spread_up = st.sidebar.slider("Call-spread upper strike moneyness (K2/B0)", 1.01, 1.50, 1.10, 0.01)
rr_call = st.sidebar.slider("Risk reversal call moneyness (Kc/B0)", 1.00, 1.30, 1.05, 0.01)
rr_put = st.sidebar.slider("Risk reversal put moneyness (Kp/B0)", 0.70, 1.00, 0.95, 0.01)

st.header("Structured products (option on basket via moment matching)")
st.info(
    "Buy & Hold always includes stochastic FX in domestic valuation. "
    "The selected FX mode affects only structured products (basket definition + basket option pricing inputs)."
)

cols = st.columns(len(STRATEGIES), gap="large")
for col, strat in zip(cols, STRATEGIES):
    with col:
        st.subheader(strat)

        V_T_sp, details = structured_product_terminal_values(
            investment_amount=investment_amount,
            protection_pct=protection_pct,
            basket_composition=basket_composition,
            weights_vector=weights_vector,
            params=params,
            fx_mode_sp=fx_mode_sp,
            T=T,
            strategy_name=strat,
            call_up=call_up,
            spread_up=spread_up,
            rr_call=rr_call,
            rr_put=rr_put,
            S_paths=S_paths,
            X_paths=X_paths,
            factors=factors,
            C_factors=C_used
        )

        sp_logret = np.log(V_T_sp / investment_amount)

        st.markdown("**Payoff vs basket level $B_T$ (option-on-basket)**")
        B_grid, V_grid = payoff_curve_vs_basket(details, strat)

        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=140)
        ax.plot(B_grid, V_grid, linewidth=2.0)
        ax.axvline(float(details["B0"]), linestyle="--", linewidth=1.0)
        ax.set_xlabel("Basket level $B_T$ (domestic)", fontsize=8)
        ax.set_ylabel("Value at T (domestic)", fontsize=8)
        ax.set_title("Payoff vs basket", fontsize=9)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Return distribution (Buy & Hold vs Structured)**")
        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=140)
        ax.hist(logret_bh, bins=35, alpha=0.6, label="Buy & Hold")
        ax.hist(sp_logret, bins=35, alpha=0.6, label="Structured")
        ax.set_xlabel("Terminal log return", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title("Histogram comparison", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"FX mode (structured): {fx_mode_sp} | "
            f"B0: {details['B0']:,.2f} | "
            f"ZCB face: {details['face']:,.2f} | "
            f"ZCB PV: {details['pv_zcb']:,.2f} | "
            f"Opt budget: {details['opt_budget']:,.2f}"
        )
        st.caption(
            f"Net option premium (per 1 unit): {details['net_premium']:,.2f} | "
            f"alpha (<=1): {details['alpha']:.4f} | "
            f"Cash at T from leftover: {details['cash_T']:,.2f}"
        )
