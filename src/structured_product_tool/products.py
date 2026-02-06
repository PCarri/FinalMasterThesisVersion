"""Structured product payoffs and valuation."""

import numpy as np

from .config import FOREIGN_REGIONS
from .pricing import price_basket_option_moment_matching


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
    C_factors=None,
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
