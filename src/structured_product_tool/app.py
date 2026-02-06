"""Streamlit app entrypoint."""

from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .config import REGIONS, FOREIGN_REGIONS, STRATEGIES, make_default_params
from .correlation import correlation_panel, factor_label
from .products import (
    buy_and_hold_basket,
    payoff_curve_vs_basket,
    structured_product_terminal_values,
)
from .simulation import simulate_equities_and_fx

mpl.rcParams["font.family"] = "serif"


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
# App
# ------------------------------------------------------------

def main():
    st.set_page_config(layout="wide")

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

    # 3) Unified correlation section
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
        rng=rng,
    )

    # 6) Buy & hold basket (always stochastic FX)
    st.subheader("Buy & Hold basket (domestic value uses stochastic FX for foreign assets)")
    N_shares, P0, V_bh, logret_bh = buy_and_hold_basket(
        investment_amount=investment_amount,
        basket_composition=basket_composition,
        weights_vector=weights_vector,
        params=params,
        S_paths=S_paths,
        X_paths=X_paths,
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
            use_container_width=True,
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
                C_factors=C_used,
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


if __name__ == "__main__":
    main()
