"""Correlation helpers and UI."""

import re
import numpy as np
import streamlit as st

from .config import FOREIGN_REGIONS


def factor_label(factor):
    # factor = ("S"/"X", "Region")
    return f"{factor[0]}:{factor[1]}"


def corr_key(a, b):
    return tuple(sorted([a, b]))


def safe_widget_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def factor_list_for_selection(basket_composition):
    """
    Factor order (stable):
      1) Equities for selected regions: ("S", region)
      2) FX for selected foreign regions: ("X", region)
    """
    factors = []
    for region in basket_composition:
        factors.append(("S", region))
    for region in basket_composition:
        if region in FOREIGN_REGIONS:
            factors.append(("X", region))
    return factors


def correlation_panel(params, basket_composition):
    factors = factor_list_for_selection(basket_composition)
    labels = [factor_label(f) for f in factors]

    with st.sidebar.expander("Correlation matrix (equities + FX factors)", expanded=True):
        st.caption(
            "Define correlations between all simulated risk-factor shocks. "
            "For 3 assets with 2 foreign currencies, this is a 5x5 factor matrix "
            "(3 equities + 2 FX). If the matrix is not PSD, the tool will regularize it for Cholesky."
        )

        # sliders for all pairwise correlations among currently active factors
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                key = corr_key(a, b)
                v0 = float(params["CORR"].get(key, 0.0))
                v = st.slider(
                    f"corr({a}, {b})",
                    -0.95, 0.95, v0, 0.01,
                    key=safe_widget_key(f"corr_{a}_{b}")
                )
                params["CORR"][key] = float(v)

    return params


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
            key = corr_key(labels[i], labels[j])
            rho = float(params["CORR"].get(key, 0.0))
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
