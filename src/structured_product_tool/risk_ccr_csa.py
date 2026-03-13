"""CCR/CSA risk engines for structured products."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import FOREIGN_REGIONS, STRATEGIES
from .pricing import bs_from_forward


def zcb_pv(face_value: float, r_cont: float, T: float) -> float:
    return float(face_value) * np.exp(-float(r_cont) * float(T))


def _grid_indices(time: np.ndarray, n_grid_points: int) -> np.ndarray:
    n_times = int(time.shape[0])
    grid_idx = np.unique(
        np.round(np.linspace(0, n_times - 1, int(n_grid_points))).astype(int)
    )
    if grid_idx.size < 2:
        raise ValueError("CCR/CSA grid must contain at least 2 points.")
    return grid_idx


def price_basket_option_moment_matching_at_t(
    N_shares: np.ndarray,
    basket_composition: List[str],
    params: Dict,
    factors: List[Tuple[str, str]],
    C_factors: np.ndarray,
    fx_mode_sp: str,
    tau: float,
    K: float,
    is_call: bool,
    S_now_vec: np.ndarray,
    X_now_vec: np.ndarray,
) -> float:
    """
    Moment-matching basket option pricing with current scenario state.
    Mirrors pricing.price_basket_option_moment_matching but uses S_now/X_now.
    """
    r_d = float(params["Local"]["r"])
    df = np.exp(-r_d * float(tau))

    idx = {factors[k]: k for k in range(len(factors))}

    n = len(basket_composition)
    F = np.zeros(n, dtype=float)
    loadings = np.zeros((n, len(factors)), dtype=float)

    for i, r in enumerate(basket_composition):
        S_now = float(S_now_vec[i])
        q = float(params[r].get("q", 0.0))
        sigS = float(params[r]["sigma"])

        loadings[i, idx[("S", r)]] += sigS

        if r not in FOREIGN_REGIONS or fx_mode_sp == "Plain (Local only)":
            F[i] = S_now * np.exp((r_d - q) * tau)
            continue

        if fx_mode_sp == "Take FX risk (Compo)":
            X_now = float(X_now_vec[i])
            sigX = float(params[r]["sigma_fx"])
            loadings[i, idx[("X", r)]] += sigX
            F[i] = (S_now * X_now) * np.exp((r_d - q) * tau)
        else:
            X0 = float(params[r]["fx0"])
            r_f = float(params[r]["r"])
            sigX = float(params[r]["sigma_fx"])
            rho_SX = float(C_factors[idx[("S", r)], idx[("X", r)]])
            adj = rho_SX * sigS * sigX
            F[i] = (S_now * X0) * np.exp((r_f - q - adj) * tau)

    cov = loadings @ C_factors @ loadings.T
    var = np.diag(cov).copy()
    sigmaP = np.sqrt(np.maximum(var, 1e-16))
    denom = np.outer(sigmaP, sigmaP)
    rhoP = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(rhoP, 1.0)

    u = N_shares * F
    M1 = float(np.sum(u))

    A = np.exp(rhoP * np.outer(sigmaP, sigmaP) * tau)
    M2 = float(np.sum(np.outer(u, u) * A))

    if M1 <= 0 or M2 <= 0:
        return 0.0

    if tau <= 0:
        sigmaB = 0.0
    else:
        sigmaB2 = (1.0 / tau) * np.log(max(M2 / (M1 * M1), 1e-16))
        sigmaB = float(np.sqrt(max(sigmaB2, 0.0)))

    FB = M1
    pv = bs_from_forward(F=FB, K=K, df=df, sigma=sigmaB, tau=tau, is_call=is_call)
    return float(pv)


class StructuredProductMTMEngine:
    """Compute MTM paths for structured product strategies."""

    def __init__(
        self,
        investment_amount: float,
        protection_pct: float,
        basket_composition: List[str],
        weights_vector: np.ndarray,
        params: Dict,
        fx_mode_sp: str,
        T: float,
        factors: List[Tuple[str, str]],
        C_factors: np.ndarray,
        call_up: float,
        spread_up: float,
        rr_call: float,
        rr_put: float,
    ) -> None:
        self.investment_amount = float(investment_amount)
        self.protection_pct = float(protection_pct)
        self.basket_composition = list(basket_composition)
        self.weights_vector = np.array(weights_vector, dtype=float)
        self.params = params
        self.fx_mode_sp = fx_mode_sp
        self.T = float(T)
        self.factors = factors
        self.C_factors = np.array(C_factors, dtype=float)
        self.call_up = float(call_up)
        self.spread_up = float(spread_up)
        self.rr_call = float(rr_call)
        self.rr_put = float(rr_put)

        self.r_d = float(params["Local"]["r"])
        self._compute_inception_terms()

    def _compute_inception_terms(self) -> None:
        S0 = np.array([self.params[r]["S0"] for r in self.basket_composition], dtype=float)
        X0 = np.array(
            [self.params[r]["fx0"] if r in FOREIGN_REGIONS else 1.0 for r in self.basket_composition],
            dtype=float,
        )
        P0 = S0 * X0

        budget_i = self.investment_amount * self.weights_vector
        self.N_shares = budget_i / P0

        self.B0 = float(np.sum(self.N_shares * P0))
        self.face = self.investment_amount * (self.protection_pct / 100.0)
        self.pv_zcb = zcb_pv(self.face, self.r_d, self.T)
        self.opt_budget = self.investment_amount - self.pv_zcb

        self.deal_terms: Dict[str, Dict[str, float]] = {}

        # ZCB + Basket Call
        K = self.call_up * self.B0
        prem = price_basket_option_moment_matching_at_t(
            self.N_shares,
            self.basket_composition,
            self.params,
            self.factors,
            self.C_factors,
            self.fx_mode_sp,
            self.T,
            K,
            True,
            S0,
            X0,
        )
        net_prem = prem
        alpha = 0.0 if net_prem <= 0 else min(1.0, self.opt_budget / net_prem)
        leftover0 = self.opt_budget - alpha * net_prem
        self.deal_terms["ZCB + Basket Call"] = {
            "alpha": float(alpha),
            "leftover0": float(leftover0),
            "K": float(K),
        }

        # ZCB + Basket Call Spread (K1 hardcoded at ATM)
        K1 = 1.00 * self.B0
        K2 = self.spread_up * self.B0
        prem1 = price_basket_option_moment_matching_at_t(
            self.N_shares,
            self.basket_composition,
            self.params,
            self.factors,
            self.C_factors,
            self.fx_mode_sp,
            self.T,
            K1,
            True,
            S0,
            X0,
        )
        prem2 = price_basket_option_moment_matching_at_t(
            self.N_shares,
            self.basket_composition,
            self.params,
            self.factors,
            self.C_factors,
            self.fx_mode_sp,
            self.T,
            K2,
            True,
            S0,
            X0,
        )
        net_prem = max(prem1 - prem2, 0.0)
        alpha = 0.0 if net_prem <= 0 else min(1.0, self.opt_budget / net_prem)
        leftover0 = self.opt_budget - alpha * net_prem
        self.deal_terms["ZCB + Basket Call Spread"] = {
            "alpha": float(alpha),
            "leftover0": float(leftover0),
            "K1": float(K1),
            "K2": float(K2),
        }

        # ZCB + Basket Risk Reversal (Long Call + Short Put)
        Kc = self.rr_call * self.B0
        Kp = self.rr_put * self.B0
        premC = price_basket_option_moment_matching_at_t(
            self.N_shares,
            self.basket_composition,
            self.params,
            self.factors,
            self.C_factors,
            self.fx_mode_sp,
            self.T,
            Kc,
            True,
            S0,
            X0,
        )
        premP = price_basket_option_moment_matching_at_t(
            self.N_shares,
            self.basket_composition,
            self.params,
            self.factors,
            self.C_factors,
            self.fx_mode_sp,
            self.T,
            Kp,
            False,
            S0,
            X0,
        )
        net_prem = premC - premP
        if net_prem <= 0:
            alpha = 1.0
            leftover0 = self.opt_budget - alpha * net_prem
        else:
            alpha = min(1.0, self.opt_budget / net_prem)
            leftover0 = self.opt_budget - alpha * net_prem
        self.deal_terms["ZCB + Basket Risk Reversal (Long Call + Short Put)"] = {
            "alpha": float(alpha),
            "leftover0": float(leftover0),
            "Kc": float(Kc),
            "Kp": float(Kp),
        }

    def _validate_paths(self, time: np.ndarray, S_paths: Dict, X_paths: Dict) -> int:
        if time.ndim != 1:
            raise ValueError("time must be a 1D array.")
        n_times = int(time.shape[0])
        n_paths = None
        for r in self.basket_composition:
            if r not in S_paths or r not in X_paths:
                raise ValueError(f"Missing paths for region: {r}")
            if S_paths[r].shape[0] != n_times or X_paths[r].shape[0] != n_times:
                raise ValueError("S_paths/X_paths time dimension mismatch.")
            if S_paths[r].ndim != 2 or X_paths[r].ndim != 2:
                raise ValueError("S_paths/X_paths must be 2D arrays.")
            if n_paths is None:
                n_paths = int(S_paths[r].shape[1])
            if S_paths[r].shape[1] != n_paths or X_paths[r].shape[1] != n_paths:
                raise ValueError("S_paths/X_paths path dimension mismatch.")
        return int(n_paths)

    def mtm_paths(
        self,
        time: np.ndarray,
        S_paths: Dict[str, np.ndarray],
        X_paths: Dict[str, np.ndarray],
        n_grid_points: int = 30,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        n_paths = self._validate_paths(time, S_paths, X_paths)
        grid_idx = _grid_indices(time, n_grid_points)
        time_grid = time[grid_idx]

        n_grid = int(time_grid.shape[0])
        V_paths = {s: np.zeros((n_grid, n_paths), dtype=float) for s in STRATEGIES}

        for k, idx in enumerate(grid_idx):
            t_k = float(time[idx])
            tau = max(self.T - t_k, 0.0)
            V_zcb = self.face * np.exp(-self.r_d * tau)

            cash_by_strat = {
                s: max(0.0, float(self.deal_terms[s]["leftover0"])) * np.exp(self.r_d * t_k)
                for s in STRATEGIES
            }

            for s_idx in range(n_paths):
                S_now_vec = np.array([S_paths[r][idx, s_idx] for r in self.basket_composition], dtype=float)
                X_now_vec = np.array([X_paths[r][idx, s_idx] for r in self.basket_composition], dtype=float)

                # ZCB + Basket Call
                terms = self.deal_terms["ZCB + Basket Call"]
                call_price = price_basket_option_moment_matching_at_t(
                    self.N_shares,
                    self.basket_composition,
                    self.params,
                    self.factors,
                    self.C_factors,
                    self.fx_mode_sp,
                    tau,
                    terms["K"],
                    True,
                    S_now_vec,
                    X_now_vec,
                )
                V_paths["ZCB + Basket Call"][k, s_idx] = (
                    V_zcb + cash_by_strat["ZCB + Basket Call"] + terms["alpha"] * call_price
                )

                # ZCB + Basket Call Spread
                terms = self.deal_terms["ZCB + Basket Call Spread"]
                call1 = price_basket_option_moment_matching_at_t(
                    self.N_shares,
                    self.basket_composition,
                    self.params,
                    self.factors,
                    self.C_factors,
                    self.fx_mode_sp,
                    tau,
                    terms["K1"],
                    True,
                    S_now_vec,
                    X_now_vec,
                )
                call2 = price_basket_option_moment_matching_at_t(
                    self.N_shares,
                    self.basket_composition,
                    self.params,
                    self.factors,
                    self.C_factors,
                    self.fx_mode_sp,
                    tau,
                    terms["K2"],
                    True,
                    S_now_vec,
                    X_now_vec,
                )
                V_paths["ZCB + Basket Call Spread"][k, s_idx] = (
                    V_zcb
                    + cash_by_strat["ZCB + Basket Call Spread"]
                    + terms["alpha"] * (call1 - call2)
                )

                # ZCB + Basket Risk Reversal (Long Call + Short Put)
                terms = self.deal_terms["ZCB + Basket Risk Reversal (Long Call + Short Put)"]
                callc = price_basket_option_moment_matching_at_t(
                    self.N_shares,
                    self.basket_composition,
                    self.params,
                    self.factors,
                    self.C_factors,
                    self.fx_mode_sp,
                    tau,
                    terms["Kc"],
                    True,
                    S_now_vec,
                    X_now_vec,
                )
                putp = price_basket_option_moment_matching_at_t(
                    self.N_shares,
                    self.basket_composition,
                    self.params,
                    self.factors,
                    self.C_factors,
                    self.fx_mode_sp,
                    tau,
                    terms["Kp"],
                    False,
                    S_now_vec,
                    X_now_vec,
                )
                V_paths["ZCB + Basket Risk Reversal (Long Call + Short Put)"][k, s_idx] = (
                    V_zcb
                    + cash_by_strat["ZCB + Basket Risk Reversal (Long Call + Short Put)"]
                    + terms["alpha"] * (callc - putp)
                )

        return time_grid, V_paths


class CCRMetricsEngine:
    """Compute CCR exposure metrics."""

    @staticmethod
    def metrics(V_paths: np.ndarray, perspective: str) -> Dict[str, np.ndarray]:
        V_adj = -V_paths if perspective == "Issuer" else V_paths
        E_pos = np.maximum(V_adj, 0.0)
        E_neg = np.maximum(-V_adj, 0.0)
        EE_pos = np.mean(E_pos, axis=1)
        PFE95_pos = np.quantile(E_pos, 0.95, axis=1)
        PFE99_pos = np.quantile(E_pos, 0.99, axis=1)
        EE_neg = np.mean(E_neg, axis=1)
        PFE95_neg = np.quantile(E_neg, 0.95, axis=1)
        PFE99_neg = np.quantile(E_neg, 0.99, axis=1)
        return {
            "V_adj": V_adj,
            "E_pos": E_pos,
            "E_neg": E_neg,
            "EE_pos": EE_pos,
            "PFE95_pos": PFE95_pos,
            "PFE99_pos": PFE99_pos,
            "EE_neg": EE_neg,
            "PFE95_neg": PFE95_neg,
            "PFE99_neg": PFE99_neg,
        }


class CSAEngine:
    """Compute collateralized exposure metrics with simplified CSA VM."""

    def __init__(
        self,
        enable: bool,
        threshold_pct: float,
        mta_pct: float,
        frequency: str,
        notional: float,
    ) -> None:
        self.enable = bool(enable)
        self.threshold_pct = float(threshold_pct)
        self.mta_pct = float(mta_pct)
        self.frequency = frequency
        self.notional = float(notional)

    def collateral_and_net_exposure(
        self,
        V_adj: np.ndarray,
        time_grid: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n_grid, n_paths = V_adj.shape
        C_pos = np.zeros((n_grid, n_paths), dtype=float)
        C_neg = np.zeros((n_grid, n_paths), dtype=float)

        if not self.enable:
            return {
                "C_pos": C_pos,
                "C_neg": C_neg,
                "E_net_pos": np.maximum(V_adj, 0.0),
                "E_net_neg": np.maximum(-V_adj, 0.0),
            }

        H = (self.threshold_pct / 100.0) * self.notional
        MTA = (self.mta_pct / 100.0) * H

        if time_grid.ndim != 1 or time_grid.shape[0] != n_grid:
            raise ValueError("time_grid must be 1D and aligned with V_adj.")

        if self.frequency == "Every 5 grid points":
            call_dates = np.zeros(n_grid, dtype=bool)
            call_dates[0] = True
            for k in range(1, n_grid):
                if k % 5 == 0:
                    call_dates[k] = True
        else:
            call_dates = np.ones(n_grid, dtype=bool)

        C_prev_pos = np.zeros(n_paths, dtype=float)
        C_prev_neg = np.zeros(n_paths, dtype=float)
        for k in range(n_grid):
            if call_dates[k]:
                Req_pos = np.maximum(V_adj[k, :] - H, 0.0)
                Req_neg = np.maximum((-V_adj[k, :]) - H, 0.0)
                if MTA <= 0:
                    C_prev_pos = Req_pos
                    C_prev_neg = Req_neg
                else:
                    update_pos = np.abs(Req_pos - C_prev_pos) >= MTA
                    update_neg = np.abs(Req_neg - C_prev_neg) >= MTA
                    C_prev_pos = np.where(update_pos, Req_pos, C_prev_pos)
                    C_prev_neg = np.where(update_neg, Req_neg, C_prev_neg)
            C_pos[k, :] = C_prev_pos
            C_neg[k, :] = C_prev_neg

        E_net_pos = np.maximum(V_adj - C_pos, 0.0)
        E_net_neg = np.maximum((-V_adj) - C_neg, 0.0)
        return {
            "C_pos": C_pos,
            "C_neg": C_neg,
            "E_net_pos": E_net_pos,
            "E_net_neg": E_net_neg,
        }

    @staticmethod
    def metrics(E: np.ndarray) -> Dict[str, np.ndarray]:
        EE = np.mean(E, axis=1)
        PFE95 = np.quantile(E, 0.95, axis=1)
        PFE99 = np.quantile(E, 0.99, axis=1)
        return {
            "EE": EE,
            "PFE95": PFE95,
            "PFE99": PFE99,
        }
