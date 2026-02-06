"""Configuration and default parameters."""

REGIONS = ["Local", "Foreign 1", "Foreign 2"]
FOREIGN_REGIONS = ["Foreign 1", "Foreign 2"]

STRATEGIES = [
    "ZCB + Basket Call",
    "ZCB + Basket Call Spread",
    "ZCB + Basket Risk Reversal (Long Call + Short Put)",
]


def make_default_corr_params():
    """
    Store correlations as a dict keyed by tuple(sorted([label_a, label_b])).
    Example key: ("S:Local","X:Foreign 1") -> 0.10
    """
    default = {}

    def corr_key(a, b):
        return tuple(sorted([a, b]))

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

    # Equity-FX (6 possible if 3 equities x 2 FX)
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
