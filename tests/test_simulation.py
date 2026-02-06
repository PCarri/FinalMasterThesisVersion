import numpy as np

from structured_product_tool.config import make_default_params
from structured_product_tool.simulation import simulate_equities_and_fx


def test_simulation_shapes():
    params = make_default_params()
    basket = ["Local", "Foreign 1"]
    time, S_paths, X_paths, factors, C, _ = simulate_equities_and_fx(
        basket_composition=basket,
        params=params,
        T=1.0,
        n_steps=10,
        n_paths=5,
        rng=np.random.default_rng(1),
    )
    assert time.shape[0] == 11
    assert S_paths["Local"].shape == (11, 5)
    assert X_paths["Foreign 1"].shape == (11, 5)
    assert C.shape == (len(factors), len(factors))
