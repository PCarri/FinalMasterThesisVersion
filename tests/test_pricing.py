import numpy as np

from structured_product_tool.config import make_default_params
from structured_product_tool.correlation import build_factor_corr_matrix, factor_list_for_selection
from structured_product_tool.pricing import bs_from_forward, price_basket_option_moment_matching


def test_bs_from_forward_intrinsic():
    price = bs_from_forward(100.0, 90.0, 0.95, 0.0, 0.0, is_call=True)
    assert np.isclose(price, 0.95 * 10.0)


def test_price_basket_option_moment_matching_positive():
    params = make_default_params()
    basket = ["Local"]
    factors = factor_list_for_selection(basket)
    C, _ = build_factor_corr_matrix(factors, params)
    N_shares = np.array([1.0])
    pv, details = price_basket_option_moment_matching(
        N_shares,
        basket,
        params,
        factors,
        C,
        "Plain (Local only)",
        1.0,
        100.0,
        is_call=True,
    )
    assert pv >= 0.0
    assert "M1" in details and "M2" in details
