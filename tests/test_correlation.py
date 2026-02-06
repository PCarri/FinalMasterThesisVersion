import numpy as np

from structured_product_tool.config import make_default_params
from structured_product_tool.correlation import build_factor_corr_matrix, factor_list_for_selection, nearest_psd_corr


def test_nearest_psd_corr_properties():
    C = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    C_psd = nearest_psd_corr(C)
    assert np.allclose(C_psd, C_psd.T)
    assert np.allclose(np.diag(C_psd), 1.0)
    eigvals = np.linalg.eigvalsh(C_psd)
    assert np.min(eigvals) >= -1e-8


def test_build_factor_corr_matrix_shape():
    params = make_default_params()
    basket = ["Local", "Foreign 1", "Foreign 2"]
    factors = factor_list_for_selection(basket)
    C, _ = build_factor_corr_matrix(factors, params)
    assert C.shape == (len(factors), len(factors))
    assert np.allclose(np.diag(C), 1.0)
