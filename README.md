# Equity Structured Product Tool (Thesis Project)

This repository contains a Streamlit application for exploring equity-linked structured products with multi-asset baskets, FX exposure, and correlation-aware simulations. It is a cleaned and modularized version of my master thesis codebase, organized to be easy to read and evaluate.

**What this demonstrates**
- Applied quantitative finance: basket construction, FX handling, and option pricing.
- Monte Carlo simulation with a unified factor correlation matrix.
- Structured product design: capital protection plus option overlays.
- A professional Python project layout with tests and documentation.

**Quickstart**
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the Streamlit app.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run StructuredProductTool.py
```

**Development and Tests**
```powershell
pip install -e .[dev]
pytest
```

**Project Structure**
- `StructuredProductTool.py` Streamlit Cloud entrypoint (thin wrapper).
- `src/structured_product_tool/app.py` Streamlit UI and main flow.
- `src/structured_product_tool/config.py` constants and default parameters.
- `src/structured_product_tool/correlation.py` correlation matrix UI and PSD handling.
- `src/structured_product_tool/simulation.py` correlated equity and FX simulation.
- `src/structured_product_tool/pricing.py` Black-style forward pricing and moment matching.
- `src/structured_product_tool/products.py` structured product definitions and payoffs.
- `tests/` lightweight unit tests.
- `docs/` thesis PDF and figures.

**Methodology Summary**
- Equities follow geometric Brownian motion with user-adjustable drift and volatility.
- FX rates follow drift based on interest rate differentials.
- A unified factor correlation matrix links equity and FX shocks.
- Basket option pricing uses moment matching (M1 and M2) and Black-style pricing.
- Structured products are built from a ZCB plus option strategies on the basket.

**Assumptions and Limitations**
- This is a research and learning tool, not a production pricing system.
- No transaction costs, liquidity constraints, or model calibration are included.
- Results are sensitive to parameter choices and correlation inputs.

**Thesis and Figures**
- Place the thesis PDF at `docs/thesis.pdf`.
- Add any screenshots or figures under `docs/figures/`.

**License**
MIT License. See `LICENSE`.
