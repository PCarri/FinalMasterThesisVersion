"""Streamlit Cloud entrypoint wrapper.

Keeps the original entrypoint while importing the refactored package code.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structured_product_tool.app import main


if __name__ == "__main__":
    main()
