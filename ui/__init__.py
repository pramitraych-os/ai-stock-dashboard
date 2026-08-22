"""Streamlit presentation layer for the AI Stock Dashboard.

The package keeps the UI split from the analysis pipeline: ``run_analysis`` and
the ``src`` modules know nothing about Streamlit, and nothing in here knows how
a score is computed. The three modules divide the screen between them:

- :mod:`ui.config` -- the selectable universe: tickers, date ranges, and the
  ticker-validation rules shared by the dropdown and the free-text box.
- :mod:`ui.sidebar` -- renders the controls and resolves them into a single
  :class:`~ui.sidebar.Selection` the page can act on.
- :mod:`ui.layout` -- lays out the three main-area sections and hands back the
  containers later phases render their content into.
"""

from __future__ import annotations

from ui.config import (
    DATE_RANGE_OPTIONS,
    DEFAULT_DATE_RANGE,
    DEFAULT_TICKER,
    TICKER_UNIVERSE,
    lookback_days_for,
    validate_ticker,
)
from ui.layout import DashboardSlots, render_layout
from ui.sidebar import Selection, render_sidebar

__all__ = [
    "DATE_RANGE_OPTIONS",
    "DEFAULT_DATE_RANGE",
    "DEFAULT_TICKER",
    "TICKER_UNIVERSE",
    "DashboardSlots",
    "Selection",
    "lookback_days_for",
    "render_layout",
    "render_sidebar",
    "validate_ticker",
]
