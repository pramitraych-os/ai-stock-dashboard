"""Streamlit presentation layer for the AI Stock Dashboard.

The package keeps the UI split from the analysis pipeline: ``run_analysis``,
``market_scan`` and the ``src`` modules know nothing about Streamlit, and
nothing in here knows how a score is computed. The modules divide the screen
between them:

- :mod:`ui.config` -- the selectable universe: tickers, date ranges, how many
  rows the overview table shows, and the ticker-validation rules shared by the
  dropdown and the free-text box.
- :mod:`ui.sidebar` -- renders the controls and resolves them into a single
  :class:`~ui.sidebar.Selection` the page can act on.
- :mod:`ui.layout` -- lays out the main-area sections and hands back the
  containers the renderers write their content into.
- :mod:`ui.market_table` -- renders the ranked overview table that opens the
  page, and translates a click on one of its rows into a selection.
- :mod:`ui.signal_card` -- renders the Daily Signal card for the selected
  stock, colour-coded by the band ``signal_blender`` puts the score in.
- :mod:`ui.price_chart` -- renders that stock's candlestick chart.
- :mod:`ui.breakdown` -- renders the two component scores below the chart.
"""

from __future__ import annotations

from ui.config import (
    DATE_RANGE_OPTIONS,
    DEFAULT_DATE_RANGE,
    DEFAULT_TICKER,
    MARKET_TABLE_SIZE,
    TICKER_UNIVERSE,
    lookback_days_for,
    ticker_for,
    validate_ticker,
)
from ui.layout import DashboardSlots, render_layout
from ui.market_table import consume_row_selection, render_market_table
from ui.sidebar import Selection, focus_ticker, render_sidebar
from ui.signal_card import render_signal_card

__all__ = [
    "DATE_RANGE_OPTIONS",
    "DEFAULT_DATE_RANGE",
    "DEFAULT_TICKER",
    "MARKET_TABLE_SIZE",
    "TICKER_UNIVERSE",
    "DashboardSlots",
    "Selection",
    "consume_row_selection",
    "focus_ticker",
    "lookback_days_for",
    "render_layout",
    "render_market_table",
    "render_sidebar",
    "render_signal_card",
    "ticker_for",
    "validate_ticker",
]
