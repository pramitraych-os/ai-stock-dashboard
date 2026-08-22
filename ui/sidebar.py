"""The dashboard's sidebar: stock selection and history window.

Two widgets can name a ticker -- the preset dropdown and the free-text box --
so this module owns the precedence rule between them and hands the page a
single resolved :class:`Selection` instead of three loose widget values.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ui.config import (
    DATE_RANGE_OPTIONS,
    DEFAULT_DATE_RANGE,
    DEFAULT_TICKER,
    MIN_LOOKBACK_FOR_TRAINING,
    TICKER_UNIVERSE,
    lookback_days_for,
    validate_ticker,
)


@dataclass(frozen=True)
class Selection:
    """What the user is asking the dashboard to analyze.

    Attributes:
    - ticker (str): The resolved, normalized symbol to analyze.
    - source (str): Where it came from -- ``'preset'`` or ``'custom'``.
    - range_label (str): The date-range label as shown in the sidebar.
    - lookback_days (int): Calendar days of history that label maps to.
    - error (str | None): Why a hand-typed symbol was rejected, if it was. The
      selection stays usable in that case: ``ticker`` falls back to the
      dropdown value.
    """

    ticker: str
    source: str
    range_label: str
    lookback_days: int
    error: str | None = None

    @property
    def is_custom(self) -> bool:
        """Whether the symbol came from the text box rather than the dropdown."""
        return self.source == "custom"

    @property
    def may_be_too_short_to_train(self) -> bool:
        """Whether the window is too short to train a first model for a ticker."""
        return self.lookback_days < MIN_LOOKBACK_FOR_TRAINING


def _default_ticker_index() -> int:
    """Finds the position of :data:`~ui.config.DEFAULT_TICKER` in the universe.

    Returns:
    - int: Index of the default symbol, or 0 if it is not in the universe.
    """
    for index, entry in enumerate(TICKER_UNIVERSE):
        if entry.symbol == DEFAULT_TICKER:
            return index
    return 0


def _render_header() -> None:
    """Writes the sidebar's title and one-line description of the dashboard."""
    st.sidebar.title("AI Stock Analysis")
    st.sidebar.caption(
        "Blends a technical ML score with LLM news sentiment into a single "
        "buy/sell signal for US and Indian equities."
    )
    st.sidebar.divider()


def render_sidebar() -> Selection:
    """Renders the sidebar controls and resolves them into one selection.

    A non-blank, valid entry in the custom-ticker box wins over the dropdown;
    a blank box defers to it. An invalid entry is reported inline and the
    dropdown value is used, so the page always has something to render.

    Returns:
    - Selection: The resolved ticker, its origin, and the lookback window.
    """
    _render_header()

    st.sidebar.subheader("Stock")

    preset = st.sidebar.selectbox(
        "Popular tickers",
        options=TICKER_UNIVERSE,
        index=_default_ticker_index(),
        format_func=lambda entry: entry.label,
        help="Large-cap US and NSE-listed names with enough history and news flow.",
    )

    typed = st.sidebar.text_input(
        "Or enter a custom ticker",
        value="",
        placeholder="e.g. NFLX or WIPRO.NS",
        help="Any yfinance symbol. Indian listings need the '.NS' suffix. "
        "Overrides the dropdown while it is filled in.",
    )

    custom, error = validate_ticker(typed)

    if error:
        st.sidebar.error(error)
        ticker, source = preset.symbol, "preset"
    elif custom:
        ticker, source = custom, "custom"
    else:
        ticker, source = preset.symbol, "preset"

    st.sidebar.divider()
    st.sidebar.subheader("History window")

    range_label = st.sidebar.selectbox(
        "Date range",
        options=[label for label, _days in DATE_RANGE_OPTIONS],
        index=[label for label, _days in DATE_RANGE_OPTIONS].index(DEFAULT_DATE_RANGE),
        help="How far back to pull daily bars for the chart and the indicators.",
    )
    lookback_days = lookback_days_for(range_label)

    selection = Selection(
        ticker=ticker,
        source=source,
        range_label=range_label,
        lookback_days=lookback_days,
        error=error,
    )

    if selection.may_be_too_short_to_train:
        st.sidebar.warning(
            f"{range_label} leaves few bars after the indicator warm-up. A ticker "
            "without a saved model may fall back to a neutral ML score."
        )

    st.sidebar.divider()
    st.sidebar.caption(f"Analyzing **{selection.ticker}** over {lookback_days} days.")

    return selection
