"""The dashboard's sidebar: stock selection and history window.

Two widgets can name a ticker -- the preset dropdown and the free-text box --
so this module owns the precedence rule between them and hands the page a
single resolved :class:`Selection` instead of three loose widget values.

A third thing can *also* name a ticker: clicking a row in the market table. It
does not get a precedence rule of its own. Instead :func:`focus_ticker` writes
the click into these two widgets' own session state, so the sidebar stays the
single source of truth for what is being analyzed and the dropdown visibly
follows the click rather than silently disagreeing with it.
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
    ticker_for,
    validate_ticker,
)

# Session-state keys for the two ticker widgets. Named rather than left implicit
# because :func:`focus_ticker` writes to them from outside this module.
PRESET_KEY = "sidebar_preset_ticker"
CUSTOM_KEY = "sidebar_custom_ticker"


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


def focus_ticker(symbol: str) -> None:
    """Points the sidebar's controls at ``symbol``.

    Must be called *before* :func:`render_sidebar` in the same script run:
    Streamlit refuses to let a widget's state be reassigned once that widget has
    been instantiated, and setting it beforehand is what makes the dropdown come
    up already showing the requested symbol.

    A preset symbol goes into the dropdown and the custom box is cleared, so the
    box cannot keep overriding the dropdown with a stale entry -- without that,
    clicking a table row while something was typed in the box would appear to do
    nothing. A symbol outside the universe has no dropdown entry to select, so it
    goes into the box instead, where it wins on its own merits.

    Parameters:
    - symbol (str): The symbol to analyze next.
    """
    entry = ticker_for(symbol)

    if entry is None:
        st.session_state[CUSTOM_KEY] = str(symbol).strip().upper()
        return

    st.session_state[PRESET_KEY] = entry
    st.session_state[CUSTOM_KEY] = ""


def _render_header() -> None:
    """Writes the sidebar's title and one-line description of the dashboard."""
    st.sidebar.title("AI Stock Analysis")
    # Describes the controls, not the app -- the page's own caption covers what
    # the dashboard does, and saying it twice on one screen just costs space.
    st.sidebar.caption(
        "Choose the stock the detail sections analyze, and how much history "
        "they read. The market table above them always covers the full universe."
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
        key=PRESET_KEY,
    )

    typed = st.sidebar.text_input(
        "Or enter a custom ticker",
        placeholder="e.g. PLTR or WIPRO.NS",
        help="Any yfinance symbol. Indian listings need the '.NS' suffix. "
        "Overrides the dropdown while it is filled in.",
        key=CUSTOM_KEY,
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
