"""Streamlit entry point for the AI Stock Dashboard.

Run with::

    streamlit run app.py

The file stays deliberately thin: it configures the page, renders the sidebar,
lays out the main area, scores the preset universe for the overview table, and
hands each payload to the widget that draws it. The pipeline stages live in
``run_analysis`` and the ``src`` modules, the multi-ticker scan in
``market_scan``, and the widgets in the ``ui`` package. What is actually
implemented here is the *caching* and *concurrency* between the two.

Streamlit re-runs this module top to bottom on every widget interaction, and
scoring one ticker is expensive in two different ways -- the ML leg may train a
Random Forest, the sentiment leg makes a paid LLM call over the network -- so
the whole page hangs on one cache:

* :func:`load_scan_row` scores a single ticker and is cached per *symbol*. That
  is the granularity that matters. The overview table and the detail sections
  below it both go through it, so the stock a reader clicks in the table is
  already scored by the time the card underneath renders -- no second fetch, no
  second LLM call, and no way for the two halves of the page to disagree about
  a number.
* The cache key is the symbol and the pipeline's own settings, deliberately
  *not* the chart's date range. Changing the range therefore re-slices bars
  already in memory rather than re-running anything; news flow has nothing to
  do with how far back the candles go.
* :func:`run_market_scan` fans the scan out across a thread pool, because a row
  is almost pure waiting -- on yfinance, then on the LLM. Twenty symbols scored
  one at a time would be minutes of idle sockets.
"""

from __future__ import annotations

import datetime
import os
import sys

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Running via ``streamlit run app.py`` does not put the project root on the
# path the way ``python app.py`` would, so ``import ui`` needs the same fix-up
# ``run_analysis`` applies for ``src``.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import market_scan  # noqa: E402
import run_analysis  # noqa: E402  (also puts ``src`` on the path)

from ui import market_table, sidebar  # noqa: E402
from ui.breakdown import render_breakdown  # noqa: E402
from ui.config import DATE_RANGE_OPTIONS, MARKET_TABLE_SIZE, TICKER_UNIVERSE  # noqa: E402
from ui.layout import render_layout  # noqa: E402
from ui.price_chart import render_stock_chart  # noqa: E402
from ui.signal_card import render_signal_card  # noqa: E402

# Extra calendar days fetched beyond the window the user asked for, so the
# 50-day moving average is already warmed up on the first bar displayed. 120
# calendar days covers 50 trading days with room for holidays. Without this a
# one-month selection would have no SMA_50 at all -- the warm-up would consume
# every row the chart was meant to draw.
INDICATOR_WARMUP_DAYS = 120

# The ML leg needs a training set, not just a display window, so the fetch never
# goes shorter than the pipeline's own default regardless of the chart's range.
# A one-month selection would otherwise leave ``train_model`` below its 50-row
# minimum and degrade every first-time ticker to a neutral score.
ANALYSIS_LOOKBACK_DAYS = run_analysis.DEFAULT_LOOKBACK_DAYS

# One fetch window for every selection, sized to the longest range the sidebar
# offers plus the warm-up, and never below what the model needs to train on.
#
# Deliberately not derived from the *current* selection: a per-range window
# would give each date range its own cache key, so switching from 1Y to 6M --
# which needs no data the 1Y fetch did not already have -- would re-download the
# ticker. A constant means one download per ticker and every range change served
# from memory, at the cost of pulling the widest window even for a short view.
FETCH_LOOKBACK_DAYS = max(
    max(days for _label, days in DATE_RANGE_OPTIONS) + INDICATOR_WARMUP_DAYS,
    ANALYSIS_LOOKBACK_DAYS,
)

# Headlines scored per pass and the ML/sentiment split, both taken from
# ``run_analysis`` so the dashboard and the CLI cannot drift apart.
NEWS_LIMIT = run_analysis.DEFAULT_NEWS_LIMIT
ML_WEIGHT = run_analysis.DEFAULT_ML_WEIGHT

# Every symbol the overview table ranks. Frozen into a tuple because it is a
# cache key argument, and a list is unhashable.
UNIVERSE_SYMBOLS: tuple[str, ...] = tuple(entry.symbol for entry in TICKER_UNIVERSE)

# A scored row is cached for half an hour, pinned to the shortest-lived thing in
# it. Daily bars only change after the close and the ML leg is a pure function
# of them, but headlines arrive through the session -- and the sentiment call is
# the one stage that costs money, which is why the window is a half hour rather
# than minutes.
ROW_CACHE_TTL_SECONDS = 1800

SPINNER_MESSAGE = "Analyzing market data & sentiment..."
SCAN_MESSAGE = "Scoring {done} of {total} stocks - price history, model, then news..."


def configure_page() -> None:
    """Sets the page-level Streamlit options.

    Must run before any other Streamlit call, so this is the first thing
    :func:`main` does. Wide layout gives the overview table room for its eleven
    columns and the price chart the horizontal room a multi-month candlestick
    series needs.
    """
    st.set_page_config(
        page_title="AI Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data(ttl=ROW_CACHE_TTL_SECONDS, show_spinner=False)
def load_scan_row(
    ticker: str, fetch_days: int, news_limit: int, ml_weight: float
) -> dict:
    """Scores one ticker end to end: bars, indicators, model, news, blend.

    The single expensive call in the app, and the only one. Both the overview
    table and the detail sections read a ticker through here, so a symbol is
    fetched once and its headlines scored once per cache window no matter how
    many parts of the page want them -- and the table's Signal column is
    literally the same value the card below renders.

    Never raises: ``market_scan.scan_ticker`` reports a failed fetch or a
    failed leg as a row with no verdict and the reason in ``error``.

    Parameters:
    - ticker (str): The symbol to score.
    - fetch_days (int): Fetch window; callers pass :data:`FETCH_LOOKBACK_DAYS`.
    - news_limit (int): Maximum headlines to score; callers pass
      :data:`NEWS_LIMIT`.
    - ml_weight (float): Share of the blend given to the ML leg; callers pass
      :data:`ML_WEIGHT`.

    Returns:
    - dict: A scan row, documented on ``market_scan.scan_ticker``.
    """
    return market_scan.scan_ticker(
        ticker,
        lookback_days=fetch_days,
        news_limit=news_limit,
        ml_weight=ml_weight,
    )


def run_market_scan(symbols: tuple[str, ...]) -> list[dict]:
    """Scores every symbol in the universe, in parallel, behind a progress bar.

    Not cached itself -- :func:`load_scan_row` is, per symbol, which is the
    better granularity: one rate-limited ticker does not invalidate the other
    nineteen, and a symbol scored by a previous scan or by the detail view is
    already paid for.

    The workers call a cached function, so each one is handed this run's
    Streamlit script context. Without it the cache lookups happen off-context
    and Streamlit logs a warning per call.

    Parameters:
    - symbols (tuple[str, ...]): The symbols to score, normally
      :data:`UNIVERSE_SYMBOLS`.

    Returns:
    - list[dict]: One scan row per symbol, in input order.
    """
    context = get_script_run_ctx()
    progress_slot = st.empty()
    progress = progress_slot.progress(0.0, text=SCAN_MESSAGE.format(done=0, total=len(symbols)))

    def score(symbol: str) -> dict:
        return load_scan_row(symbol, FETCH_LOOKBACK_DAYS, NEWS_LIMIT, ML_WEIGHT)

    def advance(done: int, total: int, _row: dict) -> None:
        progress.progress(done / total, text=SCAN_MESSAGE.format(done=done, total=total))

    try:
        return market_scan.scan_tickers(
            symbols,
            scorer=score,
            on_done=advance,
            worker_initializer=(
                (lambda: add_script_run_ctx(ctx=context)) if context is not None else None
            ),
        )
    finally:
        # On a warm cache the bar goes from 0 to 100 in a few milliseconds, so
        # it is cleared rather than left on the page as a finished 100%.
        progress_slot.empty()


def window_bars(bars: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Trims a scored frame back to the window the sidebar asked for.

    Pure, and separate from the cached scan so that changing the date range
    re-slices bars already in memory instead of re-scoring the ticker.

    Parameters:
    - bars (pd.DataFrame): A scan row's full fetched frame.
    - lookback_days (int): Calendar days of history to keep.

    Returns:
    - pd.DataFrame: The trimmed bars, oldest first.
    """
    if bars.empty or "Date" not in bars.columns:
        return bars

    cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=lookback_days))
    windowed = bars[bars["Date"] >= cutoff]

    # A stale cache can sit entirely before the cutoff. Showing the bars that do
    # exist beats showing an empty chart, so the trim is skipped in that case.
    if windowed.empty:
        return bars

    return windowed.reset_index(drop=True)


def _degraded_legs(row: dict) -> str | None:
    """Names the scoring legs that fell back to a neutral score, if any.

    Parameters:
    - row (dict): A scan row from :func:`load_scan_row`.

    Returns:
    - str | None: ``'ML'``, ``'sentiment'`` or ``'ML and sentiment'``, or None
      when both legs completed. A row with no verdict returns None too: the card
      already shows "No Signal" there, and there is no blend to qualify.
    """
    if row["blended_score"] is None:
        return None

    legs = [
        name
        for name, leg in (("ML", row["ml"]), ("sentiment", row["sentiment"]))
        if leg["status"] != "ok"
    ]
    return " and ".join(legs) if legs else None


def main() -> None:
    """Renders one pass of the dashboard.

    Streamlit re-runs this top to bottom on every widget interaction, so the
    body has to stay cheap. Everything here that can touch the network or train
    a model sits behind :func:`load_scan_row`, so the cost is paid once per
    symbol per cache window.
    """
    configure_page()

    # A click on a table row is translated into the sidebar's own widget state,
    # which has to happen before those widgets are instantiated. So it comes
    # first, reading the click that the *previous* run left behind.
    clicked = market_table.consume_row_selection()
    if clicked:
        sidebar.focus_ticker(clicked)

    selection = sidebar.render_sidebar()
    slots = render_layout(selection)

    with slots.market_table:
        scanned = run_market_scan(UNIVERSE_SYMBOLS)
        market_table.render_market_table(
            scanned, top_n=MARKET_TABLE_SIZE, focused=selection.ticker
        )

    with slots.signal_summary:
        # A universe symbol is already scored by the scan above, so this is a
        # cache hit and the spinner never appears. A hand-typed one is not, and
        # this is where that wait belongs -- where the verdict is about to be.
        with st.spinner(SPINNER_MESSAGE):
            row = load_scan_row(
                selection.ticker, FETCH_LOOKBACK_DAYS, NEWS_LIMIT, ML_WEIGHT
            )

        sentiment = row["sentiment"]
        render_signal_card(
            row["blended_score"],
            ml_probability=row["ml"]["probability"],
            # A leg that fell back returns 0.0, which the card would render as a
            # genuine neutral reading. Passing None makes it say 'n/a' instead,
            # matching what the breakdown column below reports for the same leg.
            sentiment_score=(
                sentiment["sentiment_score"] if sentiment["status"] == "ok" else None
            ),
        )

        degraded = _degraded_legs(row)
        if degraded:
            # The blend still ran -- ``blend_scores`` has no notion of a missing
            # leg -- so the headline is a real number computed from a neutral
            # stand-in. Saying which leg is standing in keeps the verdict from
            # over-claiming; the reason why is in the breakdown below.
            st.caption(
                f"Blended with a neutral stand-in for the {degraded} leg. See the "
                f"breakdown below."
            )

        if row["warning"]:
            st.warning(row["warning"])

    with slots.price_chart:
        visible = window_bars(row["bars"], selection.lookback_days)

        if visible.empty:
            st.info("No price history to chart for this selection.")
        else:
            render_stock_chart(visible, selection.ticker)

    # The table describes the bar the model scored, so it is given the full
    # fetched frame rather than the chart's visible slice.
    render_breakdown(
        slots.breakdown_technical,
        slots.breakdown_sentiment,
        row["bars"],
        row["ml"],
        row["sentiment"],
        selection.ticker,
    )


if __name__ == "__main__":
    main()
