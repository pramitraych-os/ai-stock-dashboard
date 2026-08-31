"""Streamlit entry point for the AI Stock Dashboard.

Run with::

    streamlit run app.py

The file stays deliberately thin: it configures the page, renders the sidebar,
lays out the main area, and hands the resulting containers to whatever draws
the content. The pipeline behind those containers lives in ``run_analysis`` and
the ``src`` modules; the widgets live in the ``ui`` package.
"""

from __future__ import annotations

import datetime
import os
import sys

import pandas as pd
import streamlit as st

# Running via ``streamlit run app.py`` does not put the project root on the
# path the way ``python app.py`` would, so ``import ui`` needs the same fix-up
# ``run_analysis`` applies for ``src``.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import run_analysis  # noqa: E402  (also puts ``src`` on the path)

from ui.layout import render_layout, render_placeholders  # noqa: E402
from ui.price_chart import render_stock_chart  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.signal_card import render_signal_card  # noqa: E402

# Extra calendar days fetched beyond the window the user asked for, so the
# 50-day moving average is already warmed up on the first bar displayed. 120
# calendar days covers 50 trading days with room for holidays. Without this a
# one-month selection would have no SMA_50 at all -- the warm-up would consume
# every row the chart was meant to draw.
INDICATOR_WARMUP_DAYS = 120

# Price history is cached for an hour: daily bars only change after the close,
# and re-fetching on every widget interaction would make the sidebar unusable.
PRICE_CACHE_TTL_SECONDS = 3600


def configure_page() -> None:
    """Sets the page-level Streamlit options.

    Must run before any other Streamlit call, so this is the first thing
    :func:`main` does. Wide layout gives the price chart the horizontal room a
    multi-month candlestick series needs.
    """
    st.set_page_config(
        page_title="AI Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _load_cached_csv(ticker: str) -> pd.DataFrame:
    """Reads a ticker's persisted bars from ``data/``, if they are there.

    ``data_loader.save_data_locally`` writes ``<TICKER>_daily.csv`` and the repo
    ships two of them, so this is what lets the chart draw with no network --
    on a plane, behind a proxy, or when yfinance is rate-limiting.

    Parameters:
    - ticker (str): The symbol whose cached file is wanted.

    Returns:
    - pd.DataFrame: The cached bars, or an empty frame if there is no readable
      file for this ticker.
    """
    path = os.path.join(PROJECT_ROOT, "data", f"{ticker}_daily.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except (OSError, ValueError):
        # A truncated or hand-edited file is not worth a traceback; the caller
        # reports the original fetch failure instead.
        return pd.DataFrame()


@st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
def load_price_history(ticker: str, lookback_days: int) -> tuple[pd.DataFrame, str | None]:
    """Fetches, warms up and trims the bars the price chart draws.

    Fetches ``lookback_days`` plus :data:`INDICATOR_WARMUP_DAYS` of history so
    the indicators are defined across the whole visible window, computes them,
    then trims back to the window actually requested. Trimming *after* the
    indicators are computed is the point: the moving averages on the earliest
    displayed bar are then real numbers rather than the leading NaNs a
    same-length fetch would leave.

    Never raises. A failed fetch falls back to the ticker's cached CSV when one
    exists, and returns an empty frame with a reason when it does not, so a
    network problem costs the page its chart rather than its render.

    Parameters:
    - ticker (str): The symbol to load.
    - lookback_days (int): Calendar days of history the user asked to see.

    Returns:
    - tuple[pd.DataFrame, str | None]: The bars to chart -- ``Date`` plus OHLCV
      plus indicator columns, oldest first -- and a warning to show the user,
      or None when the load was clean.
    """
    warning: str | None = None

    try:
        price_df = run_analysis.fetch_price_history(
            ticker, lookback_days=lookback_days + INDICATOR_WARMUP_DAYS
        )
    except Exception as e:
        price_df = _load_cached_csv(ticker)
        if price_df.empty:
            return pd.DataFrame(), f"Could not load price history for {ticker}: {e}"
        warning = (
            f"Live fetch for {ticker} failed ({e}). Showing the bars cached in "
            f"data/, which may be out of date."
        )

    try:
        featured = run_analysis.compute_indicators(price_df)
    except run_analysis.AnalysisError:
        # Too little history for the longest indicator window. The candles and
        # volume are still worth drawing; the overlays simply will not appear.
        featured = price_df

    # Drop the warm-up padding so the chart shows the window that was asked for.
    cutoff = pd.Timestamp(
        datetime.date.today() - datetime.timedelta(days=lookback_days)
    )
    windowed = featured[featured["Date"] >= cutoff]

    # A stale cache can sit entirely before the cutoff. Showing the bars that
    # do exist beats showing an empty chart, so the trim is skipped in that case.
    if windowed.empty:
        return featured.reset_index(drop=True), warning

    return windowed.reset_index(drop=True), warning


def main() -> None:
    """Renders one pass of the dashboard.

    Streamlit re-runs this top to bottom on every widget interaction, so the
    body has to stay cheap. The one thing here that can touch the network is
    the price-history load, and :func:`load_price_history` is cached on
    ``(ticker, lookback_days)`` so it only pays that cost when the selection
    actually changes.
    """
    configure_page()

    selection = render_sidebar()
    slots = render_layout(selection)

    with slots.signal_summary:
        # The analysis pipeline is not wired into the page yet, so there is no
        # score to show and the card renders its neutral "No Signal" state.
        # Replace these arguments with the ``analyze_ticker`` payload --
        # ``blended_score``, ``diagnostics['ml_probability']`` and ``ai_score``
        # -- once that call lands here.
        render_signal_card(None)

    with slots.price_chart:
        with st.spinner(f"Loading {selection.ticker} price history..."):
            bars, warning = load_price_history(
                selection.ticker, selection.lookback_days
            )

        if warning:
            st.warning(warning)

        if bars.empty:
            st.info("No price history to chart for this selection.")
        else:
            render_stock_chart(bars, selection.ticker)

    # Placeholder copy until the breakdown renderers land; drop these calls one
    # at a time as each section is implemented.
    render_placeholders(slots, selection)


if __name__ == "__main__":
    main()
