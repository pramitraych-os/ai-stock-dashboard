"""Streamlit entry point for the AI Stock Dashboard.

Run with::

    streamlit run app.py

The file stays deliberately thin: it configures the page, renders the sidebar,
lays out the main area, runs the analysis pipeline for the selected ticker, and
hands each payload to the widget that draws it. The pipeline stages live in
``run_analysis`` and the ``src`` modules; the widgets live in the ``ui``
package. What is actually implemented here is the *caching* between the two.

Streamlit re-runs this module top to bottom on every widget interaction, and
the pipeline's two scoring stages are expensive in different ways -- the ML leg
may train a Random Forest, the sentiment leg makes a paid LLM call over the
network. So the two legs are cached separately rather than behind one
``analyze_ticker`` call:

* Price history is keyed on the ticker and a fixed window, and shared by the
  chart and the ML leg, so a ticker is fetched once no matter how many things
  read it or which date range is on screen.
* Sentiment is keyed on the ticker alone. Changing the chart's date range
  therefore cannot trigger a second LLM call -- news flow has nothing to do with
  how far back the candles go.

:func:`analyze_selection` then composes the cached legs with
``signal_blender``, which is cheap enough to redo on every rerun.
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

import signal_blender  # noqa: E402  (from ``src``, via run_analysis's fix-up)

from ui.breakdown import render_breakdown  # noqa: E402
from ui.config import DATE_RANGE_OPTIONS  # noqa: E402
from ui.layout import render_layout  # noqa: E402
from ui.price_chart import render_stock_chart  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
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

# Price history is cached for an hour: daily bars only change after the close,
# and re-fetching on every widget interaction would make the sidebar unusable.
PRICE_CACHE_TTL_SECONDS = 3600

# The ML leg is cached as long as the bars it reads, since it is a pure function
# of them plus the saved model.
ML_CACHE_TTL_SECONDS = 3600

# Sentiment gets a shorter TTL than prices: headlines arrive through the session
# while the daily bar does not move. It is also the one stage that costs money
# per call, which is why the window is a half hour rather than minutes.
SENTIMENT_CACHE_TTL_SECONDS = 1800

SPINNER_MESSAGE = "Analyzing market data & sentiment..."


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
def load_price_history(ticker: str, fetch_days: int) -> tuple[pd.DataFrame, str | None]:
    """Fetches ``fetch_days`` of bars and appends the technical indicators.

    Returns the whole fetched window rather than the chart's slice of it,
    because the two consumers want different amounts of the same data: the ML
    leg scores the last row but trains on all of them, while the chart shows
    only the window the sidebar asked for. :func:`window_bars` does that trim,
    after the indicators are computed, so the moving averages on the earliest
    displayed bar are real numbers rather than the leading NaNs a
    display-length fetch would leave.

    Never raises. A failed fetch falls back to the ticker's cached CSV when one
    exists, and returns an empty frame with a reason when it does not, so a
    network problem costs the page its chart rather than its render.

    Parameters:
    - ticker (str): The symbol to load.
    - fetch_days (int): Calendar days of history to request; callers pass
      :data:`FETCH_LOOKBACK_DAYS`.

    Returns:
    - tuple[pd.DataFrame, str | None]: The bars -- ``Date`` plus OHLCV plus
      indicator columns, oldest first -- and a warning to show the user, or None
      when the load was clean.
    """
    warning: str | None = None

    try:
        price_df = run_analysis.fetch_price_history(ticker, lookback_days=fetch_days)
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
        # volume are still worth drawing, and the breakdown table degrades to
        # 'n/a' rows; the overlays simply will not appear.
        featured = price_df

    return featured.reset_index(drop=True), warning


def window_bars(bars: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Trims a fetched frame back to the window the sidebar asked for.

    Pure, and separate from the cached load so that changing the date range
    re-slices bars already in memory instead of re-fetching them.

    Parameters:
    - bars (pd.DataFrame): The full fetched frame from
      :func:`load_price_history`.
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


@st.cache_data(ttl=ML_CACHE_TTL_SECONDS, show_spinner=False)
def load_ml_score(ticker: str, fetch_days: int) -> dict:
    """Scores the latest bar with the ticker's Random Forest.

    Reads its bars from :func:`load_price_history` rather than taking a frame
    argument, so the cache key stays small and this call cannot be handed a
    different frame than the chart is drawing.

    Never raises: ``run_ml_inference`` already reports its own failures as a
    neutral score plus a ``status``, and an empty price frame is reported the
    same way.

    Parameters:
    - ticker (str): The symbol to score.
    - fetch_days (int): Fetch window; callers pass :data:`FETCH_LOOKBACK_DAYS`.

    Returns:
    - dict: ``run_analysis.run_ml_inference``'s payload -- ``ml_score``,
      ``probability``, ``status``, ``model_source`` and ``error``. ``status`` is
      ``'no_data'`` when there were no bars to score.
    """
    bars, _warning = load_price_history(ticker, fetch_days)

    if bars.empty:
        return {
            "ml_score": 0.0,
            "probability": None,
            "status": "no_data",
            "model_source": None,
            "error": f"No price history available for {ticker}.",
        }

    return run_analysis.run_ml_inference(ticker, bars)


@st.cache_data(ttl=SENTIMENT_CACHE_TTL_SECONDS, show_spinner=False)
def load_sentiment(ticker: str, news_limit: int) -> dict:
    """Scores recent news sentiment for a ticker.

    Keyed on the ticker and article count only -- deliberately not on the chart
    window -- so adjusting the date range never re-runs a paid LLM call.

    Parameters:
    - ticker (str): The symbol to analyze.
    - news_limit (int): Maximum number of headlines to score.

    Returns:
    - dict: ``sentiment_engine.get_llm_sentiment_score``'s payload --
      ``sentiment_score``, ``summary``, ``status``, ``article_count``,
      ``provider``, ``model``, ``error`` and ``headlines``.
    """
    return run_analysis.run_sentiment_analysis(ticker, news_limit=news_limit)


def analyze_selection(ticker: str, fetch_days: int) -> dict:
    """Runs both scoring legs for a ticker and blends them into one signal.

    The composition step of the pipeline: the fetch, the indicators, the model
    and the LLM are all done by ``run_analysis``, and this only puts their
    results together the way ``analyze_ticker`` does for the CLI. Not cached
    itself -- the two legs it reads are, and blending is arithmetic.

    Parameters:
    - ticker (str): The symbol being analyzed.
    - fetch_days (int): Fetch window; callers pass :data:`FETCH_LOOKBACK_DAYS`.

    Returns:
    - dict: With keys ``'ml'`` and ``'sentiment'`` (the two leg payloads),
      ``'blended_score'`` (float, or None when there were no bars to analyze at
      all) and ``'ml_weight'``.
    """
    ml_result = load_ml_score(ticker, fetch_days)
    sentiment_result = load_sentiment(ticker, NEWS_LIMIT)

    # With no bars, the ML leg's 0.0 is a placeholder rather than a reading, and
    # blending it with sentiment would dress a one-legged number up as a verdict.
    # The card's "No Signal" state is the honest output.
    if ml_result["status"] == "no_data":
        blended_score = None
    else:
        blended_score = signal_blender.blend_scores(
            ml_result["ml_score"],
            float(sentiment_result["sentiment_score"]),
            ml_weight=ML_WEIGHT,
        )

    return {
        "ml": ml_result,
        "sentiment": sentiment_result,
        "blended_score": blended_score,
        "ml_weight": ML_WEIGHT,
    }


def _degraded_legs(analysis: dict) -> str | None:
    """Names the scoring legs that fell back to a neutral score, if any.

    Parameters:
    - analysis (dict): An :func:`analyze_selection` payload.

    Returns:
    - str | None: ``'ML'``, ``'sentiment'`` or ``'ML and sentiment'``, or None
      when both legs completed. The no-data case returns None too: the card
      already shows "No Signal" there, and there is no blend to qualify.
    """
    if analysis["blended_score"] is None:
        return None

    legs = [
        name
        for name, leg in (("ML", analysis["ml"]), ("sentiment", analysis["sentiment"]))
        if leg["status"] != "ok"
    ]
    return " and ".join(legs) if legs else None


def main() -> None:
    """Renders one pass of the dashboard.

    Streamlit re-runs this top to bottom on every widget interaction, so the
    body has to stay cheap. Everything here that can touch the network or train
    a model sits behind a cache keyed on the selection, so the cost is paid only
    when the selection actually changes.
    """
    configure_page()

    selection = render_sidebar()
    slots = render_layout(selection)

    # One spinner covers both legs and the fetch they share: from the reader's
    # side this is a single wait, and three nested spinners would just flicker.
    # It is opened inside the signal container so the wait appears where the
    # verdict is about to.
    with slots.signal_summary:
        with st.spinner(SPINNER_MESSAGE):
            bars, warning = load_price_history(selection.ticker, FETCH_LOOKBACK_DAYS)
            analysis = analyze_selection(selection.ticker, FETCH_LOOKBACK_DAYS)

        sentiment = analysis["sentiment"]
        render_signal_card(
            analysis["blended_score"],
            ml_probability=analysis["ml"]["probability"],
            # A leg that fell back returns 0.0, which the card would render as a
            # genuine neutral reading. Passing None makes it say 'n/a' instead,
            # matching what the breakdown column below reports for the same leg.
            sentiment_score=(
                sentiment["sentiment_score"] if sentiment["status"] == "ok" else None
            ),
        )

        degraded = _degraded_legs(analysis)
        if degraded:
            # The blend still ran -- ``blend_scores`` has no notion of a missing
            # leg -- so the headline is a real number computed from a neutral
            # stand-in. Saying which leg is standing in keeps the verdict from
            # over-claiming; the reason why is in the breakdown below.
            st.caption(
                f"Blended with a neutral stand-in for the {degraded} leg. See the "
                f"breakdown below."
            )

        if warning:
            st.warning(warning)

    with slots.price_chart:
        visible = window_bars(bars, selection.lookback_days)

        if visible.empty:
            st.info("No price history to chart for this selection.")
        else:
            render_stock_chart(visible, selection.ticker)

    # The table describes the bar the model scored, so it is given the full
    # fetched frame rather than the chart's visible slice.
    render_breakdown(
        slots.breakdown_technical,
        slots.breakdown_sentiment,
        bars,
        analysis["ml"],
        analysis["sentiment"],
        selection.ticker,
    )


if __name__ == "__main__":
    main()
