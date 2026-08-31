"""The two-column Analysis Breakdown that sits below the price chart.

The signal card states a verdict and the chart shows the price action; this
section shows the *two inputs* that produced the verdict, side by side so
neither reads as the headline. The left column unpacks the quantitative leg
(the indicators the Random Forest saw, plus the score it returned), the right
column unpacks the narrative leg (the headlines the LLM read, plus the score
and rationale it returned).

Both renderers take the payloads their ``src`` stages already document --
``run_analysis.run_ml_inference`` and ``sentiment_engine.get_llm_sentiment_score``
-- rather than a UI-specific view model, so this module tracks the pipeline's
own contract and nothing has to be kept in sync by hand.

Two things worth knowing before editing.

**A neutral score is not the same as a neutral reading.** Both legs degrade to
``0.0`` on failure -- no saved model, no API key, a rate limit -- and both report
why in ``status``. So every score shown here is accompanied by its status
whenever that status is not ``'ok'``: an unexplained ``+0.00`` would otherwise
read as "the model is undecided" when it actually means "the model never ran".

**The indicator table describes the bar the model scored**, not the last bar in
the chart's visible window. Callers pass the same frame they handed
``run_ml_inference``; if the chart is zoomed to a month, the table still
describes the latest session, because that is the row the score came from.

No colour is chosen in this module. The verdict already owns the palette in
``signal_card``, and the readings here are carried by words in a table column,
which stays legible with no colour vision and in greyscale print.
"""

from __future__ import annotations

import math
import os
import re
import sys

import pandas as pd
import streamlit as st

# The currency prefix rule is already settled in ``price_chart``, so it is
# imported rather than restated -- the table and the chart's price axis must not
# be able to disagree about which market a symbol trades on. Running this file
# directly (see ``__main__`` below) puts ``ui/`` on the path rather than the
# project root, so ``import ui`` needs the same fix-up ``signal_card`` applies.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui.price_chart import currency_symbol_for  # noqa: E402

# ---------------------------------------------------------------------------
# Indicator thresholds
# ---------------------------------------------------------------------------

# Matches the window ``feature_engineering.add_rsi`` computes with, so the
# label cannot claim a period the column was not built from.
RSI_PERIOD = 14

# Wilder's conventional bounds. They are a statement about how stretched the
# recent move is, not a trade instruction, which is why the "Read" column below
# says "stretched" rather than "sell".
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# A moving-average pair closer than this is converging, not crossing: on a
# large-cap the gap sits inside a single session's noise, so calling a direction
# from it would over-read the data. Percent of the slower average.
MA_FLAT_BAND_PCT = 0.25

# The moving averages compared for the trend row. Same pair the chart overlays,
# so the table's verdict and the two lines a reader can see agree by
# construction.
MA_FAST_COLUMN, MA_FAST_PERIOD = "SMA_20", 20
MA_SLOW_COLUMN, MA_SLOW_PERIOD = "SMA_50", 50


# ---------------------------------------------------------------------------
# Status copy
# ---------------------------------------------------------------------------

# Why a leg fell back to a neutral score, in the user's terms. Keyed by the
# ``status`` values ``run_analysis.run_ml_inference`` documents. ``'ok'`` is
# absent on purpose: a successful stage needs no explanation.
ML_STATUS_NOTES = {
    "no_data": "No price history was available, so the model could not be scored.",
    "no_model": (
        "No saved model for this ticker yet, and training was not allowed on "
        "this run."
    ),
    "training_failed": (
        "Not enough usable history to train a model for this ticker. Widen the "
        "history window in the sidebar and try again."
    ),
    "inference_failed": (
        "The saved model could not score the latest bar -- most likely a stale "
        "artefact whose feature set no longer matches. Delete it from models/ "
        "to retrain."
    ),
}

# Same idea for the sentiment leg, keyed by the statuses
# ``sentiment_engine.get_llm_sentiment_score`` documents.
SENTIMENT_STATUS_NOTES = {
    "no_news": "No recent headlines found for this ticker, so the score is neutral.",
    "no_api_key": (
        "No LLM credentials found. Set GEMINI_API_KEY or OPENAI_API_KEY in your "
        "environment or a .env file to enable sentiment scoring."
    ),
    "auth_error": "The LLM provider rejected the API key.",
    "rate_limited": "The LLM provider rate-limited the request. Try again shortly.",
    "api_error": "The LLM provider returned an error.",
    "network_error": "Could not reach the LLM provider.",
    "parse_error": "The LLM's reply could not be parsed as a score.",
    "config_error": "The configured LLM provider is not usable.",
    "invalid_ticker": "The ticker symbol was rejected by the sentiment stage.",
}

# Plain-language reading of a sentiment score, ordered most bullish first with
# an inclusive lower bound. Deliberately *not*
# ``signal_blender.get_dashboard_signal``: that maps a blended score onto
# buy/sell recommendations, and one leg on its own is not a recommendation.
SENTIMENT_BANDS: tuple[tuple[float, str], ...] = (
    (0.50, "Strongly bullish"),
    (0.15, "Bullish"),
    (-0.15, "Neutral"),
    (-0.50, "Bearish"),
    (-1.00, "Strongly bearish"),
)

# How the model source reads in the caption under the ML metrics.
MODEL_SOURCE_NOTES = {
    "loaded": "saved model",
    "trained": "trained on this run",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# Characters Streamlit's markdown would interpret rather than print. ``$`` is in
# the set because a pair of them turns the text between into LaTeX, and a
# headline like "Apple nears $4T as $AAPL climbs" is exactly that shape.
#
# Deliberately narrower than "all ASCII punctuation": brackets, dots, dashes and
# parentheses are only markup in combinations that need a character already in
# this set, so escaping them too would litter ordinary prose with backslashes for
# no gain -- "no\_api\_key" is worth escaping, "Ternus\." is not.
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]$~|<>])")


def _escape_markdown(text: str) -> str:
    """Escapes text that will be interpolated into a markdown string.

    Headlines are third-party strings, so they routinely contain the characters
    markdown treats as syntax. Escaping keeps the list rendering as the
    publisher wrote it.

    Parameters:
    - text (str): Raw text to render.

    Returns:
    - str: The text with markdown syntax characters backslash-escaped.
    """
    return _MARKDOWN_SPECIALS.sub(r"\\\1", str(text))


def _format_score(value: float | None) -> str:
    """Formats a ``[-1, +1]`` score for display, e.g. ``'+0.24'``.

    Parameters:
    - value (float | None): The score, or None/non-finite if unavailable.

    Returns:
    - str: The signed two-decimal score, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.2f}"


def _format_probability(value: float | None) -> str:
    """Formats a ``[0, 1]`` probability as a percentage, e.g. ``'62.0%'``.

    Parameters:
    - value (float | None): The probability, or None if the model could not be
      scored.

    Returns:
    - str: The percentage to one decimal place, or ``'n/a'``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{min(max(value, 0.0), 1.0) * 100:.1f}%"


def _sentiment_reading(value: float | None) -> str:
    """Maps a sentiment score onto its plain-language band.

    Parameters:
    - value (float | None): The score in ``[-1, +1]``.

    Returns:
    - str: The band name, or ``'Unavailable'`` when there is no score.
    """
    if value is None or not math.isfinite(value):
        return "Unavailable"

    for lower_bound, reading in SENTIMENT_BANDS:
        if value >= lower_bound:
            return reading
    # Only reachable for a score below -1, which the engines clamp away.
    return SENTIMENT_BANDS[-1][1]


# ---------------------------------------------------------------------------
# Reading the scored bar
# ---------------------------------------------------------------------------


def _latest_value(bars: pd.DataFrame, column: str) -> float | None:
    """Reads the most recent usable value of one indicator column.

    Takes the last *non-null* value rather than the last row's: a frame built
    with ``add_technical_indicators(dropna=False)`` keeps its warm-up rows, and
    a short window can leave the longest indicator undefined on the final bar.
    Reporting the most recent value that exists beats reporting ``n/a`` for an
    indicator the frame does carry.

    Parameters:
    - bars (pd.DataFrame): The indicator frame.
    - column (str): The column to read.

    Returns:
    - float | None: The value, or None when the column is absent or empty.
    """
    if bars is None or bars.empty or column not in bars.columns:
        return None

    values = pd.to_numeric(bars[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _as_of_label(bars: pd.DataFrame) -> str | None:
    """Formats the date of the last bar in a frame, e.g. ``'29 Aug 2026'``.

    Parameters:
    - bars (pd.DataFrame): The indicator frame, with dates in a ``Date`` column
      or a ``DatetimeIndex``.

    Returns:
    - str | None: The formatted date, or None when the frame carries no dates.
    """
    if bars is None or bars.empty:
        return None

    if "Date" in bars.columns:
        stamp = pd.to_datetime(bars["Date"], errors="coerce").dropna()
        if stamp.empty:
            return None
        return stamp.iloc[-1].strftime("%d %b %Y")

    if isinstance(bars.index, pd.DatetimeIndex) and len(bars.index):
        return bars.index[-1].strftime("%d %b %Y")

    return None


# ---------------------------------------------------------------------------
# Indicator rows
# ---------------------------------------------------------------------------

# The "no value" row body, so a missing column reads the same wherever it
# appears in the table.
_UNAVAILABLE_ROW = ("n/a", "Not in this history window")


def _rsi_row(bars: pd.DataFrame) -> dict[str, str]:
    """Builds the RSI row of the indicator table.

    Parameters:
    - bars (pd.DataFrame): The indicator frame that was scored.

    Returns:
    - dict[str, str]: One ``Indicator``/``Value``/``Read`` row.
    """
    label = f"RSI ({RSI_PERIOD})"
    rsi = _latest_value(bars, "RSI")

    if rsi is None:
        value, read = _UNAVAILABLE_ROW
    elif rsi >= RSI_OVERBOUGHT:
        value, read = f"{rsi:.1f}", "Overbought - stretched to the upside"
    elif rsi <= RSI_OVERSOLD:
        value, read = f"{rsi:.1f}", "Oversold - stretched to the downside"
    else:
        value, read = f"{rsi:.1f}", "Neutral - inside the normal band"

    return {"Indicator": label, "Value": value, "Read": read}


def _ma_trend_row(bars: pd.DataFrame) -> dict[str, str]:
    """Builds the moving-average trend row of the indicator table.

    Reports the fast average's distance from the slow one as a percentage --
    the sign is the trend direction and the magnitude is its conviction, which
    a bare pair of prices does not convey.

    Parameters:
    - bars (pd.DataFrame): The indicator frame that was scored.

    Returns:
    - dict[str, str]: One ``Indicator``/``Value``/``Read`` row.
    """
    label = f"MA trend ({MA_FAST_PERIOD} vs {MA_SLOW_PERIOD})"
    fast = _latest_value(bars, MA_FAST_COLUMN)
    slow = _latest_value(bars, MA_SLOW_COLUMN)

    if fast is None or slow is None or not slow:
        return {"Indicator": label, "Value": _UNAVAILABLE_ROW[0], "Read": _UNAVAILABLE_ROW[1]}

    gap_pct = (fast - slow) / abs(slow) * 100.0

    if abs(gap_pct) < MA_FLAT_BAND_PCT:
        read = "Converging - no clear trend"
    elif gap_pct > 0:
        read = f"Bullish - {MA_FAST_PERIOD}-day above {MA_SLOW_PERIOD}-day"
    else:
        read = f"Bearish - {MA_FAST_PERIOD}-day below {MA_SLOW_PERIOD}-day"

    return {"Indicator": label, "Value": f"{gap_pct:+.2f}%", "Read": read}


def _price_vs_ma_row(bars: pd.DataFrame, symbol: str) -> dict[str, str]:
    """Builds the close-versus-slow-average row of the indicator table.

    Parameters:
    - bars (pd.DataFrame): The indicator frame that was scored.
    - symbol (str): Currency symbol for the average's price.

    Returns:
    - dict[str, str]: One ``Indicator``/``Value``/``Read`` row.
    """
    label = f"Close vs {MA_SLOW_PERIOD}-day MA"
    close = _latest_value(bars, "Close")
    slow = _latest_value(bars, MA_SLOW_COLUMN)

    if close is None or slow is None or not slow:
        return {"Indicator": label, "Value": _UNAVAILABLE_ROW[0], "Read": _UNAVAILABLE_ROW[1]}

    gap_pct = (close - slow) / abs(slow) * 100.0
    side = "above" if gap_pct >= 0 else "below"

    return {
        "Indicator": label,
        "Value": f"{gap_pct:+.2f}%",
        "Read": f"Trading {side} the {symbol}{slow:,.2f} average",
    }


def _macd_row(bars: pd.DataFrame) -> dict[str, str]:
    """Builds the MACD histogram row of the indicator table.

    The histogram, not the MACD line: it is the line's distance from its own
    signal, which is the momentum read the row is claiming.

    Parameters:
    - bars (pd.DataFrame): The indicator frame that was scored.

    Returns:
    - dict[str, str]: One ``Indicator``/``Value``/``Read`` row.
    """
    label = "MACD histogram"
    hist = _latest_value(bars, "MACD_Hist")

    if hist is None:
        value, read = _UNAVAILABLE_ROW
    elif hist > 0:
        value, read = f"{hist:+.3f}", "Momentum building - MACD above its signal"
    elif hist < 0:
        value, read = f"{hist:+.3f}", "Momentum fading - MACD below its signal"
    else:
        value, read = f"{hist:+.3f}", "Flat - MACD on its signal"

    return {"Indicator": label, "Value": value, "Read": read}


def build_indicator_table(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Summarizes the scored bar's indicators as a three-column table.

    Pure: reads the frame and returns a new one, drawing nothing. Every row
    survives a missing column, because a short history window can leave the
    longer indicators undefined and losing the whole table to that would be a
    worse outcome than losing one row.

    Parameters:
    - bars (pd.DataFrame): The indicator frame that was scored -- the output of
      ``run_analysis.compute_indicators``. A frame with no indicator columns at
      all yields a table of ``n/a`` rows rather than an error.
    - ticker (str): The symbol, used only to pick the currency prefix.

    Returns:
    - pd.DataFrame: Columns ``Indicator``, ``Value`` and ``Read``, one row per
      indicator summarized.
    """
    symbol = currency_symbol_for(ticker)

    return pd.DataFrame(
        [
            _rsi_row(bars),
            _ma_trend_row(bars),
            _price_vs_ma_row(bars, symbol),
            _macd_row(bars),
        ]
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_technical_breakdown(
    bars: pd.DataFrame, ml_result: dict, ticker: str
) -> None:
    """Renders the technical/ML column into the current Streamlit container.

    Leads with the two numbers the ML stage produced, then the indicator table
    those numbers were computed from, so a reader can see whether the score
    agrees with the indicators it saw.

    Parameters:
    - bars (pd.DataFrame): The indicator frame handed to
      ``run_analysis.run_ml_inference`` -- the same rows the model scored.
    - ml_result (dict): That function's payload -- ``ml_score``,
      ``probability``, ``status``, ``model_source`` and ``error``.
    - ticker (str): The symbol being analyzed.
    """
    st.markdown("**ML Technical Indicators**")

    status = ml_result.get("status")
    scored = status == "ok"

    score_col, probability_col = st.columns(2)
    score_col.metric(
        "ML score",
        _format_score(ml_result.get("ml_score") if scored else None),
        help="Rise probability rescaled onto [-1, +1]: 0.00 is a coin flip.",
    )
    probability_col.metric(
        "Rise probability",
        _format_probability(ml_result.get("probability")),
        help="The model's chance that the next close is higher than this one.",
    )

    as_of = _as_of_label(bars)
    source = MODEL_SOURCE_NOTES.get(ml_result.get("model_source"))
    if scored and as_of:
        provenance = f"Random Forest ({source}) scored on the {as_of} bar."
        st.caption(provenance if source else f"Scored on the {as_of} bar.")

    # A fallback ``0.0`` looks identical to genuine indecision, so the reason is
    # always shown alongside it rather than left in the payload.
    if not scored:
        note = ML_STATUS_NOTES.get(status, "The ML leg did not complete.")
        detail = ml_result.get("error")
        st.warning(f"**ML score unavailable.** {note}")
        if detail:
            with st.expander("Error detail"):
                st.code(str(detail), language=None)

    st.dataframe(
        build_indicator_table(bars, ticker),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"RSI above {RSI_OVERBOUGHT:.0f} is conventionally overbought and below "
        f"{RSI_OVERSOLD:.0f} oversold. Readings describe the latest session, not "
        "the chart's zoom window."
    )


def render_sentiment_breakdown(sentiment_result: dict) -> None:
    """Renders the AI sentiment column into the current Streamlit container.

    Shows the score, the LLM's own rationale for it, and the headlines it was
    given. The headlines are listed because they are the evidence: the score is
    one number derived from a specific set of stories, and a reader who
    disagrees with it can only tell why by seeing them.

    The engine returns one score for the batch rather than a score per story,
    so the headlines are listed without individual numbers -- attaching a
    per-headline figure here would mean inventing one.

    Parameters:
    - sentiment_result (dict): ``sentiment_engine.get_llm_sentiment_score``'s
      payload -- ``sentiment_score``, ``summary``, ``status``,
      ``article_count``, ``provider``, ``model``, ``error`` and ``headlines``.
    """
    st.markdown("**AI Sentiment Breakdown**")

    status = sentiment_result.get("status")
    scored = status == "ok"

    raw_score = sentiment_result.get("sentiment_score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else None

    headlines = sentiment_result.get("headlines") or []

    score_col, count_col = st.columns(2)
    score_col.metric(
        "Sentiment score",
        _format_score(score if scored else None),
        help="The LLM's read of recent headlines on the same [-1, +1] scale.",
    )
    count_col.metric(
        "Headlines scored",
        f"{int(sentiment_result.get('article_count') or 0)}",
        help="Number of articles sent to the model.",
    )

    if scored:
        st.caption(f"{_sentiment_reading(score)} on recent news flow.")

        provider = sentiment_result.get("provider")
        model = sentiment_result.get("model")
        if provider:
            st.caption(f"Scored by {provider}{f' / {model}' if model else ''}.")
    else:
        note = SENTIMENT_STATUS_NOTES.get(status, "The sentiment leg did not complete.")
        st.warning(f"**Sentiment score unavailable.** {note}")
        detail = sentiment_result.get("error")
        if detail:
            with st.expander("Error detail"):
                st.code(str(detail), language=None)

    rationale = sentiment_result.get("summary") or []
    if rationale:
        st.markdown("**Key drivers**")
        st.markdown("\n".join(f"- {_escape_markdown(point)}" for point in rationale))

    st.markdown("**Latest headlines**")
    if headlines:
        st.markdown(
            "\n".join(f"- {_escape_markdown(headline)}" for headline in headlines)
        )
    else:
        st.caption("No headlines were retrieved for this ticker.")


def render_breakdown(
    technical_slot,
    sentiment_slot,
    bars: pd.DataFrame,
    ml_result: dict,
    sentiment_result: dict,
    ticker: str,
) -> None:
    """Fills both breakdown columns from one analysis pass.

    A convenience over calling the two renderers directly, so the page does not
    have to repeat the ``with`` blocks or remember which payload goes where.

    Parameters:
    - technical_slot (DeltaGenerator): The left column from
      ``ui.layout.render_layout``.
    - sentiment_slot (DeltaGenerator): The right column from the same.
    - bars (pd.DataFrame): The indicator frame the model scored.
    - ml_result (dict): ``run_analysis.run_ml_inference``'s payload.
    - sentiment_result (dict): ``get_llm_sentiment_score``'s payload.
    - ticker (str): The symbol being analyzed.
    """
    with technical_slot:
        render_technical_breakdown(bars, ml_result, ticker)

    with sentiment_slot:
        render_sentiment_breakdown(sentiment_result)


if __name__ == "__main__":
    # Visual self-test: ``streamlit run ui/breakdown.py`` draws both columns
    # against a cached ticker and a stubbed sentiment payload, which is the only
    # practical way to check the fallback states without breaking credentials.
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    import feature_engineering  # noqa: E402

    st.set_page_config(page_title="Breakdown preview", layout="wide")
    st.title("Analysis Breakdown - preview")

    _probe = st.selectbox("Cached ticker", ("AAPL", "RELIANCE.NS"))
    _bars = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", f"{_probe}_daily.csv"),
        parse_dates=["Date"],
    )
    _bars = feature_engineering.add_technical_indicators(_bars)

    _ml_ok = {
        "ml_score": 0.24,
        "probability": 0.62,
        "status": "ok",
        "model_source": "loaded",
        "error": None,
    }
    _sentiment_ok = {
        "ticker": _probe,
        "sentiment_score": 0.42,
        "summary": [
            "Quarterly revenue beat consensus on services growth.",
            "Analysts flag $4T market-cap milestone as a stretch [see note].",
        ],
        "status": "ok",
        "article_count": 3,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "error": None,
        "headlines": [
            "Stock climbs 3% after earnings beat",
            "Supply chain costs *rising* into Q4, says supplier",
            "Regulator opens review of app store terms",
        ],
    }

    _left, _right = st.columns(2, gap="large")
    render_breakdown(_left, _right, _bars, _ml_ok, _sentiment_ok, _probe)

    st.divider()
    st.subheader("Fallback states")

    _left, _right = st.columns(2, gap="large")
    render_breakdown(
        _left,
        _right,
        _bars.head(3),
        {
            "ml_score": 0.0,
            "probability": None,
            "status": "training_failed",
            "model_source": None,
            "error": "Need at least 50 labelled rows to train, got 3.",
        },
        {
            "ticker": _probe,
            "sentiment_score": 0.0,
            "summary": [],
            "status": "no_api_key",
            "article_count": 0,
            "provider": None,
            "model": None,
            "error": "No GEMINI_API_KEY or OPENAI_API_KEY found.",
            "headlines": [],
        },
        _probe,
    )
