"""The Market Overview table that opens the dashboard.

Below this table the page still does what it always did -- one ticker, one
verdict, argued in three sections. This table is the step before that: it runs
the same pipeline across the whole preset universe and ranks the results, so the
first question a reader gets an answer to is "where should I be looking?" rather
than "what does AAPL look like?".

Three decisions are worth knowing about.

**The ranking is by conviction, not by direction.** ``sort_by_conviction`` in
``market_scan`` orders on the score's *magnitude*, so Strong Buy and Strong Sell
share the top of the table and the weak and neutral names sink. A reader
scanning from the top is reading the pipeline's strongest claims in order,
whichever way they point. The column headers still sort, so a directional view
is one click away.

**Every number here is the same number the sections below show.** The rows
carry the leg payloads the scan already computed, and ``app.py`` hands the
detail sections the row for the selected ticker rather than re-running the
pipeline. So the table's Signal column and the card underneath it cannot
disagree, and clicking a row costs nothing.

**The Signal cell is tinted, and the tint is not chosen here.** It comes from
the row, which got it from ``signal_blender``, and the text colour on top of it
comes from ``signal_card.ink_on`` -- the same contrast measurement the card
uses, so white text can never land on the pale-green Weak Buy band. Colour is
never the only carrier: the band name is written in the cell and the score sits
in the column beside it.
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import streamlit as st

# Running this file directly (see ``__main__`` below) puts ``ui/`` on the path
# rather than the project root, so ``import ui`` and ``import market_scan`` need
# the fix-up ``breakdown`` and ``signal_card`` apply for the same reason.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import market_scan  # noqa: E402  (also puts ``src`` on the path)

import signal_blender  # noqa: E402  (from ``src``, via market_scan's fix-up)

from ui.breakdown import (  # noqa: E402
    MA_FAST_PERIOD,
    MA_FLAT_BAND_PCT,
    MA_SLOW_PERIOD,
    RSI_PERIOD,
)
from ui.config import ticker_for  # noqa: E402
from ui.signal_card import ink_on  # noqa: E402

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------

# The dataframe widget's own key; its row selection lands in session state here.
TABLE_KEY = "market_table"

# The symbols behind the rows *as last drawn*, in display order. Written at
# render time and read back by :func:`consume_row_selection`, because Streamlit
# reports a click as a row position and the position only means something
# against the order that was on screen when it was clicked.
ORDER_KEY = "market_table_order"

# The row position already acted on, so a selection that is merely still sitting
# in state does not keep overriding the sidebar on every later rerun.
APPLIED_KEY = "market_table_applied_row"

# Whether the table is expanded past :data:`~ui.config.MARKET_TABLE_SIZE`.
SHOW_ALL_KEY = "market_table_show_all"


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

# Column order, and the display formats. Values stay numeric in the frame so the
# grid sorts on magnitude rather than on the string "n/a" or a leading '+'; the
# formatting is applied by the Styler at render time. ``None`` marks a text
# column, which needs no format.
COLUMN_FORMATS: dict[str, str | None] = {
    "Symbol": None,
    "Company": None,
    "Market": None,
    "Price": "{:,.2f}",
    "1D %": "{:+.2f}%",
    f"RSI ({RSI_PERIOD})": "{:.1f}",
    "Trend": None,
    "ML": "{:+.2f}",
    "News": "{:+.2f}",
    "Score": "{:+.2f}",
    "Signal": None,
}

# What each column means, surfaced as the header tooltip. The two currencies in
# the universe are the reason Price carries no symbol: one column cannot prefix
# both, and a formatted string would sort alphabetically instead of numerically.
COLUMN_HELP = {
    "Symbol": "The yfinance symbol. Click a row to analyze it in detail below.",
    "Company": "Company name as listed in the preset universe.",
    "Market": "Listing venue, which is also the currency of the Price column.",
    "Price": "Latest close, in the listing currency ($ for US, ₹ for NSE).",
    "1D %": "Move from the previous close to the latest one.",
    f"RSI ({RSI_PERIOD})": (
        "Relative Strength Index. Above 70 is conventionally overbought, below "
        "30 oversold."
    ),
    "Trend": (
        f"Whether the {MA_FAST_PERIOD}-day average sits above or below the "
        f"{MA_SLOW_PERIOD}-day one."
    ),
    "ML": "Random Forest score on [-1, +1]. 0.00 is a coin flip.",
    "News": "LLM news-sentiment score on the same [-1, +1] scale.",
    "Score": "The two legs blended, which is what the Signal is read from.",
    "Signal": "The blended verdict. Ranked by conviction, so the strongest claims lead.",
}

# The read shown in the Trend column, using the same band the breakdown table's
# MA row uses, so the two cannot describe the same spread differently.
TREND_BULLISH = "Bullish"
TREND_BEARISH = "Bearish"
TREND_FLAT = "Flat"

# Stand-in for a text cell with no value. Numeric cells use NaN instead, which
# the Styler renders with the same string while still sorting as a number.
NOT_AVAILABLE = "n/a"


# ---------------------------------------------------------------------------
# Table geometry
# ---------------------------------------------------------------------------

# Pixel heights of Streamlit's data grid, used to size the table to its own row
# count. Without this it stays at the default 400px -- about ten rows -- so a
# twenty-row table would arrive half-hidden behind an inner scrollbar and the
# caption's "showing 20 of 24" would be a claim the reader could not check.
ROW_HEIGHT_PX = 35
HEADER_HEIGHT_PX = 35
GRID_BORDER_PX = 3

# Past this the table alone would fill a laptop screen and push the detail
# sections out of sight, so the inner scrollbar comes back -- at a length where
# scrolling is the obvious way to read it rather than a surprise.
MAX_TABLE_HEIGHT_PX = 900


def table_height(row_count: int) -> int:
    """Sizes the grid to show ``row_count`` rows without an inner scrollbar.

    Parameters:
    - row_count (int): Rows about to be drawn.

    Returns:
    - int: A pixel height, capped at :data:`MAX_TABLE_HEIGHT_PX`.
    """
    fitted = HEADER_HEIGHT_PX + row_count * ROW_HEIGHT_PX + GRID_BORDER_PX
    return min(fitted, MAX_TABLE_HEIGHT_PX)


# ---------------------------------------------------------------------------
# Row to cells
# ---------------------------------------------------------------------------


def _number(value: object) -> float:
    """Coerces a possibly-missing reading to a float the grid can sort.

    Parameters:
    - value (object): A number, or None when the reading was unavailable.

    Returns:
    - float: The value, or NaN -- which sorts to one end and renders as
      :data:`NOT_AVAILABLE` rather than as a misleading zero.
    """
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float("nan")
    return float(value)


def _trend_label(gap_pct: float | None) -> str:
    """Reads a moving-average spread as one word.

    Parameters:
    - gap_pct (float | None): The fast average's distance from the slow one, in
      percent, from ``market_scan``.

    Returns:
    - str: :data:`TREND_BULLISH`, :data:`TREND_BEARISH`, :data:`TREND_FLAT` or
      :data:`NOT_AVAILABLE`. Spreads inside
      :data:`~ui.breakdown.MA_FLAT_BAND_PCT` read as flat: on a large-cap they
      sit inside a single session's noise, so calling a direction from them
      would over-read the data.
    """
    if gap_pct is None or not math.isfinite(gap_pct):
        return NOT_AVAILABLE
    if abs(gap_pct) < MA_FLAT_BAND_PCT:
        return TREND_FLAT
    return TREND_BULLISH if gap_pct > 0 else TREND_BEARISH


def _leg_score(leg: dict, key: str) -> float:
    """Reads one leg's score, but only if the leg actually ran.

    Both legs degrade to ``0.0`` on failure, which in a column of scores is
    indistinguishable from genuine indecision. A leg that did not complete
    therefore reports nothing at all, matching what the breakdown section says
    about the same leg further down the page.

    Parameters:
    - leg (dict): A ``ml`` or ``sentiment`` payload from a scan row.
    - key (str): The score field to read -- ``'ml_score'`` or
      ``'sentiment_score'``.

    Returns:
    - float: The score, or NaN when the leg's status is not ``'ok'``.
    """
    if leg.get("status") != "ok":
        return float("nan")
    return _number(leg.get(key))


def build_market_frame(rows: list[dict]) -> pd.DataFrame:
    """Turns scan rows into the frame the table draws.

    Pure: reads the rows and returns a new frame, drawing nothing. Company name
    and market come from the preset universe rather than from the row, because a
    scan row is a pipeline payload and the pipeline has no notion of either.

    Parameters:
    - rows (list[dict]): Rows from ``market_scan.scan_tickers``, already in the
      order they should appear.

    Returns:
    - pd.DataFrame: One row per input row, columns per :data:`COLUMN_FORMATS`,
      with a fresh ``RangeIndex`` so a click's row position indexes it directly.
    """
    records = []

    for row in rows:
        symbol = row["ticker"]
        entry = ticker_for(symbol)

        records.append(
            {
                "Symbol": symbol,
                "Company": entry.name if entry else symbol,
                # A hand-typed symbol has no universe entry, so fall back to the
                # same suffix rule ``price_chart`` picks its currency with.
                "Market": entry.market if entry else ("India" if symbol.endswith(".NS") else "US"),
                "Price": _number(row["close"]),
                "1D %": _number(row["change_pct"]),
                f"RSI ({RSI_PERIOD})": _number(row["rsi"]),
                "Trend": _trend_label(row["ma_gap_pct"]),
                "ML": _leg_score(row["ml"], "ml_score"),
                "News": _leg_score(row["sentiment"], "sentiment_score"),
                "Score": _number(row["blended_score"]),
                "Signal": row["signal"],
            }
        )

    return pd.DataFrame(records, columns=list(COLUMN_FORMATS))


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def _signal_css(color: str | None) -> str:
    """Builds the CSS for one Signal cell.

    Parameters:
    - color (str | None): The band tint from the scan row, or None when the row
      has no verdict.

    Returns:
    - str: A ``background-color``/``color`` declaration pair. The text colour is
      measured against the tint rather than assumed, since three of the six
      bands are pale enough to swallow white text.
    """
    tint = color or signal_blender.NO_SIGNAL_COLOR
    return f"background-color: {tint}; color: {ink_on(tint)}"


def style_market_frame(frame: pd.DataFrame, rows: list[dict]):
    """Applies the display formats and the Signal tints to a market frame.

    Parameters:
    - frame (pd.DataFrame): The output of :func:`build_market_frame`.
    - rows (list[dict]): The scan rows it was built from, in the same order --
      the source of each row's tint.

    Returns:
    - pandas.io.formats.style.Styler: The frame, formatted and tinted.
    """
    formats = {
        column: spec for column, spec in COLUMN_FORMATS.items() if spec is not None
    }
    tints = [_signal_css(row["color"]) for row in rows]

    return (
        frame.style
        # ``na_rep`` is why the numeric columns can hold NaN: they sort as
        # numbers but read as words where a reading is missing.
        .format(formats, na_rep=NOT_AVAILABLE)
        .apply(lambda _column: tints, subset=["Signal"])
    )


def _column_config() -> dict:
    """Builds the per-column grid configuration.

    Number *formatting* is deliberately left to the Styler, so there is one
    source of truth for how a value reads; this only supplies the header
    tooltips and pins the symbol column, which is what a reader tracks a row by
    when the table scrolls sideways on a narrow screen.

    Returns:
    - dict: A ``column_config`` mapping for ``st.dataframe``.
    """
    config = {
        column: st.column_config.Column(column, help=COLUMN_HELP.get(column))
        for column in COLUMN_FORMATS
    }
    config["Symbol"] = st.column_config.Column(
        "Symbol", help=COLUMN_HELP["Symbol"], pinned=True
    )
    return config


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


def consume_row_selection() -> str | None:
    """Reports the symbol whose row was just clicked, once.

    Must be called *before* the sidebar renders, since the caller feeds the
    answer to ``sidebar.focus_ticker`` and that has to beat the widgets it
    writes to.

    A selection stays in Streamlit's state after it has been acted on, so this
    returns a symbol only on the run that follows the click itself -- otherwise a
    row selected ten reruns ago would keep dragging the sidebar back every time
    the date range changed. Clearing the selection resets that latch, so
    re-clicking the same row works.

    The row position is resolved against the order stored when the table was
    drawn, not the current one, so a scan that refreshes between the click and
    this call cannot make the click land on a neighbouring stock.

    Returns:
    - str | None: The clicked symbol, or None when nothing new was clicked.
    """
    # Streamlit hands back a dict subclass, and nothing is there at all until the
    # table has rendered once, so both levels are defaulted rather than indexed.
    state = st.session_state.get(TABLE_KEY) or {}
    positions = (state.get("selection") or {}).get("rows") or []

    if not positions:
        st.session_state[APPLIED_KEY] = None
        return None

    position = positions[0]
    if position == st.session_state.get(APPLIED_KEY):
        return None

    st.session_state[APPLIED_KEY] = position

    order = st.session_state.get(ORDER_KEY) or ()
    if 0 <= position < len(order):
        return order[position]
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_market_table(rows: list[dict], *, top_n: int, focused: str | None = None) -> None:
    """Renders the ranked overview table into the current Streamlit container.

    Draws the strongest ``top_n`` rows, then the control that expands the table
    to the full scanned universe. The control sits *below* the table where a
    reader looks for it after running out of rows, which means its state has to
    be read from session state before the table is drawn rather than from the
    widget's return value.

    Parameters:
    - rows (list[dict]): Every scanned row, in any order; this function ranks
      them by conviction.
    - top_n (int): Rows to show before the expand control is used.
    - focused (str | None): The symbol the detail sections below are currently
      showing, named in the caption so the link between the two is explicit.
    """
    ranked = market_scan.sort_by_conviction(rows)
    show_all = bool(st.session_state.get(SHOW_ALL_KEY))
    visible = ranked if show_all else ranked[:top_n]

    if not visible:
        st.info("No stocks could be scanned. Check your connectivity and refresh.")
        return

    # The click handler resolves a row position against this, so it is written
    # from the same list that is about to be drawn.
    st.session_state[ORDER_KEY] = tuple(row["ticker"] for row in visible)

    st.dataframe(
        style_market_frame(build_market_frame(visible), visible),
        column_config=_column_config(),
        hide_index=True,
        width="stretch",
        height=table_height(len(visible)),
        on_select="rerun",
        selection_mode="single-row",
        key=TABLE_KEY,
    )

    st.caption(
        f"Showing {len(visible)} of {len(ranked)} scanned stocks, strongest signal "
        "first -- Strong Buy and Strong Sell lead, weak and neutral names follow. "
        "Click any column header to re-sort."
        + (f" Analyzing **{focused}** in detail below." if focused else "")
    )

    unscored = [row["ticker"] for row in ranked if row["blended_score"] is None]
    if unscored:
        st.caption(
            f"{len(unscored)} could not be scored and rank last: "
            f"{', '.join(unscored)}. Their rows show the reason on the ticker's "
            "own view below."
        )

    if len(ranked) > top_n:
        st.toggle(
            f"Show all {len(ranked)} stocks",
            key=SHOW_ALL_KEY,
            help=f"The table shows the top {top_n} by conviction until this is on.",
        )


if __name__ == "__main__":
    # Visual self-test: ``streamlit run ui/market_table.py`` draws the table from
    # stubbed rows, which is the only practical way to see every band's tint and
    # the unscored state at once without waiting on a real scan.
    st.set_page_config(page_title="Market table preview", layout="wide")
    st.title("Market Overview - preview")

    def _stub(symbol: str, score: float | None, *, close: float = 100.0) -> dict:
        _bars = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-04", "2026-09-05"]),
                "Close": [close * 0.99, close],
                "RSI": [50.0, 50.0 + score * 20 if score is not None else 50.0],
                market_scan.MA_FAST_COLUMN: [close, close * (1 + (score or 0) / 50)],
                market_scan.MA_SLOW_COLUMN: [close, close],
                "MACD_Hist": [0.0, score or 0.0],
            }
        )
        return market_scan._row(
            symbol,
            bars=_bars if score is not None else None,
            ml={"ml_score": (score or 0) * 0.9, "status": "ok", "probability": 0.6},
            sentiment={"sentiment_score": (score or 0) * 1.1, "status": "ok"},
            blended_score=score,
            error=None if score is not None else "No price history returned.",
        )

    _rows = [
        _stub("NVDA", 0.88, close=182.31),
        _stub("SBIN.NS", -0.91, close=812.40),
        _stub("AAPL", 0.52, close=241.10),
        _stub("TSLA", -0.44, close=333.87),
        _stub("ITC.NS", 0.11, close=409.55),
        _stub("WMT", -0.09, close=98.42),
        _stub("MARUTI.NS", None),
    ]

    render_market_table(_rows, top_n=5, focused="NVDA")
    st.divider()
    st.write("Selection state:", st.session_state.get(TABLE_KEY))
