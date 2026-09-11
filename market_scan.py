"""Multi-ticker market scan behind the dashboard's overview table.

``run_analysis`` answers "what is the signal for *this* ticker?". This module
answers the same question for a *list* of them, in the shape a table wants: one
flat row per symbol carrying the price, the handful of indicator readings worth
putting in a column, both leg payloads, and the blended verdict.

Three things shape the design.

**It is Streamlit-free, like the rest of the pipeline.** Nothing here imports
``streamlit``, so the scan is testable from a terminal and the dashboard's
caching stays in ``app.py`` where the rest of it lives.

**Scoring one ticker is I/O-bound, not CPU-bound** -- a yfinance fetch and an
LLM call, both spent waiting on a socket. So :func:`scan_tickers` runs the
symbols through a small thread pool rather than a loop; twenty tickers scored
one at a time would take minutes of almost pure waiting.

**The caller supplies the scorer.** :func:`scan_tickers` takes a ``scorer``
callable rather than always calling :func:`scan_ticker` itself, so the dashboard
can hand it a per-ticker *cached* wrapper. That is what makes a row free the
second time it is asked for -- and what lets the detail sections below the table
reuse the very legs the table already computed instead of paying for a second
LLM call on the ticker the user clicked.

A row never carries an exception. A symbol that cannot be fetched, scored or
read comes back as a row with ``blended_score`` of None, ``signal`` of
:data:`~signal_blender.NO_SIGNAL` and the reason in ``error``, because one dead
ticker should cost the table a row rather than the whole scan.
"""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Callable, Iterable

import pandas as pd

import run_analysis  # also puts ``src`` on the path

import signal_blender  # noqa: E402  (from ``src``, via run_analysis's fix-up)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The moving-average pair whose spread becomes the row's trend reading. Same
# pair ``feature_engineering`` computes and the chart overlays, so the column
# and the two lines a reader can see cannot disagree. ``ui.breakdown`` takes its
# MA row from these too, rather than restating them.
MA_FAST_COLUMN = "SMA_20"
MA_SLOW_COLUMN = "SMA_50"

# The windows those columns were built from, read off the names so no label can
# claim a period its column was not computed from.
MA_FAST_PERIOD = int(MA_FAST_COLUMN.rsplit("_", 1)[-1])
MA_SLOW_PERIOD = int(MA_SLOW_COLUMN.rsplit("_", 1)[-1])

# Threads used to score a scan. Sized for the two services a row waits on:
# high enough that twenty symbols finish in one LLM round-trip's worth of
# batches, low enough not to trip yfinance's rate limiter or the provider's
# concurrency ceiling -- both of which would turn a slow scan into a failed one.
DEFAULT_SCAN_WORKERS = 6


# ---------------------------------------------------------------------------
# Reading a frame
# ---------------------------------------------------------------------------


def latest_value(bars: pd.DataFrame, column: str) -> float | None:
    """Reads the most recent usable value of one column.

    Takes the last *non-null* value rather than the last row's, because a frame
    kept with its warm-up rows can leave the longest indicator undefined on the
    final bar. Reporting the most recent value that exists beats reporting
    nothing for an indicator the frame does carry.

    Parameters:
    - bars (pd.DataFrame): The frame to read.
    - column (str): The column to read.

    Returns:
    - float | None: The value, or None when the column is absent, empty or
      entirely non-numeric.
    """
    if bars is None or bars.empty or column not in bars.columns:
        return None

    values = pd.to_numeric(bars[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _session_change_pct(bars: pd.DataFrame) -> float | None:
    """Computes the latest bar's percentage move against the one before it.

    Parameters:
    - bars (pd.DataFrame): A frame with a ``Close`` column, oldest first.

    Returns:
    - float | None: The move in percent, or None when there are not two closes
      to compare (a single-bar frame, or a zero previous close).
    """
    if bars is None or bars.empty or "Close" not in bars.columns:
        return None

    closes = pd.to_numeric(bars["Close"], errors="coerce").dropna()
    if len(closes) < 2:
        return None

    previous = float(closes.iloc[-2])
    if not previous:
        return None

    return (float(closes.iloc[-1]) - previous) / abs(previous) * 100.0


def _as_of(bars: pd.DataFrame) -> str | None:
    """Reads the date of the last bar as ``'YYYY-MM-DD'``.

    Parameters:
    - bars (pd.DataFrame): A frame with dates in a ``Date`` column.

    Returns:
    - str | None: The formatted date, or None when the frame carries no usable
      dates.
    """
    if bars is None or bars.empty or "Date" not in bars.columns:
        return None

    stamps = pd.to_datetime(bars["Date"], errors="coerce").dropna()
    if stamps.empty:
        return None
    return stamps.iloc[-1].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def load_cached_bars(ticker: str) -> pd.DataFrame:
    """Reads a ticker's persisted bars from ``data/``, if they are there.

    ``data_loader.save_data_locally`` writes ``<TICKER>_daily.csv`` and the repo
    ships two of them, so this is what lets a row draw with no network -- on a
    plane, behind a proxy, or when yfinance is rate-limiting.

    Parameters:
    - ticker (str): The symbol whose cached file is wanted.

    Returns:
    - pd.DataFrame: The cached bars, or an empty frame when there is no readable
      file for this ticker.
    """
    path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except (OSError, ValueError):
        # A truncated or hand-edited file is not worth a traceback; the caller
        # reports the original fetch failure instead.
        return pd.DataFrame()


def prepare_bars(ticker: str, lookback_days: int) -> tuple[pd.DataFrame, str | None]:
    """Fetches ``lookback_days`` of bars and appends the technical indicators.

    The whole fetched window is returned rather than a display slice, because
    the consumers want different amounts of the same data: the ML leg scores the
    last row but trains on all of them, while a chart shows only the window its
    reader asked for.

    Never raises. A failed fetch falls back to the ticker's cached CSV when one
    exists, and returns an empty frame with a reason when it does not, so a
    network problem costs a row its numbers rather than the scan its render.

    Parameters:
    - ticker (str): The symbol to load.
    - lookback_days (int): Calendar days of history to request.

    Returns:
    - tuple[pd.DataFrame, str | None]: The bars -- ``Date`` plus OHLCV plus
      indicator columns, oldest first -- and a warning to show the user, or None
      when the load was clean.
    """
    warning: str | None = None

    try:
        price_df = run_analysis.fetch_price_history(ticker, lookback_days=lookback_days)
    except Exception as e:
        price_df = load_cached_bars(ticker)
        if price_df.empty:
            return pd.DataFrame(), f"Could not load price history for {ticker}: {e}"
        warning = (
            f"Live fetch for {ticker} failed ({e}). Showing the bars cached in "
            f"data/, which may be out of date."
        )

    try:
        featured = run_analysis.compute_indicators(price_df)
    except run_analysis.AnalysisError:
        # Too little history for the longest indicator window. The candles are
        # still worth drawing and the indicator columns degrade to 'n/a'; the
        # overlays simply will not appear.
        featured = price_df

    return featured.reset_index(drop=True), warning


# ---------------------------------------------------------------------------
# Scoring one ticker
# ---------------------------------------------------------------------------


def _row(
    ticker: str,
    *,
    bars: pd.DataFrame | None = None,
    warning: str | None = None,
    ml: dict | None = None,
    sentiment: dict | None = None,
    blended_score: float | None = None,
    ml_weight: float = run_analysis.DEFAULT_ML_WEIGHT,
    error: str | None = None,
) -> dict:
    """Assembles one scan row, filling the derived fields from ``bars``.

    Every exit from :func:`scan_ticker` goes through here, so a row that failed
    early has exactly the same keys as one that scored cleanly and no consumer
    has to guard against a missing field.

    Parameters:
    - ticker (str): The normalized symbol.
    - bars (pd.DataFrame | None): The indicator frame, if there is one.
    - warning (str | None): A caveat about the bars, e.g. a stale-cache notice.
    - ml (dict | None): ``run_analysis.run_ml_inference``'s payload; None
      synthesizes a ``'no_data'`` one.
    - sentiment (dict | None): ``get_llm_sentiment_score``'s payload; None
      synthesizes a ``'no_data'`` one.
    - blended_score (float | None): The blended score, or None when there was
      nothing to blend.
    - ml_weight (float): Weighting applied to the blend.
    - error (str | None): Why the row has no verdict, if it has none.

    Returns:
    - dict: The row, documented on :func:`scan_ticker`.
    """
    frame = pd.DataFrame() if bars is None else bars

    # A leg that never ran still gets its documented payload shape, with the
    # status saying why, so the detail sections can read a failed row the same
    # way they read a successful one instead of guarding every field access.
    if ml is None:
        ml = {
            "ml_score": 0.0,
            "probability": None,
            "status": "no_data",
            "model_source": None,
            "error": error,
        }

    if sentiment is None:
        sentiment = {
            "ticker": ticker,
            "sentiment_score": 0.0,
            "summary": [],
            "status": "no_data",
            "article_count": 0,
            "provider": None,
            "model": None,
            "error": error,
            "headlines": [],
        }

    if blended_score is None:
        signal, color = signal_blender.NO_SIGNAL, None
    else:
        verdict = signal_blender.get_dashboard_signal(blended_score)
        signal, color = verdict["signal"], verdict["color"]

    return {
        "ticker": ticker,
        "bars": frame,
        "warning": warning,
        "as_of": _as_of(frame),
        "close": latest_value(frame, "Close"),
        "change_pct": _session_change_pct(frame),
        "rsi": latest_value(frame, "RSI"),
        "ma_gap_pct": _ma_gap_pct(frame),
        "macd_hist": latest_value(frame, "MACD_Hist"),
        "ml": ml,
        "sentiment": sentiment,
        "blended_score": blended_score,
        "ml_weight": float(ml_weight),
        "signal": signal,
        "color": color,
        "error": error,
    }


def _ma_gap_pct(bars: pd.DataFrame) -> float | None:
    """Computes the fast average's distance from the slow one, in percent.

    The sign is the trend direction and the magnitude is its conviction, which
    a bare pair of prices does not convey.

    Parameters:
    - bars (pd.DataFrame): The indicator frame.

    Returns:
    - float | None: The spread in percent of the slow average, or None when
      either average is missing.
    """
    fast = latest_value(bars, MA_FAST_COLUMN)
    slow = latest_value(bars, MA_SLOW_COLUMN)

    if fast is None or slow is None or not slow:
        return None
    return (fast - slow) / abs(slow) * 100.0


def scan_ticker(
    ticker: str,
    *,
    lookback_days: int = run_analysis.DEFAULT_LOOKBACK_DAYS,
    news_limit: int = run_analysis.DEFAULT_NEWS_LIMIT,
    ml_weight: float = run_analysis.DEFAULT_ML_WEIGHT,
    allow_training: bool = True,
) -> dict:
    """Runs the full pipeline for one ticker and flattens it into a table row.

    The same five stages ``run_analysis.analyze_ticker`` runs, composed the same
    way, but returning the indicator frame and both leg payloads alongside the
    verdict so a caller can render a row *and* a detail view from one pass.

    Never raises.

    Parameters:
    - ticker (str): The symbol to score.
    - lookback_days (int): Calendar days of price history to fetch.
    - news_limit (int): Maximum number of headlines to score.
    - ml_weight (float): Share of the blend given to the ML score.
    - allow_training (bool): Whether to train and persist a model when the
      ticker has none saved yet.

    Returns:
    - dict: With keys:
      - 'ticker' (str): The normalized symbol.
      - 'bars' (pd.DataFrame): The fetched frame with indicators, empty when the
        history could not be loaded.
      - 'warning' (str | None): A caveat about those bars, e.g. a stale cache.
      - 'as_of' (str | None): Date of the latest bar, 'YYYY-MM-DD'.
      - 'close' (float | None): Its closing price, in the listing currency.
      - 'change_pct' (float | None): Its move against the previous close, in
        percent.
      - 'rsi' (float | None): Latest RSI reading.
      - 'ma_gap_pct' (float | None): 20-day average's distance from the 50-day,
        in percent.
      - 'macd_hist' (float | None): Latest MACD histogram value.
      - 'ml' (dict): ``run_analysis.run_ml_inference``'s payload.
      - 'sentiment' (dict): ``get_llm_sentiment_score``'s payload.
      - 'blended_score' (float | None): The blended score, or None when there
        were no bars to analyze.
      - 'ml_weight' (float): Weighting applied to the blend.
      - 'signal' (str): One of the six dashboard recommendations, or
        ``signal_blender.NO_SIGNAL``.
      - 'color' (str | None): Hex tint for that signal; None for no signal, so
        the caller picks its own "unavailable" colour rather than borrowing a
        band's.
      - 'error' (str | None): Why there is no verdict, if there is none.
    """
    symbol = str(ticker).strip().upper()

    bars, warning = prepare_bars(symbol, lookback_days)
    if bars.empty:
        return _row(symbol, warning=warning, ml_weight=ml_weight, error=warning)

    ml = run_analysis.run_ml_inference(symbol, bars, allow_training=allow_training)
    sentiment = run_analysis.run_sentiment_analysis(symbol, news_limit=news_limit)

    blended_score = signal_blender.blend_scores(
        ml["ml_score"], float(sentiment["sentiment_score"]), ml_weight=ml_weight
    )

    return _row(
        symbol,
        bars=bars,
        warning=warning,
        ml=ml,
        sentiment=sentiment,
        blended_score=blended_score,
        ml_weight=ml_weight,
    )


# ---------------------------------------------------------------------------
# Scoring a list
# ---------------------------------------------------------------------------


def scan_tickers(
    tickers: Iterable[str],
    *,
    lookback_days: int = run_analysis.DEFAULT_LOOKBACK_DAYS,
    news_limit: int = run_analysis.DEFAULT_NEWS_LIMIT,
    ml_weight: float = run_analysis.DEFAULT_ML_WEIGHT,
    allow_training: bool = True,
    max_workers: int = DEFAULT_SCAN_WORKERS,
    scorer: Callable[[str], dict] | None = None,
    on_done: Callable[[int, int, dict], None] | None = None,
    worker_initializer: Callable[[], None] | None = None,
) -> list[dict]:
    """Scores every ticker in parallel and returns one row each, in input order.

    Duplicates are collapsed -- the same symbol reached from two places is one
    fetch and one LLM call -- and the result keeps the order it was asked for,
    so the caller owns the sort rather than inheriting whichever row finished
    first.

    Parameters:
    - tickers (Iterable[str]): The symbols to score.
    - lookback_days (int): Passed to the default scorer.
    - news_limit (int): Passed to the default scorer.
    - ml_weight (float): Passed to the default scorer.
    - allow_training (bool): Passed to the default scorer.
    - max_workers (int): Threads to score with; see
      :data:`DEFAULT_SCAN_WORKERS` for why the ceiling is low.
    - scorer (Callable[[str], dict] | None): Called with one symbol and must
      return a row shaped like :func:`scan_ticker`'s. Defaults to
      :func:`scan_ticker` bound to the arguments above; the dashboard passes a
      cached wrapper so a symbol is scored once per cache window.
    - on_done (Callable[[int, int, dict], None] | None): Invoked in the
      *calling* thread as each row lands, with ``(completed, total, row)``.
      Called from here rather than from the workers so a progress indicator can
      touch UI state safely.
    - worker_initializer (Callable[[], None] | None): Run once in each worker
      thread before it scores anything. The dashboard uses it to attach
      Streamlit's script context, which is what lets a cached scorer be called
      off the main thread.

    Returns:
    - list[dict]: One row per unique input symbol, in input order.
    """
    symbols: list[str] = []
    for raw in tickers:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        return []

    def _score(symbol: str) -> dict:
        return scan_ticker(
            symbol,
            lookback_days=lookback_days,
            news_limit=news_limit,
            ml_weight=ml_weight,
            allow_training=allow_training,
        )

    score = scorer or _score
    rows: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_workers, len(symbols)),
        thread_name_prefix="market-scan",
        initializer=worker_initializer,
    ) as pool:
        pending = {pool.submit(score, symbol): symbol for symbol in symbols}

        for completed, future in enumerate(
            concurrent.futures.as_completed(pending), start=1
        ):
            symbol = pending[future]
            try:
                row = future.result()
            except Exception as e:
                # The scorer is documented not to raise, but it is caller-supplied
                # and one broken symbol must not take the other nineteen with it.
                row = _row(symbol, error=f"{type(e).__name__}: {e}")

            rows[symbol] = row
            if on_done is not None:
                on_done(completed, len(symbols), row)

    return [rows[symbol] for symbol in symbols]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def sort_by_conviction(rows: Iterable[dict]) -> list[dict]:
    """Orders rows strongest-first, regardless of direction.

    The table's default order, and the reason the scan exists: a reader opening
    the page wants the names the pipeline feels strongest about, and a Strong
    Sell is as strong a claim as a Strong Buy. So the key is the score's
    *magnitude*, which interleaves the two ends of the ladder and sinks the
    weak and neutral names to the bottom.

    Ties break towards the bullish side and then alphabetically, so the order is
    stable across reruns -- the table's row-click selection maps positions back
    to symbols, and an unstable sort would make a click land on the wrong stock.

    Rows with no verdict at all sort last: they are not a weak signal, they are
    the absence of one.

    Parameters:
    - rows (Iterable[dict]): Rows from :func:`scan_tickers`.

    Returns:
    - list[dict]: A new list, most-convicted first.
    """
    def key(row: dict) -> tuple[int, float, float, str]:
        score = row.get("blended_score")
        if score is None:
            return (1, 0.0, 0.0, row.get("ticker", ""))
        return (0, -abs(float(score)), -float(score), row.get("ticker", ""))

    return sorted(rows, key=key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="market_scan.py",
        description="Scan several tickers and print the ranked signals as a table.",
    )
    parser.add_argument(
        "tickers", nargs="*", default=["AAPL", "MSFT", "RELIANCE.NS"],
        help="Symbols to scan (default: AAPL MSFT RELIANCE.NS).",
    )
    parser.add_argument("--lookback-days", type=int, default=run_analysis.DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--news-limit", type=int, default=run_analysis.DEFAULT_NEWS_LIMIT)
    parser.add_argument("--workers", type=int, default=DEFAULT_SCAN_WORKERS)
    _args = parser.parse_args()

    _scanned = scan_tickers(
        _args.tickers,
        lookback_days=_args.lookback_days,
        news_limit=_args.news_limit,
        max_workers=_args.workers,
        on_done=lambda done, total, row: print(
            f"  [{done}/{total}] {row['ticker']:<14} {row['signal']}"
        ),
    )

    print(f"\n{'TICKER':<14}{'CLOSE':>10}{'1D %':>8}{'RSI':>7}{'ML':>8}{'NEWS':>8}{'SCORE':>8}  SIGNAL")
    for _r in sort_by_conviction(_scanned):
        def _f(value, spec):
            return format(value, spec) if isinstance(value, (int, float)) else "n/a"

        print(
            f"{_r['ticker']:<14}"
            f"{_f(_r['close'], '>10,.2f')}"
            f"{_f(_r['change_pct'], '>+8.2f')}"
            f"{_f(_r['rsi'], '>7.1f')}"
            f"{_f(_r['ml'].get('ml_score'), '>+8.2f')}"
            f"{_f(_r['sentiment'].get('sentiment_score'), '>+8.2f')}"
            f"{_f(_r['blended_score'], '>+8.2f')}"
            f"  {_r['signal']}"
            + (f"  ({_r['error']})" if _r["error"] else "")
        )
