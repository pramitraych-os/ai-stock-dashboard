"""End-to-end driver for the AI Stock Dashboard's Phase 2 analysis workflow.

This script is the seam between the ``src`` modules and anything that wants a
finished answer for a ticker -- the dashboard UI, a scheduled job, or a human at
a terminal. It runs the five stages of the pipeline in order:

1. **Price history** -- ``data_loader`` fetches and cleans daily OHLCV bars for
   the ticker (US symbols like ``AAPL`` or Indian ones like ``RELIANCE.NS``).
2. **Technical indicators** -- ``feature_engineering`` appends RSI, moving
   averages, MACD, Bollinger Bands and return features.
3. **ML inference** -- ``ml_engine`` loads the ticker's saved Random Forest (or
   trains and saves one on first run) and scores the latest bar into an
   ``ml_score`` on ``[-1, +1]``.
4. **LLM sentiment** -- ``sentiment_engine`` pulls recent headlines and asks the
   configured provider for an ``ai_score`` on the same ``[-1, +1]`` scale, plus
   a bullet-point rationale.
5. **Blending** -- ``signal_blender`` combines the two scores and maps the
   result onto one of six dashboard signals and its hex colour.

Only the price fetch is fatal: without bars there is nothing to analyze. The two
scoring stages degrade to a neutral ``0.0`` and report why in ``ml_status`` /
``sentiment_status``, mirroring the contract ``sentiment_engine`` already
documents, so a missing API key or an untrainable ticker still yields a usable
payload instead of a traceback.

Usage:
    python run_analysis.py --ticker AAPL
    python run_analysis.py --ticker RELIANCE.NS --lookback-days 1095
    python run_analysis.py --ticker MSFT --ml-weight 0.7 | jq .signal

Progress chatter from the underlying modules is routed to stderr, so stdout
carries nothing but the JSON payload and stays safe to pipe.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys

# The ``src`` modules import one another by bare name (``ml_engine`` does
# ``from data_loader import DATA_DIR``), so ``src`` has to sit on ``sys.path``
# directly rather than being imported as a ``src.*`` package.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd  # noqa: E402  (import after the sys.path fix-up)

import data_loader  # noqa: E402
import feature_engineering  # noqa: E402
import ml_engine  # noqa: E402
import sentiment_engine  # noqa: E402
import signal_blender  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Two years of daily bars: comfortably above the 50-row minimum ``train_model``
# needs even after the indicator warm-up window is dropped.
DEFAULT_LOOKBACK_DAYS = 730

# Headlines handed to the LLM. More context costs tokens for diminishing signal.
DEFAULT_NEWS_LIMIT = 10

# An even split between the model and the news narrative, per ``signal_blender``.
DEFAULT_ML_WEIGHT = 0.5


class AnalysisError(RuntimeError):
    """Raised when a stage fails in a way that makes the analysis meaningless."""


# ---------------------------------------------------------------------------
# Step 1: price history
# ---------------------------------------------------------------------------


def fetch_price_history(
    ticker: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS, save: bool = False
) -> pd.DataFrame:
    """Fetches and cleans daily OHLCV history for a ticker.

    The lookback window ends today and runs ``lookback_days`` calendar days
    back; weekends and holidays mean the row count lands near 70% of that.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - lookback_days (int): Calendar days of history to request.
    - save (bool): If True, also persist the cleaned frame to ``data/`` via
      ``data_loader.save_data_locally``.

    Returns:
    - pd.DataFrame: Cleaned bars with a ``Date`` column plus OHLCV, oldest
      first. ``Date`` is deliberately left as a column rather than an index
      because ``add_technical_indicators`` resets the index.

    Raises:
    - AnalysisError: If no usable bars came back for the symbol.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)

    raw_df = data_loader.fetch_stock_data(
        ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    price_df = data_loader.clean_stock_data(raw_df)

    if price_df.empty:
        raise AnalysisError(
            f"No price history returned for '{ticker}'. Check the symbol "
            f"(Indian listings need a '.NS' suffix) and your connectivity."
        )

    if save:
        data_loader.save_data_locally(price_df, ticker)

    return price_df


# ---------------------------------------------------------------------------
# Step 2: technical indicators
# ---------------------------------------------------------------------------


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Appends the technical-indicator suite to a cleaned price frame.

    Parameters:
    - price_df (pd.DataFrame): Cleaned OHLCV bars from
      :func:`fetch_price_history`.

    Returns:
    - pd.DataFrame: The bars with indicator columns appended and the leading
      warm-up rows dropped.

    Raises:
    - AnalysisError: If the warm-up period consumed every row, i.e. the history
      is shorter than the longest indicator window.
    """
    indicator_df = feature_engineering.add_technical_indicators(price_df)

    if indicator_df.empty:
        raise AnalysisError(
            f"Indicator warm-up consumed all {len(price_df)} rows of history. "
            f"Increase --lookback-days."
        )

    return indicator_df


# ---------------------------------------------------------------------------
# Step 3: ML inference
# ---------------------------------------------------------------------------


def resolve_model(ticker: str, indicator_df: pd.DataFrame, allow_training: bool = True):
    """Loads the ticker's saved model, training one on first run if needed.

    Parameters:
    - ticker (str): The stock symbol whose model is wanted.
    - indicator_df (pd.DataFrame): Indicator frame used for training when no
      saved model exists.
    - allow_training (bool): If False, a missing model is an error instead of a
      cue to train.

    Returns:
    - tuple[Pipeline, str]: The fitted pipeline and how it was obtained --
      ``'loaded'`` or ``'trained'``.

    Raises:
    - FileNotFoundError: If no model exists and ``allow_training`` is False.
    - ValueError: If training was attempted but the history was unusable (too
      few labelled rows, or a single-class target).
    """
    try:
        return ml_engine.load_model(ticker), "loaded"
    except FileNotFoundError:
        if not allow_training:
            raise

    print(f"No saved model for '{ticker}'; training one from this history...")
    model, _path = ml_engine.train_and_save(indicator_df, ticker)
    return model, "trained"


def run_ml_inference(
    ticker: str, indicator_df: pd.DataFrame, allow_training: bool = True
) -> dict:
    """Scores the most recent bar with the ticker's Random Forest.

    Never raises: a missing model that cannot be trained, or a malformed
    feature matrix, degrades to a neutral ``0.0`` with the reason in ``status``.
    Callers should check ``status == 'ok'`` before reading a ``0.0`` as genuine
    indecision rather than a fallback.

    Parameters:
    - ticker (str): The stock symbol being scored.
    - indicator_df (pd.DataFrame): Indicator frame from
      :func:`compute_indicators`; its last row is the one scored.
    - allow_training (bool): Whether a missing model may be trained on the fly.

    Returns:
    - dict: With keys ``'ml_score'`` (float in ``[-1, +1]``),
      ``'probability'`` (float in ``[0, 1]`` or None on failure),
      ``'status'`` ('ok', 'no_model', 'training_failed' or 'inference_failed'),
      ``'model_source'`` ('loaded', 'trained' or None) and ``'error'``.
    """
    try:
        model, model_source = resolve_model(
            ticker, indicator_df, allow_training=allow_training
        )
    except FileNotFoundError as e:
        return {
            "ml_score": 0.0,
            "probability": None,
            "status": "no_model",
            "model_source": None,
            "error": str(e),
        }
    except ValueError as e:
        # ``train_model`` rejects histories that are too short or one-sided.
        return {
            "ml_score": 0.0,
            "probability": None,
            "status": "training_failed",
            "model_source": None,
            "error": str(e),
        }

    try:
        probability = ml_engine.get_prediction_probability(model, indicator_df)
        ml_score = ml_engine.get_ml_score(model, indicator_df)
    except Exception as e:
        # A stale model artefact whose feature set no longer matches lands here.
        return {
            "ml_score": 0.0,
            "probability": None,
            "status": "inference_failed",
            "model_source": model_source,
            "error": f"{type(e).__name__}: {e}",
        }

    return {
        "ml_score": float(ml_score),
        "probability": float(probability),
        "status": "ok",
        "model_source": model_source,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Step 4: LLM sentiment
# ---------------------------------------------------------------------------


def run_sentiment_analysis(ticker: str, news_limit: int = DEFAULT_NEWS_LIMIT) -> dict:
    """Scores recent news sentiment for a ticker.

    A thin pass-through to :func:`sentiment_engine.get_llm_sentiment_score`,
    which already handles its own failures by returning a neutral score with a
    descriptive ``status``.

    Parameters:
    - ticker (str): The stock symbol to analyze.
    - news_limit (int): Maximum number of recent articles to send to the LLM.

    Returns:
    - dict: The engine's payload -- ``sentiment_score``, ``summary``,
      ``status``, ``article_count``, ``provider``, ``model``, ``error`` and
      ``headlines``.
    """
    return sentiment_engine.get_llm_sentiment_score(ticker, limit=news_limit)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _latest_bar(indicator_df: pd.DataFrame) -> tuple[str | None, float | None]:
    """Extracts the as-of date and close price of the most recent bar.

    Parameters:
    - indicator_df (pd.DataFrame): Indicator frame with a ``Close`` column and,
      normally, a ``Date`` column carried through from ``clean_stock_data``.

    Returns:
    - tuple[str | None, float | None]: The bar's date as ``'YYYY-MM-DD'`` (None
      if the frame carries no ``Date`` column) and its closing price.
    """
    latest = indicator_df.iloc[-1]

    as_of = None
    if "Date" in indicator_df.columns:
        as_of = pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d")

    close = latest.get("Close")
    return as_of, None if close is None else float(close)


def analyze_ticker(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    news_limit: int = DEFAULT_NEWS_LIMIT,
    ml_weight: float = DEFAULT_ML_WEIGHT,
    allow_training: bool = True,
    save_data: bool = False,
) -> dict:
    """Runs the full Phase 2 workflow for one ticker.

    Executes all five stages -- fetch, indicators, ML inference, LLM sentiment,
    blending -- and packages the result into a single JSON-serializable dict
    ready for the dashboard to render.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - lookback_days (int): Calendar days of price history to fetch.
    - news_limit (int): Maximum number of headlines to score.
    - ml_weight (float): Share of the blend given to ``ml_score``; the rest
      goes to ``ai_score``. Defaults to an even 0.5 split.
    - allow_training (bool): Whether to train and persist a model when the
      ticker has none saved yet.
    - save_data (bool): Whether to also write the fetched bars to ``data/``.

    Returns:
    - dict: With keys:
      - 'ticker' (str): The normalized symbol.
      - 'as_of' (str | None): Date of the latest bar, 'YYYY-MM-DD'.
      - 'latest_close' (float | None): Closing price of that bar.
      - 'ml_score' (float): Raw ML score in [-1, +1].
      - 'ai_score' (float): Raw sentiment score in [-1, +1].
      - 'ml_weight' (float): Weighting actually applied to the blend.
      - 'blended_score' (float): The combined score in [-1, +1].
      - 'signal' (str): One of the six dashboard recommendations.
      - 'color' (str): Hex colour the UI tints the card with.
      - 'rationale' (list[str]): Sentiment bullet points behind ``ai_score``.
      - 'headlines' (list[str]): Headlines the sentiment score is based on.
      - 'diagnostics' (dict): Per-stage detail -- statuses, LLM provider and
        model, article count, ML probability, model source, history size and
        any error strings.

    Raises:
    - AnalysisError: If the ticker is blank, or no price history is available.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise AnalysisError("A non-empty ticker symbol is required.")

    ticker = ticker.strip().upper()

    # Step 1: historical prices.
    print(f"[1/5] Fetching {lookback_days}d of history for {ticker}...")
    price_df = fetch_price_history(ticker, lookback_days=lookback_days, save=save_data)

    # Step 2: technical indicators.
    print(f"[2/5] Computing technical indicators on {len(price_df)} bars...")
    indicator_df = compute_indicators(price_df)

    # Step 3: quantitative score from the Random Forest.
    print("[3/5] Running ML inference on the latest bar...")
    ml_result = run_ml_inference(ticker, indicator_df, allow_training=allow_training)
    ml_score = ml_result["ml_score"]

    # Step 4: narrative score from the LLM.
    print(f"[4/5] Scoring news sentiment (up to {news_limit} articles)...")
    sentiment_result = run_sentiment_analysis(ticker, news_limit=news_limit)
    ai_score = float(sentiment_result["sentiment_score"])

    # Step 5: blend the two into the signal the dashboard renders.
    print(f"[5/5] Blending ml={ml_score:+.4f} and ai={ai_score:+.4f}...")
    blended_score = signal_blender.blend_scores(ml_score, ai_score, ml_weight=ml_weight)
    signal = signal_blender.get_dashboard_signal(blended_score)

    as_of, latest_close = _latest_bar(indicator_df)

    return {
        "ticker": ticker,
        "as_of": as_of,
        "latest_close": latest_close,
        "ml_score": round(ml_score, 4),
        "ai_score": round(ai_score, 4),
        "ml_weight": float(ml_weight),
        "blended_score": round(signal["blended_score"], 4),
        "signal": signal["signal"],
        "color": signal["color"],
        "rationale": sentiment_result["summary"],
        "headlines": sentiment_result["headlines"],
        "diagnostics": {
            "ml_status": ml_result["status"],
            "ml_probability": ml_result["probability"],
            "ml_model_source": ml_result["model_source"],
            "ml_error": ml_result["error"],
            "sentiment_status": sentiment_result["status"],
            "sentiment_provider": sentiment_result["provider"],
            "sentiment_model": sentiment_result["model"],
            "article_count": sentiment_result["article_count"],
            "sentiment_error": sentiment_result["error"],
            "price_rows": int(len(price_df)),
            "indicator_rows": int(len(indicator_df)),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line interface for the driver.

    Returns:
    - argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="run_analysis.py",
        description=(
            "Run the full AI Stock Dashboard analysis for a ticker and print "
            "the result as JSON."
        ),
        epilog=(
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY (in the environment or a .env "
            "file at the project root) to enable LLM sentiment; without a key "
            "the sentiment leg falls back to a neutral 0.0."
        ),
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock symbol to analyze, e.g. AAPL or RELIANCE.NS.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Calendar days of price history to fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=DEFAULT_NEWS_LIMIT,
        help="Maximum number of recent articles to score (default: %(default)s).",
    )
    parser.add_argument(
        "--ml-weight",
        type=float,
        default=DEFAULT_ML_WEIGHT,
        help=(
            "Share of the blend given to the ML score, 0.0-1.0; the remainder "
            "goes to sentiment (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Fail the ML leg instead of training a model when none is saved.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Also persist the fetched price history to data/.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation; 0 emits a single compact line (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: analyze one ticker and print the JSON payload.

    Progress output from every stage is redirected to stderr so stdout holds
    only the JSON document and remains pipeable into tools like ``jq``.

    Parameters:
    - argv (list[str] | None): Argument list to parse; defaults to ``sys.argv``.

    Returns:
    - int: ``0`` on success, ``1`` if the analysis could not be completed.
    """
    args = build_arg_parser().parse_args(argv)

    if not 0.0 <= args.ml_weight <= 1.0:
        print(
            f"--ml-weight must be between 0.0 and 1.0, got {args.ml_weight}.",
            file=sys.stderr,
        )
        return 1

    try:
        # Keep the modules' own ``print`` calls off stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = analyze_ticker(
                args.ticker,
                lookback_days=args.lookback_days,
                news_limit=args.news_limit,
                ml_weight=args.ml_weight,
                allow_training=not args.no_train,
                save_data=args.save_data,
            )
    except AnalysisError as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=args.indent or None, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
