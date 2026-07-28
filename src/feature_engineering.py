"""Feature engineering for historical daily stock data.

This module turns a raw OHLCV DataFrame (``Open``, ``High``, ``Low``, ``Close``,
``Volume``) into a feature-rich frame by appending common technical indicators
used for analysis and modelling:

- Relative Strength Index (RSI, 14 periods)
- Simple Moving Averages (SMA_20, SMA_50)
- Exponential Moving Averages (EMA_12, EMA_26)
- MACD line, Signal line and Histogram
- Bollinger Bands (Upper, Middle, Lower)
- Daily returns (%) and Volume change (%)

The heavy lifting is delegated to the `ta` library so the indicator maths stays
well-tested and consistent, while this module owns the orchestration, column
naming and NaN handling. Each indicator lives in its own small function so it
can be reused or tested in isolation, and :func:`add_technical_indicators` is
the single entry point that composes them all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands

# Columns the indicators are derived from. ``Close`` is required by every
# indicator here; the rest are validated for completeness / future use.
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validates that the input frame has the expected OHLCV columns.

    Parameters:
    - df (pd.DataFrame): The candidate DataFrame to validate.

    Raises:
    - ValueError: If any of the required OHLCV columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}."
        )


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Appends the Relative Strength Index (RSI) as an ``RSI`` column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a ``Close`` column.
    - window (int): Lookback period for the RSI (default 14).

    Returns:
    - pd.DataFrame: The input DataFrame with an added ``RSI`` column.
    """
    df["RSI"] = RSIIndicator(close=df["Close"], window=window).rsi()
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Appends simple and exponential moving averages.

    Adds ``SMA_20``, ``SMA_50``, ``EMA_12`` and ``EMA_26`` columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a ``Close`` column.

    Returns:
    - pd.DataFrame: The input DataFrame with the moving-average columns added.
    """
    close = df["Close"]
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["EMA_12"] = EMAIndicator(close=close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close=close, window=26).ema_indicator()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Appends MACD line, Signal line and Histogram.

    Uses the standard MACD configuration (fast=12, slow=26, signal=9) and adds
    ``MACD``, ``MACD_Signal`` and ``MACD_Hist`` columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a ``Close`` column.

    Returns:
    - pd.DataFrame: The input DataFrame with the MACD columns added.
    """
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> pd.DataFrame:
    """Appends Bollinger Bands.

    Adds ``BB_Upper``, ``BB_Middle`` and ``BB_Lower`` columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a ``Close`` column.
    - window (int): Moving-average window for the middle band (default 20).
    - window_dev (int): Number of standard deviations for the bands (default 2).

    Returns:
    - pd.DataFrame: The input DataFrame with the Bollinger Band columns added.
    """
    bb = BollingerBands(close=df["Close"], window=window, window_dev=window_dev)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Appends daily returns and volume change as percentages.

    Adds ``Daily_Return_Pct`` (percentage change in ``Close``) and
    ``Volume_Change_Pct`` (percentage change in ``Volume``). Infinite values,
    which can arise from a division by a zero-volume day, are converted to NaN
    so they are handled consistently by the downstream NaN cleaning step.

    Parameters:
    - df (pd.DataFrame): DataFrame containing ``Close`` and ``Volume`` columns.

    Returns:
    - pd.DataFrame: The input DataFrame with the return columns added.
    """
    df["Daily_Return_Pct"] = df["Close"].pct_change() * 100
    df["Volume_Change_Pct"] = df["Volume"].pct_change() * 100
    # A prior volume of 0 produces +/-inf; treat those as missing.
    df["Volume_Change_Pct"] = df["Volume_Change_Pct"].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def _handle_nans(df: pd.DataFrame, dropna: bool) -> pd.DataFrame:
    """Cleans NaN values produced by rolling-window indicators.

    Indicators such as SMA_50 have an unavoidable warm-up period at the start of
    the series where no value can be computed. These leading NaNs are dropped
    when ``dropna`` is True. Any remaining interior NaNs (rare, e.g. from source
    data gaps) are back-filled so no NaN "holes" are left mid-series.

    Parameters:
    - df (pd.DataFrame): DataFrame with freshly computed indicator columns.
    - dropna (bool): If True, drop the leading warm-up rows entirely; if False,
      keep every row and only back-fill interior gaps.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    if dropna:
        # Drop the leading warm-up window where long indicators are undefined.
        df = df.dropna().reset_index(drop=True)
    else:
        # Preserve all rows; fill interior gaps from later valid observations.
        df = df.bfill()
    return df


def add_technical_indicators(
    df: pd.DataFrame, dropna: bool = True
) -> pd.DataFrame:
    """Appends a full suite of technical indicators to an OHLCV DataFrame.

    This is the primary entry point of the module. It computes RSI, moving
    averages (SMA/EMA), MACD, Bollinger Bands and percentage-based return
    features, appending each as new columns, then cleans up the NaN values that
    rolling indicators inevitably produce during their warm-up period.

    The input DataFrame is not mutated; a modified copy is returned.

    Parameters:
    - df (pd.DataFrame): Historical daily data with ``Open``, ``High``, ``Low``,
      ``Close`` and ``Volume`` columns. The index is preserved (though it is
      reset when ``dropna`` is True).
    - dropna (bool): If True (default), drop the leading warm-up rows where the
      longest-window indicators are undefined. If False, keep all rows and
      back-fill instead.

    Returns:
    - pd.DataFrame: A copy of the input with the technical indicator columns
      appended and NaNs handled.

    Raises:
    - ValueError: If any required OHLCV column is missing.
    """
    # Empty input has nothing to compute; return an untouched copy.
    if df.empty:
        return df.copy()

    _validate_ohlcv(df)

    # Work on a copy so the caller's DataFrame is never mutated.
    df = df.copy()

    # Ensure numeric dtypes so the indicator maths behaves predictably even if
    # the source data arrived as strings/objects.
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compose the indicators. Order is independent, but returns are placed last
    # so the frame reads price -> indicators -> derived changes.
    df = add_rsi(df)
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_returns(df)

    df = _handle_nans(df, dropna=dropna)
    return df


def _make_sample_data(periods: int = 120) -> pd.DataFrame:
    """Builds a deterministic synthetic OHLCV frame for self-testing.

    Parameters:
    - periods (int): Number of business days to generate.

    Returns:
    - pd.DataFrame: A random-walk OHLCV DataFrame indexed by business day.
    """
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=periods, freq="B")

    # Build a gently trending random walk for Close, then derive OHLC from it.
    close = 100 + np.cumsum(rng.standard_normal(periods))
    spread = np.abs(rng.standard_normal(periods))
    high = close + spread
    low = close - spread
    open_ = close + rng.standard_normal(periods) * 0.5
    volume = rng.integers(1_000_000, 5_000_000, size=periods)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


if __name__ == "__main__":
    print("--- Testing feature_engineering on synthetic market data ---\n")

    sample = _make_sample_data()
    print(f"Input shape:  {sample.shape}  (rows, cols)")
    print(f"Input columns: {list(sample.columns)}\n")

    featured = add_technical_indicators(sample)

    added = [c for c in featured.columns if c not in REQUIRED_COLUMNS]
    print(f"Output shape: {featured.shape}  (rows, cols)")
    print(f"Added feature columns ({len(added)}): {added}\n")

    print("Tail of engineered features:")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(
            featured[
                [
                    "Close",
                    "RSI",
                    "SMA_20",
                    "EMA_12",
                    "MACD",
                    "MACD_Signal",
                    "MACD_Hist",
                    "BB_Upper",
                    "BB_Lower",
                    "Daily_Return_Pct",
                    "Volume_Change_Pct",
                ]
            ].tail()
        )

    # Sanity checks: no NaNs should remain and RSI must stay within [0, 100].
    remaining_nans = int(featured.isna().sum().sum())
    rsi_in_range = featured["RSI"].between(0, 100).all()
    print(f"\nRemaining NaNs: {remaining_nans}")
    print(f"RSI within [0, 100]: {bool(rsi_in_range)}")
    print("\n--- Self-test complete ---")
