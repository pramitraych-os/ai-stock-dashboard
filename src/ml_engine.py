"""Machine-learning engine for the AI Stock Dashboard.

This module turns the technical features produced by
:mod:`feature_engineering` into a single, normalized directional signal:

1. A binary target is built by looking ``HORIZON_DAYS`` bars into the future:
   ``1`` if the close price is higher than today's close, else ``0``.
2. A :class:`~sklearn.ensemble.RandomForestClassifier` is trained on the
   technical features inside a scikit-learn :class:`~sklearn.pipeline.Pipeline`
   (imputer -> scaler -> forest), so preprocessing always travels with the
   model.
3. The model's probability ``p`` of the price rising is rescaled into an
   ``ml_score`` in ``[-1.0, +1.0]`` via ``2 * (p - 0.5)``, which lets the score
   be blended with other bounded indicator scores in the dashboard.

Because this is time-series data, the train/test split is chronological (no
shuffling) and the target rows whose future is unknown are dropped, so the
model is never trained on information that leaks from the future.

Trained pipelines are persisted with :mod:`joblib` under ``models/`` at the
project root.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Directory where trained models are persisted (sibling of ``data/``).
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

# How many trading days ahead the target looks. A 5-day horizon corresponds to
# roughly one trading week.
HORIZON_DAYS = 5

# Name of the generated target column.
TARGET_COLUMN = "Target"

# Technical features the model is trained on. These are produced by
# ``feature_engineering.add_technical_indicators``. Raw price levels (Close,
# SMA, Bollinger Bands, ...) are deliberately expressed relative to price by
# ``build_feature_matrix`` so the model learns scale-free patterns that
# generalise across tickers and across time.
BASE_FEATURES = (
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Daily_Return_Pct",
    "Volume_Change_Pct",
)

# Ratio features derived as ``Close / <column> - 1``, i.e. how far price sits
# from each reference level in fractional terms.
RATIO_FEATURES = {
    "Close_vs_SMA_20": "SMA_20",
    "Close_vs_SMA_50": "SMA_50",
    "Close_vs_EMA_12": "EMA_12",
    "Close_vs_EMA_26": "EMA_26",
    "Close_vs_BB_Upper": "BB_Upper",
    "Close_vs_BB_Lower": "BB_Lower",
}

# The full, ordered feature list the pipeline expects at fit and predict time.
FEATURE_COLUMNS = list(BASE_FEATURES) + list(RATIO_FEATURES)


def create_target(
    df: pd.DataFrame, horizon: int = HORIZON_DAYS
) -> pd.DataFrame:
    """Appends the binary directional target column.

    The target is ``1`` when the close price ``horizon`` bars ahead is strictly
    greater than the current close, and ``0`` otherwise. The final ``horizon``
    rows have no known future and are therefore left as NaN for the caller to
    drop (see :func:`prepare_training_data`).

    Parameters:
    - df (pd.DataFrame): Feature frame containing a ``Close`` column.
    - horizon (int): Number of trading days to look ahead (default 5).

    Returns:
    - pd.DataFrame: A copy of the input with a ``Target`` column added.

    Raises:
    - ValueError: If ``Close`` is missing or ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    # Comparison against NaN yields False, so mask the unknown tail explicitly
    # rather than silently labelling it as "price fell".
    target = (future_close > df["Close"]).astype(float)
    df[TARGET_COLUMN] = target.where(future_close.notna())
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Derives the model's feature matrix from a technical-indicator frame.

    Absolute price levels are converted into price-relative ratios so the
    features are scale-free: a stock trading at 3,000 and one trading at 30
    produce comparable inputs, and the model does not have to relearn its rules
    every time the price level drifts.

    Parameters:
    - df (pd.DataFrame): Output of
      ``feature_engineering.add_technical_indicators``, containing ``Close``
      plus the indicator columns.

    Returns:
    - pd.DataFrame: A frame with exactly the :data:`FEATURE_COLUMNS`, in order,
      sharing the input's index.

    Raises:
    - ValueError: If any required source column is missing.
    """
    required = {"Close", *BASE_FEATURES, *RATIO_FEATURES.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required feature column(s): {missing}."
        )

    close = pd.to_numeric(df["Close"], errors="coerce")

    features = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        features[col] = pd.to_numeric(df[col], errors="coerce")

    for name, reference in RATIO_FEATURES.items():
        ref = pd.to_numeric(df[reference], errors="coerce")
        # Guard against a zero reference level, which would produce +/-inf.
        features[name] = (close / ref.replace(0, np.nan)) - 1.0

    # Any remaining inf (e.g. from an extreme volume jump) is treated as
    # missing so the pipeline's imputer handles it consistently.
    features = features.replace([np.inf, -np.inf], np.nan)
    return features[FEATURE_COLUMNS]


def prepare_training_data(
    df: pd.DataFrame, horizon: int = HORIZON_DAYS
) -> tuple[pd.DataFrame, pd.Series]:
    """Builds the aligned ``(X, y)`` pair used for training.

    Rows whose future outcome is unknown (the last ``horizon`` bars) are
    dropped. Feature NaNs are left in place for the pipeline's imputer.

    Parameters:
    - df (pd.DataFrame): Technical-indicator frame with a ``Close`` column.
    - horizon (int): Number of trading days to look ahead (default 5).

    Returns:
    - tuple[pd.DataFrame, pd.Series]: The feature matrix and the integer target
      series, sharing the same index.
    """
    labelled = create_target(df, horizon=horizon)
    features = build_feature_matrix(labelled)

    # Keep only rows with a known outcome; features may still contain NaNs.
    known = labelled[TARGET_COLUMN].notna()
    X = features.loc[known]
    y = labelled.loc[known, TARGET_COLUMN].astype(int)
    return X, y


def _build_pipeline(
    n_estimators: int, max_depth: int | None, random_state: int
) -> Pipeline:
    """Assembles the untrained preprocessing + classifier pipeline.

    The imputer keeps indicator warm-up NaNs from crashing prediction. The
    scaler is not required by a random forest but makes the saved artefact
    reusable if the classifier is ever swapped for a scale-sensitive model.

    Parameters:
    - n_estimators (int): Number of trees in the forest.
    - max_depth (int | None): Maximum tree depth; None grows trees fully.
    - random_state (int): Seed for reproducible fits.

    Returns:
    - Pipeline: The unfitted scikit-learn pipeline.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=5,
                    # Financial up/down classes are rarely balanced; reweight so
                    # the forest does not simply learn the majority direction.
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int | None = 8,
    random_state: int = 42,
    verbose: bool = True,
) -> Pipeline:
    """Trains the Random Forest pipeline on historical technical features.

    The split is chronological rather than random: the earliest
    ``1 - test_size`` fraction of rows trains the model and the most recent
    rows act as an out-of-sample holdout. This mirrors live use, where only the
    past is available. After reporting holdout metrics the pipeline is refit on
    the full history so the returned model uses every available observation.

    Parameters:
    - df (pd.DataFrame): Technical-indicator frame (output of
      ``feature_engineering.add_technical_indicators``) with a ``Close`` column.
    - horizon (int): Trading days ahead used to define the target (default 5).
    - test_size (float): Fraction of the most recent rows held out for
      evaluation. Set to 0 to skip evaluation entirely.
    - n_estimators (int): Number of trees in the forest (default 300).
    - max_depth (int | None): Maximum tree depth (default 8) to limit
      overfitting on noisy market data.
    - random_state (int): Seed for reproducible fits.
    - verbose (bool): If True, print holdout metrics and top feature
      importances.

    Returns:
    - Pipeline: The fitted pipeline, ready for :func:`get_ml_score`.

    Raises:
    - ValueError: If there are too few labelled rows, or the target contains a
      single class (the model could not learn a direction).
    """
    X, y = prepare_training_data(df, horizon=horizon)

    # A forest needs a meaningful sample; anything smaller is noise.
    min_rows = 50
    if len(X) < min_rows:
        raise ValueError(
            f"Not enough labelled rows to train: got {len(X)}, need at least "
            f"{min_rows}. Fetch a longer history."
        )
    if y.nunique() < 2:
        raise ValueError(
            "Target contains a single class; the price never changed direction "
            "over the chosen horizon. Try a different horizon or history."
        )

    pipeline = _build_pipeline(n_estimators, max_depth, random_state)

    if 0 < test_size < 1:
        split_at = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

        # Only evaluate when the chronological split leaves usable data on both
        # sides and the training slice still has both classes.
        if len(X_test) > 0 and y_train.nunique() == 2:
            pipeline.fit(X_train, y_train)
            if verbose:
                _report_metrics(pipeline, X_test, y_test)

    # Refit on the complete history so the deployed model sees the latest bars.
    pipeline.fit(X, y)

    if verbose:
        print(f"Trained on {len(X)} rows, {len(FEATURE_COLUMNS)} features.")
        print(f"Class balance (1 = price rose in {horizon}d): "
              f"{y.mean():.1%} up / {1 - y.mean():.1%} down")
        _report_importances(pipeline)

    return pipeline


def _report_metrics(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Prints out-of-sample metrics for the holdout slice.

    Parameters:
    - pipeline (Pipeline): Pipeline fitted on the training slice only.
    - X_test (pd.DataFrame): Holdout features.
    - y_test (pd.Series): Holdout targets.
    """
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Holdout rows: {len(X_test)}  |  Accuracy: {accuracy:.3f}")

    # ROC-AUC is undefined when the holdout window contains a single class.
    if y_test.nunique() == 2:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        print(f"Holdout ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    else:
        print("Holdout ROC-AUC: n/a (holdout window has a single class)")


def _report_importances(pipeline: Pipeline, top_n: int = 5) -> None:
    """Prints the most influential features of a fitted pipeline.

    Parameters:
    - pipeline (Pipeline): A fitted pipeline whose classifier exposes
      ``feature_importances_``.
    - top_n (int): How many features to list.
    """
    importances = pd.Series(
        pipeline.named_steps["classifier"].feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print(f"Top {top_n} features:")
    for name, value in importances.head(top_n).items():
        print(f"  {name:<20} {value:.4f}")


def get_prediction_probability(pipeline: Pipeline, latest_features_df: pd.DataFrame) -> float:
    """Returns the model's probability that the price will rise.

    Parameters:
    - pipeline (Pipeline): A fitted pipeline from :func:`train_model` or
      :func:`load_model`.
    - latest_features_df (pd.DataFrame): One or more rows of technical
      indicators (as produced by
      ``feature_engineering.add_technical_indicators``) or a frame that already
      holds the :data:`FEATURE_COLUMNS`. When several rows are supplied, the
      last (most recent) one is scored.

    Returns:
    - float: The probability ``p`` of the "price rose" class, in ``[0.0, 1.0]``.

    Raises:
    - ValueError: If the input frame is empty or lacks the required columns.
    """
    if latest_features_df is None or latest_features_df.empty:
        raise ValueError("latest_features_df is empty; nothing to score.")

    # Accept either a raw indicator frame or a pre-built feature matrix.
    if set(FEATURE_COLUMNS).issubset(latest_features_df.columns):
        features = latest_features_df[FEATURE_COLUMNS]
    else:
        features = build_feature_matrix(latest_features_df)

    # Score the most recent row only, keeping it 2-D for scikit-learn.
    latest_row = features.iloc[[-1]]

    probabilities = pipeline.predict_proba(latest_row)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)

    # A degenerate single-class fit has no column for the missing class.
    if 1 not in classes:
        return 0.0
    return float(probabilities[classes.index(1)])


def get_ml_score(model: Pipeline, latest_features_df: pd.DataFrame) -> float:
    """Converts the model's rise probability into a normalized ML score.

    The probability ``p`` in ``[0, 1]`` is mapped onto ``[-1, +1]`` with
    ``2 * (p - 0.5)``: ``+1`` is maximum conviction that the price rises,
    ``-1`` maximum conviction that it falls, and ``0`` a coin flip. This puts
    the model on the same scale as the dashboard's other indicator scores.

    Parameters:
    - model (Pipeline): A fitted pipeline from :func:`train_model` or
      :func:`load_model`.
    - latest_features_df (pd.DataFrame): The latest technical-indicator row(s);
      the most recent row is scored.

    Returns:
    - float: The ``ml_score``, clipped to ``[-1.0, +1.0]``.
    """
    probability = get_prediction_probability(model, latest_features_df)
    ml_score = 2.0 * (probability - 0.5)
    # Clip defensively so downstream consumers can rely on the bounds.
    return float(np.clip(ml_score, -1.0, 1.0))


def get_model_path(ticker: str, model_dir: str = MODEL_DIR) -> str:
    """Builds the on-disk path for a ticker's persisted model.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - model_dir (str): Directory holding the model files.

    Returns:
    - str: Full path to the ``.joblib`` file for this ticker.
    """
    # Normalise symbols like 'RELIANCE.NS' into a safe, flat filename.
    safe_ticker = ticker.replace(".", "_").replace("/", "_").upper()
    return os.path.join(model_dir, f"{safe_ticker}_rf.joblib")


def save_model(model: Pipeline, ticker: str, model_dir: str = MODEL_DIR) -> str:
    """Persists a trained pipeline to disk with joblib.

    Parameters:
    - model (Pipeline): The fitted pipeline to persist.
    - ticker (str): The stock symbol, used to name the output file.
    - model_dir (str): Directory to write into; created if absent.

    Returns:
    - str: The path to the written file.
    """
    os.makedirs(model_dir, exist_ok=True)
    file_path = get_model_path(ticker, model_dir)
    joblib.dump(model, file_path)
    print(f"Saved model for '{ticker}' to {file_path}")
    return file_path


def load_model(ticker: str, model_dir: str = MODEL_DIR) -> Pipeline:
    """Loads a previously saved pipeline from disk.

    Parameters:
    - ticker (str): The stock symbol whose model should be loaded.
    - model_dir (str): Directory holding the model files.

    Returns:
    - Pipeline: The deserialized, fitted pipeline.

    Raises:
    - FileNotFoundError: If no saved model exists for the ticker.
    """
    file_path = get_model_path(ticker, model_dir)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No saved model for '{ticker}' at {file_path}. "
            f"Train one first with train_model() and save it with save_model()."
        )
    return joblib.load(file_path)


def train_and_save(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = HORIZON_DAYS,
    model_dir: str = MODEL_DIR,
    **train_kwargs,
) -> tuple[Pipeline, str]:
    """Convenience wrapper: train on a frame and persist the result.

    Parameters:
    - df (pd.DataFrame): Technical-indicator frame with a ``Close`` column.
    - ticker (str): The stock symbol, used to name the model file.
    - horizon (int): Trading days ahead used to define the target.
    - model_dir (str): Directory to write the model into.
    - **train_kwargs: Extra keyword arguments forwarded to :func:`train_model`.

    Returns:
    - tuple[Pipeline, str]: The fitted pipeline and the path it was saved to.
    """
    model = train_model(df, horizon=horizon, **train_kwargs)
    path = save_model(model, ticker, model_dir)
    return model, path


def _load_local_ohlcv(ticker: str) -> pd.DataFrame:
    """Loads a locally persisted OHLCV CSV written by ``data_loader``.

    Parameters:
    - ticker (str): The stock symbol whose CSV should be read.

    Returns:
    - pd.DataFrame: The OHLCV frame indexed by Date, or an empty frame if the
      file is missing or predates the full-OHLCV format.
    """
    from data_loader import DATA_DIR
    from feature_engineering import REQUIRED_COLUMNS

    file_path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Older CSVs stored only Close/Volume; they can't feed the indicator suite.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"Local CSV for {ticker} is missing {missing}; re-run data_loader.py.")
        return pd.DataFrame()
    return df


if __name__ == "__main__":
    from feature_engineering import _make_sample_data, add_technical_indicators

    print("--- Testing ml_engine ---\n")

    # Prefer real local data (written by data_loader.py); fall back to the
    # synthetic random walk so this self-test runs on a fresh checkout too.
    TICKER = "AAPL"
    raw = _load_local_ohlcv(TICKER)
    if raw.empty:
        print("No usable local CSV; falling back to synthetic data.\n")
        TICKER = "SYNTH"
        raw = _make_sample_data(periods=600)
    else:
        print(f"Loaded {len(raw)} rows of local data for {TICKER}.\n")

    featured = add_technical_indicators(raw)
    print(f"Feature frame: {featured.shape} (rows, cols)\n")

    print("Training Random Forest pipeline...")
    model, model_path = train_and_save(featured, TICKER, horizon=HORIZON_DAYS)

    print("\nScoring the latest bar:")
    probability = get_prediction_probability(model, featured)
    score = get_ml_score(model, featured)
    print(f"  P(price rises in {HORIZON_DAYS}d) = {probability:.4f}")
    print(f"  ml_score                    = {score:+.4f}")

    # Round-trip the artefact to confirm save/load produces identical scores.
    reloaded = load_model(TICKER)
    reloaded_score = get_ml_score(reloaded, featured)
    print(f"\nReloaded model ml_score       = {reloaded_score:+.4f}")
    print(f"Round-trip identical: {np.isclose(score, reloaded_score)}")

    # Sanity check: every historical bar must score inside the declared bounds.
    all_scores = [
        get_ml_score(model, featured.iloc[: i + 1]) for i in range(len(featured) - 20, len(featured))
    ]
    print(
        f"Last 20 scores within [-1, 1]: "
        f"{all(-1.0 <= s <= 1.0 for s in all_scores)} "
        f"(min {min(all_scores):+.3f}, max {max(all_scores):+.3f})"
    )
    print("\n--- Self-test complete ---")
