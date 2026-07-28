# Data loader script for AI Stock Dashboard
# This module will handle data loading and preprocessing tasks
import datetime
import os
import pandas as pd
import yfinance as yf

# Directory where persisted data files are written
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_stock_data(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetches historical daily closing prices and trading volume for a given ticker symbol.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing 'Open', 'High', 'Low', 'Close' and
      'Volume' columns, indexed by Date.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    try:
        # Initialize the Ticker object
        stock = yf.Ticker(ticker)

        # Fetch historical data for the specified date range
        df = stock.history(start=start_date, end=end_date, interval="1d")

        # Check if the returned DataFrame is empty (e.g., invalid ticker or no data)
        if df.empty:
            print(
                f"Warning: No data found for ticker '{ticker}'. Please check the symbol or date range."
            )
            return pd.DataFrame()

        # Keep the full OHLCV set so downstream feature engineering (e.g.
        # technical indicators) has access to Open/High/Low as well.
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        df_filtered = df[required_columns]

        return df_filtered

    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and validates the structure of fetched stock data.

    Performs the following steps:
    - Converts the index to a timezone-naive datetime format.
    - Handles missing values in the price columns ('Open', 'High', 'Low',
      'Close' are forward-filled) and 'Volume' (filled with 0).
    - Drops any rows that remain NaN in 'Close' after filling.
    - Ensures the output columns are 'Date', 'Open', 'High', 'Low', 'Close'
      and 'Volume'.

    Parameters:
    - df (pd.DataFrame): Raw DataFrame with OHLCV columns, indexed by Date.

    Returns:
    - pd.DataFrame: A cleaned DataFrame with 'Date' plus the OHLCV columns.
    """
    # Guard against empty input to avoid downstream errors
    if df.empty:
        return df

    # Work on a copy so the caller's DataFrame is not mutated
    df = df.copy()

    price_columns = ["Open", "High", "Low", "Close"]

    # 1. Ensure the index is a clean, timezone-naive datetime
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        # Strip timezone-aware offsets to make the index timezone-naive
        df.index = df.index.tz_localize(None)

    # 2. Handle missing values
    # Forward-fill price columns so gaps carry the last known price
    df[price_columns] = df[price_columns].ffill()
    # Missing volume is treated as zero trading activity
    df["Volume"] = df["Volume"].fillna(0)
    # Drop any remaining NaN rows (e.g. leading NaNs that forward-fill can't cover)
    df = df.dropna(subset=["Close"])

    # 3. Ensure columns are ordered and named consistently
    df.index.name = "Date"
    df = df[price_columns + ["Volume"]].reset_index()

    return df


def save_data_locally(
    df: pd.DataFrame, ticker: str, storage_type: str = "csv"
) -> str:
    """Persists a cleaned stock DataFrame to local storage.

    Parameters:
    - df (pd.DataFrame): Cleaned DataFrame (expects a 'Date' column, as produced by clean_stock_data).
    - ticker (str): The stock symbol, used to name the output file (e.g., 'AAPL').
    - storage_type (str): The storage format. Currently only 'csv' is supported.

    Returns:
    - str: The path to the written file, or an empty string if nothing was saved.
    """
    # Nothing to persist for empty results
    if df.empty:
        print(f"Warning: No data to save for '{ticker}'. Skipping write.")
        return ""

    if storage_type == "csv":
        # Ensure the target directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        file_path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")

        # clean_stock_data returns 'Date' as a regular column, so no index is written.
        # index=False keeps the CSV free of an extra unnamed index column while still
        # explicitly persisting the date as a 'Date' column.
        df.to_csv(file_path, index=False)

        print(f"Saved {len(df)} rows for '{ticker}' to {file_path}")
        return file_path

    raise ValueError(
        f"Unsupported storage_type '{storage_type}'. Supported types: 'csv'."
    )


def run_pipeline(
    ticker: str, start_date: str, end_date: str, storage_type: str = "csv"
) -> pd.DataFrame:
    """Runs the full data pipeline: fetch, clean, and save.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - storage_type (str): The storage format passed to save_data_locally.

    Returns:
    - pd.DataFrame: The cleaned DataFrame that was persisted (empty if fetching failed).
    """
    raw_df = fetch_stock_data(ticker, start_date, end_date)
    clean_df = clean_stock_data(raw_df)
    save_data_locally(clean_df, ticker, storage_type)
    return clean_df


if __name__ == "__main__":
    # Define a 2-year lookback window for testing
    today = datetime.date.today()
    two_years_ago = today - datetime.timedelta(days=2 * 365)

    start_str = two_years_ago.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    print("--- Running Data Pipeline (fetch -> clean -> save) --- \n")

    # Run the full pipeline for each ticker so running this file populates data/
    for ticker in ["AAPL", "RELIANCE.NS"]:
        data = run_pipeline(ticker, start_str, end_str)

        if not data.empty:
            print(f"\nSuccessfully processed {ticker} data!")
            print(data.head())
            print("-" * 50)
        else:
            print(f"Failed to retrieve data for {ticker}.\n")