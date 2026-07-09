# Data loader script for AI Stock Dashboard
# This module will handle data loading and preprocessing tasks
# Test
import datetime
import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetches historical daily closing prices and trading volume for a given ticker symbol.

    Parameters:
    - ticker (str): The stock symbol (e.g., 'AAPL', 'RELIANCE.NS').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing 'Close' and 'Volume' columns, indexed by Date.
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

        # Extract only the required columns: 'Close' and 'Volume'
        # Using a subset list ensures we only keep what is needed for the dashboard foundations
        required_columns = ["Close", "Volume"]
        df_filtered = df[required_columns]

        return df_filtered

    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Define a 2-year lookback window for testing
    today = datetime.date.today()
    two_years_ago = today - datetime.timedelta(days=2 * 365)

    start_str = two_years_ago.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    print("--- Testing yfinance Data Fetching Pipeline --- \n")

    # Test 1: US Stock (Apple)
    us_ticker = "AAPL"
    aapl_data = fetch_stock_data(us_ticker, start_str, end_str)

    if not aapl_data.empty:
        print(f"\nSuccessfully fetched {us_ticker} data!")
        print(aapl_data.head())
        print("-" * 50)
    else:
        print(f"Failed to retrieve data for {us_ticker}.\n")

    # Test 2: Indian Stock (Reliance Industries)
    in_ticker = "RELIANCE.NS"
    reliance_data = fetch_stock_data(in_ticker, start_str, end_str)

    if not reliance_data.empty:
        print(f"\nSuccessfully fetched {in_ticker} data!")
        print(reliance_data.head())
        print("-" * 50)
    else:
        print(f"Failed to retrieve data for {in_ticker}.\n")