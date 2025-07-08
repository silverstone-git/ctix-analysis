import pandas as pd
import json
from pandas.core.indexes.interval import Timestamp
import requests
from typing import Optional
import yfinance as yf
import pytz
from smolagents import tool
import os

# Tool: Enhanced Yahoo Finance Data Fetcher with Alpha Vantage Fallback
@tool
def get_yahoo_finance_data(ticker: str, start_date: str, end_date: str) -> str:
    """
    Fetches historical stock data for a given ticker from Yahoo Finance.
    Falls back to Alpha Vantage API if >30% of entries are NaN or <70% expected date coverage.

    Args:
        ticker: Stock symbol (e.g., 'IBM', 'RELIANCE.BSE')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        JSON string containing the stock data or error message
        JSON Format is: {Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits}
    """
    # try:
    print(f"Fetching Yahoo Finance data for {ticker} from {start_date} to {end_date}...")

    start_date_aware= pd.to_datetime(start_date).tz_localize(pytz.utc)
    end_date_aware = pd.to_datetime(end_date).tz_localize(pytz.utc)

    # Primary data fetch from Yahoo Finance
    yahoo_df = _fetch_yahoo_finance_data(ticker, start_date_aware, end_date_aware)

    # Check data quality - ensure _check_data_quality handles timezone-aware dates
    data_quality_issues = _check_data_quality(yahoo_df, start_date_aware, end_date_aware)

    if data_quality_issues:
        print(f"Data quality issues detected: {data_quality_issues}")
        print("Falling back to Alpha Vantage API...")

        # Fallback to Alpha Vantage
        # Ensure _fetch_alpha_vantage_data returns a DataFrame with a timezone-aware index
        alpha_vantage_df = _fetch_alpha_vantage_data(ticker)

        if alpha_vantage_df is not None:
            # Filter Alpha Vantage data to requested date range
            # Ensure _filter_date_range handles timezone-aware dates
            filtered_df = _filter_date_range(alpha_vantage_df, start_date_aware, end_date_aware)
            print("Successfully retrieved data from Alpha Vantage")
            return filtered_df.to_json()
        else:
            print("Alpha Vantage fallback failed, returning Yahoo Finance data with warnings")
            result = {
                "data": json.loads(yahoo_df.to_json()),
                "warnings": data_quality_issues,
                "source": "yahoo_finance_with_issues"
            }
            return json.dumps(result)

    print("Yahoo Finance data quality is acceptable")
    return yahoo_df.to_json()

    # except Exception as e:
    #     return json.dumps({"error": f"Failed to fetch stock data: {e}"})


def _fetch_yahoo_finance_data(ticker: str, start_date: Timestamp, end_date: Timestamp) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT', 'RELIANCE.NS').
        start_date (str): The start date for data in 'YYYY-MM-DD' format.
        end_date (str): The end date for data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data (Open, High, Low, Close, Volume, etc.),
                      or an empty DataFrame if no data is found or an error occurs.
    """
    try:
        # Create a Ticker object for the given ticker symbol
        stock_ticker = yf.Ticker(ticker)

        # Fetch historical data
        # interval: '1d' for daily, '1wk' for weekly, '1mo' for monthly. Default is '1d'.
        # auto_adjust: If True, automatically adjusts Close prices for splits and dividends.
        #              If False, you get 'Adj Close' column.
        historical_data = stock_ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)

        if historical_data.empty:
            print(f"Warning: No data found for ticker '{ticker}' between {start_date} and {end_date}. Check ticker or date range.")
            return pd.DataFrame() # Return empty DataFrame

        # Ensure index is datetime and has a name (optional but good practice)
        historical_data.index = pd.to_datetime(historical_data.index)
        historical_data.index.name = 'Date'

        # Optional: Rename columns to match common conventions (e.g., 'Adj Close' to 'Close')
        # if 'Adj Close' in historical_data.columns:
        #     historical_data['Close'] = historical_data['Adj Close']
        #     historical_data = historical_data.drop(columns=['Adj Close'])
        # Or, if auto_adjust=True, 'Adj Close' becomes 'Close' directly.

        # You might want to explicitly select or reorder columns if your downstream
        # agents expect a specific subset or order.
        # E.g., return historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]

        return historical_data

    except Exception as e:
        print(f"Error fetching data for '{ticker}': {e}")
        return pd.DataFrame() # Return an empty DataFrame on error


def _check_data_quality(df: pd.DataFrame, start_date: Timestamp, end_date: Timestamp) -> list:
    """
    Check data quality and return list of issues found.

    Returns:
        List of data quality issues (empty list if no issues)
    """
    issues = []

    # Calculate expected number of trading days (rough estimate: ~252 trading days per year)
    start_dt = start_date.to_pydatetime()
    end_dt = end_date.to_pydatetime()
    total_days = (end_dt - start_dt).days + 1
    expected_trading_days = total_days * 0.7  # Rough estimate accounting for weekends/holidays

    # Check 1: NaN percentage
    total_values = df.size
    nan_count = df.isnull().sum().sum()
    nan_percentage = (nan_count / total_values) * 100

    if nan_percentage > 30:
        issues.append(f"High NaN percentage: {nan_percentage:.1f}%")

    # Check 2: Date coverage
    actual_data_points = len(df.dropna())
    coverage_percentage = (actual_data_points / expected_trading_days) * 100

    if coverage_percentage < 70:
        issues.append(f"Low date coverage: {coverage_percentage:.1f}% of expected trading days")

    # Check 3: Recent data availability
    if not df.tail(5).dropna().empty:
        latest_data_date = df.dropna().index[-1]
        days_since_latest = (end_dt - latest_data_date).days
        if days_since_latest > 7:
            issues.append(f"Stale data: latest data is {days_since_latest} days older than end limit")

    return issues


def _fetch_alpha_vantage_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch data from Alpha Vantage API.

    Returns:
        DataFrame with Alpha Vantage data or None if failed
    """
    try:
        # Construct API URL
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'full',  # Get full historical data
            'datatype': 'json',
            'apikey': api_key
        }

        print(f"Calling Alpha Vantage API for {ticker}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            print(f"Alpha Vantage API error: {data['Error Message']}")
            return None

        if "Note" in data:
            print(f"Alpha Vantage API note: {data['Note']}")
            return None

        # Parse the time series data
        if "Time Series (Daily)" not in data:
            print("No time series data found in Alpha Vantage response")
            return None

        time_series = data["Time Series (Daily)"]

        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })

        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        print(f"Successfully fetched {len(df)} days of data from Alpha Vantage")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request error when calling Alpha Vantage: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error parsing Alpha Vantage data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error with Alpha Vantage: {e}")
        return None


def _filter_date_range(df: pd.DataFrame, start_date: Timestamp, end_date: Timestamp) -> pd.DataFrame:
    """Filter DataFrame to specified date range."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter the dataframe to the requested date range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    filtered_df = df.loc[mask]

    return filtered_df


if __name__ == "__main__":

    # Example usage:
    result = get_yahoo_finance_data(
        ticker="TCS",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(result)
