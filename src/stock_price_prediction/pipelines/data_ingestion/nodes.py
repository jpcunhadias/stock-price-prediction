"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.11
"""

import logging
from datetime import datetime
import pandas as pd
import yfinance as yf


logging.basicConfig(level=logging.INFO)


def fetch_stock_prices(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads the stock prices for a specific symbol within the indicated period.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL" for Apple).
        start_date (str): Start date in the format "YYYY-MM-DD".
        end_date (str): End date in the format "YYYY-MM-DD".

    Returns:
        pd.DataFrame: DataFrame containing the historical stock prices.
    """
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(
        end_date, "%Y-%m-%d"
    ):
        raise ValueError("Start date should be before end date")

    logging.info(f"Fetching stock prices for {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data was fetched for symbol: {symbol}")

    df.reset_index(inplace=True)

    return df
