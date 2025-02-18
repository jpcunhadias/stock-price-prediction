import os
import pickle

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

from config import bucket_name, MODEL_PATH, SCALER_PATH, local_scaler_path, local_model_path

TRAINED_SYMBOL = "PETR4.SA"

s3_client = boto3.client("s3")

try:
    s3_client.download_file(bucket_name, MODEL_PATH, local_model_path)
    print(f"Downloaded model to {local_model_path}")
except Exception as e:
    print(f"Error downloading model file: {e}")
    raise

# Download the scaler file from S3
try:
    s3_client.download_file(bucket_name, SCALER_PATH, local_scaler_path)
    print(f"Downloaded scaler to {local_scaler_path}")
except Exception as e:
    print(f"Error downloading scaler file: {e}")
    raise

# Load the model using Keras
try:
    model = tf.keras.models.load_model(local_model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load the scaler using pickle
try:
    with open(local_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    raise

# Now that the model and scaler are loaded, remove the local files
try:
    os.remove(local_model_path)
    print(f"Removed local model file: {local_model_path}")
except Exception as e:
    print(f"Error removing model file: {e}")

try:
    os.remove(local_scaler_path)
    print(f"Removed local scaler file: {local_scaler_path}")
except Exception as e:
    print(f"Error removing scaler file: {e}")


def preprocess_input(data: list[float]) -> np.ndarray:
    """
    Preprocesses the input data for the model.
    Args:
        data: list[float]: The last 60 trading days of stock prices.

    Returns:
        np.ndarray: The preprocessed data.
    """
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    return np.reshape(scaled_data, (1, len(data), 1))


def predict_price(data: list[float]) -> float:
    """
    Predicts the stock price for the next trading day.
    Args:
        data: list[float]: The last 60 trading days of stock prices.

    Returns:
        float: The predicted stock price.
    """
    processed_data = preprocess_input(data)

    prediction = model.predict(processed_data)
    return float(scaler.inverse_transform(prediction)[0][0])


def get_last_60_days_prices() -> list[float]:
    """
    Fetches the last 60 trading days of stock prices for the trained stock.

    Returns:
        List[float]: The last 60 closing prices.
    """
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime(
        "%Y-%m-%d"
    )

    df = yf.download(TRAINED_SYMBOL, start=start_date, end=end_date)

    if len(df) < 60:
        raise ValueError(
            f"Not enough trading days for {TRAINED_SYMBOL} to fetch 60 days"
        )

    return df["Close"].tail(60).to_numpy().reshape(-1).tolist()
