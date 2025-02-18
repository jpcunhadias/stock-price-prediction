"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def drop_invalid_first_row(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the first row if it contains non-numeric values."""
    if df.iloc[0].str.contains("[A-Za-z]", regex=True).any():
        df = df.iloc[1:].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The complete DataFrame.
        test_size (float): Proportion of data to reserve for validation (e.g., 0.2 for 20%).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def normalize_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature: str
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Normalizes the specified feature in the training and validation DataFrames using MinMaxScaler.
    The scaler is fit only on the training data to avoid data leakage.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        feature (str): The name of the feature to be normalized.

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
            - Normalized training data array.
            - Normalized validation data array.
            - The fitted scaler.
    """
    scaler = MinMaxScaler()

    # Fit only on the training data
    train_scaled = scaler.fit_transform(train_df[[feature]].values)

    # Transform the validation data with the same scaler
    val_scaled = scaler.transform(val_df[[feature]].values)

    return train_scaled, val_scaled, scaler


def save_scaled_data(scaled_array: np.ndarray) -> pd.DataFrame:
    """
    Saves the scaled data into a DataFrame.

    Args:
        scaled_array (np.ndarray): The scaled data array.

    Returns:
        pd.DataFrame: A DataFrame containing the scaled data.
    """
    return pd.DataFrame(scaled_array, columns=["scaled_price"])


def create_sequences(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for time series prediction.

    Args:
        data (np.ndarray): Array of data to create sequences from.
        window_size (int): The size of the window to create sequences.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences (X) and the corresponding target values (y).
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y
