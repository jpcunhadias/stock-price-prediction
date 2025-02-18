"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

import logging
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_lstm_model(X_train: np.ndarray, params: dict) -> tf.keras.Model:
    """
    Builds an LSTM model with the given hyperparameters.

    Args:
        X_train (np.ndarray): Training features.
        params (dict): Model hyperparameters.

    Returns:
        tf.keras.Model: LSTM model.
    """
    logger.info("Building LSTM model with params: %s", params)

    model = Sequential(
        [
            LSTM(
                params["lstm_units"],
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            LSTM(params["lstm_units"], return_sequences=False),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mean_squared_error",
    )

    logger.info("LSTM model built successfully.")
    return model


def train_lstm_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
) -> tf.keras.Model:
    """
    Trains an LSTM model to predict stock prices.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        params (dict): Model hyperparameters.

    Returns:
        tf.keras.Model: Trained LSTM model.
    """
    logger.info(
        "Training model with batch size: %d and epochs: %d",
        params["batch_size"],
        params["epochs"],
    )

    model = build_lstm_model(X_train, params)

    early_stop = EarlyStopping(**params["early_stopping"])
    history = model.fit(
        X_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    logger.info("Training completed.")
    return model


def evaluate_model(
        model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray, scaler
) -> Dict[str, float]:
    """
    Evaluates the trained model using MAE, RMSE, and MAPE after denormalization.

    Args:
        model (tf.keras.Model): Trained LSTM model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        scaler: Scaler object used for normalization.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    logger.info("Evaluating model performance.")

    y_pred = model.predict(X_val)
    logger.info("Model prediction completed.")

    # Denormalizing predictions and actual values
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_denorm = scaler.inverse_transform(y_val.reshape(-1, 1))

    mae = mean_absolute_error(y_val_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_val_denorm, y_pred_denorm))

    # Avoid division by zero in MAPE
    y_val_denorm = np.where(y_val_denorm == 0, 1e-6, y_val_denorm)
    mape = np.mean(np.abs((y_val_denorm - y_pred_denorm) / y_val_denorm)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    logger.info("Evaluation metrics: %s", metrics)

    return metrics


def save_model(model: tf.keras.Model, s3_bucket: str, model_key: str):
    """
    Saves the trained LSTM model to an S3 bucket instead of local storage.

    Args:
        model (tf.keras.Model): Trained model.
        s3_bucket (str): The name of the S3 bucket.
        model_key (str): The file path inside the bucket.
    """
    BASE_DIR = Path(__file__).resolve().parents[4]
    full_path = f"{BASE_DIR}/{s3_bucket}/{model_key}"
    print(full_path)
    model.save(full_path)
