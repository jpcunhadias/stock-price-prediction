"""
Hyperparameter Tuning Pipeline
-------------------------------

This module defines a function to perform hyperparameter tuning for an LSTM-based model
using grid search. It is designed to work with training and validation datasets,
evaluating model performance based on MAE, RMSE, and MAPE metrics.

Generated with Kedro 0.19.11.
"""

import logging
from itertools import product
from typing import Dict, Tuple, Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from ..model_training.nodes import build_lstm_model

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float], tf.keras.Model]:
    """
    Runs hyperparameter tuning to find the best model configuration.

    Returns:
        Tuple containing:
            - Best hyperparameters found
            - Best model performance metrics
            - Best trained model (tf.keras.Model)
    """

    best_metrics = {"MAE": float("inf"), "RMSE": float("inf"), "MAPE": float("inf")}
    best_params: Dict[str, Any] = {}
    best_model: Sequential = None  # Initialize as None

    hyperparameter_combinations = list(
        product(
            params["lstm_units"],
            params["batch_size"],
            params["epochs"],
            params["learning_rate"],
            params["dropout_rate"],
        )
    )

    logger.info("=" * 80)
    logger.info(
        f"Starting hyperparameter tuning over {len(hyperparameter_combinations)} combinations."
    )
    logger.info("=" * 80)

    for idx, (lstm_units, batch_size, epochs, learning_rate, dropout_rate) in enumerate(
        hyperparameter_combinations
    ):
        if idx % 10 == 0 or idx == 0:
            logger.info("-" * 80)
            logger.info(
                f"Iteration {idx + 1}/{len(hyperparameter_combinations)}\n"
                f"LSTM units={lstm_units}, Batch size={batch_size}, "
                f"Epochs={epochs}, Learning rate={learning_rate}, Dropout rate={dropout_rate}"
            )
            logger.info("-" * 80)

        current_params = {
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
        }
        model = build_lstm_model(X_train, current_params)

        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        epsilon = 1e-8
        mape = (
            np.mean(np.abs((y_val - y_pred) / (np.maximum(np.abs(y_val), epsilon))))
            * 100
        )

        logger.info(f"Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        logger.info("-" * 80)

        if mae < best_metrics["MAE"]:
            best_metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            best_params = {
                "lstm_units": lstm_units,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
            }
            best_model = model  # Store the best model
            logger.info("New best configuration found!")
            logger.info(f"Params used: {best_params}")
            logger.info("=" * 80)

    logger.info("Hyperparameter tuning completed.")
    logger.info(f"Best Hyperparameters: {best_params}")
    logger.info(f"Best Metrics: {best_metrics}")
    logger.info("=" * 80)

    return best_params, best_metrics, best_model
