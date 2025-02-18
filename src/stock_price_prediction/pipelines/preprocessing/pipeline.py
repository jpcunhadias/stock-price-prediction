"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (
    drop_invalid_first_row,
    split_data,
    normalize_data,
    save_scaled_data,
    create_sequences,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=drop_invalid_first_row,
                inputs="stock_prices",
                outputs="stock_prices_no_first_row",
                name="drop_invalid_first_row_node",
            ),
            node(
                func=split_data,
                inputs=["stock_prices_no_first_row", "params:test_size"],
                outputs=["train_df", "val_df"],
                name="split_data_node",
            ),
            node(
                func=normalize_data,
                inputs=["train_df", "val_df", "params:feature"],
                outputs=["train_scaled", "val_scaled", "scaler"],
                name="normalize_data_node",
            ),
            node(
                func=save_scaled_data,
                inputs="train_scaled",
                outputs="train_scaled_df",
                name="save_train_scaled_data_node",
            ),
            node(
                func=save_scaled_data,
                inputs="val_scaled",
                outputs="val_scaled_df",
                name="save_val_scaled_data_node",
            ),
            node(
                func=create_sequences,
                inputs=["train_scaled", "params:window_size"],
                outputs=["X_train", "y_train"],
                name="create_train_sequences_node",
            ),
            node(
                func=create_sequences,
                inputs=["val_scaled", "params:window_size"],
                outputs=["X_val", "y_val"],
                name="create_val_sequences_node",
            ),
        ]
    )
