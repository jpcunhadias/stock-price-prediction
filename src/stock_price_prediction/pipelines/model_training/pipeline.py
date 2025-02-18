"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_lstm_model, evaluate_model, save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=train_lstm_model,
            #     inputs=[
            #         "X_train",
            #         "y_train",
            #         "X_val",
            #         "y_val",
            #         "params:model_training",
            #     ],
            #     outputs="lstm_model",
            #     name="train_lstm_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["lstm_model", "X_val", "y_val", "scaler"],
            #     outputs=None,
            #     name="evaluate_model_node",
            # ),
            node(
                func=save_model,
                inputs=["lstm_model", "params:model_training.s3_bucket", "params:model_training.model_key"],
                outputs=None,
                name="save_model_node",
            ),
        ]
    )
