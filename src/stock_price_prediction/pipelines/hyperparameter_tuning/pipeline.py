"""
This is a boilerplate pipeline 'hyperparameter_tuning'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import optimize_hyperparameters
from ..model_training.nodes import save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                optimize_hyperparameters,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:hyperparameter_search",
                ],
                outputs=["best_params", "best_metrics", "best_model"],
                name="optimize_hyperparameters_node",
            ),
            node(
                save_model,
                inputs=["best_model", "params:hyperparameter_search.s3_bucket",
                        "params:hyperparameter_search.model_key"],
                outputs=None,
                name="save_model_node",
            ),
        ]
    )
