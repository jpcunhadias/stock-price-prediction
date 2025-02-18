"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import fetch_stock_prices


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_stock_prices,
                inputs=["params:symbol", "params:start_date", "params:end_date"],
                outputs="stock_prices",
                name="fetch_stock_prices_node",
            )
        ]
    )
