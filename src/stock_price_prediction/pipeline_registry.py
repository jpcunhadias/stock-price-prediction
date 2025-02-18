"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_ingestion import pipeline as data_ingestion
from .pipelines.preprocessing import pipeline as preprocessing
from .pipelines.model_training import pipeline as model_training
from .pipelines.hyperparameter_tuning import pipeline as hyperparameter_tuning


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_ingestion": data_ingestion.create_pipeline(),
        "preprocessing": preprocessing.create_pipeline(),
        "model_training": model_training.create_pipeline(),
        "hyperparameter_tuning": hyperparameter_tuning.create_pipeline(),
        "__default__": data_ingestion.create_pipeline()
        + preprocessing.create_pipeline()
        + model_training.create_pipeline(),
    }
