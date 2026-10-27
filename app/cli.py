"""CLI adapter that exposes pipeline stages to Airflow's BashOperator."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from app.data import (
    exploratory_data_analysis,
    feature_engineering,
    feature_selection,
    preprocess_data,
    sync_data_from_s3,
    validate_source_data,
)
from app.train import (
    handle_imbalance_data,
    publish_if_quality_passes,
    split_data,
    train_model_with_mlflow,
)


STAGES: dict[str, Callable[[], dict[str, Any]]] = {
    "ingest": sync_data_from_s3,
    "validate": validate_source_data,
    "eda": exploratory_data_analysis,
    "preprocessing": preprocess_data,
    "feature_engineering": feature_engineering,
    "feature_selection": feature_selection,
    "split_data": split_data,
    "handle_class_imbalance": handle_imbalance_data,
    "train_model": train_model_with_mlflow,
    "publish": publish_if_quality_passes,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    stage = parser.parse_args().stage
    result = STAGES[stage]()
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
