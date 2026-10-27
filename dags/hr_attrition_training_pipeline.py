"""Airflow orchestration for the HR attrition data and ML pipelines."""
from __future__ import annotations

from datetime import timedelta
import os

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag


PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")


@dag(
    dag_id="hr_attrition_prediction_orchestration",
    start_date=pendulum.datetime(2026, 10, 2, tz="Asia/Ho_Chi_Minh"),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "people-analytics",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["employees-data", "data-processing", "model-training"],
)
def hr_attrition_prediction_orchestration():
    def stage(task_id: str, command: str) -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f"python -m app.cli {command}",
            cwd=PROJECT_ROOT,
            do_xcom_push=False,
        )

    ingest = stage("ingest_raw_data_from_s3", "ingest")
    validate = stage("validate_source_data", "validate")
    eda = stage("exploratory_data_analysis", "eda")
    preprocess = stage("preprocess_data", "preprocessing")
    feature_engineering = stage("feature_engineering", "feature_engineering")
    feature_selection = stage("feature_selection", "feature_selection")
    split = stage("create_train_test_split", "split_data")
    handle_imbalance = stage("handle_class_imbalance", "handle_class_imbalance")
    train = stage("train_and_track_mlflow", "train_model")
    publish = stage("quality_gate_and_publish", "publish")
    (
        ingest
        >> validate
        >> eda
        >> preprocess
        >> feature_engineering
        >> feature_selection
        >> split
        >> handle_imbalance
        >> train
        >> publish
    )


hr_attrition_prediction_orchestration()
