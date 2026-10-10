import json
import socket
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from app.config import settings
from app.evaluate import evaluate_classifier, evaluate_risk_bands
from app.features import (
    CATEGORICAL_FEATURES,
    ID_COLUMN, 
    NUMERIC_FEATURES, 
    TARGET_COLUMN, 
    DOWN_SAMPLE_STRATIFY_FEATURES, 
    FINAL_FEATURES
)

from helper.helper_functions import write_json


def configure_mlflow_tracking() -> None:
    """Configure MLflow and fail early when an HTTP tracking server is offline."""
    tracking_uri = settings.MLFLOW_TRACKING_URI
    parsed_uri = urlparse(tracking_uri)

    if parsed_uri.scheme in {"http", "https"}:
        host = parsed_uri.hostname
        port = parsed_uri.port or (443 if parsed_uri.scheme == "https" else 80)
        if not host:
            raise ValueError(f"Invalid MLFLOW_TRACKING_URI: {tracking_uri!r}")

        try:
            with socket.create_connection((host, port), timeout=3):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"MLflow tracking server is unavailable at {tracking_uri}. "
                "Start it with 'docker compose up -d postgres mlflow' or run "
                f"'mlflow server --host 127.0.0.1 --port {port}', then retry."
            ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

def split_data() -> dict[str, Any]:
    if not settings.FEATURE_SELECTED_PATH.exists():
        raise FileNotFoundError("Processed features are missing; run prepare_features first")
    frame = pd.read_parquet(settings.FEATURE_SELECTED_PATH)
    train, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.label)
    settings.TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(settings.TRAIN_PATH, index=False)
    test.to_parquet(settings.TEST_PATH, index=False)
    return {"train_rows": int(len(train)), "test_rows": int(len(test)), "random_state": 42}

def _stratified_downsample(
    df,
    label_col="label",
    majority_label=0,
    target_size=33_000,
    stratify_cols=None,
    random_state=42
) -> pd.DataFrame:
    majority = df[df[label_col] == majority_label].copy()

    if stratify_cols:
        majority["_stratify_key"] = (
            majority[stratify_cols]
            .fillna("UNKNOWN")
            .astype(str)
            .agg("_".join, axis=1)
        )

        # Tỷ lệ của từng nhóm
        proportions = (
            majority["_stratify_key"]
            .value_counts(normalize=True)
        )

        # Số lượng sample cho từng nhóm
        sample_sizes = (
            (proportions * target_size)
            .round()
            .astype(int)
        )

        sampled_parts = []

        for key, n in sample_sizes.items():
            group = majority[
                majority["_stratify_key"] == key
            ]

            n = min(n, len(group))

            sampled_parts.append(
                group.sample(
                    n=n,
                    random_state=random_state
                )
            )

        sampled = pd.concat(
            sampled_parts,
            ignore_index=True
        )

        sampled = sampled.drop(
            columns="_stratify_key"
        )

    else:
        sampled = majority.sample(
            n=min(target_size, len(majority)),
            random_state=random_state
        )

    return sampled

def _upsample_minority(
    df,
    label_col="label",
    minority_label=1,
    target_size=16_500,
    random_state=42
) -> pd.DataFrame:
    minority = df[
        df[label_col] == minority_label
    ].copy()

    if len(minority) >= target_size:
        return minority.sample(
            n=target_size,
            random_state=random_state
        )

    # Lấy lại mẫu với replacement
    extra = minority.sample(
        n=target_size - len(minority),
        replace=True,
        random_state=random_state
    )

    return pd.concat(
        [minority, extra],
        ignore_index=True
    )

def _downsample_test_majority(
    df,
    label_col,
    majority_label=0,
    target_size=8_000,
    random_state=42
) -> pd.DataFrame:
    """
    Downsample chỉ majority class trong tập test.

    Parameters
    ----------
    df : pd.DataFrame
        Test dataframe.
    label_col : str
        Tên cột label.
    majority_label : int
        Nhãn majority cần downsample.
    target_size : int
        Số lượng sample tối đa muốn giữ lại của majority class.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Test dataframe sau khi downsample majority.
    """

    majority_df = df[df[label_col] == majority_label]
    minority_df = df[df[label_col] != majority_label]

    # Không thể lấy nhiều sample hơn số lượng hiện có
    target_size = min(target_size, len(majority_df))

    majority_sampled = majority_df.sample(
        n=target_size,
        random_state=random_state
    )

    result = pd.concat(
        [majority_sampled, minority_df],
        axis=0
    )

    # Shuffle lại dataset
    result = result.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return result

def _hybrid_balance(
    df,
    label_col="label",
    majority_label=0,
    minority_label=1,
    majority_target=33_000,
    minority_target=16_500,
    stratify_cols=None,
    random_state=42
) -> pd.DataFrame:
    majority = _stratified_downsample(
        df=df,
        label_col=label_col,
        majority_label=majority_label,
        target_size=majority_target,
        stratify_cols=stratify_cols,
        random_state=random_state
    )

    minority = _upsample_minority(
        df=df,
        label_col=label_col,
        minority_label=minority_label,
        target_size=minority_target,
        random_state=random_state
    )

    balanced = pd.concat(
        [majority, minority],
        ignore_index=True
    )

    # Shuffle
    balanced = balanced.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return balanced

def handle_imbalance_data() -> dict[str, Any]:
    if not settings.TRAIN_PATH.exists():
        raise FileNotFoundError("Training data is missing; run split_data first")
    train = pd.read_parquet(settings.TRAIN_PATH)
    train_balanced = _hybrid_balance(
        df=train,
        label_col="label",
        majority_label=0,
        minority_label=1,
        majority_target=33_000,
        minority_target=16_500,
        stratify_cols=list(DOWN_SAMPLE_STRATIFY_FEATURES),
        random_state=42
    )
    settings.TRAIN_BALANCED_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_balanced.to_parquet(settings.TRAIN_BALANCED_PATH, index=False)

    test = pd.read_parquet(settings.TEST_PATH)
    test_balanced = _downsample_test_majority(
        df=test,
        label_col=TARGET_COLUMN,
        majority_label=0,
        target_size=2300,
        random_state=42
    )
    settings.TEST_BALANCED_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_balanced.to_parquet(settings.TEST_BALANCED_PATH, index=False)

    summary = {
        "status": "completed",
        "train_original_rows": int(len(train)),
        "train_balanced_rows": int(len(train_balanced)),
        "train_label_distribution": {
            "train": train["label"].value_counts().to_dict(),
            "train_balanced": train_balanced["label"].value_counts().to_dict(),
        },
        "test_original_rows": int(len(test)),
        "test_balanced_rows": int(len(test_balanced)),
        "test_label_distribution": {
            "test": test["label"].value_counts().to_dict(),
            "test_balanced": test_balanced["label"].value_counts().to_dict(),
        },
        "output": list([str(settings.TRAIN_BALANCED_PATH), str(settings.TEST_BALANCED_PATH)]),
    }
    write_json(settings.ARTIFACTS_DIR / "imbalance_handling_summary.json", summary)
    return summary

def build_preprocessor() -> ColumnTransformer:
    """Tạo tiền xử lý kết hợp StandardScaler và OneHotEncoder."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])


def build_model_pipelines() -> Dict[str, Pipeline]:
    """Khởi tạo danh sách 5 mô hình phân lớp."""
    return {
        "DecisionTree": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
                max_depth=5
            ))
        ]),
        "RandomForest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, 
                random_state=42, 
                n_jobs=-1, 
                class_weight="balanced_subsample"
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", GradientBoostingClassifier(
                n_estimators=400, 
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", XGBClassifier(
            n_estimators=400, 
            random_state=42,
            scale_pos_weight=2.0,
            eval_metric="logloss"
            ))
        ]),

        "LightGBM": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", lgb.LGBMClassifier(
            n_estimators=400,
            random_state=42, 
            class_weight="balanced",
            verbose=-1
            ))
        ])
    }

def train_and_track_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Huấn luyện các mô hình và log đầy đủ metrics lên MLflow với Nested Runs."""
    X_train = train_df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    models = build_model_pipelines()

    # Cấu hình MLflow
    configure_mlflow_tracking()

    model_results = []
    trained_pipelines = {}

    with mlflow.start_run(run_name="model_training_batch") as parent_run:
        for name, pipe in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # 1. Fit
                pipe.fit(X_train, y_train)
                trained_pipelines[name] = pipe

                # 2. Predict & Evaluate
                y_pred = pipe.predict(X_test)
                y_proba = pipe.predict_proba(X_test)[:, 1]

                metrics = evaluate_classifier(y_test, y_pred, y_proba)
                risk_eval = evaluate_risk_bands(y_test, y_proba)

                # 3. Log to MLflow
                mlflow.log_params({
                    "model_type": name,
                    "num_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, artifact_path=f"{name}_model", serialization_format="cloudpickle")

                model_results.append({
                    "Model": name,
                    **metrics,
                    "risk_distribution": risk_eval["bands_summary"]
                })

        # Xếp hạng mô hình tốt nhất theo ROC-AUC
        metrics_df = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)
        best_row = metrics_df.iloc[0]
        best_name = str(best_row["Model"])
        best_pipe = trained_pipelines[best_name]

        # Lưu model tốt nhất
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, settings.BEST_MODEL_PATH)

        def _to_native(d: dict) -> dict:
            return {k: float(v) if hasattr(v, "item") else v for k, v in d.items()}

        summary = {
            "run_id": parent_run.info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_model": _to_native(best_row.to_dict()),
            "full_metrics": [_to_native(r) for r in metrics_df.to_dict(orient="records")]
        }

        settings.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # The serving API reads this file directly.  Training and serving must
        # therefore publish their feature contract together.
        schema = {
            "model_name": best_name,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "feature_count": len(FINAL_FEATURES),
            "quality_gate_roc_auc": float(settings.MIN_ROC_AUC),
        }
        settings.SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        # Lưu Production Manifest
        manifest = {
            "model_name": best_name,
            "artifact_path": str(settings.BEST_MODEL_PATH),
            "quality_gate": settings.MIN_ROC_AUC,
            "quality_passed": bool(best_row["roc_auc"] >= settings.MIN_ROC_AUC),
            "published_at": datetime.now(timezone.utc).isoformat(),
            **_to_native(best_row.to_dict())
        }
        settings.MODEL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.MODEL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return summary

def training_pipeline() -> Dict[str, Any]:
    """Huấn luyện mô hình và lưu trữ kết quả."""
    _ = split_data()
    _ = handle_imbalance_data()

    train_df = pd.read_parquet(settings.TRAIN_BALANCED_PATH)
    test_df = pd.read_parquet(settings.TEST_BALANCED_PATH)

    training_summary = train_and_track_models(train_df, test_df)

    return {
        "run_id": training_summary["run_id"],
        "model_path": str(settings.BEST_MODEL_PATH),
        "best_model": training_summary["best_model"].get("Model"),
        "accuracy": training_summary["best_model"].get("accuracy"),
        "precision": training_summary["best_model"].get("precision"),
        "recall": training_summary["best_model"].get("recall"),
        "f1": training_summary["best_model"].get("f1"),
        "roc_auc": training_summary["best_model"].get("roc_auc"),
    }
