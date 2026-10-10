from functools import lru_cache
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app.config import settings
from app.config import SOURCE_FILES
from app.features import RAW_FEATURES, OUTPUT_FEATURES, TARGET_COLUMN, DOWN_SAMPLE_STRATIFY_FEATURES
from helper.helper_functions import write_json, read_parquet
from app.schemas import (
    IngestDataResponse,
    ValidateDataResponse,
)


def _active_data_dir() -> Path:
    """Return local mounted data or the S3-synchronized raw-data directory."""
    return settings.RAW_DATA_DIR if settings.DATA_SOURCE.lower() == "s3" else settings.DATA_DIR

def sync_data_from_s3() -> dict[str, Any]:
    """Download the expected training files from S3/S3-compatible storage.

    The task is idempotent: a new DAG run replaces only the known source files
    inside its raw-data workspace; it never writes outside that directory.
    """
    if settings.DATA_SOURCE.lower() != "s3":
        return {"status": "skipped", "reason": "DATA_SOURCE is local", "data_dir": str(settings.RAW_DATA_DIR)}
    bucket = settings.S3_BUCKET
    if not bucket:
        raise ValueError("S3_BUCKET is required when DATA_SOURCE=s3")
    prefix = settings.S3_PREFIX.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    import boto3

    client = boto3.client(
        "s3",
        region_name=settings.AWS_DEFAULT_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
    )
    objects: dict[str, str] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            filename = Path(key).name
            if filename in SOURCE_FILES:
                objects[filename] = key
    missing = sorted(set(SOURCE_FILES).difference(objects.keys()))
    if missing:
        raise FileNotFoundError(f"S3 source is missing required files under s3://{bucket}/{prefix}: {missing}")

    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, key in objects.items():
        client.download_file(bucket, key, str(settings.RAW_DATA_DIR / filename))
    report = {
        "status": "downloaded",
        "bucket": bucket,
        "prefix": prefix,
        "files": sorted(objects),
        "data_dir": str(settings.RAW_DATA_DIR),
    }
    write_json(settings.ARTIFACTS_DIR / "ingestion_report.json", report)
    return report


def _merge_temporal_data(
    combined_data,
    source_data,
    source_date_col,
    avg_cols=None,
    mode_cols=None
):
    combined_data = combined_data.copy()
    source_data = source_data.copy()

    avg_cols = avg_cols or []
    mode_cols = mode_cols or []

    # =========================
    # 1. Convert datetime
    # =========================
    combined_data["snapshot_date"] = pd.to_datetime(
        combined_data["snapshot_date"],
        errors="coerce"
    )

    source_data[source_date_col] = pd.to_datetime(
        source_data[source_date_col],
        errors="coerce"
    )

    # Remove invalid records
    combined_data = combined_data.dropna(
        subset=["employee_id", "snapshot_date"]
    )

    source_data = source_data.dropna(
        subset=["employee_id", source_date_col]
    )

    # =========================
    # 2. Sort
    # =========================
    combined_data = (
        combined_data
        .sort_values(["snapshot_date", "employee_id"])
        .reset_index(drop=True)
    )

    source_data = (
        source_data
        .sort_values([source_date_col, "employee_id"])
        .reset_index(drop=True)
    )

    # =========================
    # 3. Merge
    # =========================
    merged = pd.merge_asof(
        combined_data,
        source_data,
        left_on="snapshot_date",
        right_on=source_date_col,
        by="employee_id",
        direction="backward"
    )

    return merged


def _load_and_merge_all_sources(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or _active_data_dir()
    attrition_labels = pd.read_csv(data_dir / "attrition_labels.csv").drop(columns=["termination_date", "attrition_type"], errors="ignore").rename(columns={"left_within_90_days": "label"})
    employees = pd.read_csv(data_dir / "employees.csv").drop(columns=['department', 'job_level'], errors="ignore")

    frame = attrition_labels.merge(employees, on="employee_id", how="left")

    # Merge engagement surveys
    engagement_surveys = pd.read_csv(data_dir / "engagement_surveys.csv")
    frame = _merge_temporal_data(
        combined_data=frame,
        source_data=engagement_surveys,
        source_date_col="survey_date",
        avg_cols=[
            "engagement_survey",
            "work_life_balance",
            "manager_support_score",
            "career_growth_score",
            "compensation_satisfaction_score"
        ]
    )

    # Merge leave logs
    leave_logs = pd.read_csv(data_dir / "leave_logs.csv")
    frame = _merge_temporal_data(
        combined_data=frame,
        source_data=leave_logs,
        source_date_col="leave_date",
        avg_cols=["leave_days"],
        mode_cols=[
            "leave_type",
            "is_unplanned",
            "approved_by_manager"
        ]
    )

    # Merge performance reviews
    performance_reviews = pd.read_csv(data_dir / "performance_reviews.csv")
    frame = _merge_temporal_data(
        combined_data=frame,
        source_data=performance_reviews,
        source_date_col="review_date",
        avg_cols=[
            "performance_score",
            "goal_completion_rate",
            "manager_rating",
            "peer_feedback_score",
            "review_sentiment_score"
        ]
    )

    # Merge work activity
    work_activity = pd.read_csv(data_dir / "work_activity.csv")
    frame = _merge_temporal_data(
        combined_data=frame,
        source_data=work_activity,
        source_date_col="week_start",
        avg_cols=[
            "tasks_completed",
            "late_tasks",
            "overtime_hours",
            "meeting_hours",
            "focus_hours",
            "after_hours_messages",
            "absent_days"
        ]
    )

    return frame


def validate_source_data() -> dict[str, Any]:
    frame = _load_and_merge_all_sources()
    required = (set(RAW_FEATURES))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required features in source data: {missing}")
    if frame.empty:
        raise ValueError("Source data is empty")
    if not set(frame.label.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Source data contains unexpected label values: {set(frame.label.dropna().unique())}")
    if frame.label.nunique() != 2:
        raise ValueError(f"Source data does not contain all expected label classes: {set(frame.label.dropna().unique())}")
    
    settings.MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.MERGED_DATA_PATH, index=False)

    report = {
        "status": "passed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "source_file": len(SOURCE_FILES),
    }

    write_json(settings.ARTIFACTS_DIR / "validation_report.json", report)
    return report


def _bivariate_analysis(
    frame: pd.DataFrame,
    x_col: str,
    label_col: str,
    chart_name: str,
    title: str,
    chart_type: str = "countplot",
    figsize: tuple[int, int] = (10, 6),
    output_dir: Path | None = None,
) -> str:
    """
    Vẽ biểu đồ phân tích mối quan hệ giữa một feature và label,
    sau đó lưu biểu đồ vào thư mục EDA.
    """

    output_dir = output_dir or settings.ARTIFACTS_DIR / "eda_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = frame[[x_col, label_col]].dropna()

    plt.figure(figsize=figsize)

    if chart_type == "countplot":
        sns.countplot(
            data=data,
            x=x_col,
            hue=label_col
        )

    elif chart_type == "histplot":
        sns.histplot(
            data=data,
            x=x_col,
            hue=label_col,
            multiple="stack",
            bins=30
        )

    elif chart_type == "boxplot":
        sns.boxplot(
            data=data,
            x=label_col,
            y=x_col
        )

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    plt.title(title)
    plt.tight_layout()

    chart_path = output_dir / chart_name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(chart_path)

def exploratory_data_analysis() -> dict[str, Any]:
    """
    Thực hiện EDA bivariate analysis giữa các feature và label,
    lưu các biểu đồ vào artifacts/eda_charts.
    """

    # =========================
    # 1. Load merged data
    # =========================

    # frame = _load_and_merge_all_sources()
    frame = read_parquet(settings.MERGED_DATA_PATH)

    eda_dir = settings.ARTIFACTS_DIR / "eda_charts"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. Feature engineering cho EDA
    # =========================

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce"
    )

    frame["hire_date"] = pd.to_datetime(
        frame["hire_date"],
        errors="coerce"
    )

    frame["tenure_months"] = (
        (frame["snapshot_date"] - frame["hire_date"]).dt.days / 30
    ).clip(lower=0)

    # Overtime tier
    if "overtime_hours" in frame.columns:
        frame["overtime_tier"] = pd.cut(
            frame["overtime_hours"],
            bins=[-1, 2, 5, 8, float("inf")],
            labels=[
                "Low (<=2h)",
                "Medium (2-5h)",
                "High (5-8h)",
                "Very High (>8h)"
            ]
        )

    # Unplanned leave tier
    if "is_unplanned" in frame.columns:
        frame["unplanned_tier"] = pd.cut(
            pd.to_numeric(frame["is_unplanned"], errors="coerce").fillna(0),
            bins=[-1, 0, 2, 5, float("inf")],
            labels=[
                "0",
                "1-2",
                "3-5",
                ">5"
            ]
        )

    # =========================
    # 3. Bivariate analysis
    # =========================

    charts = []

    chart_configs = [
        {
            "x_col": "department",
            "chart_name": "attrition_by_department.png",
            "title": "Tỷ lệ thôi việc theo phòng ban",
            "chart_type": "countplot",
            "figsize": (10, 6),
        },
        {
            "x_col": "job_level",
            "chart_name": "attrition_by_level.png",
            "title": "Tỷ lệ thôi việc theo cấp bậc (Job Level)",
            "chart_type": "countplot",
            "figsize": (8, 5),
        },
        {
            "x_col": "tenure_years",
            "chart_name": "attrition_by_tenure.png",
            "title": "Phân bố thôi việc theo thâm niên",
            "chart_type": "histplot",
            "figsize": (10, 5),
        },
        {
            "x_col": "overtime_tier",
            "chart_name": "attrition_by_overtime.png",
            "title": "Tỷ lệ thôi việc theo mức làm thêm giờ",
            "chart_type": "countplot",
            "figsize": (8, 5),
        },
        {
            "x_col": "unplanned_tier",
            "chart_name": "attrition_by_leave_pattern.png",
            "title": "Tỷ lệ thôi việc theo số lần nghỉ đột xuất",
            "chart_type": "countplot",
            "figsize": (8, 5),
        },
        {
            "x_col": "engagement_score",
            "chart_name": "attrition_by_engagement_score.png",
            "title": "Phân bố điểm gắn kết theo trạng thái thôi việc",
            "chart_type": "histplot",
            "figsize": (8, 5),
        },
    ]

    for config in chart_configs:
        x_col = config["x_col"]

        # Bỏ qua chart nếu feature không tồn tại
        if x_col not in frame.columns:
            continue

        chart_path = _bivariate_analysis(
            frame=frame,
            label_col="label",
            output_dir=eda_dir,
            **config
        )

        charts.append(chart_path)

    # =========================
    # 4. Save summary
    # =========================

    result = {
        "status": "completed",
        "eda_dir": str(eda_dir),
        "charts": charts,
        "sample_count": len(frame),
    }

    write_json(
        settings.ARTIFACTS_DIR / "eda_summary.json",
        result
    )

    return result

def preprocess_data() -> dict[str, Any]:
    # 1. Đọc dữ liệu cơ sở
    frame = read_parquet(settings.MERGED_DATA_PATH)

    # 2. Chuẩn hóa kiểu dữ liệu ngày tháng
    for date_col in ["snapshot_date", "hire_date", "last_promotion_date"]:
        if date_col in frame.columns:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")

    # 3. Loại bỏ trùng lặp khóa chính (employee_id, snapshot_date)
    frame = frame.drop_duplicates(subset=["employee_id", "snapshot_date"], keep="first")

    # 4. Loại bỏ dòng rỗng ở các khóa bắt buộc và lọc nhãn hợp lệ
    frame = frame.dropna(subset=["employee_id", "snapshot_date", "label"])
    frame = frame[frame["label"].isin([0, 1])]

    # 5. Lưu kết quả làm sạch ra tệp trung gian
    settings.PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.PREPROCESSED_PATH, index=False)

    summary = {
        "status": "completed",
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "target_column": TARGET_COLUMN,
        "output": str(settings.PREPROCESSED_PATH),
    }
    write_json(settings.ARTIFACTS_DIR / "preprocessing_summary.json", summary)
    return summary

def feature_engineering() -> dict[str, Any]:
    if not settings.PREPROCESSED_PATH.exists():
        raise FileNotFoundError(f"Preprocessed base data missing at {settings.PREPROCESSED_PATH}; run preprocess_data first")

    frame = pd.read_parquet(settings.PREPROCESSED_PATH)

    frame["tenure_months"] = (frame["snapshot_date"] - frame["hire_date"]).dt.days / 30

    frame["days_since_last_promotion"] = (frame["snapshot_date"] - frame["last_promotion_date"]).dt.days

    settings.FEATURE_ENGINEERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.FEATURE_ENGINEERED_PATH, index=False)

    summary = {
        "status": "completed",
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "output": str(settings.FEATURE_ENGINEERED_PATH),
    }
    write_json(settings.ARTIFACTS_DIR / "feature_engineering_summary.json", summary)
    return summary


def feature_selection() -> dict[str, Any]:
    if not settings.FEATURE_ENGINEERED_PATH.exists():
        raise FileNotFoundError(f"Feature-engineered data missing at {settings.FEATURE_ENGINEERED_PATH}; run feature_engineering first")

    frame = read_parquet(settings.FEATURE_ENGINEERED_PATH)

    frame = frame[OUTPUT_FEATURES]

    settings.FEATURE_SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.FEATURE_SELECTED_PATH, index=False)

    summary = {
        "status": "completed",
        "selected_features": OUTPUT_FEATURES,
        "num_features": len(OUTPUT_FEATURES),
        "rows": int(len(frame)),
    }
    write_json(settings.ARTIFACTS_DIR / "feature_selection_summary.json", summary)
    return summary


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

def data_pipeline() -> dict[str, Any]:
    """
    Thực hiện toàn bộ pipeline xử lý dữ liệu:
    1. Đồng bộ dữ liệu từ S3 (nếu cần)
    2. Validate dữ liệu nguồn
    3. Tiền xử lý dữ liệu
    4. Feature engineering
    5. Feature selection

    Returns
    -------
    dict[str, Any]
        Báo cáo tổng hợp của toàn bộ pipeline.
    """
    ingestion_report = sync_data_from_s3()
    validation_report = validate_source_data()
    preprocessing_report = preprocess_data()
    feature_engineering_report = feature_engineering()
    feature_selection_report = feature_selection()

    summary = {
        "ingestion": ingestion_report,
        "validation": validation_report,
        "preprocessing": preprocessing_report,
        "feature_engineering": feature_engineering_report,
        "feature_selection": feature_selection_report,
    }

    write_json(settings.ARTIFACTS_DIR / "data_pipeline_summary.json", summary)
    return summary


class DataPipelineService:
    def __init__(self):
        self.ingestion_report: dict[str, Any] | None = None
        self.validation_report: dict[str, Any] | None = None
        self.preprocessing_report: dict[str, Any] | None = None
        self.feature_engineering_report: dict[str, Any] | None = None
        self.feature_selection_report: dict[str, Any] | None = None

    def ingest_data(self) -> IngestDataResponse:
        self.ingestion_report = sync_data_from_s3()
        return IngestDataResponse(**self.ingestion_report)

    def validate_data(self) -> ValidateDataResponse:
        self.validation_report = validate_source_data()
        return ValidateDataResponse(**self.validation_report)

    def preprocess_data(self) -> dict[str, Any]:
        self.preprocessing_report = preprocess_data()
        return self.preprocessing_report

    def feature_engineering(self) -> dict[str, Any]:
        self.feature_engineering_report = feature_engineering()
        return self.feature_engineering_report

    def feature_selection(self) -> dict[str, Any]:
        self.feature_selection_report = feature_selection()
        return self.feature_selection_report
    
@lru_cache()
def get_data_service() -> DataPipelineService:
    return DataPipelineService()