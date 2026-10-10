from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


def _project_path(value: str) -> Path:
    """Resolve relative configuration paths from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


load_dotenv(BASE_DIR / ".env")

SOURCE_FILES = [
    "attrition_labels.csv",
    "employees.csv",
    "engagement_surveys.csv",
    "leave_logs.csv",
    "performance_reviews.csv",
    "work_activity.csv"
]

class Settings(BaseSettings):
    # Data & Directories
    DATA_SOURCE: str = os.getenv("DATA_SOURCE", "local")
    ARTIFACTS_DIR: Path = _project_path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    RAW_DATA_DIR: Path = _project_path(os.getenv("RAW_DATA_DIR", "data/raw"))
    MERGED_DATA_DIR: Path = _project_path(os.getenv("MERGED_DATA_DIR", "data/merged"))
    PROCESSED_DATA_DIR: Path = _project_path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    MODELS_DIR: Path = _project_path(os.getenv("MODELS_DIR", "models"))
    REPORTS_DIR: Path = _project_path(os.getenv("REPORTS_DIR", "reports"))

    # Artifact paths
    MERGED_DATA_PATH: Path = _project_path(os.getenv("MERGED_DATA_PATH", "data/merged/merged_data.parquet"))
    PREPROCESSED_PATH: Path = _project_path(os.getenv("PREPROCESSED_PATH", "data/processed/preprocessed.parquet"))
    FEATURE_ENGINEERED_PATH: Path = _project_path(os.getenv("FEATURE_ENGINEERED_PATH", "data/processed/feature_engineered.parquet"))
    FEATURE_SELECTED_PATH: Path = _project_path(os.getenv("FEATURE_SELECTED_PATH", "data/processed/feature_selected.parquet"))
    TRAIN_PATH: Path = _project_path(os.getenv("TRAIN_PATH", "data/processed/train.parquet"))
    TEST_PATH: Path = _project_path(os.getenv("TEST_PATH", "data/processed/test.parquet"))
    TRAIN_BALANCED_PATH: Path = _project_path(os.getenv("TRAIN_BALANCED_PATH", "data/processed/train_balanced.parquet"))
    TEST_BALANCED_PATH: Path = _project_path(os.getenv("TEST_BALANCED_PATH", "data/processed/test_balanced.parquet"))
    BEST_MODEL_PATH: Path = _project_path(os.getenv("BEST_MODEL_PATH", "models/best_model.joblib"))
    METRICS_PATH: Path = _project_path(os.getenv("METRICS_PATH", "models/metrics.json"))
    SCHEMA_PATH: Path = _project_path(os.getenv("SCHEMA_PATH", "models/schema.json"))
    MODEL_MANIFEST_PATH: Path = _project_path(os.getenv("MODEL_MANIFEST_PATH", "models/production/model_manifest.json"))

    # AWS S3 Settings
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "data/")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "hr-attrition-risk-prediction-v1")

    # Quality Gate
    MIN_ROC_AUC: float = os.getenv("MIN_ROC_AUC", 0.70)

    # PostgreSQL Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_URL: str = os.getenv("LLM_URL", "https://api.openai.com/v1/chat/completions")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
