from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from app.config import DEFAULT_RAW_DATA_DIR, PROCESSED_DATA_PATH
from app.data import (
    data_pipeline
)

def main():
    data_pipeline_report = data_pipeline()
    sync_data_report = data_pipeline_report.get("ingestion")
    validate_report = data_pipeline_report.get("validation")
    preprocess_report = data_pipeline_report.get("preprocessing")
    feature_engineering_report = data_pipeline_report.get("feature_engineering")
    feature_selection_report = data_pipeline_report.get("feature_selection")

    final_report = {
        "processed_data_path": str(preprocess_report.get("output")),
        "rows": preprocess_report.get("rows"),
        "columns": len(preprocess_report.get("columns")),
        "target_column": preprocess_report.get("target_column"),
        "selected_features": feature_selection_report.get("selected_features"),
    }

    print(f"- Saved processed dataset: {final_report['processed_data_path']}")
    print(f"- Rows: {final_report['rows']}")
    print(f"- Columns: {final_report['columns']}")
    print(f"- Target column: {final_report['target_column']}")
    print(f"- Selected features: {final_report['selected_features']}")

if __name__ == "__main__":
    main()