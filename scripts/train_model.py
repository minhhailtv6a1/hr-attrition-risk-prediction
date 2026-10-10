from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.train import (
    training_pipeline
)

def main() -> None:
    training_report = training_pipeline()
    
    print(f"- Training run ID: {training_report['run_id']}")
    print(f"- Model saved at: {training_report['model_path']}")
    print(f"- Best model: {training_report['best_model']}")
    print(f"- Accuracy: {training_report['accuracy']:.4f}")
    print(f"- Precision: {training_report['precision']:.4f}")
    print(f"- Recall: {training_report['recall']:.4f}")
    print(f"- F1 Score: {training_report['f1']:.4f}")
    print(f"- ROC AUC: {training_report['roc_auc']:.4f}")

if __name__ == "__main__":
    main()