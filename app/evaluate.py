from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

def evaluate_classifier(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Tính toán các metrics phù hợp cho Imbalance Data."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba))
    }


def assign_risk_band(probability: float) -> str:
    """Ánh xạ xác suất dự đoán sang 4 dải rủi ro theo chuẩn đề bài."""
    if probability >= 0.75:
        return "CRITICAL"
    elif probability >= 0.50:
        return "HIGH"
    elif probability >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def evaluate_risk_bands(y_true, y_proba) -> Dict[str, Any]:
    """
    Phân tích rủi ro theo 4 dải: CRITICAL, HIGH, MEDIUM, LOW
    và xây dựng ma trận phân bổ theo dải rủi ro.
    """
    y_true_arr = np.array(y_true)
    bands = [assign_risk_band(p) for p in y_proba]

    df_eval = pd.DataFrame({
        "actual": y_true_arr,
        "probability": y_proba,
        "risk_band": bands
    })

    band_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    summary = {}

    for b in band_order:
        sub = df_eval[df_eval["risk_band"] == b]
        total_in_band = len(sub)
        attrition_count = int(sub["actual"].sum())
        rate = float(attrition_count / total_in_band) if total_in_band > 0 else 0.0

        summary[b] = {
            "total_employees": total_in_band,
            "actual_attrition_count": attrition_count,
            "actual_attrition_rate": round(rate, 4)
        }

    return {
        "bands_summary": summary,
        "total_evaluated": len(df_eval)
    }

def get_recommended_action(risk_band: str) -> str:
        actions = {
            "CRITICAL": "Lên lịch gặp 1-1 khẩn cấp trong 48h; rà soát chế độ đãi ngộ; chuẩn bị phương án giữ chân đặc biệt.",
            "HIGH": "Trao đổi với quản lý trực tiếp; tìm hiểu vướng mắc về khối lượng công việc và lộ trình thăng tiến.",
            "MEDIUM": "Khảo sát mức độ hài lòng; kiểm tra cân bằng công việc - đời sống qua các kỳ đánh giá tiếp theo.",
            "LOW": "Duy trì lộ trình đào tạo và môi trường làm việc bình thường."
        }
        return actions.get(risk_band, "Theo dõi định kỳ.")