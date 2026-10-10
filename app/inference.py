from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from pathlib import Path
from functools import lru_cache
from app.config import settings
from app.features import FINAL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from app.evaluate import assign_risk_band, get_recommended_action
from app.explain import extract_feature_drivers
from app.schemas import AttritionInput, AttritionPrediction

class HRAttritionServices:
    def __init__(self, model_path: Path = settings.BEST_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        artifact = joblib.load(model_path)
        self.model_path = model_path
        self.pipeline = artifact
        self.model_version = getattr(self.pipeline, "version", "unknown")
        self.features: list[str] = FINAL_FEATURES
        self.categorical_features: list[str] = CATEGORICAL_FEATURES
        self.numeric_features: list[str] = NUMERIC_FEATURES

    def predict_one(self, attrition: AttritionInput | dict[str, Any]) -> AttritionPrediction:
        if isinstance(attrition, AttritionInput):
            payload = attrition.model_dump()
        else:
            payload = AttritionInput(**attrition).model_dump()

        row = pd.DataFrame([payload], columns=self.features)
        probabilities = float(self.pipeline.predict_proba(row)[0, 1])
        risk_band = assign_risk_band(probabilities)
        return AttritionPrediction(
            churn_probability=round(probabilities, 4),
            risk_band=risk_band,
            top_drivers=extract_feature_drivers(self.pipeline, payload),
            recommended_action=get_recommended_action(risk_band),  # <-- Added
            model_name=self.pipeline.__class__.__name__,
            model_version=self.model_version
        )

    def predict_many(self, attrition_lists: list[AttritionInput | dict[str, Any]]) -> list[AttritionPrediction]:
        return [self.predict_one(attrition) for attrition in attrition_lists]
    
    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.pipeline.__class__.__name__,
            "model_version": self.model_version,
            "model_path": str(self.model_path),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
        }

    # def 
    

@lru_cache(maxsize=1)
def get_inference_services() -> HRAttritionServices:
    return HRAttritionServices()