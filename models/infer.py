import numpy as np
from typing import Dict, Optional
from models.model_loader import ModelLoader


class InferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_loader = ModelLoader(model_path)
        self.model_loader.load()
    
    def predict(self, features: Dict) -> Dict:
        if not features:
            return {
                "MS_risk": 0.0,
                "PD_risk": 0.0,
                "PSP_risk": 0.0,
                "CN6_risk": 0.0,
                "confidence": 0.0,
                "top_features": []
            }
        
        result = self.model_loader.predict(features)
        
        risk_scores = [
            ("MS", result["MS_risk"]),
            ("PD", result["PD_risk"]),
            ("PSP", result["PSP_risk"]),
            ("CN6", result["CN6_risk"])
        ]
        
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_features = [name for name, score in risk_scores[:3] if score > 0.1]
        result["top_features"] = top_features
        
        return result

