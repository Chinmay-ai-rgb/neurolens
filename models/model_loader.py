import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class ModelLoader:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = Path(__file__).parent / "neurolens_model.onnx"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        
    def load(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_shape = input_shape
        
    def prepare_input(self, features: Dict) -> np.ndarray:
        feature_order = [
            'fix_std_x', 'fix_std_y', 'fix_drift', 'fix_microsaccades', 'fix_swj', 'fix_nystagmus_hz',
            'sac_latency', 'sac_peak_vel', 'sac_amp_error', 'sac_direction_error', 'sac_dysmetria_type',
            'pursuit_gain', 'pursuit_phase_lag', 'pursuit_catchups', 'pursuit_smoothness_r2',
            'anti_error_rate', 'anti_corr_latency', 'anti_reflexive_count'
        ]
        
        feature_vector = []
        for key in feature_order:
            value = features.get(key, 0.0)
            if np.isnan(value):
                value = 0.0
            feature_vector.append(float(value))
        
        feature_array = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        
        return feature_array
    
    def predict(self, features: Dict) -> Dict:
        if self.session is None:
            self.load()
        
        input_array = self.prepare_input(features)
        
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        output = outputs[0][0]
        
        if len(output) >= 4:
            result = {
                "MS_risk": float(output[0]),
                "PD_risk": float(output[1]),
                "PSP_risk": float(output[2]),
                "CN6_risk": float(output[3]) if len(output) > 3 else 0.0,
                "confidence": float(np.max(output)),
                "top_features": []
            }
        else:
            result = {
                "MS_risk": 0.0,
                "PD_risk": 0.0,
                "PSP_risk": 0.0,
                "CN6_risk": 0.0,
                "confidence": 0.0,
                "top_features": []
            }
        
        return result

