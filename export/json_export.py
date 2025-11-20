import json
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime


def export_json(features: Dict, predictions: Dict = None, output_path: str = None) -> str:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/neurolens_session_{timestamp}.json"
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "features": convert_numpy(features),
        "predictions": convert_numpy(predictions) if predictions else None
    }
    
    with open(output_path, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return str(output_path)

