# NeuroLens+ - Neurological Biomarker Assessment System

A comprehensive eye-tracking based neurological assessment platform that measures eye movement biomarkers for early detection of neurological conditions including Multiple Sclerosis (MS), Parkinson's Disease (PD), Progressive Supranuclear Palsy (PSP), and Sixth Nerve Palsy (CN6).

## Features

- **Eye Movement Tracking**: Real-time pupil tracking using MediaPipe FaceMesh
- **Four Assessment Tasks**:
  - Fixation stability test
  - Saccade latency and accuracy test
  - Smooth pursuit tracking test
  - Anti-saccade inhibitory control test
- **Feature Extraction**: Automated biomarker computation from raw eye movement data
- **AI-Powered Risk Assessment**: Machine learning-based neurological risk prediction
- **Interactive UI**: Streamlit-based web interface for task execution and results visualization
- **Export Capabilities**: PDF reports and JSON data export

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam
- Good lighting conditions

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd neurolens
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have a trained ONNX model file (optional, for AI predictions):
   - Place `neurolens_model.onnx` in the `models/` directory
   - If no model is present, the system will use dummy predictions

## Usage

### Running the Streamlit UI

Start the main interface:

```bash
streamlit run interface/app.py
```

The interface will open in your web browser. Follow these steps:

1. **Run IPD Calibration**: Click "Run IPD Calibration" to measure inter-pupillary distance
2. **Execute Tasks**: Run each eye movement test in sequence:
   - Fixation Test
   - Saccade Test
   - Smooth Pursuit Test
   - Anti-Saccade Test
3. **Process Features**: Click "Process Latest Features" to extract biomarkers
4. **Run AI Prediction**: Get neurological risk assessment
5. **Export Results**: Generate PDF report or JSON export

### Running Tasks Directly

You can also run tasks programmatically:

```python
from tasks.ipd_calibration import IPDCalibration
from tasks.fixation import FixationTask

# Calibrate IPD
calibrator = IPDCalibration()
ipd = calibrator.run_calibration(duration=5.0)

# Run fixation task
task = FixationTask()
output_file = task.run_task(ipd, fixation_duration=10.0)
```

### Feature Extraction

Extract features from raw CSV data:

```python
from extraction.extract_features import extract_all_features

features = extract_all_features("data/raw_logs/fixation_20240101_120000.csv")
```

### Model Inference

Run predictions with extracted features:

```python
from models.infer import InferenceEngine

engine = InferenceEngine(model_path="models/neurolens_model.onnx")
prediction = engine.predict(features)
```

## Project Structure

```
neurolens/
├── core/
│   └── eye_tracker.py          # Core eye tracking engine
├── tasks/
│   ├── ipd_calibration.py      # IPD measurement task
│   ├── fixation.py             # Fixation stability task
│   ├── saccade.py              # Saccade task
│   ├── smooth_pursuit.py       # Smooth pursuit task
│   └── anti_saccade.py         # Anti-saccade task
├── extraction/
│   ├── fixation_features.py    # Fixation biomarker extraction
│   ├── saccade_features.py     # Saccade biomarker extraction
│   ├── smooth_features.py      # Smooth pursuit biomarker extraction
│   ├── anti_saccade_features.py # Anti-saccade biomarker extraction
│   └── extract_features.py     # Feature extraction orchestrator
├── models/
│   ├── model_loader.py         # ONNX model loader
│   └── infer.py                # Inference engine
├── interface/
│   ├── app.py                  # Streamlit main app
│   └── ui_helpers.py           # UI helper functions
├── export/
│   ├── pdf_report.py           # PDF report generator
│   └── json_export.py          # JSON export
├── data/
│   ├── raw_logs/               # Raw CSV data from tasks
│   ├── processed_features/     # Extracted feature vectors
│   └── neuro_signatures/       # Final processed signatures
├── reports/                    # Generated PDF reports
├── results/                    # JSON exports
└── README.md
```

## Extracted Biomarkers

### Fixation Features
- Fixation stability (STD X/Y)
- Drift magnitude
- Microsaccade count
- Square-wave jerks
- Nystagmus frequency (1-8 Hz)

### Saccade Features
- Latency (stimulus onset to saccade onset)
- Peak velocity
- Amplitude error
- Directional accuracy
- Dysmetria classification (hypometric/hypermetric)

### Smooth Pursuit Features
- Pursuit gain (eye velocity / target velocity)
- Phase lag
- Catch-up saccade frequency
- Smoothness (R² correlation)

### Anti-Saccade Features
- Error rate (incorrect reflexive responses)
- Correction latency
- Reflexive prosaccade count

## Data Format

### Raw CSV Format

Tasks record frame-by-frame data:
- `timestamp`: Frame timestamp
- `left_x`, `left_y`: Left pupil coordinates
- `right_x`, `right_y`: Right pupil coordinates
- `ipd_px`: Inter-pupillary distance in pixels
- Task-specific fields (target positions, velocities, etc.)

### Feature Vector Format

Extracted features are saved as CSV with 19 biomarkers:
- 6 fixation features
- 5 saccade features
- 4 smooth pursuit features
- 3 anti-saccade features
- Missing features filled with NaN

## Model Requirements

The ONNX model should accept:
- **Input**: 19-dimensional feature vector (normalized float32)
- **Output**: 4-dimensional risk scores (MS, PD, PSP, CN6)

## Notes

- Ensure good lighting and camera positioning for accurate tracking
- Keep face centered in frame during tests
- Tests require stable head position
- Tracking may be lost if face moves out of frame

## License

[Specify your license here]

## Citation

If using this system in research, please cite appropriately.

## Contact

[Your contact information]

