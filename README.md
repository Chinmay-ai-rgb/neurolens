# NeuroLens+

A research-grade webcam-based eye biomarker platform for collecting oculomotor and pupillary biomarkers using standard laptop webcams.

## Overview

NeuroLens+ is designed for research applications requiring standardized eye tracking tasks with robust validity handling and research-ready outputs. It uses MediaPipe FaceMesh with iris refinement for gaze estimation and implements comprehensive calibration, head motion compensation, and quality assurance.

## Features

- **6 Standardized Eye Tracking Tasks**
  - Fixation Task - Measure fixation stability
  - Smooth Pursuit Task - Track moving targets
  - Saccade Task - Measure rapid eye movements
  - Anti-Saccade Task - Assess inhibitory control
  - 9-Point Gaze Grid - Assess gaze accuracy (screen-safe with 10% margins)
  - Visual Search Task - Covert blink measurement during natural search behavior

- **Research-Grade Features**
  - Per-frame raw signal logging (33 columns)
  - Per-trial biomarker summaries
  - Session metadata with calibration quality
  - Validity checking with explicit reason codes
  - Replay mode for reprocessing frame logs
  - Head motion compensation
  - Calibration with degeneracy detection

- **Quality Assurance**
  - Real-time quality indicators
  - Clamp rate monitoring
  - FPS tracking
  - Face detection status
  - Automatic validity flagging

## Requirements

- Python 3.11+
- Webcam (built-in laptop camera works)
- Display (1920x1080 recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/neurolens-plus.git
cd neurolens-plus
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Menu

Run the main launcher to select tasks interactively:

```bash
python main.py
```

Use arrow keys to navigate, SPACE to select, Q to quit.

### Command Line

Run a specific task directly:

```bash
# Run saccade task
python main.py -t 3

# Run in fullscreen mode
python main.py -t 3 -f

# Run all tasks in sequence
python main.py -t all

# List available tasks
python main.py --list
```

### Task Numbers

1. Fixation Task
2. Smooth Pursuit Task
3. Saccade Task
4. Anti-Saccade Task
5. 9-Point Gaze Grid
6. Visual Search Task

## Controls

During tasks:
- **SPACE**: Start/continue
- **Q** or **ESC**: Quit (saves partial data)
- **R**: Redo calibration
- **S**: Skip current trial
- **F**: Toggle fullscreen

## Output Files

Data is saved to `data/sessions/<session_id>/`:

```
data/sessions/20241220_143052_abc123/
├── meta.json              # Session metadata
├── calibration.json       # Calibration parameters
├── frame_log_saccade.csv  # Per-frame raw data
├── summary_saccade.csv    # Per-trial biomarkers
└── debug_saccade.log      # Debug information
```

### Frame Log Columns

The frame log CSV contains 33 columns including:
- Timestamps and frame indices
- Gaze positions (raw and compensated)
- Velocity estimates
- Pupil proxy measurements
- Quality flags (valid_sample, clamp_flag, etc.)

### Summary Columns

Task-specific biomarkers including:
- Latency, duration, amplitude (saccade tasks)
- Gain, position error (pursuit task)
- RMS stability, BCEA (fixation task)
- Validity status and reason codes

## Calibration

NeuroLens+ uses multi-point calibration:

1. **Center baseline** (2s): Establishes baseline gaze and anchor positions
2. **3-point calibration**: Center, left, right for horizontal mapping
3. **9-point calibration**: Full grid for comprehensive mapping (grid task)

Calibration is rejected if:
- Gaze spread is insufficient (degeneracy)
- Calibration error exceeds 120px
- Slope magnitude is too small

## Validity Checking

Each trial is validated with explicit reason codes:

- `VALID` - Trial passed all checks
- `NO_FACE` - Face not detected
- `TOO_MANY_BLINKS` - Excessive blinking
- `LOW_FPS` - Frame rate too low
- `CALIBRATION_REJECTED` - Calibration failed
- `CLAMP_RATE_HIGH` - Too many clamped samples
- `WRONG_DIRECTION` - Saccade went wrong way
- `LATENCY_OUT_OF_RANGE` - Response too fast/slow
- `DURATION_OUT_OF_RANGE` - Movement too short/long
- And more...

## Acceptance Criteria

For typical webcam with stable lighting:
- Saccade valid trials: ≥85%
- Antisaccade valid trials: ≥80%
- Fixation valid fraction: ≥0.85
- Grid mean error: ≤120px
- Clamp rate: ≤2% average, ≤5% per trial

## Replay Mode

Reprocess frame logs to recompute biomarkers:

```python
from core.replay import ReplayEngine

engine = ReplayEngine()
results = engine.replay_and_compute('path/to/frame_log.csv', task='saccade')
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_core.py -v
pytest tests/test_synthetic.py -v
pytest tests/test_replay.py -v
```

## Project Structure

```
neurolens-plus/
├── core/
│   ├── tracker.py      # MediaPipe eye tracking
│   ├── calibration.py  # Calibration system
│   ├── mapping.py      # Gaze mapping
│   ├── validity.py     # Validity checking
│   ├── logging.py      # Data logging
│   ├── replay.py       # Replay engine
│   └── utils.py        # Utility functions
├── tasks/
│   ├── base.py         # Base task class
│   ├── fixation.py     # Fixation task
│   ├── pursuit.py      # Smooth pursuit task
│   ├── saccade.py      # Saccade task
│   ├── antisaccade.py  # Anti-saccade task
│   ├── grid9.py        # 9-point grid task (screen-safe)
│   └── visual_search.py # Visual search task (covert blink)
├── ui/
│   └── ui.py           # Pygame UI
├── tests/
│   ├── test_core.py    # Unit tests
│   ├── test_synthetic.py # Synthetic tests
│   └── test_replay.py  # Replay tests
├── main.py             # Entry point
├── requirements.txt    # Dependencies
├── README.md           # This file
└── METHODS.md          # Research methods documentation
```

## Known Limitations

1. **Webcam Resolution**: Standard webcams provide lower precision than research-grade eye trackers
2. **Pupil Measurement**: Uses iris diameter as proxy since webcams cannot directly measure pupil size
3. **Lighting Sensitivity**: Performance varies with ambient lighting conditions
4. **Head Movement**: While compensated, large head movements may affect accuracy

## Troubleshooting

**Camera not detected:**
- Ensure webcam is connected and not in use by another application
- Try `cv2.VideoCapture(1)` if multiple cameras are present

**Low FPS:**
- Close other applications using the camera
- Ensure adequate lighting
- Check CPU usage

**Calibration fails:**
- Ensure face is clearly visible
- Look directly at calibration points
- Maintain stable head position

**High clamp rate:**
- Recalibrate with better fixation
- Check for head drift during task

## License

MIT License

## Citation

If you use NeuroLens+ in your research, please cite:

```
NeuroLens+: A Webcam-based Eye Biomarker Platform
https://github.com/your-repo/neurolens-plus
```

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.
