# Multimodal Stress Detection System

This project implements a complete local multimodal stress-detection pipeline using only a laptop webcam and laptop microphone. It trains a facial emotion model, trains a voice emotion or stress-tendency model, combines both branches with late fusion, runs live inference, and stores results through a FastAPI backend backed by MongoDB.

## Why this is still a multimodal system

The system is multimodal because it uses two different input modalities:

- vision from cropped face images
- audio from short microphone clips converted to mel-spectrograms

Each branch is trained separately, produces its own probabilities, and the final stress decision is made by fusing both branch outputs.

## Project structure

```text
project_root/
  README.md
  requirements.txt
  .env.example
  app/
    api/
    db/
    fusion/
    inference/
    utils/
  training/
    face/
    voice/
  data/
    raw/
    processed/
  models/
  scripts/
  notebooks/
  tests/
```

## Dataset assumptions

### Face dataset

Expected default layout:

```text
data/raw/face/
  angry/
  happy/
  neutral/
  sad/
```

Each class folder should contain `.jpg`, `.jpeg`, `.png`, or `.bmp` images.

Current project setup:

- primary face training uses a 4-class layout: `angry`, `happy`, `neutral`, `sad`
- the workspace currently contains merged face data from multiple source datasets
- one additional numeric-label dataset was merged with this mapping:
  - `4 -> happy`
  - `5 -> sad`
  - `6 -> angry`
  - `7 -> neutral`
- labels outside this 4-class target set were intentionally excluded instead of being merged blindly

### Voice dataset

Expected default layout:

```text
data/raw/voice/
  calm/
  stressed/
  neutral/
  happy/
  sad/
```

Each class folder should contain `.wav`, `.mp3`, `.flac`, or `.ogg` files.

Prepared target folders already exist:

```text
data/raw/voice/
  calm/
  neutral/
  stressed/
  happy/
  sad/
```

If you add a source voice dataset in another layout later, it can be reorganized into these folders before training.

Current project setup:

- the workspace currently contains merged voice data from multiple source datasets
- one additional CREMA-D-like dataset was merged with this mapping:
  - `ANG`, `DIS`, `FEA -> stressed`
  - `HAP -> happy`
  - `NEU -> neutral`
  - `SAD -> sad`
- `calm` is still preserved as its own class from the original voice dataset

### Optional paired multimodal calibration set

If your main face and voice datasets are separate, that is fine. The recommended way to improve fusion is to collect a smaller paired multimodal validation set for tuning only.

Expected layout:

```text
data/raw/multimodal/
  manifest.csv
  sample_001_face.jpg
  sample_001_audio.wav
  sample_002_face.jpg
  sample_002_audio.wav
```

`manifest.csv` format:

```csv
face_image,audio_file,stress_label
sample_001_face.jpg,sample_001_audio.wav,low
sample_002_face.jpg,sample_002_audio.wav,high
```

This paired set is used only to tune fusion weights and thresholds, not to retrain the face or voice models.

If datasets are missing, the code falls back to a synthetic demo mode so the project still runs end to end.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

## Run the API

```bash
python scripts/run_api.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run the MindPulse frontend

The frontend lives in a separate folder:

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Then open:

```text
http://localhost:3000
```

MindPulse includes registration, login, and an authenticated wellbeing workspace that reads from the FastAPI backend.

Frontend environment:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Train and evaluate

```bash
python scripts/train_face.py --epochs 5 --batch-size 32 --max-files-per-class 750 --fine-tune-epochs 4 --fine-tune-layers 60
python scripts/evaluate_face.py --max-files-per-class 750
python scripts/train_voice.py --epochs 5 --batch-size 16 --max-files-per-class 192
python scripts/evaluate_voice.py --max-files-per-class 192
python scripts/train_fusion.py --max-pairs 1500 --face-max-files-per-class 750 --voice-max-files-per-class 192
python scripts/tune_fusion.py
```

Saved artifacts include trained models, label files, training history, evaluation JSON, and confusion matrix images under `models/`.

### Active face model artifact

The face branch currently uses a weights-based artifact for reliability on this environment:

- `models/face_emotion_model.weights.h5`

The application rebuilds the face architecture in code and then loads these saved weights.

## Optional learned fusion model

The default system uses rule-based late fusion in production, especially when the main face and voice datasets are separate. This is the main presentation-ready multimodal method in this repo.

If you also want a tiny learned fusion classifier:

```bash
python scripts/train_fusion.py
```

Demo version:

```bash
python scripts/train_fusion.py --demo
```

This script:

- loads the trained face and voice models
- gets branch probability outputs on their test sets
- creates paired fusion features from face probabilities plus voice probabilities
- generates stress targets from the existing rule-based fusion
- trains and saves a logistic-regression fusion model

Saved artifacts:

- `models/fusion_model.pkl`
- `models/fusion_model.json`

Note: because the face and voice datasets are usually separate, this optional logistic model is trained to approximate the rule-based fusion unless you later replace it with truly aligned multimodal labels. Treat it as an experimental add-on, not the main system claim.

## Recommended fusion tuning for separate datasets

When your main datasets are separate, the recommended path is:

1. train the face model on folder-per-emotion face data
2. train the voice model on folder-per-emotion voice data
3. collect a smaller paired multimodal calibration set
4. tune rule-based fusion weights and thresholds on that paired set

Run:

```bash
python scripts/tune_fusion.py
```

This script:

- loads the trained face and voice models
- runs them on a paired multimodal manifest
- searches for better face versus voice weights
- searches for better `medium` and `high` stress thresholds
- saves the tuned rule config to `models/fusion_rules.json`

Saved artifacts:

- `models/fusion_rules.json`
- `models/fusion_rules.metrics.json`

This is the most defensible approach when your face and voice training datasets are not naturally aligned sample by sample.

## Current merged-data results

Latest evaluated branch metrics in this workspace:

- face accuracy: `0.4650`
- face precision: `0.5575`
- face recall: `0.4650`
- face F1: `0.4301`
- voice accuracy: `0.3385`
- voice precision: `0.2093`
- voice recall: `0.3385`
- voice F1: `0.2548`

Files:

- `models/face_evaluation.json`
- `models/voice_evaluation.json`

These numbers are intentionally reported as-is. They are modest, and that is normal for a practical laptop baseline built from merged emotion datasets with different sources and label schemes.

## Current fusion policy

The default production path is rule-based late fusion:

- face and voice are trained separately
- each branch outputs class probabilities
- the final stress label is produced from configurable mappings, branch weights, and thresholds
- stronger voice evidence is given slightly more weight by default
- the rule now includes a confidence-aware adjustment so strong `stressed` or `angry` predictions can reach `high`

The optional logistic-regression fusion model remains available, but it is documented and stored as experimental.

After rebalancing the default rules, the optional learned fusion training set now produces all three stress labels instead of collapsing to only `low` and `medium`.

## Run live inference

Start the API first, then run:

```bash
python scripts/run_live_inference.py
```

Demo mode:

```bash
python scripts/run_live_inference.py --demo
python scripts/run_live_inference.py --demo --max-cycles 2
```

Rule-based live inference automatically uses `models/fusion_rules.json` if you have tuned and saved one.

The face runtime loads `models/face_emotion_model.weights.h5` by rebuilding the architecture and applying the saved weights.

Use the optional logistic fusion model during live inference:

```bash
python scripts/run_live_inference.py --use-learned-fusion
```

For clean validation without leaving the process running forever:

```bash
python scripts/run_live_inference.py --demo --max-cycles 2
```

## API endpoints

- `POST /inference-result`
- `GET /latest-result`
- `GET /history`
- `GET /health`

Sample inference payload:

```json
{
  "timestamp": "2026-03-30T10:15:00Z",
  "face_emotion": "sad",
  "face_confidence": 0.81,
  "voice_emotion": "stressed",
  "voice_confidence": 0.76,
  "stress_level": "high",
  "source": "laptop_webcam_mic"
}
```

## Evaluation

Both branch evaluators report:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

Performance depends on the local dataset you provide. The project does not invent final numbers.

## Presentation notes

- local-first and laptop-friendly
- separate face and voice branches
- clear late-fusion multimodal design
- honest handling of separate datasets through paired calibration for fusion
- FastAPI plus MongoDB backend for dashboard integration
- demo mode available before full datasets are ready
