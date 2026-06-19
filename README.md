# MindPulse

MindPulse is a multimodal stress monitoring project built for a normal laptop.

It uses:

- a face model to read facial emotion from webcam images
- a voice model to read emotion or stress tendency from microphone audio
- a late-fusion layer to combine both results into a final stress level
- a FastAPI backend
- a MongoDB database
- a Next.js frontend with login, profile, history, and mobile-friendly dashboard navigation

The project is designed to be easy to explain, easy to run step by step, and suitable for a final year project presentation.

## What the system does

MindPulse works in this order:

1. The face model predicts an emotion from a cropped face image.
2. The voice model predicts an emotion or stress tendency from a short audio clip.
3. The fusion logic combines both outputs.
4. The final result is labeled as:
   - `low`
   - `medium`
   - `high`
5. The result is saved through the backend and shown in the dashboard.

## Why this is still a multimodal system

This system is still multimodal because it uses two different input types:

- image input from the webcam
- audio input from the microphone

These two branches are trained separately. Each branch makes its own prediction first. The final stress result is produced only after both outputs are combined with late fusion.

## Main features

- facial emotion recognition with MobileNetV2 transfer learning
- voice emotion or stress tendency recognition from mel-spectrograms
- rule-based late fusion as the main stress prediction method
- optional learned fusion model as an experiment
- live browser-based webcam and microphone capture
- FastAPI backend for auth, inference storage, and dashboard data
- MongoDB persistence so hosted accounts and history survive redeploys
- Next.js frontend with:
  - sign up
  - log in
  - profile update
  - password change
  - history page
  - mobile bottom navigation

## Project documents

- `PROJECT_PRESENTATION.md` gives a short presentation-ready overview.
- `PROJECT_RESULTS.md` summarizes the current model results and project discussion.
- `MODEL_PIPELINE_AND_EVALUATION.md` explains the model-building process, preprocessing, evaluation metrics, and final results in detail.

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
  frontend/
```

## Dataset format

### Face dataset

Expected folder layout:

```text
data/raw/face/
  angry/
  happy/
  neutral/
  sad/
```

Each emotion folder should contain image files such as:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

### Voice dataset

Expected folder layout:

```text
data/raw/voice/
  calm/
  happy/
  neutral/
  sad/
  stressed/
```

Each class folder should contain audio files such as:

- `.wav`
- `.mp3`
- `.flac`
- `.ogg`

### Important note about datasets

The face and voice datasets do not need to come from the same source.

This project supports training the two branches separately and then combining them later with late fusion. That is the intended design.

## Local setup

### 1. Create and activate a Python virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Create backend environment file

Create a root `.env` file using `.env.example`.

Example:

```env
APP_ENV=development
API_HOST=127.0.0.1
API_PORT=8000

MONGODB_URI=mongodb://127.0.0.1:27017/mindpulse
MONGODB_DATABASE=mindpulse

FACE_DATA_DIR=./data/raw/face
VOICE_DATA_DIR=./data/raw/voice
FACE_PROCESSED_DIR=./data/processed/face
VOICE_PROCESSED_DIR=./data/processed/voice

FACE_MODEL_PATH=./models/face_emotion_model.weights.h5
FACE_LABELS_PATH=./models/face_labels.json
VOICE_MODEL_PATH=./models/voice_emotion_model.weights.h5
VOICE_LABELS_PATH=./models/voice_labels.json
VOICE_PREPROCESSING_PATH=./models/voice_preprocessing.json
FUSION_MODEL_PATH=./models/fusion_model.pkl
FUSION_RULES_PATH=./models/fusion_rules.json

LIVE_CAPTURE_INTERVAL_SECONDS=5
AUDIO_RECORD_SECONDS=3
AUDIO_SAMPLE_RATE=16000
API_BASE_URL=http://127.0.0.1:8000
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
LOG_LEVEL=INFO
```

## Frontend setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

For full local development:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

If you want the frontend to use the hosted backend instead:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.onrender.com
```

Start the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

## Run the backend

```bash
python scripts/run_api.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Train the models

If new AffectNet/IEMOCAP-style datasets are present in the workspace, prepare them first:

```bash
python scripts/prepare_new_datasets.py
```

This can merge supported labels into the existing training folders:

- face: `angry`, `happy`, `neutral`, `sad`
- voice: `happy`, `neutral`, `sad`, `stressed`

The script writes a preparation report to `data/processed/dataset_prepare_report.json`.

For the latest higher-accuracy training run, clean dataset folders were also prepared under:

- `data/processed/face_affectnet_clean`
- `data/processed/voice_iemocap_clean`

### Face model

```bash
python scripts/train_face.py --data-dir data/processed/face_affectnet_clean --epochs 12 --batch-size 32 --max-files-per-class 3200 --fine-tune-epochs 12 --fine-tune-layers 110
```

Evaluate face model:

```bash
python scripts/evaluate_face.py --data-dir data/processed/face_affectnet_clean --max-files-per-class 3200
```

### Voice model

By default, the voice branch now trains a simpler binary stress-tendency classifier:

- `not_stressed`
- `stressed`

This is more reliable for this project than trying to separate many detailed voice emotions.

```bash
python scripts/train_voice.py --data-dir data/processed/voice_iemocap_clean --epochs 14 --batch-size 32 --max-files-per-class 2994 --label-mode binary_stress --augment-copies 1
```

Evaluate voice model:

```bash
python scripts/evaluate_voice.py --data-dir data/processed/voice_iemocap_clean --max-files-per-class 2994 --label-mode binary_stress
```

### Current local results

These are the latest local evaluation results after retraining. They are useful for a final year project report, but they should not be described as medical-grade accuracy.

| Branch | Labels | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Face | angry, happy, neutral, sad | 81.7% | 82.1% | 81.7% | 81.6% |
| Voice | not_stressed, stressed | 72.5% | 72.5% | 72.5% | 72.5% |

Recent accuracy-focused improvements:

- Face training now uses a cleaner AffectNet-only four-class folder, lighting normalization, balanced sampling, class weighting, and longer fine-tuning.
- Voice training now uses the cleaner IEMOCAP-only voice folder, trims silence, normalizes audio, adds light waveform augmentation, and trains a binary stress-tendency CNN.
- Live inference rejects weak face crops and very quiet microphone samples instead of forcing unreliable predictions.
- These numbers are still dataset-dependent and should not be presented as medical-grade performance.

### Fusion

Main method:

- rule-based late fusion

Optional experiment:

```bash
python scripts/train_fusion.py --max-pairs 3000 --face-max-files-per-class 3200 --voice-max-files-per-class 2994 --voice-label-mode binary_stress
```

### True overall multimodal accuracy

To measure the combined face + voice accuracy, add a small paired validation set:

```text
data/raw/fusion_eval/
  sample_001/
    face.jpg
    voice.wav
    label.txt
  sample_002/
    face.jpg
    voice.wav
    label.txt
```

Each `label.txt` must contain exactly one stress label:

```text
low
medium
high
```

Then run:

```bash
python scripts/evaluate_fusion.py
```

This saves:

- `models/fusion_evaluation.json`
- `models/fusion_confusion_matrix.png`

This evaluation is optional and does not affect normal model training, the backend, or live inference. It only gives a real combined accuracy score when the paired validation samples are present.

Tune rule-based fusion:

```bash
python scripts/tune_fusion.py
```

## Run live inference

### Local Python live pipeline

```bash
python scripts/run_live_inference.py
```

Demo mode:

```bash
python scripts/run_live_inference.py --demo
python scripts/run_live_inference.py --demo --max-cycles 2
```

### Browser-based live session

The web app can also capture:

- webcam frame
- microphone sample

Then it sends them to the backend endpoint:

```text
POST /sessions/analyze-sample
```

This is the main hosted flow used by the frontend dashboard.

## API endpoints

### Auth and profile

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `PATCH /auth/profile`
- `POST /auth/change-password`
- `POST /auth/logout`

### Inference and dashboard

- `POST /inference-result`
- `POST /sessions/analyze-sample`
- `GET /latest-result`
- `GET /history`
- `GET /dashboard/summary`
- `GET /health`

### Sample inference payload

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

## Current saved model files

Important model artifacts used by the app:

- `models/face_emotion_model.weights.h5`
- `models/face_labels.json`
- `models/voice_emotion_model.weights.h5`
- `models/voice_labels.json`
- `models/voice_preprocessing.json`
- `models/fusion_rules.json`
- `models/fusion_model.pkl` (optional experimental fusion model)

## Current frontend behavior

The web app includes:

- sign up
- log in
- profile editing
- password change
- latest reading summary
- history page
- mobile-friendly navigation

### Mobile dashboard behavior

On larger screens:

- the dashboard uses top navigation

On mobile:

- the dashboard uses a bottom navigation bar
- the bottom navigation has icons and labels
- the tabs are:
  - `Overview`
  - `Session`
  - `History`
  - `Profile`

The `Session` tab is separated on mobile so the app feels more like a real phone app.

## Hosting summary

### Backend

Recommended:

- Render

The backend now uses MongoDB instead of SQLite so that user accounts and history do not disappear after redeploys.

Important Render environment variables:

```env
APP_ENV=production
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/mindpulse?retryWrites=true&w=majority
MONGODB_DATABASE=mindpulse
FACE_MODEL_PATH=./models/face_emotion_model.weights.h5
FACE_LABELS_PATH=./models/face_labels.json
VOICE_MODEL_PATH=./models/voice_emotion_model.weights.h5
VOICE_LABELS_PATH=./models/voice_labels.json
VOICE_PREPROCESSING_PATH=./models/voice_preprocessing.json
FUSION_RULES_PATH=./models/fusion_rules.json
AUDIO_SAMPLE_RATE=16000
AUDIO_RECORD_SECONDS=3
FRONTEND_ORIGINS=https://your-frontend-url.vercel.app
```

### Frontend

Recommended:

- Vercel

Frontend environment variable:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.onrender.com
```

## MongoDB Atlas note

If you use MongoDB Atlas, you usually need:

- a database user
- a connection string
- network access configured

For testing, Atlas network access is often set to:

```text
0.0.0.0/0
```

That allows connections from anywhere.

## Evaluation

The evaluation scripts report:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

The project does not invent performance numbers. Results depend on the datasets used.

## Presentation points

If you are presenting this project, these are the main talking points:

- It is a multimodal system.
- Face and voice are trained separately.
- Stress is predicted with late fusion.
- The system works with only a laptop webcam and microphone.
- It supports live browser capture.
- It includes authentication, dashboard history, and profile management.
- It uses MongoDB for hosted persistence.
- It is practical and lightweight rather than overengineered.
