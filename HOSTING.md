# MindPulse Hosting Guide

This project is prepared for a hosted frontend and backend.

## Architecture

```text
Browser on user device
  captures webcam frame + microphone WAV sample
        |
Next.js frontend
        |
FastAPI backend
  runs face model + voice model + fusion
        |
SQLite database
```

Camera and microphone access requires HTTPS in deployment. `localhost` works during development.

## 1. Backend Hosting

Recommended first deployment target: Render Docker web service.

Backend files:

- `Dockerfile`
- `.dockerignore`
- `requirements-backend.txt`
- `render.yaml`

Required environment variables:

```text
APP_ENV=production
DATABASE_URL=sqlite:///./stress_results.db
FACE_MODEL_PATH=./models/face_emotion_model.weights.h5
FACE_LABELS_PATH=./models/face_labels.json
VOICE_MODEL_PATH=./models/voice_emotion_model.weights.h5
VOICE_LABELS_PATH=./models/voice_labels.json
VOICE_PREPROCESSING_PATH=./models/voice_preprocessing.json
FUSION_RULES_PATH=./models/fusion_rules.json
AUDIO_SAMPLE_RATE=16000
AUDIO_RECORD_SECONDS=3
FRONTEND_ORIGINS=https://your-frontend-domain.vercel.app
```

Important:

- The active model files are included in the repository because the hosted backend needs them.
- SQLite is acceptable for a demo deployment, but a production deployment should move to a persistent managed database.
- Free/low-memory hosts may struggle with TensorFlow. Use a plan with enough RAM.

## 2. Frontend Hosting

Recommended target: Vercel.

Use `frontend/` as the project root.

Required environment variable:

```text
NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.onrender.com
```

After setting the env var, redeploy the frontend.

## 3. Deployment Order

1. Push project to GitHub.
2. Deploy backend first.
3. Copy backend URL.
4. Set frontend `NEXT_PUBLIC_API_BASE_URL` to backend URL.
5. Deploy frontend.
6. Copy frontend URL.
7. Set backend `FRONTEND_ORIGINS` to frontend URL.
8. Redeploy backend.

## 4. Smoke Test

Backend:

```text
GET /health
```

Frontend:

1. Register or login.
2. Start a session.
3. Allow camera and microphone permissions.
4. Wait for a reading to save.
5. Confirm dashboard and history update.
