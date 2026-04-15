# MindPulse Frontend

Next.js + TypeScript interface for authenticated wellbeing readings and stress-risk review.

## Setup

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

The frontend expects the FastAPI backend to be running at:

```bash
http://127.0.0.1:8000
```

Change `NEXT_PUBLIC_API_BASE_URL` in `.env.local` if your backend runs elsewhere.

## Features

- account registration
- login with bearer-token session storage
- authenticated workspace summary
- browser webcam and microphone capture
- uploaded sample analysis through the FastAPI backend
- latest risk reading
- stress distribution cards
- face and voice confidence cards
- recent reading history table
- mobile-responsive layout

## Session Capture

The browser captures a webcam frame and a short microphone clip, then uploads both to:

```text
POST /sessions/analyze-sample
```

Camera and microphone permissions require HTTPS in deployment. `localhost` works during development.

## Backend Endpoints Used

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /dashboard/summary`

## Design Direction

The UI uses a calm wellness-dashboard style:

- warm neutral background
- deep ink typography
- teal for low/calm states
- amber for medium/caution states
- coral for high stress states
- rounded dashboard cards
- responsive grid layout

The design is original, but the direction was informed by modern wellness, mental-health, and health-monitoring dashboard references.
