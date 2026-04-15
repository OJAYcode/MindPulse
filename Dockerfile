FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements-backend.txt

COPY app ./app
COPY training ./training
COPY models ./models
COPY .env.example ./.env.example

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
