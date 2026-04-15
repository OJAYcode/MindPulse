from fastapi.testclient import TestClient
from uuid import uuid4

from app.inference.upload_runtime import UploadedInferenceResult
from app.api.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_and_dashboard_flow():
    client = TestClient(create_app())
    suffix = uuid4().hex[:8]
    email = f"demo.{suffix}@example.com"
    username = f"demo_{suffix}"
    updated_username = f"updated_{suffix}"
    register_response = client.post(
        "/auth/register",
        json={"name": "Demo User", "username": username, "email": email, "password": "password123"},
    )
    assert register_response.status_code == 201
    auth_payload = register_response.json()

    token = auth_payload["token"]
    headers = {"Authorization": f"Bearer {token}"}
    me_response = client.get("/auth/me", headers=headers)
    assert me_response.status_code == 200
    assert me_response.json()["email"] == email
    assert me_response.json()["username"] == username

    profile_response = client.patch(
        "/auth/profile",
        headers=headers,
        json={"name": "Demo Updated", "username": updated_username, "email": email},
    )
    assert profile_response.status_code == 200
    assert profile_response.json()["username"] == updated_username

    password_response = client.post(
        "/auth/change-password",
        headers=headers,
        json={"current_password": "password123", "new_password": "password1234"},
    )
    assert password_response.status_code == 200

    login_response = client.post(
        "/auth/login",
        json={"email": email, "password": "password1234"},
    )
    assert login_response.status_code == 200
    auth_payload = login_response.json()
    token = auth_payload["token"]
    headers = {"Authorization": f"Bearer {token}"}

    inference_response = client.post(
        "/inference-result",
        headers=headers,
        json={
            "timestamp": "2026-04-15T10:00:00Z",
            "face_emotion": "neutral",
            "face_confidence": 0.72,
            "voice_emotion": "calm",
            "voice_confidence": 0.68,
            "stress_level": "low",
            "source": "test_client",
        },
    )
    assert inference_response.status_code == 200
    assert inference_response.json()["user_id"] == auth_payload["user"]["id"]

    dashboard_response = client.get("/dashboard/summary", headers=headers)
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert dashboard["total_results"] >= 1
    assert "stress_distribution" in dashboard


def test_upload_sample_endpoint(monkeypatch):
    from datetime import datetime, timezone
    import app.api.routes as routes

    def fake_analyze_uploaded_sample(*_args, **_kwargs):
        return UploadedInferenceResult(
            timestamp=datetime.now(timezone.utc),
            face_emotion="neutral",
            face_confidence=0.8,
            voice_emotion="calm",
            voice_confidence=0.75,
            stress_level="low",
            source="browser_capture",
        )

    monkeypatch.setattr(routes, "analyze_uploaded_sample", fake_analyze_uploaded_sample)
    client = TestClient(create_app())
    suffix = uuid4().hex[:8]
    auth_response = client.post(
        "/auth/register",
        json={
            "name": "Upload User",
            "username": f"upload_{suffix}",
            "email": f"upload.{suffix}@example.com",
            "password": "password123",
        },
    )
    assert auth_response.status_code == 201
    token = auth_response.json()["token"]
    response = client.post(
        "/sessions/analyze-sample",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "face_image": ("frame.jpg", b"fake-image", "image/jpeg"),
            "audio_file": ("audio.webm", b"fake-audio", "audio/webm"),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "browser_capture"
    assert payload["stress_level"] == "low"
    assert payload["user_id"] == auth_response.json()["user"]["id"]
