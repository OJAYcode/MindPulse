from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status

from app.api.schemas import (
    AuthResponse,
    DashboardSummary,
    HealthResponse,
    InferenceResultIn,
    InferenceResultOut,
    PasswordChange,
    StatusResponse,
    StressDistribution,
    UserCreate,
    UserLogin,
    UserOut,
    UserProfileUpdate,
)
from app.db.database import get_connection, initialize_database
from app.inference.upload_runtime import analyze_uploaded_sample
from app.utils.security import create_token, hash_password, hash_token, verify_password


router = APIRouter()
SESSION_DAYS = 7


def _row_to_user(row) -> UserOut:
    username = row["username"] if "username" in row.keys() and row["username"] else row["email"].split("@", 1)[0]
    return UserOut(id=row["id"], name=row["name"], username=username, email=row["email"], created_at=row["created_at"])


def _row_to_result(row) -> InferenceResultOut:
    return InferenceResultOut(**dict(row))


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def get_current_user(authorization: str | None = Header(default=None)) -> UserOut:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    token_digest = hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.name, users.username, users.email, users.created_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?
            """,
            (token_digest, now),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    return _row_to_user(row)


def get_optional_user(authorization: str | None = Header(default=None)) -> UserOut | None:
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    token_digest = hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.name, users.username, users.email, users.created_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?
            """,
            (token_digest, now),
        ).fetchone()
    return _row_to_user(row) if row is not None else None


def _insert_inference_result(payload: InferenceResultIn, user_id: int | None) -> InferenceResultOut:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO inference_results (
                user_id,
                timestamp,
                face_emotion,
                face_confidence,
                voice_emotion,
                voice_confidence,
                stress_level,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                payload.timestamp.isoformat(),
                payload.face_emotion,
                payload.face_confidence,
                payload.voice_emotion,
                payload.voice_confidence,
                payload.stress_level,
                payload.source,
            ),
        )
        connection.commit()
        record_id = cursor.lastrowid
    return InferenceResultOut(id=record_id, user_id=user_id, **payload.model_dump())
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    initialize_database()
    return HealthResponse(status="ok", database="ready")


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate) -> AuthResponse:
    email = payload.email.strip().lower()
    username = payload.username.strip().lower()
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")
        existing_username = connection.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing_username is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
        cursor = connection.execute(
            """
            INSERT INTO users (name, username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (payload.name.strip(), username, email, hash_password(payload.password), created_at),
        )
        user_id = cursor.lastrowid
        token = create_token()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
        connection.execute(
            """
            INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, hash_token(token), created_at, expires_at),
        )
        connection.commit()
        row = connection.execute(
            "SELECT id, name, username, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return AuthResponse(token=token, user=_row_to_user(row))


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: UserLogin) -> AuthResponse:
    email = payload.email.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, name, username, email, password_hash, created_at FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None or not verify_password(payload.password, row["password_hash"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
        token = create_token()
        created_at = datetime.now(timezone.utc).isoformat()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
        connection.execute(
            """
            INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (row["id"], hash_token(token), created_at, expires_at),
        )
        connection.commit()
    return AuthResponse(token=token, user=_row_to_user(row))


@router.get("/auth/me", response_model=UserOut)
def me(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    return current_user


@router.patch("/auth/profile", response_model=UserOut)
@router.put("/auth/profile", response_model=UserOut)
@router.post("/auth/profile", response_model=UserOut)
def update_profile(
    payload: UserProfileUpdate,
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    email = payload.email.strip().lower()
    username = payload.username.strip().lower()
    with get_connection() as connection:
        existing_email = connection.execute(
            "SELECT id FROM users WHERE email = ? AND id != ?",
            (email, current_user.id),
        ).fetchone()
        if existing_email is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")
        existing_username = connection.execute(
            "SELECT id FROM users WHERE username = ? AND id != ?",
            (username, current_user.id),
        ).fetchone()
        if existing_username is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
        connection.execute(
            """
            UPDATE users
            SET name = ?, username = ?, email = ?
            WHERE id = ?
            """,
            (payload.name.strip(), username, email, current_user.id),
        )
        connection.commit()
        row = connection.execute(
            "SELECT id, name, username, email, created_at FROM users WHERE id = ?",
            (current_user.id,),
        ).fetchone()
    return _row_to_user(row)


@router.post("/auth/change-password", response_model=StatusResponse)
def change_password(
    payload: PasswordChange,
    current_user: UserOut = Depends(get_current_user),
) -> StatusResponse:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (current_user.id,),
        ).fetchone()
        if row is None or not verify_password(payload.current_password, row["password_hash"]):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(payload.new_password), current_user.id),
        )
        connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (current_user.id,))
        connection.commit()
    return StatusResponse(status="password_updated")


@router.post("/auth/logout", response_model=StatusResponse)
def logout(authorization: str | None = Header(default=None)) -> StatusResponse:
    token = _extract_bearer_token(authorization)
    if token:
        with get_connection() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (hash_token(token),))
            connection.commit()
    return StatusResponse(status="ok")


@router.post("/inference-result", response_model=InferenceResultOut)
def create_inference_result(
    payload: InferenceResultIn,
    current_user: UserOut | None = Depends(get_optional_user),
) -> InferenceResultOut:
    return _insert_inference_result(payload, current_user.id if current_user else None)


@router.post("/sessions/analyze-sample", response_model=InferenceResultOut)
async def analyze_sample(
    face_image: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
) -> InferenceResultOut:
    image_bytes = await face_image.read()
    audio_bytes = await audio_file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty.")
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")
    try:
        result = analyze_uploaded_sample(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            audio_filename=audio_file.filename or "audio.webm",
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    payload = InferenceResultIn(
        timestamp=result.timestamp,
        face_emotion=result.face_emotion,
        face_confidence=result.face_confidence,
        voice_emotion=result.voice_emotion,
        voice_confidence=result.voice_confidence,
        stress_level=result.stress_level,
        source=result.source,
    )
    return _insert_inference_result(payload, current_user.id)


@router.get("/latest-result", response_model=InferenceResultOut)
def latest_result() -> InferenceResultOut:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, timestamp, face_emotion, face_confidence, voice_emotion,
                   voice_confidence, stress_level, source, user_id
            FROM inference_results
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="No inference results found.")

    return _row_to_result(row)


@router.get("/history", response_model=list[InferenceResultOut])
def history(
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[InferenceResultOut]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp, face_emotion, face_confidence, voice_emotion,
                   voice_confidence, stress_level, source, user_id
            FROM inference_results
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return [_row_to_result(row) for row in rows]


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(
    current_user: UserOut = Depends(get_current_user),
    limit: int = Query(default=8, ge=1, le=50),
) -> DashboardSummary:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp, face_emotion, face_confidence, voice_emotion,
                   voice_confidence, stress_level, source, user_id
            FROM inference_results
            WHERE user_id = ? OR user_id IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (current_user.id, limit),
        ).fetchall()
        aggregate = connection.execute(
            """
            SELECT
                COUNT(*) AS total_results,
                AVG(face_confidence) AS average_face_confidence,
                AVG(voice_confidence) AS average_voice_confidence,
                SUM(CASE WHEN stress_level = 'low' THEN 1 ELSE 0 END) AS low_count,
                SUM(CASE WHEN stress_level = 'medium' THEN 1 ELSE 0 END) AS medium_count,
                SUM(CASE WHEN stress_level = 'high' THEN 1 ELSE 0 END) AS high_count
            FROM inference_results
            WHERE user_id = ? OR user_id IS NULL
            """,
            (current_user.id,),
        ).fetchone()

    recent_results = [_row_to_result(row) for row in rows]
    total_results = int(aggregate["total_results"] or 0)
    high_count = int(aggregate["high_count"] or 0)
    return DashboardSummary(
        total_results=total_results,
        latest_result=recent_results[0] if recent_results else None,
        stress_distribution=StressDistribution(
            low=int(aggregate["low_count"] or 0),
            medium=int(aggregate["medium_count"] or 0),
            high=high_count,
        ),
        average_face_confidence=float(aggregate["average_face_confidence"] or 0.0),
        average_voice_confidence=float(aggregate["average_voice_confidence"] or 0.0),
        high_stress_rate=float(high_count / total_results) if total_results else 0.0,
        recent_results=recent_results,
    )
