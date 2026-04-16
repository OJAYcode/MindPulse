from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from pymongo.errors import DuplicateKeyError

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
from app.db.database import (
    create_inference_result as create_inference_result_record,
    create_session,
    create_user,
    delete_session_by_token_hash,
    delete_sessions_by_user_id,
    get_dashboard_summary_data,
    get_latest_inference_result,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    get_user_for_session,
    initialize_database,
    list_inference_history,
    update_user_password,
    update_user_profile,
)
from app.inference.upload_runtime import analyze_uploaded_sample
from app.utils.security import create_token, hash_password, hash_token, verify_password


router = APIRouter()
SESSION_DAYS = 7


def _row_to_user(row) -> UserOut:
    username = row["username"] if row.get("username") else row["email"].split("@", 1)[0]
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
    row = get_user_for_session(token_digest, datetime.now(timezone.utc))
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    return _row_to_user(row)


def get_optional_user(authorization: str | None = Header(default=None)) -> UserOut | None:
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    token_digest = hash_token(token)
    row = get_user_for_session(token_digest, datetime.now(timezone.utc))
    return _row_to_user(row) if row is not None else None


def _insert_inference_result(payload: InferenceResultIn, user_id: int | None) -> InferenceResultOut:
    record = create_inference_result_record(payload.model_dump(), user_id)
    return InferenceResultOut(**record)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    initialize_database()
    return HealthResponse(status="ok", database="ready")


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate) -> AuthResponse:
    email = payload.email.strip().lower()
    username = payload.username.strip().lower()
    created_at = datetime.now(timezone.utc)
    if get_user_by_email(email) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")
    if get_user_by_username(username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
    try:
        row = create_user(payload.name.strip(), username, email, hash_password(payload.password), created_at)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account details already exist.") from exc
    token = create_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    create_session(row["id"], hash_token(token), created_at, expires_at)
    return AuthResponse(token=token, user=_row_to_user(row))


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: UserLogin) -> AuthResponse:
    email = payload.email.strip().lower()
    row = get_user_by_email(email, include_password_hash=True)
    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    token = create_token()
    created_at = datetime.now(timezone.utc)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    create_session(row["id"], hash_token(token), created_at, expires_at)
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
    existing_email = get_user_by_email(email)
    if existing_email is not None and existing_email["id"] != current_user.id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")
    existing_username = get_user_by_username(username)
    if existing_username is not None and existing_username["id"] != current_user.id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
    try:
        row = update_user_profile(current_user.id, payload.name.strip(), username, email)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Profile details already exist.") from exc
    return _row_to_user(row)


@router.post("/auth/change-password", response_model=StatusResponse)
def change_password(
    payload: PasswordChange,
    current_user: UserOut = Depends(get_current_user),
) -> StatusResponse:
    row = get_user_by_id(current_user.id, include_password_hash=True)
    if row is None or not verify_password(payload.current_password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")
    update_user_password(current_user.id, hash_password(payload.new_password))
    delete_sessions_by_user_id(current_user.id)
    return StatusResponse(status="password_updated")


@router.post("/auth/logout", response_model=StatusResponse)
def logout(authorization: str | None = Header(default=None)) -> StatusResponse:
    token = _extract_bearer_token(authorization)
    if token:
        delete_session_by_token_hash(hash_token(token))
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
    row = get_latest_inference_result()
    if row is None:
        raise HTTPException(status_code=404, detail="No inference results found.")
    return _row_to_result(row)


@router.get("/history", response_model=list[InferenceResultOut])
def history(
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[InferenceResultOut]:
    rows = list_inference_history(limit=limit, offset=offset)
    return [_row_to_result(row) for row in rows]


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(
    current_user: UserOut = Depends(get_current_user),
    limit: int = Query(default=8, ge=1, le=50),
) -> DashboardSummary:
    rows, aggregate = get_dashboard_summary_data(current_user.id, limit=limit)
    recent_results = [_row_to_result(row) for row in rows]
    total_results = int(aggregate.get("total_results", 0) or 0)
    high_count = int(aggregate.get("high_count", 0) or 0)
    return DashboardSummary(
        total_results=total_results,
        latest_result=recent_results[0] if recent_results else None,
        stress_distribution=StressDistribution(
            low=int(aggregate.get("low_count", 0) or 0),
            medium=int(aggregate.get("medium_count", 0) or 0),
            high=high_count,
        ),
        average_face_confidence=float(aggregate.get("average_face_confidence", 0.0) or 0.0),
        average_voice_confidence=float(aggregate.get("average_voice_confidence", 0.0) or 0.0),
        high_stress_rate=float(high_count / total_results) if total_results else 0.0,
        recent_results=recent_results,
    )
