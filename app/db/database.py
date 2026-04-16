from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection

from app.utils.config import get_settings

try:
    import mongomock
except ImportError:  # pragma: no cover - optional dependency for tests only
    mongomock = None


_CLIENT: MongoClient | None = None
_DATABASE = None
_ACTIVE_URI: str | None = None
_ACTIVE_DB_NAME: str | None = None


def _database_name_from_uri(uri: str, configured_name: str | None) -> str:
    if configured_name:
        return configured_name
    parsed = urlparse(uri)
    path = parsed.path.lstrip("/")
    if path:
        return path
    return "mindpulse"


def _create_client(uri: str) -> MongoClient:
    if uri.startswith("mongomock://"):
        if mongomock is None:
            raise RuntimeError("mongomock is required for in-memory MongoDB tests.")
        return mongomock.MongoClient()
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_database():
    global _CLIENT, _DATABASE, _ACTIVE_URI, _ACTIVE_DB_NAME

    settings = get_settings()
    db_name = _database_name_from_uri(settings.mongodb_uri, settings.mongodb_database)
    if _CLIENT is None or _DATABASE is None or _ACTIVE_URI != settings.mongodb_uri or _ACTIVE_DB_NAME != db_name:
        _CLIENT = _create_client(settings.mongodb_uri)
        _DATABASE = _CLIENT[db_name]
        _ACTIVE_URI = settings.mongodb_uri
        _ACTIVE_DB_NAME = db_name
    return _DATABASE


def reset_database_state() -> None:
    global _CLIENT, _DATABASE, _ACTIVE_URI, _ACTIVE_DB_NAME

    if _CLIENT is not None:
        _CLIENT.close()
    _CLIENT = None
    _DATABASE = None
    _ACTIVE_URI = None
    _ACTIVE_DB_NAME = None


def _users() -> Collection:
    return get_database()["users"]


def _sessions() -> Collection:
    return get_database()["auth_sessions"]


def _results() -> Collection:
    return get_database()["inference_results"]


def _counters() -> Collection:
    return get_database()["counters"]


def _next_sequence(name: str) -> int:
    counter = _counters().find_one_and_update(
        {"_id": name},
        {"$inc": {"value": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(counter["value"])


def initialize_database() -> None:
    db = get_database()
    db.command("ping")
    _users().create_index([("email", ASCENDING)], unique=True, name="idx_users_email_unique")
    _users().create_index([("username", ASCENDING)], unique=True, name="idx_users_username_unique")
    _sessions().create_index([("token_hash", ASCENDING)], unique=True, name="idx_sessions_token_hash_unique")
    _sessions().create_index([("expires_at", ASCENDING)], name="idx_sessions_expires_at")
    _results().create_index([("id", DESCENDING)], unique=True, name="idx_results_id_unique")
    _results().create_index([("user_id", ASCENDING), ("id", DESCENDING)], name="idx_results_user_id_id")


def _normalize_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _normalize_user(document: dict[str, Any] | None, include_password_hash: bool = False) -> dict[str, Any] | None:
    if document is None:
        return None
    payload = {
        "id": int(document["id"]),
        "name": document["name"],
        "username": document["username"],
        "email": document["email"],
        "created_at": _normalize_datetime(document["created_at"]),
    }
    if include_password_hash:
        payload["password_hash"] = document["password_hash"]
    return payload


def _normalize_result(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {
        "id": int(document["id"]),
        "user_id": int(document["user_id"]) if document.get("user_id") is not None else None,
        "timestamp": _normalize_datetime(document["timestamp"]),
        "face_emotion": document["face_emotion"],
        "face_confidence": float(document["face_confidence"]),
        "voice_emotion": document["voice_emotion"],
        "voice_confidence": float(document["voice_confidence"]),
        "stress_level": document["stress_level"],
        "source": document["source"],
    }


def get_user_by_email(email: str, include_password_hash: bool = False) -> dict[str, Any] | None:
    document = _users().find_one({"email": email})
    return _normalize_user(document, include_password_hash=include_password_hash)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    document = _users().find_one({"username": username})
    return _normalize_user(document)


def get_user_by_id(user_id: int, include_password_hash: bool = False) -> dict[str, Any] | None:
    document = _users().find_one({"id": int(user_id)})
    return _normalize_user(document, include_password_hash=include_password_hash)


def create_user(name: str, username: str, email: str, password_hash: str, created_at: datetime) -> dict[str, Any]:
    user_id = _next_sequence("users")
    document = {
        "id": user_id,
        "name": name,
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "created_at": created_at,
    }
    _users().insert_one(document)
    return _normalize_user(document)


def update_user_profile(user_id: int, name: str, username: str, email: str) -> dict[str, Any] | None:
    updated = _users().find_one_and_update(
        {"id": int(user_id)},
        {"$set": {"name": name, "username": username, "email": email}},
        return_document=ReturnDocument.AFTER,
    )
    return _normalize_user(updated)


def update_user_password(user_id: int, password_hash: str) -> None:
    _users().update_one({"id": int(user_id)}, {"$set": {"password_hash": password_hash}})


def create_session(user_id: int, token_hash: str, created_at: datetime, expires_at: datetime) -> None:
    document = {
        "id": _next_sequence("auth_sessions"),
        "user_id": int(user_id),
        "token_hash": token_hash,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    _sessions().insert_one(document)


def get_user_for_session(token_hash: str, now: datetime) -> dict[str, Any] | None:
    session = _sessions().find_one({"token_hash": token_hash, "expires_at": {"$gt": now}})
    if session is None:
        return None
    return get_user_by_id(int(session["user_id"]))


def delete_session_by_token_hash(token_hash: str) -> None:
    _sessions().delete_one({"token_hash": token_hash})


def delete_sessions_by_user_id(user_id: int) -> None:
    _sessions().delete_many({"user_id": int(user_id)})


def create_inference_result(payload: dict[str, Any], user_id: int | None) -> dict[str, Any]:
    result_id = _next_sequence("inference_results")
    document = {
        "id": result_id,
        "user_id": int(user_id) if user_id is not None else None,
        "timestamp": payload["timestamp"],
        "face_emotion": payload["face_emotion"],
        "face_confidence": float(payload["face_confidence"]),
        "voice_emotion": payload["voice_emotion"],
        "voice_confidence": float(payload["voice_confidence"]),
        "stress_level": payload["stress_level"],
        "source": payload["source"],
    }
    _results().insert_one(document)
    return _normalize_result(document)


def get_latest_inference_result() -> dict[str, Any] | None:
    document = _results().find_one(sort=[("id", DESCENDING)])
    return _normalize_result(document)


def list_inference_history(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    cursor = _results().find().sort("id", DESCENDING).skip(int(offset)).limit(int(limit))
    return [_normalize_result(document) for document in cursor]


def get_dashboard_summary_data(user_id: int, limit: int = 8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    match_filter = {"$or": [{"user_id": int(user_id)}, {"user_id": None}, {"user_id": {"$exists": False}}]}
    recent_cursor = _results().find(match_filter).sort("id", DESCENDING).limit(int(limit))
    recent_results = [_normalize_result(document) for document in recent_cursor]
    aggregate = list(
        _results().aggregate(
            [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "total_results": {"$sum": 1},
                        "average_face_confidence": {"$avg": "$face_confidence"},
                        "average_voice_confidence": {"$avg": "$voice_confidence"},
                        "low_count": {
                            "$sum": {"$cond": [{"$eq": ["$stress_level", "low"]}, 1, 0]}
                        },
                        "medium_count": {
                            "$sum": {"$cond": [{"$eq": ["$stress_level", "medium"]}, 1, 0]}
                        },
                        "high_count": {
                            "$sum": {"$cond": [{"$eq": ["$stress_level", "high"]}, 1, 0]}
                        },
                    }
                },
            ]
        )
    )
    aggregate_payload = aggregate[0] if aggregate else {}
    return recent_results, aggregate_payload
