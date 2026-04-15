from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from app.utils.config import get_settings


def _sqlite_path() -> Path:
    settings = get_settings()
    if settings.database_url.startswith("sqlite:///"):
        raw_path = settings.database_url.replace("sqlite:///", "", 1)
        return Path(raw_path).resolve()
    parsed = urlparse(settings.database_url)
    return Path(parsed.path).resolve()


def initialize_database() -> None:
    db_path = _sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT NOT NULL,
                face_emotion TEXT NOT NULL,
                face_confidence REAL NOT NULL,
                voice_emotion TEXT NOT NULL,
                voice_confidence REAL NOT NULL,
                stress_level TEXT NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(inference_results)").fetchall()
        }
        if "user_id" not in columns:
            connection.execute("ALTER TABLE inference_results ADD COLUMN user_id INTEGER")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        user_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if "username" not in user_columns:
            connection.execute("ALTER TABLE users ADD COLUMN username TEXT")
            rows = connection.execute("SELECT id, email FROM users").fetchall()
            for row in rows:
                base_username = row[1].split("@", 1)[0].replace(".", "_").replace("-", "_")
                candidate = base_username
                suffix = 1
                while connection.execute("SELECT id FROM users WHERE username = ?", (candidate,)).fetchone():
                    suffix += 1
                    candidate = f"{base_username}{suffix}"
                connection.execute("UPDATE users SET username = ? WHERE id = ?", (candidate, row[0]))
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        else:
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON auth_sessions(token_hash)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_results_user_id ON inference_results(user_id)")
        connection.commit()


@contextmanager
def get_connection():
    db_path = _sqlite_path()
    initialize_database()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
