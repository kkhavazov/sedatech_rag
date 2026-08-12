import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path(
    os.getenv("CACHE_DATABASE_PATH", Path(__file__).with_name("cache.db"))
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                last_message_id TEXT NOT NULL,
                message_ids_json TEXT NOT NULL,
                synced_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                role TEXT NOT NULL,
                body TEXT NOT NULL,
                position INTEGER NOT NULL,
                visible INTEGER NOT NULL DEFAULT 1,
                cached_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_ticket_position
            ON messages(ticket_id, position);

            CREATE TABLE IF NOT EXISTS drafts (
                ticket_id TEXT NOT NULL,
                based_on_message_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (ticket_id, based_on_message_id)
            );
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "visible" not in columns:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN visible INTEGER NOT NULL DEFAULT 1"
            )


def get_ticket_revision(ticket_id: str) -> str | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT last_message_id FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
    return row["last_message_id"] if row else None


def get_cached_messages(ticket_id: str, message_ids: list[str]) -> list[dict] | None:
    if not message_ids:
        return []

    placeholders = ",".join("?" for _ in message_ids)
    with _connection() as connection:
        rows = connection.execute(
            f"""
            SELECT message_id, role, body, visible
            FROM messages
            WHERE ticket_id = ? AND message_id IN ({placeholders})
            """,
            (ticket_id, *message_ids),
        ).fetchall()

    by_id = {row["message_id"]: row for row in rows}
    if any(message_id not in by_id for message_id in message_ids):
        return None
    return [
        {"role": by_id[message_id]["role"], "text": by_id[message_id]["body"]}
        for message_id in message_ids
        if by_id[message_id]["visible"]
    ]


def get_cached_message_ids(ticket_id: str, message_ids: list[str]) -> set[str]:
    if not message_ids:
        return set()
    placeholders = ",".join("?" for _ in message_ids)
    with _connection() as connection:
        rows = connection.execute(
            f"""
            SELECT message_id FROM messages
            WHERE ticket_id = ? AND message_id IN ({placeholders})
            """,
            (ticket_id, *message_ids),
        ).fetchall()
    return {row["message_id"] for row in rows}


def store_ticket_messages(
    ticket_id: str,
    message_ids: list[str],
    messages_by_id: dict[str, dict],
) -> None:
    if not message_ids:
        return

    now = _now()
    with _connection() as connection:
        for position, message_id in enumerate(message_ids):
            message = messages_by_id.get(message_id)
            if message is None:
                continue
            connection.execute(
                """
                INSERT INTO messages (
                    message_id, ticket_id, role, body, position, visible, cached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    ticket_id = excluded.ticket_id,
                    role = excluded.role,
                    body = excluded.body,
                    position = excluded.position,
                    visible = excluded.visible,
                    cached_at = excluded.cached_at
                """,
                (
                    message_id,
                    ticket_id,
                    message["role"],
                    message["text"],
                    position,
                    int(message.get("visible", True)),
                    now,
                ),
            )

        connection.execute(
            """
            INSERT INTO tickets (
                ticket_id, last_message_id, message_ids_json, synced_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                last_message_id = excluded.last_message_id,
                message_ids_json = excluded.message_ids_json,
                synced_at = excluded.synced_at
            """,
            (ticket_id, message_ids[-1], json.dumps(message_ids), now),
        )


def get_cached_draft(ticket_id: str, last_message_id: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT response_json FROM drafts
            WHERE ticket_id = ? AND based_on_message_id = ?
            """,
            (ticket_id, last_message_id),
        ).fetchone()
    return json.loads(row["response_json"]) if row else None


def store_draft(ticket_id: str, last_message_id: str, response: dict) -> None:
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO drafts (
                ticket_id, based_on_message_id, response_json, created_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(ticket_id, based_on_message_id) DO UPDATE SET
                response_json = excluded.response_json,
                created_at = excluded.created_at
            """,
            (ticket_id, last_message_id, json.dumps(response), _now()),
        )
