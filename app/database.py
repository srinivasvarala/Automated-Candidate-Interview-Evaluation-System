import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "interviews.db"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS interviews (
    id TEXT PRIMARY KEY,
    job_position TEXT NOT NULL,
    job_description TEXT DEFAULT '',
    interview_type TEXT DEFAULT 'mixed',
    questions_asked INTEGER DEFAULT 0,
    summary_data TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT NOT NULL,
    role TEXT NOT NULL,
    name TEXT DEFAULT '',
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (interview_id) REFERENCES interviews(id)
);
"""


async def init_db():
    """Initialize database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_CREATE_TABLES)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)


async def save_interview(
    interview_id: str,
    job_position: str,
    job_description: str,
    interview_type: str,
    questions_asked: int,
    summary_data: dict | None,
    messages: list[dict],
):
    """Save a completed interview to the database."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO interviews
               (id, job_position, job_description, interview_type, questions_asked, summary_data, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                interview_id,
                job_position,
                job_description,
                interview_type,
                questions_asked,
                json.dumps(summary_data) if summary_data else None,
                now,
                now,
            ),
        )
        for msg in messages:
            await db.execute(
                """INSERT INTO messages (interview_id, role, name, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (interview_id, msg["role"], msg.get("name", ""), msg["content"], now),
            )
        await db.commit()
    logger.info("Saved interview %s to database", interview_id)


async def list_interviews(limit: int = 20, offset: int = 0) -> list[dict]:
    """List past interviews, most recent first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, job_position, interview_type, questions_asked, summary_data, created_at, completed_at
               FROM interviews
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            if item["summary_data"]:
                item["summary_data"] = json.loads(item["summary_data"])
            results.append(item)
        return results


async def get_interview(interview_id: str) -> dict | None:
    """Get a full interview with messages."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM interviews WHERE id = ?", (interview_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        interview = dict(row)
        if interview["summary_data"]:
            interview["summary_data"] = json.loads(interview["summary_data"])

        cursor = await db.execute(
            "SELECT role, name, content, created_at FROM messages WHERE interview_id = ? ORDER BY id",
            (interview_id,),
        )
        interview["messages"] = [dict(r) for r in await cursor.fetchall()]
        return interview
