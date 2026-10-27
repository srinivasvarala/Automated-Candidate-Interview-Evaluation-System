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
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    updated_at TEXT
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


STATUSES = ("in_progress", "completed", "abandoned")


async def _migrate(db: aiosqlite.Connection) -> None:
    """Add columns introduced after the first release to pre-existing databases.

    Rows that predate `status` were only ever saved on completion, so they
    default to 'completed'.
    """
    cursor = await db.execute("PRAGMA table_info(interviews)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "status" not in columns:
        await db.execute("ALTER TABLE interviews ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'")
    if "updated_at" not in columns:
        await db.execute("ALTER TABLE interviews ADD COLUMN updated_at TEXT")
    await db.execute(
        "UPDATE interviews SET updated_at = COALESCE(completed_at, created_at) WHERE updated_at IS NULL"
    )


async def init_db():
    """Initialize database, creating and migrating tables as needed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_CREATE_TABLES)
        await _migrate(db)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)


async def mark_abandoned(interview_id: str) -> None:
    """Mark one in-progress interview as abandoned."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE interviews SET status = 'abandoned', updated_at = ? WHERE id = ? AND status = 'in_progress'",
            (now, interview_id),
        )
        await db.commit()


async def abandon_in_progress() -> int:
    """Mark every in-progress interview as abandoned. Returns how many changed.

    Live session state is held in memory, so after a restart no in-progress
    interview can be resumed.
    """
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE interviews SET status = 'abandoned', updated_at = ? WHERE status = 'in_progress'",
            (now,),
        )
        await db.commit()
        count = cursor.rowcount
    if count:
        logger.info("Marked %d in-progress interview(s) as abandoned", count)
    return count


async def create_interview(
    interview_id: str,
    job_position: str,
    job_description: str,
    interview_type: str,
) -> None:
    """Insert a new in-progress interview."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO interviews
               (id, job_position, job_description, interview_type, questions_asked,
                created_at, status, updated_at)
               VALUES (?, ?, ?, ?, 0, ?, 'in_progress', ?)""",
            (interview_id, job_position, job_description, interview_type, now, now),
        )
        await db.commit()
    logger.info("Created interview %s", interview_id)


async def append_messages(interview_id: str, messages: list[dict], questions_asked: int) -> None:
    """Append transcript messages after a turn and record progress."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        for msg in messages:
            await db.execute(
                """INSERT INTO messages (interview_id, role, name, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (interview_id, msg["role"], msg.get("name", ""), msg["content"], now),
            )
        await db.execute(
            "UPDATE interviews SET questions_asked = ?, updated_at = ? WHERE id = ?",
            (questions_asked, now, interview_id),
        )
        await db.commit()


async def finish_interview(interview_id: str, questions_asked: int, summary_data: dict | None) -> None:
    """Mark an interview completed and store its scorecard."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE interviews
               SET status = 'completed', questions_asked = ?, summary_data = ?,
                   completed_at = ?, updated_at = ?
               WHERE id = ?""",
            (questions_asked, json.dumps(summary_data) if summary_data else None, now, now, interview_id),
        )
        await db.commit()
    logger.info("Completed interview %s", interview_id)


async def list_interviews(limit: int = 20, offset: int = 0) -> list[dict]:
    """List past interviews, most recent first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, job_position, interview_type, questions_asked, summary_data, created_at, completed_at,
                      status, updated_at
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
