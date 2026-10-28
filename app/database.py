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

CREATE TABLE IF NOT EXISTS prep_plans (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    seniority TEXT NOT NULL,
    job_description TEXT DEFAULT '',
    resume_text TEXT DEFAULT '',
    outline TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prep_lessons (
    plan_id TEXT NOT NULL,
    area TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    generated_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (plan_id, area),
    FOREIGN KEY (plan_id) REFERENCES prep_plans(id)
);

CREATE TABLE IF NOT EXISTS prep_checks (
    plan_id TEXT NOT NULL,
    area TEXT NOT NULL,
    idx INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    score INTEGER,
    feedback TEXT NOT NULL,
    answered_at TEXT NOT NULL,
    PRIMARY KEY (plan_id, area, idx),
    FOREIGN KEY (plan_id) REFERENCES prep_plans(id)
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


# --- Prep plans ---

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_prep_plan(
    plan_id: str,
    role: str,
    seniority: str,
    job_description: str,
    resume_text: str,
    outline: list[dict],
) -> None:
    """Insert a new Prep plan with its outline (a list of Area dicts)."""
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO prep_plans
               (id, role, seniority, job_description, resume_text, outline, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (plan_id, role, seniority, job_description, resume_text, json.dumps(outline), now, now),
        )
        await db.commit()
    logger.info("Created prep plan %s", plan_id)


async def list_prep_plans(limit: int = 20, offset: int = 0) -> list[dict]:
    """List Prep plans, most recent first, with progress counts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT p.id, p.role, p.seniority, p.outline, p.created_at, p.updated_at,
                      (SELECT COUNT(*) FROM prep_lessons l WHERE l.plan_id = p.id) AS lessons_generated,
                      (SELECT COUNT(*) FROM prep_lessons l WHERE l.plan_id = p.id AND l.status = 'done') AS lessons_done
               FROM prep_plans p
               ORDER BY p.created_at DESC, p.rowid DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        plans = []
        for row in await cursor.fetchall():
            item = dict(row)
            item["areas_total"] = len(json.loads(item.pop("outline")))
            plans.append(item)
        return plans


async def get_prep_plan(plan_id: str) -> dict | None:
    """A Prep plan with each Area's Lesson, status and Checks. Includes the private resume text."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM prep_plans WHERE id = ?", (plan_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        plan = dict(row)
        outline = json.loads(plan.pop("outline"))

        cursor = await db.execute("SELECT * FROM prep_lessons WHERE plan_id = ?", (plan_id,))
        lessons = {r["area"]: dict(r) for r in await cursor.fetchall()}
        cursor = await db.execute(
            "SELECT area, idx, question, answer, score, feedback, answered_at FROM prep_checks "
            "WHERE plan_id = ? ORDER BY area, idx",
            (plan_id,),
        )
        checks: dict[str, list[dict]] = {}
        for r in await cursor.fetchall():
            item = dict(r)
            checks.setdefault(item.pop("area"), []).append(item)

        areas = []
        for area in outline:
            lesson = lessons.get(area["key"])
            areas.append({
                **area,
                "status": lesson["status"] if lesson else "not_started",
                "lesson": json.loads(lesson["content"]) if lesson else None,
                "checks": checks.get(area["key"], []),
            })
        plan["areas"] = areas
        return plan


async def save_lesson(plan_id: str, area: str, content: dict) -> dict:
    """Store a generated Lesson unless one exists already; returns the Lesson now stored.

    If two requests generate the same Lesson at once, the first one saved wins.
    """
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO prep_lessons (plan_id, area, content, status, generated_at) "
            "VALUES (?, ?, ?, 'ready', ?)",
            (plan_id, area, json.dumps(content), now),
        )
        await db.execute("UPDATE prep_plans SET updated_at = ? WHERE id = ?", (now, plan_id))
        cursor = await db.execute(
            "SELECT content FROM prep_lessons WHERE plan_id = ? AND area = ?", (plan_id, area)
        )
        stored = json.loads((await cursor.fetchone())[0])
        await db.commit()
        return stored


async def save_check(
    plan_id: str, area: str, idx: int, question: str, answer: str, score: int | None, feedback: str
) -> None:
    """Record the answer to a Check; answering again replaces the earlier answer."""
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO prep_checks
               (plan_id, area, idx, question, answer, score, feedback, answered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (plan_id, area, idx, question, answer, score, feedback, now),
        )
        await db.execute("UPDATE prep_plans SET updated_at = ? WHERE id = ?", (now, plan_id))
        await db.commit()


async def complete_lesson(plan_id: str, area: str) -> bool:
    """Mark a generated Lesson done. Returns False if the Lesson does not exist."""
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE prep_lessons SET status = 'done', completed_at = COALESCE(completed_at, ?) "
            "WHERE plan_id = ? AND area = ?",
            (now, plan_id, area),
        )
        if cursor.rowcount:
            await db.execute("UPDATE prep_plans SET updated_at = ? WHERE id = ?", (now, plan_id))
        await db.commit()
        return cursor.rowcount > 0
