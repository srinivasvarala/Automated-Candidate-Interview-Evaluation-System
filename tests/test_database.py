import sqlite3

import pytest

from app import database

pytestmark = pytest.mark.asyncio


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    return path


def _insert(path, interview_id, status):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO interviews (id, job_position, created_at, status) VALUES (?, 'SWE', 'now', ?)",
            (interview_id, status),
        )


def _columns(path):
    with sqlite3.connect(path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(interviews)")}


async def test_init_creates_status_columns(db_path):
    await database.init_db()
    assert {"status", "updated_at"} <= _columns(db_path)


async def test_init_migrates_legacy_database_as_completed(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """CREATE TABLE interviews (
                id TEXT PRIMARY KEY, job_position TEXT NOT NULL, job_description TEXT DEFAULT '',
                interview_type TEXT DEFAULT 'mixed', questions_asked INTEGER DEFAULT 0,
                summary_data TEXT, created_at TEXT NOT NULL, completed_at TEXT);
            INSERT INTO interviews (id, job_position, created_at, completed_at)
                VALUES ('old', 'SWE', '2026-01-01', '2026-01-02');"""
        )
    await database.init_db()
    await database.init_db()  # idempotent
    interview = await database.get_interview("old")
    assert interview["status"] == "completed"
    assert interview["updated_at"] == "2026-01-02"


async def test_save_interview_marks_completed(db_path):
    await database.init_db()
    await database.save_interview("a", "SWE", "", "mixed", 1, None, [])
    assert (await database.get_interview("a"))["status"] == "completed"
    listed = await database.list_interviews()
    assert listed[0]["status"] == "completed"
    assert listed[0]["updated_at"]


async def test_abandon_in_progress_only_touches_in_progress(db_path):
    await database.init_db()
    _insert(db_path, "live", "in_progress")
    _insert(db_path, "done", "completed")
    assert await database.abandon_in_progress() == 1
    assert (await database.get_interview("live"))["status"] == "abandoned"
    assert (await database.get_interview("done"))["status"] == "completed"
    assert await database.abandon_in_progress() == 0
