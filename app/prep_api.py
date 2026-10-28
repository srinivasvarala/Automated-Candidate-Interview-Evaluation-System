"""REST routes for Prep plans. Lessons are not a conversation, so no WebSocket."""

import asyncio
import logging
import uuid
import weakref

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app import database, prep
from app.config import settings
from app.llm_service import LLMOutputError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/prep")

# One lock per (plan, area) so a double click generates a Lesson once, not twice.
_lesson_locks: "weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock]" = weakref.WeakValueDictionary()


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _generation_failed(what: str, exc: Exception) -> JSONResponse:
    if isinstance(exc, LLMOutputError):
        logger.warning("Could not generate %s: %s", what, exc)
    else:
        logger.error("Could not generate %s", what, exc_info=exc)
    return _error(f"Couldn't generate {what}. Please try again.", 502)


def _public_plan(plan: dict) -> dict:
    """The plan as the browser sees it: never the resume text."""
    shown = {k: v for k, v in plan.items() if k != "resume_text"}
    shown["has_resume"] = bool(plan.get("resume_text"))
    return shown


def _plan_input(plan: dict) -> prep.PlanInput:
    return prep.PlanInput(plan["role"], plan["seniority"], plan["job_description"], plan["resume_text"])


def _area(plan: dict, key: str) -> dict | None:
    return next((a for a in plan["areas"] if a["key"] == key), None)


async def _json_object(request: Request) -> dict | None:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


@router.get("/templates")
async def templates():
    return {
        "templates": prep.role_templates(),
        "seniorities": list(prep.SENIORITIES),
        "default_seniority": prep.DEFAULT_SENIORITY,
    }


@router.post("/plans")
async def create_plan(request: Request):
    body = await _json_object(request)
    if body is None:
        return _error("Request body must be a JSON object.", 400)
    fields = {k: body.get(k, "") for k in ("role", "seniority", "job_description", "resume_text")}
    if not all(isinstance(v, str) for v in fields.values()):
        return _error("All fields must be text.", 400)
    try:
        plan_input = prep.validate_plan_input(**fields)
    except prep.PrepInputError as e:
        return _error(str(e), 400)

    try:
        outline = await prep.generate_outline(plan_input)
    except Exception as e:
        return _generation_failed("your prep plan", e)

    plan_id = str(uuid.uuid4())
    await database.create_prep_plan(
        plan_id, plan_input.role, plan_input.seniority, plan_input.job_description,
        plan_input.resume_text, [a.model_dump() for a in outline.areas],
    )
    plan = await database.get_prep_plan(plan_id)
    return JSONResponse(_public_plan(plan), status_code=201)


@router.get("/plans")
async def list_plans(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    plans = await database.list_prep_plans(limit=limit, offset=offset)
    return {"plans": plans, "count": len(plans)}


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str):
    plan = await database.get_prep_plan(plan_id)
    if not plan:
        return _error("Prep plan not found", 404)
    return _public_plan(plan)


@router.post("/plans/{plan_id}/lessons/{area_key}")
async def open_lesson(plan_id: str, area_key: str):
    """Return the Lesson for an Area, generating and saving it the first time."""
    plan = await database.get_prep_plan(plan_id)
    if not plan:
        return _error("Prep plan not found", 404)
    if not _area(plan, area_key):
        return _error("Area not found", 404)

    lock = _lesson_locks.get((plan_id, area_key))
    if lock is None:
        lock = _lesson_locks[(plan_id, area_key)] = asyncio.Lock()
    async with lock:
        plan = await database.get_prep_plan(plan_id)  # another request may have finished meanwhile
        area = _area(plan, area_key)
        if not area["lesson"]:
            try:
                lesson = await prep.generate_lesson(
                    _plan_input(plan), prep.OutlineArea(**{k: area[k] for k in ("key", "title", "case", "goal")})
                )
            except Exception as e:
                return _generation_failed("this lesson", e)
            await database.save_lesson(plan_id, area_key, lesson.model_dump())
            area = _area(await database.get_prep_plan(plan_id), area_key)
    return area


@router.post("/plans/{plan_id}/lessons/{area_key}/checks/{idx}")
async def answer_check(plan_id: str, area_key: str, idx: str, request: Request):
    body = await _json_object(request)
    answer = body.get("answer") if body else None
    if not isinstance(answer, str):
        return _error("Please provide an answer.", 400)
    answer = answer.strip()
    if not answer:
        return _error("Please provide an answer.", 400)
    if len(answer) > settings.max_answer_length:
        return _error(f"Answer too long. Maximum {settings.max_answer_length} characters.", 400)

    plan = await database.get_prep_plan(plan_id)
    if not plan:
        return _error("Prep plan not found", 404)
    area = _area(plan, area_key)
    if not area:
        return _error("Area not found", 404)
    if not area["lesson"]:
        return _error("Open the lesson before answering its checks.", 409)
    questions = area["lesson"]["checks"]
    if not idx.isdigit() or int(idx) >= len(questions):
        return _error("Check not found", 404)
    idx = int(idx)

    try:
        result = await prep.score_check(
            _plan_input(plan),
            prep.OutlineArea(**{k: area[k] for k in ("key", "title", "case", "goal")}),
            prep.LessonContent(**area["lesson"]),
            questions[idx],
            answer,
        )
    except Exception as e:
        return _generation_failed("feedback", e)

    await database.save_check(plan_id, area_key, idx, questions[idx], answer, result.score, result.feedback)
    return {"idx": idx, "question": questions[idx], "answer": answer,
            "score": result.score, "feedback": result.feedback}


@router.post("/plans/{plan_id}/lessons/{area_key}/complete")
async def complete_lesson(plan_id: str, area_key: str):
    plan = await database.get_prep_plan(plan_id)
    if not plan:
        return _error("Prep plan not found", 404)
    if not _area(plan, area_key):
        return _error("Area not found", 404)
    if not await database.complete_lesson(plan_id, area_key):
        return _error("Open the lesson before marking it done.", 409)
    return _area(await database.get_prep_plan(plan_id), area_key)
