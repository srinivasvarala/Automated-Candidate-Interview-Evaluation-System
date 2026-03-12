import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from app.graph import interview_graph
from app.models import GraphState
from app.config import settings
from app import database
from app import resume_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    logger.info("Application started, model=%s", settings.llm_model)
    yield


app = FastAPI(title="AI Interview Coach", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


# --- Resume Upload ---

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Parse an uploaded resume (PDF or text) and return extracted text."""
    if not file.filename:
        return {"error": "No file provided"}, 400

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5 MB limit
        return {"error": "File too large. Maximum 5 MB."}, 400

    try:
        text = resume_parser.extract_text(contents, file.filename)
        logger.info("Resume parsed: %d chars from %s", len(text), file.filename)
        return {"text": text, "filename": file.filename, "chars": len(text)}
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        logger.error("Resume parse failed: %s", e)
        return {"error": "Failed to parse resume. Please try a different file."}, 500


# --- Interview History REST API ---

@app.get("/api/interviews")
async def api_list_interviews(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    interviews = await database.list_interviews(limit=limit, offset=offset)
    return {"interviews": interviews, "count": len(interviews)}


@app.get("/api/interviews/{interview_id}")
async def api_get_interview(interview_id: str):
    interview = await database.get_interview(interview_id)
    if not interview:
        return {"error": "Interview not found"}, 404
    return interview


# --- WebSocket Interview Handler ---

async def _send(ws: WebSocket, msg: dict) -> None:
    await ws.send_text(json.dumps(msg))


async def _process_stream_events(
    stream, websocket: WebSocket, session_id: str, collected_messages: list
) -> bool:
    """Process stream events and send messages to client.
    Returns True if an interrupt was hit."""
    hit_interrupt = False

    async for chunk in stream:
        logger.debug("[%s] stream chunk keys: %s", session_id, list(chunk.keys()))
        for node_name, updates in chunk.items():
            if node_name == "__interrupt__":
                hit_interrupt = True
                logger.debug("[%s] interrupt detected", session_id)
                continue

            if not isinstance(updates, dict):
                logger.debug("[%s] skipping non-dict node=%s", session_id, node_name)
                continue

            new_messages = updates.get("messages", [])
            for msg in new_messages:
                if isinstance(msg, AIMessage) and msg.name:
                    source = msg.name
                    logger.info("[%s] sending message from: %s", session_id, source)
                    collected_messages.append({
                        "role": "assistant", "name": source, "content": msg.content,
                    })

                    if source == "summary":
                        summary_data = updates.get("summary_data")
                        await _send(websocket, {
                            "type": "summary",
                            "content": msg.content,
                            "metadata": summary_data or {},
                        })
                    else:
                        # Include performance data for evaluator messages
                        msg_metadata = {}
                        if source == "evaluator":
                            tracker = updates.get("performance_tracker")
                            if tracker and tracker.get("scores"):
                                last_score = tracker["scores"][-1]
                                tier_idx = tracker["tiers"][-1] if tracker.get("tiers") else 0
                                tier_names = ["Warm-up", "Foundation", "Mid-level", "Advanced", "Expert"]
                                msg_metadata = {
                                    "score": last_score,
                                    "tier": tier_names[min(tier_idx, 4)],
                                    "tier_index": tier_idx,
                                }
                        await _send(websocket, {
                            "type": "agent_message",
                            "source": source,
                            "content": msg.content,
                            "metadata": msg_metadata,
                        })
                elif isinstance(msg, HumanMessage):
                    collected_messages.append({
                        "role": "user", "name": getattr(msg, "name", ""), "content": msg.content,
                    })

    logger.debug("[%s] stream done, hit_interrupt=%s", session_id, hit_interrupt)
    return hit_interrupt


@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    collected_messages: list[dict] = []
    interview_meta = {}

    try:
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)

        job_position = start_msg.get("job_position", "").strip()
        if not job_position:
            await _send(websocket, {"type": "error", "content": "Job position is required."})
            await websocket.close()
            return
        if len(job_position) > 200:
            await _send(websocket, {"type": "error", "content": "Job position must be under 200 characters."})
            await websocket.close()
            return

        job_description = start_msg.get("job_description", "").strip()
        if len(job_description) > 10000:
            await _send(websocket, {"type": "error", "content": "Job description must be under 10,000 characters."})
            await websocket.close()
            return

        resume_text = start_msg.get("resume_text", "").strip()
        if len(resume_text) > 10000:
            resume_text = resume_text[:10000]

        interview_type = start_msg.get("interview_type", "mixed").strip()
        if interview_type not in ("mixed", "behavioral", "technical", "system_design"):
            interview_type = "mixed"

        interview_meta = {
            "job_position": job_position,
            "job_description": job_description,
            "interview_type": interview_type,
            "has_resume": bool(resume_text),
        }

        logger.info("[%s] Starting interview: position=%s, type=%s, has_resume=%s", session_id, job_position, interview_type, bool(resume_text))

        await _send(websocket, {
            "type": "system_event",
            "event": "interview_started",
            "content": f"Starting {interview_type} interview for {job_position}...",
            "metadata": {"num_questions": settings.num_questions},
        })

        initial_state: GraphState = {
            "messages": [HumanMessage(content=f"Start the interview for a {job_position} position.")],
            "job_position": job_position,
            "job_description": job_description,
            "resume_text": resume_text,
            "interview_type": interview_type,
            "questions_asked": 0,
            "current_phase": "interviewer",
            "end_requested": False,
            "summary_data": None,
            "performance_tracker": None,
        }

        config = {"configurable": {"thread_id": session_id}}

        logger.info("[%s] Running graph with model: %s", session_id, settings.llm_model)
        await _send(websocket, {"type": "system_event", "event": "agent_typing"})

        stream = interview_graph.astream(initial_state, config=config, stream_mode="updates")
        hit_interrupt = await _process_stream_events(stream, websocket, session_id, collected_messages)
        questions_asked = 0
        summary_data = None

        while hit_interrupt:
            await _send(websocket, {
                "type": "system_event",
                "event": "waiting_for_input",
            })
            logger.debug("[%s] Waiting for user input...", session_id)

            raw = await websocket.receive_text()
            user_msg = json.loads(raw)

            if user_msg.get("type") == "end_interview":
                user_answer = "__END_INTERVIEW__"
                logger.info("[%s] User requested end of interview", session_id)
            else:
                user_answer = user_msg.get("content", "").strip()
                if len(user_answer) > settings.max_answer_length:
                    await _send(websocket, {
                        "type": "error",
                        "content": f"Answer too long. Maximum {settings.max_answer_length} characters.",
                    })
                    continue
                if not user_answer:
                    await _send(websocket, {
                        "type": "error",
                        "content": "Please provide an answer.",
                    })
                    continue
                logger.info("[%s] Got user answer (%d chars)", session_id, len(user_answer))
                questions_asked += 1

            await _send(websocket, {"type": "system_event", "event": "agent_typing"})

            stream = interview_graph.astream(
                Command(resume=user_answer),
                config=config,
                stream_mode="updates",
            )
            hit_interrupt = await _process_stream_events(stream, websocket, session_id, collected_messages)

        # Extract summary_data from collected messages
        for msg in collected_messages:
            if msg.get("name") == "summary":
                try:
                    summary_data = json.loads(msg["content"])
                except (json.JSONDecodeError, KeyError):
                    pass

        logger.info("[%s] Interview complete", session_id)
        await _send(websocket, {"type": "system_event", "event": "interview_complete"})

        # Save to database
        try:
            await database.save_interview(
                interview_id=session_id,
                job_position=job_position,
                job_description=job_description,
                interview_type=interview_type,
                questions_asked=questions_asked,
                summary_data=summary_data,
                messages=collected_messages,
            )
        except Exception as e:
            logger.error("[%s] Failed to save interview: %s", session_id, e)

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", session_id)
    except Exception as e:
        logger.error("[%s] Error: %s", session_id, e, exc_info=True)
        try:
            await _send(websocket, {
                "type": "error",
                "content": f"Something went wrong: {str(e)}",
            })
        except Exception:
            pass
