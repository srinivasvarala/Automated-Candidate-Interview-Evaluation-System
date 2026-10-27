import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.graph import interview_graph
from app.config import settings
from app import database
from app import session
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
    await database.abandon_in_progress()
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
        return JSONResponse({"error": "No file provided"}, status_code=400)

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5 MB limit
        return JSONResponse({"error": "File too large. Maximum 5 MB."}, status_code=400)

    try:
        text = resume_parser.extract_text(contents, file.filename)
        logger.info("Resume parsed: %d chars from %s", len(text), file.filename)
        return {"text": text, "filename": file.filename, "chars": len(text)}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error("Resume parse failed: %s", e)
        return JSONResponse({"error": "Failed to parse resume. Please try a different file."}, status_code=500)


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
        return JSONResponse({"error": "Interview not found"}, status_code=404)
    return interview


# --- WebSocket Interview Handler ---

def _wire(event: session.Event) -> dict:
    """Map a session event to the JSON message the browser understands."""
    match event:
        case session.Started():
            return {
                "type": "system_event", "event": "interview_started",
                "content": event.content, "metadata": {"num_questions": event.num_questions},
            }
        case session.Typing():
            return {"type": "system_event", "event": "agent_typing"}
        case session.AgentMessage():
            metadata = {}
            if event.score is not None:
                metadata = {"score": event.score, "tier": event.tier, "tier_index": event.tier_index}
            return {
                "type": "agent_message", "source": event.source,
                "content": event.content, "metadata": metadata,
            }
        case session.WaitingForInput():
            return {"type": "system_event", "event": "waiting_for_input"}
        case session.Summary():
            return {"type": "summary", "content": event.content, "metadata": event.data}
        case session.Complete():
            return {"type": "system_event", "event": "interview_complete"}
        case session.Error():
            return {"type": "error", "content": event.content}


async def _forward(ws: WebSocket, events) -> bool:
    """Send events to the client. Returns False if a fatal error was sent."""
    ok = True
    async for event in events:
        await ws.send_text(json.dumps(_wire(event)))
        if isinstance(event, session.Error) and event.fatal:
            ok = False
    return ok


@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    await websocket.accept()
    interview = session.InterviewSession(interview_graph)

    try:
        start_msg = json.loads(await websocket.receive_text())
        ok = await _forward(websocket, interview.start(
            job_position=start_msg.get("job_position", ""),
            job_description=start_msg.get("job_description", ""),
            resume_text=start_msg.get("resume_text", ""),
            interview_type=start_msg.get("interview_type", "mixed"),
        ))
        if not ok:
            await websocket.close()
            return

        while interview.awaiting_input:
            msg = json.loads(await websocket.receive_text())
            if msg.get("type") == "end_interview":
                events = interview.end()
            else:
                events = interview.answer(msg.get("content", ""))
            if not await _forward(websocket, events):
                await websocket.close()
                return

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", interview.id)
    except Exception as e:
        logger.error("[%s] Error: %s", interview.id, e, exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "content": "Something went wrong. Please try again.",
            }))
        except Exception:
            pass
