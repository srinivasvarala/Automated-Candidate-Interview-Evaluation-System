import json
import uuid
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from app.graph import interview_graph
from app.models import GraphState
from app.config import settings
from app import session_store

app = FastAPI(title="AI Interview Coach")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def _send(ws: WebSocket, msg: dict) -> None:
    await ws.send_text(json.dumps(msg))


async def _process_stream_events(stream, websocket: WebSocket) -> bool:
    """Process stream events and send messages to client.
    Returns True if an interrupt was hit."""
    hit_interrupt = False

    async for chunk in stream:
        print(f"  [stream] chunk keys: {list(chunk.keys())}")
        for node_name, updates in chunk.items():
            if node_name == "__interrupt__":
                hit_interrupt = True
                print(f"  [stream] >>> interrupt detected")
                continue

            if not isinstance(updates, dict):
                print(f"  [stream] skipping non-dict node={node_name} type={type(updates)}")
                continue

            new_messages = updates.get("messages", [])
            for msg in new_messages:
                if not isinstance(msg, AIMessage) or not msg.name:
                    continue

                source = msg.name
                print(f"  [stream] sending message from: {source}")

                if source == "summary":
                    summary_data = updates.get("summary_data")
                    await _send(websocket, {
                        "type": "summary",
                        "content": msg.content,
                        "metadata": summary_data or {},
                    })
                else:
                    await _send(websocket, {
                        "type": "agent_message",
                        "source": source,
                        "content": msg.content,
                        "metadata": {},
                    })

    print(f"  [stream] done, hit_interrupt={hit_interrupt}")
    return hit_interrupt


@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)
        job_position = start_msg.get("job_position", "Software Engineer")
        print(f"[ws] Starting interview for: {job_position}")

        session_store.create_session(session_id, job_position)

        await _send(websocket, {
            "type": "system_event",
            "event": "interview_started",
            "content": f"Starting interview for {job_position}...",
        })

        initial_state: GraphState = {
            "messages": [HumanMessage(content=f"Start the interview for a {job_position} position.")],
            "job_position": job_position,
            "questions_asked": 0,
            "current_phase": "interviewer",
            "end_requested": False,
            "summary_data": None,
        }

        config = {"configurable": {"thread_id": session_id}}

        print(f"[ws] Running graph with model: {settings.llm_model}")
        await _send(websocket, {"type": "system_event", "event": "agent_typing"})

        stream = interview_graph.astream(initial_state, config=config, stream_mode="updates")
        print("[ws] Stream created, processing events...")
        hit_interrupt = await _process_stream_events(stream, websocket)
        print(f"[ws] First run complete, hit_interrupt={hit_interrupt}")

        while hit_interrupt:
            await _send(websocket, {"type": "system_event", "event": "waiting_for_input"})
            print("[ws] Waiting for user input...")

            raw = await websocket.receive_text()
            user_msg = json.loads(raw)

            # Check if user wants to end the interview
            if user_msg.get("type") == "end_interview":
                user_answer = "__END_INTERVIEW__"
                print("[ws] User requested end of interview")
            else:
                user_answer = user_msg.get("content", "")
                print(f"[ws] Got user answer: {user_answer[:50]}...")

            await _send(websocket, {"type": "system_event", "event": "agent_typing"})

            stream = interview_graph.astream(
                Command(resume=user_answer),
                config=config,
                stream_mode="updates",
            )
            hit_interrupt = await _process_stream_events(stream, websocket)
            print(f"[ws] Resume complete, hit_interrupt={hit_interrupt}")

        print("[ws] Interview complete")
        await _send(websocket, {"type": "system_event", "event": "interview_complete"})

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print(f"[ws] ERROR: {e}")
        traceback.print_exc()
        try:
            await _send(websocket, {
                "type": "error",
                "content": f"Something went wrong: {str(e)}",
            })
        except Exception:
            pass
    finally:
        session_store.remove_session(session_id)
