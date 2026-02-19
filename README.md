# AI Interview Coach

An AI-powered mock interview system that simulates a real interview experience with an **Interviewer**, real-time **Evaluator** feedback, and a structured **Scorecard** at the end. Built to help candidates practice and improve their interview skills for any role.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-orange)
![LiteLLM](https://img.shields.io/badge/LiteLLM-Multi_Provider-purple)

## How It Works

1. **Enter a job role** (e.g., "AI Engineer", "Product Manager", "Backend Developer")
2. **An AI interviewer** asks you questions — technical, behavioral, problem-solving, and culture fit
3. **After each answer**, an AI evaluator gives brief constructive feedback (strengths + improvements)
4. **Continue as long as you want** — click **End** when you're ready
5. **Get a scorecard** with scores across 4 categories, strengths, areas for improvement, and an overall summary

## Architecture

```
Browser (Alpine.js)
    │ WebSocket (JSON)
    │
FastAPI ──── WebSocket handler
    │
LangGraph ── Interview State Machine
    ├── interviewer_node → asks questions (LLM)
    ├── candidate_node   → waits for human input (interrupt)
    ├── evaluator_node   → gives feedback (LLM)
    └── summary_node     → generates scorecard (LLM)
    │
LiteLLM ──── Provider-agnostic LLM calls
```

The interview is modeled as a **LangGraph state machine** with human-in-the-loop via `interrupt()`. The graph loops through interviewer → candidate → evaluator until the user ends the interview, then routes to a summary node that produces a structured scorecard.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | **FastAPI** | Async HTTP + WebSocket server |
| Agent Orchestration | **LangGraph** | Graph-based state machine with interrupt/resume |
| LLM | **LiteLLM** | Unified API — swap between OpenAI, Anthropic, Ollama, etc. |
| Frontend | **Alpine.js** | Lightweight reactivity, no build step |
| Transport | **WebSocket + JSON** | Bidirectional structured messaging |
| Config | **pydantic-settings** | Typed, validated config from `.env` |

## Project Structure

```
app/
├── main.py            # FastAPI app, routes, WebSocket handler
├── config.py          # Settings: LLM model, API key, interview params
├── models.py          # Pydantic models + LangGraph state definition
├── graph.py           # LangGraph interview state machine
├── agents.py          # Node functions + system prompts
├── llm_service.py     # LiteLLM wrapper
└── session_store.py   # In-memory session management
static/
├── app.js             # Alpine.js interview UI component
└── style.css          # Styling (chat, scorecard, animations)
templates/
└── index.html         # Single-page HTML with Alpine.js directives
```

## Getting Started

### Prerequisites

- Python 3.11+
- An API key from OpenAI, Anthropic, or any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Automated-Candidate-Interview-Evaluation-System.git
cd Automated-Candidate-Interview-Evaluation-System

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` with your settings:

```env
# LiteLLM model string (see https://docs.litellm.ai/docs/providers)
LLM_MODEL=openai/gpt-4o

# Your API key
LLM_API_KEY=sk-your-key-here

# Interview settings
MAX_EVALUATOR_WORDS=50
```

**Supported model examples:**
| Provider | Model String |
|----------|-------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| Ollama (local) | `ollama/llama3`, `ollama/mistral` |

### Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

To share on your local network:

```bash
uvicorn app.main:app --reload --host 0.0.0.0
```

## JSON Message Protocol

All communication uses structured JSON over WebSocket:

```jsonc
// Server → Client
{"type": "agent_message", "source": "interviewer", "content": "..."}
{"type": "agent_message", "source": "evaluator", "content": "..."}
{"type": "system_event", "event": "waiting_for_input"}
{"type": "system_event", "event": "agent_typing"}
{"type": "summary", "content": "...", "metadata": {"scores": {...}, "strengths": [...], "improvements": [...]}}
{"type": "error", "content": "Something went wrong"}

// Client → Server
{"type": "user_input", "content": "My answer..."}
{"type": "end_interview"}
```

## Scorecard

After ending the interview, you receive a scorecard with:

- **Scores (1-10)** across Technical, Problem Solving, Communication, and Culture Fit
- **Top strengths** identified from your answers
- **Areas for improvement** with actionable suggestions
- **Overall summary** of your performance

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
