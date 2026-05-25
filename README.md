# Devin's Younger Brother

**Autonomous AI Software Engineer** — An enterprise-grade LangGraph pipeline that autonomously writes, executes, and debugs Python code inside an isolated Docker sandbox, with multi-model LLM failover.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                      │
│          (app.py — Control Center + Live Console)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               LangGraph State Machine                        │
│                  (src/core/graph.py)                          │
│                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│   │ Planner  │──▸│  Coder   │──▸│ Terminal  │──▸│Debugger │ │
│   └──────────┘   └──────────┘   └────┬─────┘   └────┬────┘ │
│                                      │               │      │
│                                      │   ◂────────── ┘      │
│                                      │  (repair loop,       │
│                                      │   capped at 5)       │
└──────────────────────────────────────┼──────────────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │   Docker Sandbox            │
                         │   (python:3.11-slim)        │
                         │   Ephemeral, read-only      │
                         │   10s execution timeout     │
                         └────────────────────────────┘
```

### Agent Pipeline

| Stage | Role | Implementation |
|-------|------|----------------|
| **Planner** | Proposes target artifact and augments the user prompt | `src/core/graph.py` |
| **Coder** | Generates Python via LLM, sanitizes markdown fences, writes to disk | `src/agents/coder.py` |
| **Terminal** | Executes code in an ephemeral Docker container with 10s timeout | `src/agents/terminal.py` |
| **Debugger** | Analyzes errors, rewrites code, and re-runs (max 5 attempts) | `src/agents/debugger.py` |

### Multi-Model LLM Failover

The system uses a two-tier LLM strategy implemented in `src/core/llm_fallback.py`:

1. **Primary**: Google Gemini 2.5 Flash via `ChatGoogleGenerativeAI`
2. **Fallback**: Meta Llama-3-8B-Instruct via `ChatHuggingFace` (triggered on 429 / RESOURCE_EXHAUSTED / 503)

A graceful **Visual Portfolio Simulation** fallback activates automatically if all backends are unavailable, ensuring the dashboard always renders a complete demonstration.

---

## Prerequisites

- **Python** 3.9+
- **Docker Desktop** running (required for sandboxed code execution)
- API keys configured in `.env`:
  ```env
  GEMINI_API_KEY=your_gemini_api_key
  HUGGINGFACEHUB_API_TOKEN=your_hf_token
  ```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Harsh-Sharma29/Devin-s.git
cd Devin-s

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Launch the dashboard
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Project Structure

```
├── app.py                        # Streamlit dashboard entry point
├── requirements.txt              # Pinned production dependencies
├── .env                          # API keys (gitignored)
├── src/
│   ├── core/
│   │   ├── graph.py              # LangGraph state machine & routing
│   │   ├── llm_fallback.py       # Multi-model LLM wrapper with failover
│   │   ├── config.py             # Configuration constants
│   │   ├── logger.py             # Logging configuration
│   │   └── prompts.py            # System prompt templates
│   ├── agents/
│   │   ├── coder.py              # Code generation agent
│   │   ├── debugger.py           # Autonomous repair agent
│   │   ├── terminal.py           # Docker sandbox execution agent
│   │   └── planner.py            # Task planning agent
│   └── tools/
│       └── file_ops.py           # File I/O & Docker execution helpers
├── Dockerfile                    # Container build config
├── docker-compose.yml            # Multi-service orchestration
└── tests/                        # Test suite
```

---

## Key Design Decisions

- **Python 3.9 Compatibility**: A global `importlib.metadata.packages_distributions` monkey-patch in `app.py` isolates dependency conflicts from `google-auth` / `langchain-core` on older runtimes.
- **Immediate Termination Routing**: The LangGraph `route_from_terminal` function returns `"END"` immediately when `detected_errors` is empty or `is_verified` is `True`, preventing unnecessary recursion loops.
- **Dual State Access**: All agent nodes support both Pydantic model objects and raw dictionaries for maximum compatibility across LangGraph versions.
- **Session State Hygiene**: On each pipeline execution, `st.session_state` buffers are fully reset before invocation to prevent stale error displays.

---

## License

MIT
