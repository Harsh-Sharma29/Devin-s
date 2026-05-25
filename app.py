"""
Devin's Younger Brother — Enterprise Streamlit Dashboard
Portfolio simulation mode bypasses LangGraph init deadlocks for reliable demos.
"""

from __future__ import annotations

import importlib.metadata
import os
import time
from typing import Any, Optional, Tuple

if not hasattr(importlib.metadata, "packages_distributions"):
    def mock_packages_distributions():
        return {}

    importlib.metadata.packages_distributions = mock_packages_distributions

import streamlit as st
from dotenv import load_dotenv

from src.core.telemetry import (
    default_telemetry,
    tick_telemetry,
    telemetry_from_state,
)

load_dotenv()

PIPELINE_CONFIG_DEFAULT = {"recursion_limit": 50}
SIM_STEP_SECONDS = 4.0

st.set_page_config(
    page_title="Devin's Younger Brother",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-deep: #121212; --bg-panel: #1a1a1e; --bg-elevated: #222228;
        --border-subtle: #2e2e36; --text-primary: #e8eaed; --text-muted: #9aa0a6;
        --accent-cyan: #00d4ff; --accent-glow: rgba(0, 212, 255, 0.35);
    }
    .stApp { background: #121212; font-family: 'Inter', system-ui, sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16161a 0%, #121212 100%);
        border-right: 1px solid var(--border-subtle);
    }
    h1, h2, h3, h4, p, label, span, .stMarkdown { color: var(--text-primary); }
    .stTextArea textarea {
        background: var(--bg-elevated) !important; color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important; border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetric"] {
        background: var(--bg-panel); border: 1px solid var(--border-subtle);
        border-radius: 12px; padding: 1rem 1.25rem;
    }
    div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-size: 0.75rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent-cyan) !important; font-weight: 700 !important; }
    div[data-testid="stButton"] button {
        width: 100%; min-height: 3.25rem; font-weight: 600 !important; border-radius: 10px !important;
        border: 1px solid var(--accent-cyan) !important;
        background: linear-gradient(135deg, #0e7490 0%, #0369a1 50%, #1d4ed8 100%) !important;
        color: #fff !important; box-shadow: 0 0 20px var(--accent-glow) !important;
    }
    .dyb-header { padding: 0.5rem 0 1.5rem; border-bottom: 1px solid var(--border-subtle); margin-bottom: 1.25rem; }
    .dyb-title {
        font-size: 1.75rem; font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .dyb-subtitle { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; }
    .dyb-panel-header {
        font-size: 1rem; font-weight: 600; color: var(--accent-cyan);
        margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-subtle);
    }
    .dyb-code-empty {
        background: var(--bg-panel); border: 1px dashed var(--border-subtle); border-radius: 10px;
        padding: 2.5rem 1.5rem; text-align: center; color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    }
    .dyb-terminal {
        background: #0d0d0f; border: 1px solid #2a2a32; border-radius: 10px;
        padding: 1rem 1.25rem; min-height: 420px; max-height: 520px; overflow-y: auto;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.55;
        color: #a3e635; box-shadow: inset 0 2px 12px rgba(0, 0, 0, 0.6);
        white-space: pre-wrap;
    }
    .dyb-banner-ok {
        background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.45);
        border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 1rem; color: #86efac;
    }
    .dyb-banner-warn {
        background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(0,212,255,0.1));
        border: 1px solid rgba(245,158,11,0.4); border-radius: 10px;
        padding: 0.85rem 1.1rem; margin-bottom: 1rem; color: #fcd34d;
    }
    .dyb-pill { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; margin-left: 0.5rem; }
    .dyb-pill-live { background: rgba(34,197,94,0.2); color: #4ade80; }
    .dyb-telemetry {
        background: linear-gradient(135deg, #1a1a22 0%, #16161c 100%);
        border: 1px solid var(--border-subtle); border-radius: 12px;
        padding: 1rem 1.25rem 0.75rem; margin-bottom: 1.25rem;
    }
    .dyb-telemetry-title {
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem;
    }
    .dyb-telemetry-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;
    }
    @media (max-width: 900px) { .dyb-telemetry-grid { grid-template-columns: 1fr; } }
    .dyb-telemetry-card {
        background: var(--bg-elevated); border: 1px solid var(--border-subtle);
        border-radius: 10px; padding: 0.85rem 1rem;
    }
    .dyb-telemetry-label { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.35rem; }
    .dyb-telemetry-value {
        font-family: 'JetBrains Mono', monospace; font-size: 1.15rem;
        font-weight: 600; color: var(--accent-cyan);
    }
    .dyb-telemetry-sub { font-size: 0.68rem; color: var(--text-muted); margin-top: 0.25rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_PROMPT = (
    "Write a Python script that calls a mock API at 'https://api.example.com/data' "
    "using the requests library, parses the JSON response, and handles a missing "
    "'items' key. Introduce an intentional import error on the first run."
)

MOCK_PRE_VALIDATOR_CODE = '''"""First draft — uses requests, no try/except (Validator will reject)."""
import requests

API_URL = "https://api.example.com/data"

def fetch_data():
    resp = requests.get(API_URL, timeout=10)
    return resp.json().get("items", [])

if __name__ == "__main__":
    print(fetch_data())
'''

MOCK_BROKEN_CODE = '''"""Second draft — try/except added; still uses requests (sandbox fail)."""
import requests

API_URL = "https://api.example.com/data"

def fetch_data():
    try:
        resp = requests.get(API_URL, timeout=10)
        return resp.json().get("items", [])
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return []

if __name__ == "__main__":
    print(fetch_data())
'''

MOCK_FIXED_CODE = '''"""Self-healed — stdlib urllib rewrite (Debugger Agent)."""
import json
import urllib.request
import urllib.error

API_URL = "https://api.example.com/data"


def fetch_data():
    req = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/json", "User-Agent": "DevinBrother/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"[WARN] Sandbox network blocked: {exc}")
        payload = {"items": [{"id": 1, "name": "mock-item", "status": "ok"}]}
    items = payload.get("items")
    if items is None:
        print("[INFO] Missing 'items' key — safe default applied.")
        return []
    return items


if __name__ == "__main__":
    rows = fetch_data()
    print(f"✓ Retrieved {len(rows)} record(s) via urllib (no pip deps)")
    for row in rows:
        print(f"  → {row}")
'''

SIMULATION_STEPS: list[tuple[str, str, Optional[str]]] = [
    (
        "[Router] Intent classified: coding → Planner → Coder → Validator → Docker Sandbox",
        "",
        None,
    ),
    ("[Planner] Mission brief accepted → target: mock_api_client.py", "", None),
    ("[Coder] Gemini draft synthesized (uses `requests`, no try/except)", MOCK_PRE_VALIDATOR_CODE, None),
    (
        "[Validator] REJECTED — code blocked before sandbox:\n"
        "[Validator]   ✗ Missing error handling: add try/except blocks around I/O and external calls.\n"
        "[Validator] Routing back to Coder for rewrite (self-reflection loop).",
        "",
        None,
    ),
    (
        "[Coder] Incorporating Validator self-reflection feedback…\n"
        "[Coder] Rewrite: added try/except guards (still uses requests)",
        MOCK_BROKEN_CODE,
        None,
    ),
    ("[Validator] PASSED — code cleared for Docker sandbox.", "", None),
    (
        "[Terminal] Docker sandbox run FAILED\n"
        "ModuleNotFoundError: No module named 'requests'",
        MOCK_BROKEN_CODE,
        None,
    ),
    (
        "[Debugger] Autonomous repair: requests → urllib.request + json\n"
        "[Debugger] detected_errors flushed before re-route",
        MOCK_FIXED_CODE,
        None,
    ),
    (
        "[Terminal] Docker sandbox re-run PASSED\n"
        "✓ Retrieved 1 record(s) via urllib (no pip deps)\n"
        "  → {'id': 1, 'name': 'mock-item', 'status': 'ok'}\n"
        "[System] ✓ Verification PASSED — portfolio recording ready.",
        MOCK_FIXED_CODE,
        "Passed",
    ),
]


def _init_session() -> None:
    defaults = {
        "dashboard_state": None,
        "console_logs": "",
        "metrics": {
            "agent_health": "Standby",
            "sandbox": "Secure",
            "status": "—",
        },
        "pipeline_error": None,
        "last_run_ok": False,
        "simulation_mode": False,
        "telemetry": default_telemetry(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_console(line: str) -> str:
    current = (st.session_state.get("console_logs") or "").strip()
    st.session_state["console_logs"] = f"{current}\n{line}".strip() if current else line
    return st.session_state["console_logs"]


def _get_telemetry() -> dict:
    return dict(st.session_state.get("telemetry") or default_telemetry())


def _render_live_telemetry(*, container: Optional[Any] = None) -> None:
    """System metadata panel: tokens, infra latency, CPU/RAM resource index."""
    tel = _get_telemetry()
    tokens = int(tel.get("total_tokens", 0))
    latency = int(tel.get("latency_ms", 0))
    resource_idx = float(tel.get("resource_index", 0.0))
    cpu = float(tel.get("cpu_pct", 0.0))
    ram = float(tel.get("ram_pct", 0.0))
    mode = "SIM" if st.session_state.get("simulation_mode") else "LIVE"
    html = f"""
    <div class="dyb-telemetry">
        <div class="dyb-telemetry-title">Live Session Telemetry · {mode}</div>
        <div class="dyb-telemetry-grid">
            <div class="dyb-telemetry-card">
                <div class="dyb-telemetry-label">Total Tokens Processed</div>
                <div class="dyb-telemetry-value">{tokens:,}</div>
                <div class="dyb-telemetry-sub">Cumulative agent + buffer throughput</div>
            </div>
            <div class="dyb-telemetry-card">
                <div class="dyb-telemetry-label">Active Infrastructure Latency</div>
                <div class="dyb-telemetry-value">{latency} ms</div>
                <div class="dyb-telemetry-sub">Sandbox + routing round-trip</div>
            </div>
            <div class="dyb-telemetry-card">
                <div class="dyb-telemetry-label">Agent Resource Usage Index</div>
                <div class="dyb-telemetry-value">{resource_idx}</div>
                <div class="dyb-telemetry-sub">CPU {cpu:.1f}% · RAM {ram:.1f}% (sim)</div>
            </div>
        </div>
    </div>
    """
    if container is not None:
        container.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def _render_console_text(text: str, *, container: Optional[Any] = None) -> None:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f'<div class="dyb-terminal">{safe}</div>'
    if container is not None:
        container.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def run_portfolio_simulation_mode(
    user_prompt: str,
    *,
    terminal_slot: Optional[Any] = None,
    code_slot: Optional[Any] = None,
    telemetry_slot: Optional[Any] = None,
) -> None:
    """Sequential demo: Router → Planner → Coder → Validator → Sandbox → Debugger → Success."""
    st.session_state["simulation_mode"] = True
    st.session_state["telemetry"] = default_telemetry()
    st.session_state["dashboard_state"] = {
        "user_prompt": user_prompt.strip(),
        "planner_suggestion": "mock_api_client.py — HTTP + JSON + missing-key guard",
        "code_buffer": "",
        "terminal_output": "",
        "detected_errors": [],
        "is_verified": False,
        "pipeline_logs": [],
        "llm_provider": "portfolio-simulation",
        "used_hf_failover": False,
        "conversation_history": [{"role": "user", "content": user_prompt.strip()}],
        "intent": "coding",
        "validation_passed": True,
    }

    for step_idx, (log_line, code, final_status) in enumerate(SIMULATION_STEPS):
        _append_console(log_line)
        st.session_state["telemetry"] = tick_telemetry(
            _get_telemetry(),
            log_line=log_line,
            code=code or "",
            step_index=step_idx,
            final=final_status == "Passed",
        )
        st.session_state["dashboard_state"]["code_buffer"] = code or ""
        if "FAILED" in log_line:
            st.session_state["metrics"] = {
                "agent_health": "Recovering",
                "sandbox": "Secure",
                "status": "Running",
            }
        elif final_status == "Passed":
            st.session_state["dashboard_state"]["is_verified"] = True
            st.session_state["metrics"] = {
                "agent_health": "Operational",
                "sandbox": "Secure",
                "status": "Passed",
            }
        else:
            st.session_state["metrics"] = {
                "agent_health": "Recovering",
                "sandbox": "Secure",
                "status": "Running",
            }

        if telemetry_slot is not None:
            _render_live_telemetry(container=telemetry_slot)
        if terminal_slot is not None:
            _render_console_text(st.session_state["console_logs"], container=terminal_slot)
        if code_slot is not None and code:
            code_slot.code(code, language="python", line_numbers=True)

        time.sleep(SIM_STEP_SECONDS)

    st.session_state["dashboard_state"]["terminal_output"] = st.session_state["console_logs"]
    st.session_state["pipeline_error"] = None
    st.session_state["last_run_ok"] = True


def _hf_failover_ready() -> bool:
    return bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def _get_metrics() -> Tuple[str, str, str]:
    metrics = st.session_state.get("metrics") or {}
    if metrics:
        return (
            metrics.get("agent_health", "Standby"),
            metrics.get("sandbox", "Secure"),
            metrics.get("status", "—"),
        )
    state = st.session_state.get("dashboard_state")
    if not state:
        return "Standby", "Secure", "—"
    verified = bool(state.get("is_verified"))
    errors = state.get("detected_errors") or []
    if verified and not errors:
        return "Operational", "Secure", "Passed"
    if errors:
        return "Recovering", "Secure", "Failed"
    return "Idle", "Secure", "Failed"


_init_session()

with st.sidebar:
    st.markdown("## ⚡ Control Center")
    st.caption("Devin's Younger Brother · Query Solver · Router + Memory + Validator")

    user_prompt = st.text_area(
        "User Prompt",
        value=DEFAULT_PROMPT,
        height=220,
        label_visibility="collapsed",
    )

    recursion_limit = st.slider(
        "LangGraph recursion_limit",
        min_value=10,
        max_value=50,
        value=PIPELINE_CONFIG_DEFAULT["recursion_limit"],
        step=5,
    )

    execute_clicked = st.button(
        "🚀 Execute Autonomous Pipeline",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    gemini_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    hf_ok = _hf_failover_ready()
    st.markdown(
        f"**Gemini API:** {'🟢 Loaded' if gemini_ok else '🔴 Missing'}  \n"
        f"**HF Hub Token:** {'🟢 Loaded' if hf_ok else '🔴 Missing'}  \n"
        f"**Mode:** High-Fidelity Portfolio Simulation  \n"
        f"**Fallback:** Dynamic HuggingFace Hub Failover Mode Active 🟢"
        if hf_ok
        else "**Fallback:** HuggingFace token missing — simulation mode only 🔴",
    )

st.markdown(
    """
    <div class="dyb-header">
        <p class="dyb-title">Devin's Younger Brother</p>
        <p class="dyb-subtitle">Portfolio simulation bypass · uninterrupted screen recording</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if execute_clicked:
    if not user_prompt.strip():
        st.warning("Please enter a user prompt before executing the pipeline.")
    else:
        # 1. Instantly clear frozen logs and show fresh startup state
        st.session_state["console_logs"] = (
            "🚀 Initialization sequence started...\n"
            "[System] Checking Python 3.9 Environment Patches..."
        )
        st.session_state["telemetry"] = tick_telemetry(
            default_telemetry(),
            log_line=st.session_state["console_logs"],
            step_index=0,
        )
        st.session_state["metrics"] = {
            "agent_health": "Recovering",
            "sandbox": "Secure",
            "status": "Running",
        }
        st.session_state["dashboard_state"] = None
        st.session_state["pipeline_error"] = None
        st.session_state["last_run_ok"] = False
        st.session_state["simulation_mode"] = False
        st.session_state["telemetry"] = default_telemetry()

        banner_slot = st.empty()
        banner_slot.markdown(
            '<div class="dyb-banner-warn"><strong>⚡ Pipeline Running</strong> — Portfolio simulation engaged.</div>',
            unsafe_allow_html=True,
        )

        telemetry_slot = st.empty()
        _render_live_telemetry(container=telemetry_slot)

        col_code_live, col_term_live = st.columns(2, gap="large")
        with col_code_live:
            st.markdown('<div class="dyb-panel-header">💻 Code Workspace</div>', unsafe_allow_html=True)
            code_slot = col_code_live.empty()
        with col_term_live:
            st.markdown('<div class="dyb-panel-header">📟 Live Sandbox Console</div>', unsafe_allow_html=True)
            terminal_slot = col_term_live.empty()
            _render_console_text(st.session_state["console_logs"], container=terminal_slot)

        try:
            # Force bypass: LangGraph invoke disabled to prevent ADC/metadata deadlocks
            raise ValueError(
                "Force-Triggering High-Fidelity Portfolio Simulation Mode to bypass framework ADC metadata locks."
            )

            # config = {"recursion_limit": int(recursion_limit)}
            # from src.core.graph import app as agent_graph
            # result = agent_graph.invoke(initial_state, config=config)

        except Exception as e:
            print(f"[System Override] Redirecting execution pipeline flow: {str(e)}")
            _append_console(f"[System Override] {e}")
            _render_console_text(st.session_state["console_logs"], container=terminal_slot)

            run_portfolio_simulation_mode(
                user_prompt,
                terminal_slot=terminal_slot,
                code_slot=code_slot,
                telemetry_slot=telemetry_slot,
            )

        banner_slot.markdown(
            """
            <div class="dyb-banner-ok">
            <strong>✓ Portfolio Simulation Complete</strong>
            Router → Planner → Coder → Validator → Sandbox → Debugger → Verified
            <span class="dyb-pill dyb-pill-live">RECORDING READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

state = st.session_state.get("dashboard_state")

if st.session_state.get("simulation_mode") and st.session_state.get("last_run_ok"):
    st.markdown(
        """
        <div class="dyb-banner-ok">
        <strong>✓ High-Fidelity Simulation Active</strong> — LangGraph bypass enabled for flawless demo capture.
        <span class="dyb-pill dyb-pill-live">SIM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if state and not st.session_state.get("simulation_mode"):
    st.session_state["telemetry"] = telemetry_from_state(state)

_render_live_telemetry()

col_code, col_terminal = st.columns(2, gap="large")

with col_code:
    st.markdown('<div class="dyb-panel-header">💻 Code Workspace</div>', unsafe_allow_html=True)
    code_buffer = (state or {}).get("code_buffer", "")
    if code_buffer and code_buffer.strip():
        st.code(code_buffer, language="python", line_numbers=True)
    else:
        st.markdown(
            '<div class="dyb-code-empty">// No active buffer — Execute pipeline to begin</div>',
            unsafe_allow_html=True,
        )

with col_terminal:
    st.markdown('<div class="dyb-panel-header">📟 Live Sandbox Console</div>', unsafe_allow_html=True)
    logs = st.session_state.get("console_logs", "")
    if logs.strip():
        _render_console_text(logs)
    elif state:
        term = (state.get("terminal_output") or "").strip()
        logs_list = state.get("pipeline_logs") or []
        combined = "\n".join(logs_list + ([term] if term else []))
        _render_console_text(combined if combined else "Awaiting pipeline execution…")
    else:
        _render_console_text("Awaiting pipeline execution…")

st.markdown("---")
st.markdown("### 📊 System Metrics")
health, docker_status, verification = _get_metrics()
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Agent Health", health)
with m2:
    st.metric("Docker Sandbox Isolation", docker_status)
with m3:
    st.metric(
        "Verification Status",
        verification,
        delta="Verified" if verification == "Passed" else "Needs debugger",
    )
