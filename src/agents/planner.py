"""
Planner agent — mission brief with sliding-window conversation memory.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.memory import format_history_context, get_state_field, record_assistant_turn


def planner_agent(state: Any) -> Dict[str, Any]:
    user_prompt = get_state_field(state, "user_prompt", "") or ""
    pipeline_logs = list(get_state_field(state, "pipeline_logs", []) or [])
    history = get_state_field(state, "conversation_history", []) or []

    context = format_history_context(history)
    suggestion = (
        "Generate 'mock_api_client.py' — HTTP fetch, JSON parse, "
        "handle missing 'items' key; include try/except for all I/O."
    )

    pipeline_logs.append(f"[Planner] Mission brief accepted → target: mock_api_client.py")
    pipeline_logs.append(f"[Planner] Memory window: {min(len(history), 10)} message(s) loaded")

    brief = (
        f"{user_prompt}\n[Planner]: {suggestion}\n"
        f"[Context — last exchanges]:\n{context}"
    )
    history = record_assistant_turn(history, f"Plan: {suggestion}")

    return {
        "planner_suggestion": suggestion,
        "user_prompt": brief,
        "pipeline_logs": pipeline_logs,
        "conversation_history": history,
    }
