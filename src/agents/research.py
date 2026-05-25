"""
Research agent — technical Q&A without Docker sandbox.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.llm_fallback import call_agent_llm
from src.core.memory import (
    format_history_context,
    get_state_field,
    record_assistant_turn,
)

RESEARCH_SYSTEM = (
    "You are a senior technical research assistant. Provide accurate, structured "
    "answers with concise bullets when helpful. Cite concepts clearly; do not "
    "generate executable code unless explicitly asked."
)


def research_agent(state: Any) -> Dict[str, Any]:
    user_prompt = get_state_field(state, "user_prompt", "") or ""
    pipeline_logs = list(get_state_field(state, "pipeline_logs", []) or [])
    history = get_state_field(state, "conversation_history", []) or []

    context = format_history_context(history)
    pipeline_logs.append("[Research] Invoking LLM (primary: Gemini)…")

    message = (
        f"Conversation context (last exchanges):\n{context}\n\n"
        f"Research query:\n{user_prompt}"
    )

    try:
        llm_result = call_agent_llm(RESEARCH_SYSTEM, message)
        provider = "Hugging Face Hub" if llm_result.used_failover else "Gemini"
        answer = llm_result.content.strip()
        pipeline_logs.append(f"[Research] Response via {provider}")
        history = record_assistant_turn(history, answer)

        return {
            "terminal_output": answer,
            "code_buffer": "",
            "is_verified": True,
            "detected_errors": [],
            "pipeline_logs": pipeline_logs,
            "conversation_history": history,
            "llm_provider": llm_result.provider,
            "used_hf_failover": llm_result.used_failover
            or bool(get_state_field(state, "used_hf_failover", False)),
        }
    except Exception as exc:
        pipeline_logs.append(f"[Research] LLM failure: {exc}")
        return {
            "terminal_output": f"[Research] Error: {exc}",
            "detected_errors": [str(exc)],
            "is_verified": False,
            "pipeline_logs": pipeline_logs,
        }
