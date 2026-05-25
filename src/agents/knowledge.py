"""
Knowledge agent — generic queries via base LLM (no sandbox).
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.llm_fallback import call_agent_llm
from src.core.memory import (
    format_history_context,
    get_state_field,
    record_assistant_turn,
)

KNOWLEDGE_SYSTEM = (
    "You are a helpful AI assistant. Answer clearly and concisely using your "
    "general knowledge. Do not invoke code execution or Docker."
)


def knowledge_agent(state: Any) -> Dict[str, Any]:
    user_prompt = get_state_field(state, "user_prompt", "") or ""
    pipeline_logs = list(get_state_field(state, "pipeline_logs", []) or [])
    history = get_state_field(state, "conversation_history", []) or []

    context = format_history_context(history)
    pipeline_logs.append("[Knowledge] Invoking base LLM (primary: Gemini)…")

    message = (
        f"Conversation context (last exchanges):\n{context}\n\n"
        f"User query:\n{user_prompt}"
    )

    try:
        llm_result = call_agent_llm(KNOWLEDGE_SYSTEM, message)
        provider = "Hugging Face Hub" if llm_result.used_failover else "Gemini"
        answer = llm_result.content.strip()
        pipeline_logs.append(f"[Knowledge] Response via {provider}")
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
        pipeline_logs.append(f"[Knowledge] LLM failure: {exc}")
        return {
            "terminal_output": f"[Knowledge] Error: {exc}",
            "detected_errors": [str(exc)],
            "is_verified": False,
            "pipeline_logs": pipeline_logs,
        }
