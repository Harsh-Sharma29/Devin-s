"""
Sliding-window conversation memory for multi-turn agent context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MAX_EXCHANGES = 5


def exchange(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": (content or "").strip()}


def append_exchange(
    history: Optional[List[Dict[str, str]]],
    role: str,
    content: str,
) -> List[Dict[str, str]]:
    updated = list(history or [])
    if content and content.strip():
        updated.append(exchange(role, content))
    return sliding_window(updated)


def sliding_window(
    history: List[Dict[str, str]],
    max_exchanges: int = MAX_EXCHANGES,
) -> List[Dict[str, str]]:
    """Keep the last N user/assistant pairs (up to 2*N messages)."""
    if not history:
        return []
    cap = max(1, max_exchanges) * 2
    return history[-cap:]


def format_history_context(
    history: Optional[List[Dict[str, str]]],
    max_exchanges: int = MAX_EXCHANGES,
) -> str:
    window = sliding_window(list(history or []), max_exchanges=max_exchanges)
    if not window:
        return "(no prior conversation)"

    lines = []
    for item in window:
        role = (item.get("role") or "unknown").upper()
        content = item.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def record_user_turn(
    history: Optional[List[Dict[str, str]]],
    user_prompt: str,
) -> List[Dict[str, str]]:
    return append_exchange(history, "user", user_prompt)


def record_assistant_turn(
    history: Optional[List[Dict[str, str]]],
    response: str,
) -> List[Dict[str, str]]:
    return append_exchange(history, "assistant", response)


def get_state_field(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)
