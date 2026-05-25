"""
Intent router — classifies user queries before planner/coder/sandbox execution.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.core.memory import get_state_field, record_user_turn

INTENT_CODING = "coding"
INTENT_RESEARCH = "research"
INTENT_GENERIC = "generic"

_CODING_PATTERNS = [
    r"\b(write|implement|build|create|fix|debug|refactor)\b.*\b(code|script|function|class|api|bug)\b",
    r"\bpython\b.*\b(script|code|program)\b",
    r"\b(module|import|syntax|traceback|exception|error)\b",
    r"\b(docker|sandbox|execute|run)\b.*\b(code|script)\b",
    r"```",
    r"\bdef\s+\w+",
    r"\bunittest\b",
    r"\bpip\s+install\b",
]

_RESEARCH_PATTERNS = [
    r"\b(explain|describe|compare|document|research|survey|overview)\b",
    r"\b(how does|what is|what are|why does|architecture of)\b",
    r"\b(best practice|trade-?off|difference between)\b",
    r"\b(readme|documentation|spec|whitepaper)\b",
    r"\b(langgraph|langchain|gemini|hugging\s*face)\b.*\b(work|architecture)\b",
]


def classify_intent(user_prompt: str) -> str:
    text = (user_prompt or "").strip().lower()
    if not text:
        return INTENT_GENERIC

    coding_score = sum(1 for p in _CODING_PATTERNS if re.search(p, text, re.I))
    research_score = sum(1 for p in _RESEARCH_PATTERNS if re.search(p, text, re.I))

    if coding_score >= 2 or (
        coding_score >= 1
        and any(k in text for k in ("python", "script", "debug", "code", "implement"))
    ):
        return INTENT_CODING

    if research_score >= 2 or (
        research_score >= 1
        and not any(k in text for k in ("write a python", "python script", "fix the code"))
    ):
        return INTENT_RESEARCH

    if coding_score >= 1:
        return INTENT_CODING

    return INTENT_GENERIC


def router_node(state: Any) -> Dict[str, Any]:
    """Brain: route coding → execution pipeline; research/generic → LLM-only paths."""
    user_prompt = get_state_field(state, "user_prompt", "") or ""
    pipeline_logs: List[str] = list(get_state_field(state, "pipeline_logs", []) or [])
    history = record_user_turn(
        get_state_field(state, "conversation_history", []) or [],
        user_prompt,
    )

    intent = classify_intent(user_prompt)
    route_labels = {
        INTENT_CODING: "Planner → Coder → Validator → Docker Sandbox",
        INTENT_RESEARCH: "Research Agent (no sandbox)",
        INTENT_GENERIC: "Knowledge Agent (base LLM, no sandbox)",
    }
    pipeline_logs.append(
        f"[Router] Intent classified: {intent} → {route_labels[intent]}"
    )

    return {
        "intent": intent,
        "conversation_history": history,
        "pipeline_logs": pipeline_logs,
        "validation_passed": False,
        "validator_feedback": [],
        "validator_attempts": 0,
    }
