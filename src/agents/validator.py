"""
Self-reflection validator — static QA before Docker sandbox execution.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.core.memory import get_state_field

MAX_VALIDATOR_ATTEMPTS = 3

_DANGEROUS_PATTERNS = [
    (r"\bos\.remove\b", "Dangerous operation: os.remove"),
    (r"\bos\.unlink\b", "Dangerous operation: os.unlink"),
    (r"\bshutil\.rmtree\b", "Dangerous operation: shutil.rmtree"),
    (r"\bsubprocess\.(call|run|Popen)\b.*\bshell\s*=\s*True", "Dangerous: subprocess with shell=True"),
    (r"\brm\s+-rf\b", "Dangerous operation: rm -rf"),
    (r"\beval\s*\(", "Dangerous operation: eval()"),
    (r"\bexec\s*\(", "Dangerous operation: exec()"),
    (r"\b__import__\s*\(", "Dangerous operation: dynamic __import__"),
]

_API_KEY_PATTERNS = [
    (r'(?i)(api[_-]?key|secret|token|password)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded credential assignment"),
    (r"\bsk-[a-zA-Z0-9]{20,}\b", "Possible OpenAI-style API key"),
    (r"\bAIza[0-9A-Za-z\-_]{20,}\b", "Possible Google API key"),
    (r"\bghp_[a-zA-Z0-9]{20,}\b", "Possible GitHub token"),
    (r"\bhf_[a-zA-Z0-9]{20,}\b", "Possible Hugging Face token"),
]


def _has_try_except(code: str) -> bool:
    return bool(re.search(r"\btry\s*:", code)) and bool(re.search(r"\bexcept\b", code))


def validate_code(code: str) -> Tuple[bool, List[str]]:
    """Return (passed, list of rejection reasons)."""
    reasons: List[str] = []
    if not (code or "").strip():
        return False, ["Empty code buffer — nothing to validate."]

    for pattern, message in _DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.I | re.M):
            reasons.append(message)

    for pattern, message in _API_KEY_PATTERNS:
        if re.search(pattern, code, re.I | re.M):
            reasons.append(message)

    if not _has_try_except(code):
        reasons.append(
            "Missing error handling: add try/except blocks around I/O and external calls."
        )

    return (len(reasons) == 0, reasons)


def validator_node(state: Any) -> Dict[str, Any]:
    """QA gate before sandbox — logs rejection reasons to pipeline_logs."""
    code = get_state_field(state, "code_buffer", "") or ""
    pipeline_logs: List[str] = list(get_state_field(state, "pipeline_logs", []) or [])
    attempts = int(get_state_field(state, "validator_attempts", 0) or 0)

    passed, reasons = validate_code(code)

    if passed:
        pipeline_logs.append("[Validator] PASSED — code cleared for Docker sandbox.")
        return {
            "validation_passed": True,
            "validator_feedback": [],
            "detected_errors": [],
            "pipeline_logs": pipeline_logs,
            "validator_attempts": attempts,
        }

    feedback = reasons
    pipeline_logs.append("[Validator] REJECTED — code blocked before sandbox:")
    for reason in feedback:
        pipeline_logs.append(f"[Validator]   ✗ {reason}")
    pipeline_logs.append(
        "[Validator] Routing back to Coder for rewrite (self-reflection loop)."
    )

    return {
        "validation_passed": False,
        "validator_feedback": feedback,
        "detected_errors": feedback,
        "pipeline_logs": pipeline_logs,
        "validator_attempts": attempts + 1,
    }
