"""
Session telemetry helpers for the Streamlit dashboard.
Estimates token throughput and simulates infra latency / resource load for demos.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def default_telemetry() -> Dict[str, Any]:
    return {
        "total_tokens": 0,
        "latency_ms": 0,
        "cpu_pct": 0.0,
        "ram_pct": 0.0,
        "resource_index": 0.0,
    }


def estimate_tokens(*chunks: Optional[str]) -> int:
    combined = "".join(c for c in chunks if c)
    if not combined.strip():
        return 0
    return max(1, len(combined) // 4)


def tick_telemetry(
    telemetry: Dict[str, Any],
    *,
    log_line: str = "",
    code: str = "",
    step_index: int = 0,
    final: bool = False,
) -> Dict[str, Any]:
    """Advance telemetry counters for one pipeline step."""
    tel = dict(telemetry or default_telemetry())
    tel["total_tokens"] = int(tel.get("total_tokens", 0)) + estimate_tokens(log_line, code)

    base_latency = int(tel.get("latency_ms", 0))
    if "FAILED" in log_line:
        tel["latency_ms"] = min(2400, base_latency + 380 + step_index * 45)
    elif final:
        tel["latency_ms"] = max(38, base_latency // 3 if base_latency else 48)
    else:
        tel["latency_ms"] = min(1850, 95 + step_index * 88 + (tel["total_tokens"] % 180))

    cpu = min(99.0, 24.0 + step_index * 12.0 + (14.0 if "Debugger" in log_line else 0.0))
    ram = min(96.0, 18.0 + step_index * 8.5 + (tel["total_tokens"] / 420.0))
    tel["cpu_pct"] = round(cpu, 1)
    tel["ram_pct"] = round(ram, 1)
    tel["resource_index"] = round((tel["cpu_pct"] + tel["ram_pct"]) / 2.0, 1)
    return tel


def telemetry_from_state(dashboard_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive telemetry snapshot from a LangGraph result when live mode is used."""
    if not dashboard_state:
        return default_telemetry()

    logs = dashboard_state.get("pipeline_logs") or []
    code = dashboard_state.get("code_buffer") or ""
    term = dashboard_state.get("terminal_output") or ""
    tokens = estimate_tokens("\n".join(logs), code, term)

    errors = dashboard_state.get("detected_errors") or []
    verified = bool(dashboard_state.get("is_verified"))
    latency = 52 if verified and not errors else min(2200, 140 + len(errors) * 220 + tokens % 90)

    cpu = 32.0 if verified else min(94.0, 40.0 + len(errors) * 12.0)
    ram = min(88.0, 26.0 + tokens / 380.0)
    return {
        "total_tokens": tokens,
        "latency_ms": int(latency),
        "cpu_pct": round(cpu, 1),
        "ram_pct": round(ram, 1),
        "resource_index": round((cpu + ram) / 2.0, 1),
    }
