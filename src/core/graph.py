from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Import agents cleanly at the top. Circular dependency is avoided because
# agents only import DevinBrotherState inside `if TYPE_CHECKING:`.
from src.agents.coder import coder_agent
from src.agents.terminal import terminal_agent
from src.agents.debugger import debugger_agent
from src.core.llm_fallback import is_api_error_payload, sanitize_code_for_buffer

MAX_REPAIR_ATTEMPTS = 5

class DevinBrotherState(BaseModel):
	user_prompt: str = Field(default="", description="The original user prompt")
	planner_suggestion: str = Field(default="", description="Planner output / file plan")
	code_buffer: str = Field(default="", description="The generated code")
	terminal_output: str = Field(default="", description="Output from execution")
	detected_errors: List[str] = Field(default_factory=list, description="List of detected errors")
	is_verified: bool = Field(default=False, description="Flag indicating if the code is verified")
	repair_attempts: int = Field(default=0, description="Debugger cycles to cap infinite loops")
	last_code_buffer: str = Field(default="", description="Snapshot before last debugger fix")
	pipeline_logs: List[str] = Field(default_factory=list, description="Live agent log stream for UI")
	llm_provider: str = Field(default="gemini", description="Last LLM provider used")
	used_hf_failover: bool = Field(default=False, description="True when HF Hub handled a request")

def get_initial_state(user_prompt: str) -> Dict[str, Any]:
	"""
	Exposes a strict initial state dictionary handling tracking keys:
	user_prompt, code_buffer, terminal_output, detected_errors, is_verified, and repair_attempts.
	"""
	return {
		"user_prompt": user_prompt.strip(),
		"planner_suggestion": "",
		"code_buffer": "",
		"terminal_output": "",
		"detected_errors": [],
		"is_verified": False,
		"repair_attempts": 0,
		"last_code_buffer": "",
		"pipeline_logs": [],
		"llm_provider": "gemini",
		"used_hf_failover": False,
	}

def sanitize_state_code_buffer(state: Any) -> Dict[str, Any]:
	"""
	Last-line defense: strip API error JSON from code_buffer before terminal execution.
	"""
	if isinstance(state, dict):
		raw = state.get("code_buffer") or ""
		pipeline_logs = state.get("pipeline_logs") or []
	else:
		raw = getattr(state, "code_buffer", "") or ""
		pipeline_logs = getattr(state, "pipeline_logs", []) or []

	if not raw.strip():
		return {}

	if not is_api_error_payload(raw):
		return {}

	code, err = sanitize_code_for_buffer(raw)
	logs = list(pipeline_logs)
	logs.append("[Graph] Intercepted API error payload in code_buffer — flushed.")

	if code:
		return {
			"code_buffer": code,
			"detected_errors": [],
			"pipeline_logs": logs,
		}

	return {
		"code_buffer": "",
		"detected_errors": [err or "Invalid code_buffer content removed."],
		"pipeline_logs": logs,
	}

def planner_agent(state: Any) -> Dict[str, Any]:
	"""Planner: proposes target artifact and augments the mission brief."""
	suggestion = "Generate 'mock_api_client.py' — HTTP fetch, JSON parse, handle missing 'items' key."
	if isinstance(state, dict):
		user_prompt = state.get("user_prompt") or ""
		pipeline_logs = state.get("pipeline_logs") or []
	else:
		user_prompt = getattr(state, "user_prompt", "") or ""
		pipeline_logs = getattr(state, "pipeline_logs", []) or []

	logs = list(pipeline_logs)
	logs.append(f"[Planner] {suggestion}")
	return {
		"planner_suggestion": suggestion,
		"user_prompt": f"{user_prompt}\n[Planner]: {suggestion}",
		"pipeline_logs": logs,
	}

def terminal_agent_guarded(state: Any) -> Dict[str, Any]:
	"""Run terminal after sanitizing code_buffer so error JSON never executes."""
	patch = sanitize_state_code_buffer(state)
	
	if isinstance(state, dict):
		merged = {**state, **patch}
		logs = list(merged.get("pipeline_logs") or [])
	else:
		merged = state.model_copy(update=patch) if patch else state
		logs = list(getattr(merged, "pipeline_logs", []) or [])

	result = terminal_agent(merged)
	status = "PASSED" if result.get("is_verified") else "FAILED"
	logs.append(f"[Terminal] Docker sandbox run {status}.")
	result["pipeline_logs"] = logs
	return result

def route_from_terminal(state: Any) -> str:
	"""
	End when verified or no errors remain; debugger only with actionable errors.
	"""
	if isinstance(state, dict):
		detected_errors = state.get("detected_errors") or []
		is_verified = state.get("is_verified") or False
		repair_attempts = state.get("repair_attempts") or 0
		code_buffer = state.get("code_buffer") or ""
	else:
		detected_errors = getattr(state, "detected_errors", []) or []
		is_verified = getattr(state, "is_verified", False) or False
		repair_attempts = getattr(state, "repair_attempts", 0) or 0
		code_buffer = getattr(state, "code_buffer", "") or ""

	# In the routing logic, if state["detected_errors"] is empty or state["is_verified"] is True,
	# instantly return "END" to enforce immediate termination and avoid hitting the recursion limit.
	if not detected_errors or is_verified:
		return "END"

	if repair_attempts >= MAX_REPAIR_ATTEMPTS:
		return "END"

	if not code_buffer.strip():
		return "END"

	return "debugger"

workflow = StateGraph(DevinBrotherState)

workflow.add_node("planner_agent", planner_agent)
workflow.add_node("coder_agent", coder_agent)
workflow.add_node("terminal_agent", terminal_agent_guarded)
workflow.add_node("debugger_agent", debugger_agent)

workflow.add_edge(START, "planner_agent")
workflow.add_edge("planner_agent", "coder_agent")
workflow.add_edge("coder_agent", "terminal_agent")

workflow.add_conditional_edges(
	"terminal_agent",
	route_from_terminal,
	{
		"END": END,
		"debugger": "debugger_agent",
	},
)

workflow.add_edge("debugger_agent", "terminal_agent")

app = workflow.compile()
