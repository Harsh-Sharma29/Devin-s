from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from src.agents.coder import coder_agent
from src.agents.terminal import terminal_agent
from src.agents.debugger import debugger_agent
from src.agents.router import router_node, INTENT_CODING, INTENT_RESEARCH, INTENT_GENERIC
from src.agents.planner import planner_agent
from src.agents.research import research_agent
from src.agents.knowledge import knowledge_agent
from src.agents.validator import validator_node, MAX_VALIDATOR_ATTEMPTS
from src.core.llm_fallback import is_api_error_payload, sanitize_code_for_buffer

MAX_REPAIR_ATTEMPTS = 5

IntentType = Literal["coding", "research", "generic"]


class DevinBrotherState(BaseModel):
	user_prompt: str = Field(default="", description="The original user prompt")
	planner_suggestion: str = Field(default="", description="Planner output / file plan")
	code_buffer: str = Field(default="", description="The generated code")
	terminal_output: str = Field(default="", description="Output from execution or LLM answer")
	detected_errors: List[str] = Field(default_factory=list, description="List of detected errors")
	is_verified: bool = Field(default=False, description="Flag indicating if the code is verified")
	repair_attempts: int = Field(default=0, description="Debugger cycles to cap infinite loops")
	last_code_buffer: str = Field(default="", description="Snapshot before last debugger fix")
	pipeline_logs: List[str] = Field(default_factory=list, description="Live agent log stream for UI")
	llm_provider: str = Field(default="gemini", description="Last LLM provider used")
	used_hf_failover: bool = Field(default=False, description="True when HF Hub handled a request")
	conversation_history: List[Dict[str, str]] = Field(
		default_factory=list,
		description="Sliding-window user/assistant exchanges",
	)
	intent: IntentType = Field(default="generic", description="Router classification")
	validation_passed: bool = Field(default=False, description="Validator QA gate status")
	validator_feedback: List[str] = Field(default_factory=list, description="Last validator rejections")
	validator_attempts: int = Field(default=0, description="Coder↔Validator rewrite cycles")


def get_initial_state(
	user_prompt: str,
	conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
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
		"conversation_history": list(conversation_history or []),
		"intent": "generic",
		"validation_passed": False,
		"validator_feedback": [],
		"validator_attempts": 0,
	}


def sanitize_state_code_buffer(state: Any) -> Dict[str, Any]:
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


def terminal_agent_guarded(state: Any) -> Dict[str, Any]:
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


def route_after_router(state: Any) -> str:
	if isinstance(state, dict):
		intent = state.get("intent") or INTENT_GENERIC
	else:
		intent = getattr(state, "intent", INTENT_GENERIC) or INTENT_GENERIC

	if intent == INTENT_CODING:
		return "planner_agent"
	if intent == INTENT_RESEARCH:
		return "research_agent"
	return "knowledge_agent"


def route_after_validator(state: Any) -> str:
	if isinstance(state, dict):
		passed = state.get("validation_passed", False)
		attempts = state.get("validator_attempts", 0) or 0
	else:
		passed = getattr(state, "validation_passed", False)
		attempts = getattr(state, "validator_attempts", 0) or 0

	if passed:
		return "terminal_agent"
	if attempts >= MAX_VALIDATOR_ATTEMPTS:
		return "terminal_agent"
	return "coder_agent"


def route_from_terminal(state: Any) -> str:
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

	if not detected_errors or is_verified:
		return "END"

	if repair_attempts >= MAX_REPAIR_ATTEMPTS:
		return "END"

	if not code_buffer.strip():
		return "END"

	return "debugger"


workflow = StateGraph(DevinBrotherState)

workflow.add_node("router_node", router_node)
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("coder_agent", coder_agent)
workflow.add_node("validator_node", validator_node)
workflow.add_node("terminal_agent", terminal_agent_guarded)
workflow.add_node("debugger_agent", debugger_agent)
workflow.add_node("research_agent", research_agent)
workflow.add_node("knowledge_agent", knowledge_agent)

workflow.add_edge(START, "router_node")

workflow.add_conditional_edges(
	"router_node",
	route_after_router,
	{
		"planner_agent": "planner_agent",
		"research_agent": "research_agent",
		"knowledge_agent": "knowledge_agent",
	},
)

workflow.add_edge("planner_agent", "coder_agent")
workflow.add_edge("coder_agent", "validator_node")

workflow.add_conditional_edges(
	"validator_node",
	route_after_validator,
	{
		"terminal_agent": "terminal_agent",
		"coder_agent": "coder_agent",
	},
)

workflow.add_conditional_edges(
	"terminal_agent",
	route_from_terminal,
	{
		"END": END,
		"debugger": "debugger_agent",
	},
)

workflow.add_edge("debugger_agent", "terminal_agent")
workflow.add_edge("research_agent", END)
workflow.add_edge("knowledge_agent", END)

app = workflow.compile()
