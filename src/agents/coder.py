from typing import Dict, Any, TYPE_CHECKING
from src.tools.file_ops import write_code_to_disk
from src.core.llm_fallback import call_agent_llm, sanitize_code_for_buffer
from src.core.memory import format_history_context, record_assistant_turn

if TYPE_CHECKING:
	from src.core.graph import DevinBrotherState

CODER_SYSTEM_PROMPT = (
	"You are an expert Python developer. Return ONLY valid Python inside markdown code fences. "
	"Code must run in an isolated Docker sandbox without external pip packages. "
	"Always wrap network and file I/O in try/except blocks. Never embed API keys or secrets. "
	"Never use os.remove, shutil.rmtree, eval, exec, or shell=True subprocess calls."
)

def _append_log(state: Any, line: str) -> list[str]:
	if isinstance(state, dict):
		logs = list(state.get("pipeline_logs") or [])
	else:
		logs = list(getattr(state, "pipeline_logs", []) or [])
	logs.append(line)
	return logs

def coder_agent(state: Any) -> Dict[str, Any]:
	"""Generate Python via Gemini with Hugging Face Hub failover and memory context."""
	if isinstance(state, dict):
		user_prompt = state.get("user_prompt") or "Write a basic python script."
		pipeline_logs = state.get("pipeline_logs") or []
		used_hf_failover = state.get("used_hf_failover") or False
		terminal_output = state.get("terminal_output") or ""
		llm_provider = state.get("llm_provider") or "gemini"
		validator_feedback = state.get("validator_feedback") or []
		history = state.get("conversation_history") or []
	else:
		user_prompt = getattr(state, "user_prompt", "") or "Write a basic python script."
		pipeline_logs = getattr(state, "pipeline_logs", []) or []
		used_hf_failover = getattr(state, "used_hf_failover", False) or False
		terminal_output = getattr(state, "terminal_output", "") or ""
		llm_provider = getattr(state, "llm_provider", "gemini") or "gemini"
		validator_feedback = getattr(state, "validator_feedback", []) or []
		history = getattr(state, "conversation_history", []) or []

	logs = _append_log(state, "[Coder] Invoking LLM (primary: Gemini)…")
	context = format_history_context(history)

	user_message = (
		f"Conversation context (last exchanges):\n{context}\n\n"
		f"Task:\n{user_prompt}"
	)
	if validator_feedback:
		feedback_block = "\n".join(f"- {item}" for item in validator_feedback)
		user_message += (
			f"\n\n[Validator — rewrite required before sandbox]:\n{feedback_block}\n"
			"Fix every issue above. Return the full corrected Python file only."
		)
		logs.append("[Coder] Incorporating Validator self-reflection feedback…")

	try:
		llm_result = call_agent_llm(CODER_SYSTEM_PROMPT, user_message)
		provider_label = "Hugging Face Hub" if llm_result.used_failover else "Gemini"
		logs.append(
			f"[Coder] Response received via {provider_label}"
			+ (" (failover active)" if llm_result.used_failover else "")
		)

		code, err = sanitize_code_for_buffer(llm_result.content)
		if err or not code:
			logs.append(f"[Coder] Invalid payload rejected: {err}")
			return {
				"code_buffer": "",
				"detected_errors": [err or "Coder returned non-Python payload."],
				"pipeline_logs": logs,
				"llm_provider": llm_result.provider,
				"used_hf_failover": llm_result.used_failover or used_hf_failover,
				"terminal_output": terminal_output + "\n[Coder] Blocked invalid LLM output — not written to disk.",
				"validation_passed": False,
			}

		filename = "generated_script.py"
		write_code_to_disk(filename, code)
		logs.append(f"[Coder] Wrote sanitized code → {filename}")
		history = record_assistant_turn(history, f"Generated {filename} ({len(code)} chars)")

		return {
			"code_buffer": code,
			"detected_errors": [],
			"pipeline_logs": logs,
			"conversation_history": history,
			"llm_provider": llm_result.provider,
			"used_hf_failover": llm_result.used_failover or used_hf_failover,
			"terminal_output": terminal_output + f"\n[Coder] Generated via {provider_label}.",
			"validation_passed": False,
			"validator_feedback": [],
		}

	except Exception as exc:
		logs.append(f"[Coder] LLM failure: {exc}")
		return {
			"code_buffer": "",
			"detected_errors": [str(exc)],
			"pipeline_logs": logs,
			"llm_provider": llm_provider,
			"used_hf_failover": used_hf_failover,
			"terminal_output": terminal_output + f"\n[Coder] Error: {exc}",
			"validation_passed": False,
		}
