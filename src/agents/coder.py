from typing import Dict, Any, TYPE_CHECKING
from src.tools.file_ops import write_code_to_disk
from src.core.llm_fallback import call_agent_llm, sanitize_code_for_buffer

if TYPE_CHECKING:
	from src.core.graph import DevinBrotherState

CODER_SYSTEM_PROMPT = (
	"You are an expert Python developer. Return ONLY valid Python inside markdown code fences. "
	"Code must run in an isolated Docker sandbox without external pip packages."
)

def _append_log(state: Any, line: str) -> list[str]:
	if isinstance(state, dict):
		logs = list(state.get("pipeline_logs") or [])
	else:
		logs = list(getattr(state, "pipeline_logs", []) or [])
	logs.append(line)
	return logs

def coder_agent(state: Any) -> Dict[str, Any]:
	"""Generate Python via Gemini with Hugging Face Hub failover."""
	# Handle both dictionary and Pydantic object state access
	if isinstance(state, dict):
		user_prompt = state.get("user_prompt") or "Write a basic python script."
		pipeline_logs = state.get("pipeline_logs") or []
		used_hf_failover = state.get("used_hf_failover") or False
		terminal_output = state.get("terminal_output") or ""
		llm_provider = state.get("llm_provider") or "gemini"
	else:
		user_prompt = getattr(state, "user_prompt", "") or "Write a basic python script."
		pipeline_logs = getattr(state, "pipeline_logs", []) or []
		used_hf_failover = getattr(state, "used_hf_failover", False) or False
		terminal_output = getattr(state, "terminal_output", "") or ""
		llm_provider = getattr(state, "llm_provider", "gemini") or "gemini"

	logs = _append_log(state, "[Coder] Invoking LLM (primary: Gemini)…")

	try:
		# Parse rules correctly and invoke LLM fallback wrapper
		llm_result = call_agent_llm(CODER_SYSTEM_PROMPT, user_prompt)
		provider_label = "Hugging Face Hub" if llm_result.used_failover else "Gemini"
		logs.append(
			f"[Coder] Response received via {provider_label}"
			+ (" (failover active)" if llm_result.used_failover else "")
		)

		# Clean resulting text using the sanitizer
		code, err = sanitize_code_for_buffer(llm_result.content)
		if err or not code:
			logs.append(f"[Coder] Invalid payload rejected: {err}")
			return {
				"code_buffer": "",
				"detected_errors": [err or "Coder returned non-Python payload."],
				"pipeline_logs": logs,
				"llm_provider": llm_result.provider,
				"used_hf_failover": llm_result.used_failover or used_hf_failover,
				"terminal_output": terminal_output + f"\n[Coder] Blocked invalid LLM output — not written to disk.",
			}

		# Write clean script code safely
		filename = "generated_script.py"
		write_code_to_disk(filename, code)
		logs.append(f"[Coder] Wrote sanitized code → {filename}")

		return {
			"code_buffer": code,
			"detected_errors": [],
			"pipeline_logs": logs,
			"llm_provider": llm_result.provider,
			"used_hf_failover": llm_result.used_failover or used_hf_failover,
			"terminal_output": terminal_output + f"\n[Coder] Generated via {provider_label}.",
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
		}
