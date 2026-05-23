import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

@dataclass
class LLMResult:
	content: str
	provider: str
	used_failover: bool

def call_agent_llm(system_prompt: str, user_message: str) -> LLMResult:
	"""
	Primary LLM call to Gemini with a clean failover mechanism 
	to Hugging Face Hub (Llama-3) if a 429/Quota error triggers.
	"""
	try:
		# 1. Primary Engine: Gemini 2.5 Flash
		llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
		response = llm.invoke([
			("system", system_prompt),
			("user", user_message)
		])
		return LLMResult(
			content=response.content,
			provider="gemini",
			used_failover=False
		)

	except Exception as e:
		error_str = str(e)
		# Checking for any variation of quota limits or service exhaustion
		if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
			print("⚠️ Gemini Quota Exceeded! Switching to ChatHuggingFace Fallback Layer...")
			
			hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
			if not hf_token:
				print("❌ Critical: HUGGINGFACEHUB_API_TOKEN is missing in .env!")
				raise e
			
			# 2. Secondary Engine: Llama-3-8B wrapped cleanly using text-generation task to avoid validation clash
			llm_endpoint = HuggingFaceEndpoint(
				repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
				huggingfacehub_api_token=hf_token,
				max_new_tokens=500,
				temperature=0.2,
				task="text-generation"
			)
			
			# Wrapping endpoint to force proper format constraints
			chat_model = ChatHuggingFace(llm=llm_endpoint)
			
			messages = [
				SystemMessage(content=system_prompt),
				HumanMessage(content=user_message)
			]
			
			fallback_response = chat_model.invoke(messages)
			return LLMResult(
				content=fallback_response.content,
				provider="huggingface",
				used_failover=True
			)
		
		# If it's a completely different error, raise it
		raise e

def sanitize_code_for_buffer(response_text: str) -> Tuple[str, Optional[str]]:
	"""
	Utility function to strip out Markdown code block wraps 
	so the Docker sandbox only receives clean, runnable Python execution.
	Returns a tuple: (sanitized_code, error_message_or_None)
	"""
	if not response_text:
		return "", "Empty response received."
	
	if "```python" in response_text:
		try:
			parts = response_text.split("```python")
			code = parts[1].split("```")[0].strip()
			return code, None
		except Exception as e:
			return "", f"Failed to parse markdown python block: {str(e)}"
	elif "```" in response_text:
		try:
			parts = response_text.split("```")
			code = parts[1].split("```")[0].strip()
			return code, None
		except Exception as e:
			return "", f"Failed to parse markdown generic block: {str(e)}"
			
	return response_text.strip(), None

def is_api_error_payload(text: str) -> bool:
	"""
	Check if the text appears to be an API error payload (e.g. JSON containing error details)
	to avoid running it in the terminal execution buffer.
	"""
	if not text:
		return False
	
	text_stripped = text.strip()
	if text_stripped.startswith("{") and text_stripped.endswith("}"):
		try:
			data = json.loads(text_stripped)
			if isinstance(data, dict):
				error_keys = {"error", "errors", "message", "status", "code", "detail"}
				if any(k in data for k in error_keys) or "error" in text_stripped.lower():
					return True
		except Exception:
			pass
			
	if "RESOURCE_EXHAUSTED" in text or "Quota Exceeded" in text or "API error" in text:
		return True
		
	return False