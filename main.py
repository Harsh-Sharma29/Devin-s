import os
from dotenv import load_dotenv

load_dotenv()

from src.core.graph import app

PIPELINE_CONFIG = {"recursion_limit": 50}

DEFAULT_PROMPT = (
    "Write a python script that calls a mock API at 'https://api.example.com/data' "
    "using the requests library, parses the JSON response, and handles the potential "
    "missing 'items' key error. Make sure to intentionally pass an un-imported module "
    "name or a syntax error in the first line so that it fails initially."
)


def main():
    print("🚀 Initializing Devin's Younger Brother...\n" + "-" * 40)

    initial_state = {
        "user_prompt": DEFAULT_PROMPT,
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

    print(f"👤 User Prompt: {initial_state['user_prompt']}")
    print(f"⚙️  recursion_limit: {PIPELINE_CONFIG['recursion_limit']}")
    print("⏳ Invoking LangGraph pipeline...\n")

    try:
        final_state = app.invoke(initial_state, config=PIPELINE_CONFIG)

        if hasattr(final_state, "model_dump"):
            state_dict = final_state.model_dump()
        elif hasattr(final_state, "dict"):
            state_dict = final_state.dict()
        else:
            state_dict = final_state

        print("✅ Pipeline execution completed!\n")
        print("====== FINAL STATE OVERVIEW ======")
        print(f"Planner Suggestion: {state_dict.get('planner_suggestion')}\n")
        print(f"Updated User Prompt:\n{state_dict.get('user_prompt')}\n")
        print(f"Terminal Output: {state_dict.get('terminal_output')}")
        print(f"Detected Errors: {state_dict.get('detected_errors')}")
        print(f"Repair Attempts: {state_dict.get('repair_attempts', 0)}")
        print(f"Is Verified:     {state_dict.get('is_verified')}\n")

        print("====== 💻 GENERATED CODE BUFFER 💻 ======")
        code_buffer = state_dict.get("code_buffer", "")
        if code_buffer:
            print(code_buffer)
        else:
            print("[No code was generated]")
        print("=========================================")

    except Exception as e:
        print(f"❌ Error during graph invocation: {e}")


if __name__ == "__main__":
    main()
