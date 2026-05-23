from typing import Dict, Any, TYPE_CHECKING
from src.tools.file_ops import execute_python_code

if TYPE_CHECKING:
    from src.core.graph import DevinBrotherState

def terminal_agent(state: 'DevinBrotherState') -> Dict[str, Any]:
    """
    Executes the generated code via the custom subprocess tool.
    If it succeeds, it verifies the code. If it fails, it populates errors.
    """
    filename = "generated_script.py"
    
    # Run the script
    result = execute_python_code(filename)
    
    if result["returncode"] == 0:
        return {
            "terminal_output": result["stdout"],
            "detected_errors": [],
            "is_verified": True
        }
    else:
        # Include both stderr and stdout in case the error is logged to stdout
        error_msg = result["stderr"] if result["stderr"] else result["stdout"]
        
        return {
            "terminal_output": result["stdout"],
            "detected_errors": [error_msg],
            "is_verified": False
        }
