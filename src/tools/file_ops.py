import os
import tempfile

def write_code_to_disk(filename: str, code: str) -> str:
    """
    Writes the provided code to a specified filename inside a temporary directory.
    Returns the absolute path to the saved file.
    """
    # Create a temporary directory or use a dedicated one
    temp_dir = tempfile.gettempdir()
    # Create a 'devin_brother_sandbox' folder if we want to be organized
    workspace_dir = os.path.join(temp_dir, "devin_brother_sandbox")
    os.makedirs(workspace_dir, exist_ok=True)
    
    file_path = os.path.join(workspace_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    return file_path

import docker
import time

def execute_python_code(filename: str) -> dict:
    """
    Executes a python file inside an ephemeral Docker container for sandbox isolation.
    Returns a dict with stdout, stderr, and returncode.
    """
    temp_dir = tempfile.gettempdir()
    workspace_dir = os.path.join(temp_dir, "devin_brother_sandbox")
    file_path = os.path.join(workspace_dir, filename)
    
    if not os.path.exists(file_path):
        return {"stdout": "", "stderr": f"File not found: {file_path}", "returncode": 1}
        
    try:
        # Initialize Docker client
        client = docker.from_env()
        
        # Spin up ephemeral container, mount workspace as read-only volume
        container = client.containers.run(
            image="python:3.11-slim",
            command=["python", f"/sandbox/{filename}"],
            volumes={workspace_dir: {'bind': '/sandbox', 'mode': 'ro'}},
            detach=True
        )
        
        # Wait with 10-second timeout constraint
        start_time = time.time()
        timeout = 10
        
        while True:
            container.reload()
            if container.status == 'exited':
                break
            if time.time() - start_time > timeout:
                container.kill()
                container.remove(force=True)
                return {"stdout": "", "stderr": "Execution timed out (10s limit).", "returncode": 124}
            time.sleep(0.2)
            
        # Extract exit code
        result = container.wait()
        returncode = result.get('StatusCode', 1)
        
        # Capture logs
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
        
        # Cleanup
        container.remove()
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode
        }
    except docker.errors.DockerException as e:
        return {"stdout": "", "stderr": f"Docker error: {str(e)}\nIs Docker running?", "returncode": 1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": 1}

