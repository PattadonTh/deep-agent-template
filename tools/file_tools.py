from pathlib import Path
from langchain_core.tools import tool

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)


@tool
def read_file(path: str) -> str:
    """
    Read a file from the workspace directory.
    Use for: inspecting existing code, reading config, loading data.
    Always use relative paths (e.g. 'main.py', not '/workspace/main.py').
    """
    try:
        full_path = WORKSPACE / path
        if not full_path.exists():
            return f"File not found: {path}"
        return full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read failed: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the workspace directory.
    Use for: saving code, writing configs, creating output files.
    Always use relative paths. Creates parent directories if needed.
    """
    try:
        full_path = WORKSPACE / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Written: {path}"
    except Exception as e:
        return f"Write failed: {e}"
