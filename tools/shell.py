import shlex
import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

TIMEOUT = 30  # seconds
MAX_OUTPUT = 8_000  # characters

BLOCKED = [
    "rm -rf",
    "rmdir /s",
    "sudo",
    "su ",
    "chmod 777",
    "chmod 000",
    "mkfs",
    "dd if=",
    "> /dev",
    "curl",
    "wget",
    ":(){ :|:& };:",  # fork bomb
]


@tool
def run_command(command: str) -> str:
    """
    Run a shell command inside the workspace directory.
    Use for: running scripts, executing tests, installing packages.
    Commands are sandboxed to the workspace folder with a 30s timeout.
    Note: pipe operators (|) are not supported — run each command separately.
    Do NOT use for: deleting files, system changes, or anything outside workspace.
    """
    # Block dangerous patterns
    for blocked in BLOCKED:
        if blocked in command:
            return f"❌ Blocked command: '{blocked}' is not allowed."

    try:
        args = shlex.split(command)
    except ValueError as e:
        return f"❌ Invalid command syntax: {e}"

    try:
        result = subprocess.run(
            args,
            shell=False,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            return "Command completed with no output."
        if len(output) > MAX_OUTPUT:
            output = (
                output[:MAX_OUTPUT] + f"\n... [truncated — {len(output)} chars total]"
            )
        return output
    except FileNotFoundError:
        return f"❌ Command not found: '{args[0]}'"
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {TIMEOUT}s."
    except Exception as e:
        return f"❌ Command failed: {e}"
