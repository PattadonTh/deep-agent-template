"""
agent.py — Deep Agent Definition
==================================
Sections:
  1. Output Schema    — override per project
  2. Model            — get_model()
  3. Paths & Setup    — SKILLS_DIR
  4. System Prompt    — SYSTEM_PROMPT
  5. Subagents        — get_subagent() (optional)
  6. Factory Function — create_agent()
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from tools import web_search, read_file, write_file

load_dotenv()

# ═══════════════════════════════════════════════════════
# 1. Output Schema
# ═══════════════════════════════════════════════════════


# ✏️  TODO: Rename this class and redefine its fields to match your use case.
#      Examples:
#        class ResearchReport(BaseModel): title, summary, sources
#        class CodingResult(BaseModel):   code, explanation, language
#        class EmailDraft(BaseModel):      subject, body, recipients
#
#      After renaming here, also update the import in main.py:
#        from agent import create_agent, <YourClassName>
class AgentOutput(BaseModel):
    """
    Replace this with your project-specific output schema.
    Example: ResearchReport, CodingResult, etc.
    """

    # ✏️  TODO: Replace with your own fields.
    #      Each field becomes a key in the final JSON output saved to outputs/.
    result: str = Field(description="Agent output")


# ═══════════════════════════════════════════════════════
# 2. Model
# ═══════════════════════════════════════════════════════


# ✏️  TODO: Set MODEL and PROVIDER in your .env file (copy .env.example → .env).
#      Common values:
#        MODEL=claude-haiku-4-5      PROVIDER=anthropic
#        MODEL=gpt-4o-mini           PROVIDER=openai
#        MODEL=gemini-2.0-flash      PROVIDER=google-vertexai
#      You can also pass a different model at runtime without changing this function.
def get_model(
    model: str = os.getenv("MODEL", "claude-haiku-4-5"),
    provider: str = os.getenv("PROVIDER", "anthropic"),
    max_retries: int = 3,
) -> BaseChatModel:
    """
    Build and return the LLM that powers the agent.

    Values are read from environment variables.
    Defaults fall back to Anthropic Claude Haiku
    if nothing is set in .env.

    max_retries=3 means the SDK will automatically retry transient network /
    rate-limit errors before raising an exception — leave it as-is unless you
    have a specific reason to change it.
    """
    return init_chat_model(
        model,
        model_provider=provider,
        max_retries=max_retries,
    )


# ═══════════════════════════════════════════════════════
# 3. Paths & Setup
# ═══════════════════════════════════════════════════════

# Skills are reusable instruction files (Markdown) the agent can read at runtime.
# Drop .md files into the skills/ folder — they are auto-loaded via the backend.
# No code change needed here unless you want to point to a different directory.
SKILLS_DIR = Path(__file__).parent / "skills"


# ═══════════════════════════════════════════════════════
# 4. System Prompt
# ═══════════════════════════════════════════════════════

# ✏️  TODO: Replace the placeholders below with your agent's purpose and rules.
#      Be specific — the more detailed the prompt, the better the output quality.
#      Tips:
#        • Describe the agent's role in one sentence.
#        • List the exact steps it should follow (numbered list works well).
#        • Add hard rules for things it must never do (tone, scope, format, etc.).
SYSTEM_PROMPT = """
You are a [describe your agent here].

## Process
1. [step 1]
2. [step 2]
3. [step 3]

## Rules
- [rule 1]
- [rule 2]
"""


# ═══════════════════════════════════════════════════════
# 5. Subagents (optional)
# ═══════════════════════════════════════════════════════


# ✏️  TODO (optional): Define a subagent if your workflow needs a specialized helper.
#      Common patterns:
#        • A "researcher" subagent that only does web search.
#        • A "writer" subagent that only drafts content.
#        • A "critic" subagent that reviews and scores output.
#      If you don't need subagents, delete this function and set use_subagents=False
#      (or just ignore it — create_agent() defaults to no subagents).
def get_subagent() -> dict:
    """
    Return a configuration dict that describes a subagent.

    A subagent is a smaller, focused LLM worker that the main agent can
    delegate tasks to.  Think of it like hiring a specialist: the main agent
    decides WHAT to delegate, and the subagent handles HOW to do that
    specific piece of work.

    The dict keys are:
      name         — identifier used when the main agent calls the subagent
      description  — tells the main agent when it should use this subagent
      system_prompt — instructions given to the subagent (keep it narrow)
      tools        — list of tool functions the subagent is allowed to call
      model        — can differ from the main agent (e.g. use a cheaper model)
    """
    return {
        "name": "subagent",
        "description": "Describe what this subagent does.",
        "system_prompt": "Focused subagent. [instructions]",
        "tools": [],  # ✏️  TODO: add tool functions the subagent is allowed to use
        "model": f"{os.getenv('PROVIDER', 'anthropic')}:{os.getenv('MODEL', 'claude-haiku-4-5')}",
    }


# ═══════════════════════════════════════════════════════
# 6. Factory Function
# ═══════════════════════════════════════════════════════


def create_agent(use_subagents: bool = False):
    """
    Assemble and return the configured deep agent.

    This is the main factory called by main.py every time a task runs.
    It wires together four things:

      1. Model      — the LLM returned by get_model()
      2. Tools      — Python functions the agent can call (web_search, etc.)
      3. Backend    — where the agent reads/writes memory and skills
      4. Subagents  — optional specialist workers (disabled by default)

    CompositeBackend explanation:
      • StateBackend()      — default in-memory store (cleared between runs)
      • FilesystemBackend  — maps the virtual path "/skills/" to the real
                             skills/ folder on disk so the agent can read
                             your Markdown skill files at runtime.

    You almost never need to change the backend wiring unless you want
    to plug in a database or remote storage.
    """
    subagents = [get_subagent()] if use_subagents else None
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            # virtual_mode=True means the agent sees a clean path ("/skills/")
            # instead of the real OS path — keeps prompts portable.
            "/skills/": FilesystemBackend(root_dir=str(SKILLS_DIR), virtual_mode=True),
        },
    )
    return create_deep_agent(
        model=get_model(),
        # ✏️  TODO: Add or remove tools to control what the agent can do.
        #      Built-in options: web_search, read_file, write_file
        #      Custom tools: define them in tools.py and import here.
        tools=[web_search, read_file, write_file],
        system_prompt=SYSTEM_PROMPT,
        # ✏️  TODO: Replace AgentOutput with your renamed schema class.
        response_format=AgentOutput,
        backend=backend,
        skills=["/skills"],
        subagents=subagents,
        interrupt_on=None,
        checkpointer=None,
    )
