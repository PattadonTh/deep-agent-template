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


class AgentOutput(BaseModel):
    """
    Replace this with your project-specific output schema.
    Example: ResearchReport, CodingResult, etc.
    """

    result: str = Field(description="Agent output")


# ═══════════════════════════════════════════════════════
# 2. Model
# ═══════════════════════════════════════════════════════


def get_model(
    model: str = os.getenv("MODEL", "claude-haiku-4-5"),
    provider: str = os.getenv("PROVIDER", "anthropic"),
    max_retries: int = 3,
) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider=provider,
        max_retries=max_retries,
    )


# ═══════════════════════════════════════════════════════
# 3. Paths & Setup
# ═══════════════════════════════════════════════════════

SKILLS_DIR = Path(__file__).parent / "skills"


# ═══════════════════════════════════════════════════════
# 4. System Prompt
# ═══════════════════════════════════════════════════════

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


def get_subagent() -> dict:
    """
    Optional — remove if not using subagents.
    """
    return {
        "name": "subagent",
        "description": "Describe what this subagent does.",
        "system_prompt": "Focused subagent. [instructions]",
        "tools": [],
        "model": f"{os.getenv('PROVIDER', 'anthropic')}:{os.getenv('MODEL', 'claude-haiku-4-5')}",
    }


# ═══════════════════════════════════════════════════════
# 6. Factory Function
# ═══════════════════════════════════════════════════════


def create_agent(use_subagents: bool = False):
    subagents = [get_subagent()] if use_subagents else None
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=str(SKILLS_DIR), virtual_mode=True),
        },
    )
    return create_deep_agent(
        model=get_model(),
        tools=[web_search, read_file, write_file],  # add or remove tools here
        system_prompt=SYSTEM_PROMPT,
        response_format=AgentOutput,
        backend=backend,
        skills=["/skills"],
        subagents=subagents,
        interrupt_on=None,
        checkpointer=None,
    )
