"""
main.py — LangGraph Orchestration + CLI
========================================
Graph:
  START → load_memory → agent → validate → human_review → save → END

  Auto-reject loop:  validate → agent (fail, max 3 retries with feedback)
  Human-reject loop: human_review → agent (max 3 retries with feedback)

Usage:
  python main.py --task "your task here"
  python main.py --task "your task here" --hitl
"""

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent import create_agent, AgentOutput
from memory import get_memory_store

load_dotenv()

MAX_RETRIES = 3
OUTPUTS_DIR = Path("outputs")


# ═══════════════════════════════════════════════════════
# 1. State
# ═══════════════════════════════════════════════════════


class AgentState(TypedDict):
    task: str
    past_memories: str
    result: Optional[AgentOutput]
    feedback: Optional[str]
    status: str
    retry_count: int


# ═══════════════════════════════════════════════════════
# 2. Nodes
# ═══════════════════════════════════════════════════════


def load_memory_node(state: AgentState) -> dict:
    """Node 1: Query vector memory for relevant context."""
    print(f"🧠 [Memory] Querying for: '{state['task']}'...")
    store = get_memory_store()
    memories = store.query(topic=state["task"], top_k=5)
    context = store.format_context(memories)
    if context:
        print("🧠 [Memory] Relevant context found.")
    else:
        print("🧠 [Memory] No relevant context found.")
    return {"past_memories": context}


async def agent_node(state: AgentState) -> dict:
    """Node 2: Run the deep agent with task + memory + feedback."""
    agent = create_agent()

    parts = [f"Task: {state['task']}"]
    if state.get("past_memories"):
        parts.append(f"--- PAST MEMORIES ---\n{state['past_memories']}")
    if state.get("feedback"):
        parts.append(f"--- FEEDBACK (fix this) ---\n{state['feedback']}")

    full_prompt = "\n\n".join(parts)

    print("🤖 [Agent] Working...")
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": full_prompt}]}
    )

    return {
        "result": response.get("structured_response"),
        "status": "pending",
        "feedback": None,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def validate_node(state: AgentState) -> dict:
    """Node 3: Validate agent output before human review."""
    result = state.get("result")

    if result is None:
        print("❌ [Validate] No result generated.")
        return {
            "status": "rejected",
            "feedback": "Agent returned no result. Try again.",
        }

    # TODO: add project-specific validation rules here

    print("✅ [Validate] Passed.")
    return {}


def human_review_node(state: AgentState) -> dict:
    """Node 4: Auto-approve or interrupt for human review (--hitl)."""
    if state["status"] == "pending":
        return {"status": "approved"}
    return {}


def save_node(state: AgentState) -> dict:
    """Node 5: Save result to outputs/ and update vector memory."""
    result = state["result"]

    # Save to outputs/
    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug = state["task"][:50].lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"{slug}_{timestamp}.json"
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"📄 [Save] Output → {output_path}")

    # Update vector memory
    store = get_memory_store()
    num_chunks = store.save(
        text=result.model_dump_json(),
        topic=state["task"],
    )
    print(f"💾 [Memory] Indexed {num_chunks} chunks.")

    return {"status": "approved"}


# ═══════════════════════════════════════════════════════
# 3. Conditional Edge
# ═══════════════════════════════════════════════════════


def route_after_review(state: AgentState) -> str:
    if state["status"] == "approved":
        return "save"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"⚠️  [Route] Max retries reached — saving best result.")
        return "save"
    return "agent"


# ═══════════════════════════════════════════════════════
# 4. Build Graph
# ═══════════════════════════════════════════════════════


def build_graph(use_hitl: bool = False):
    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("save", save_node)

    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "agent")
    workflow.add_edge("agent", "validate")
    workflow.add_edge("validate", "human_review")
    workflow.add_conditional_edges(
        "human_review",
        route_after_review,
        {"save": "save", "agent": "agent"},
    )
    workflow.add_edge("save", END)

    interrupt_before = ["human_review"] if use_hitl else []
    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interrupt_before,
    )


# ═══════════════════════════════════════════════════════
# 5. Runners
# ═══════════════════════════════════════════════════════


def _init_state(task: str) -> AgentState:
    return {
        "task": task,
        "past_memories": "",
        "result": None,
        "feedback": None,
        "status": "pending",
        "retry_count": 0,
    }


async def run_without_hitl(task: str) -> AgentOutput:
    app = build_graph(use_hitl=False)
    config = {"configurable": {"thread_id": "agent-1"}}
    final = await app.ainvoke(_init_state(task), config=config)
    return final["result"]


async def run_with_hitl(task: str) -> AgentOutput:
    app = build_graph(use_hitl=True)
    config = {"configurable": {"thread_id": "agent-1"}}

    await app.ainvoke(_init_state(task), config=config)

    while True:
        snapshot = app.get_state(config)
        result = snapshot.values.get("result")

        print(f"\n{result}")
        print("\n" + "─" * 50)
        decision = (
            input("✅ Approve (a) / ❌ Reject with feedback (r): ").strip().lower()
        )

        if decision == "a":
            app.update_state(config, {"status": "approved"}, as_node="human_review")
            await app.ainvoke(None, config=config)
            break
        else:
            feedback = input("📝 Enter feedback: ").strip()
            app.update_state(
                config,
                {"status": "rejected", "feedback": feedback},
                as_node="human_review",
            )
            await app.ainvoke(None, config=config)
            snapshot = app.get_state(config)
            if snapshot.next == ():
                break

    return app.get_state(config).values["result"]


# ═══════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="Deep Agent — LangGraph orchestration",
        epilog="""
examples:
  python main.py --task "your task here"
  python main.py --task "your task here" --hitl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="Task for the agent")
    parser.add_argument(
        "--hitl", action="store_true", help="Enable human-in-the-loop review"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.hitl:
        result = asyncio.run(run_with_hitl(args.task))
    else:
        result = asyncio.run(run_without_hitl(args.task))

    print(f"\n✅ Done:\n{result}")


if __name__ == "__main__":
    main()
