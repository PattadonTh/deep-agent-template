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

# ✏️  TODO: Tune MAX_RETRIES if you want more or fewer auto-retry attempts
#      before the graph gives up and saves the best result it has.
MAX_RETRIES = 3
# All successful outputs are written here as timestamped JSON files.
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
    """
    Node 1 — Load relevant past memories before running the agent.

    Before the agent does any work, we search the vector memory store for
    results from *previous* runs that are similar to the current task.
    These are injected into the agent's prompt as context so it can build
    on past work instead of starting from scratch every time.

    top_k=5 means we pull at most 5 relevant memory chunks.  Raise this
    number if your tasks benefit from more history; lower it to reduce
    prompt size and cost.

    Returns a partial state update — only the keys you return here are
    merged into the graph state; everything else stays unchanged.
    """
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
    """
    Node 2 — Run the deep agent and get a structured response.

    This node builds the full prompt from three optional sections:
      • Task         — the user's original request (always present)
      • Past memories — relevant context from previous runs (if any)
      • Feedback      — rejection reason from a previous attempt (on retry)

    The sections are joined and sent as a single user message so the agent
    has everything it needs in one shot.

    Why `ainvoke` (async)?  The agent may call tools (web search, file I/O)
    which are I/O-bound.  Async lets the event loop handle other work while
    waiting for API responses instead of blocking the whole process.

    `structured_response` is the parsed instance of your AgentOutput class.
    If the model fails to produce a valid response it will be None, which
    the validate_node catches on the next step.

    retry_count is incremented here (not in validate) so it accurately
    reflects how many times the agent has actually run, regardless of why.
    """
    agent = create_agent()

    parts = [f"Task: {state['task']}"]
    if state.get("past_memories"):
        parts.append(f"--- PAST MEMORIES ---\n{state['past_memories']}")
    if state.get("feedback"):
        # On a retry, the previous rejection reason is appended so the agent
        # knows exactly what to fix instead of repeating the same mistake.
        parts.append(f"--- FEEDBACK (fix this) ---\n{state['feedback']}")

    full_prompt = "\n\n".join(parts)

    print("🤖 [Agent] Working...")
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": full_prompt}]}
    )

    return {
        "result": response.get("structured_response"),
        "status": "pending",  # always reset to pending; validate/review will update it
        "feedback": None,  # clear previous feedback so it doesn't bleed into next node
        "retry_count": state.get("retry_count", 0) + 1,
    }


def validate_node(state: AgentState) -> dict:
    """
    Node 3 — Programmatically check the agent's output before human review.

    This is your quality gate.  It runs automatically (no human needed) and
    can reject the result with a specific feedback message that gets fed back
    into the agent on the next attempt.

    Returning {"status": "rejected", "feedback": "..."}  → triggers a retry
    Returning {}  (empty dict)                           → passes to human_review

    Why separate from human_review?  Automated checks are fast and cheap.
    Catch obvious failures here (empty fields, wrong format, missing data)
    so humans only see results that already passed basic sanity checks.
    """
    result = state.get("result")

    if result is None:
        # The model returned nothing — this usually means an API error or
        # the model refused to follow the response_format schema.
        print("❌ [Validate] No result generated.")
        return {
            "status": "rejected",
            "feedback": "Agent returned no result. Try again.",
        }

    # ✏️  TODO: Add your own validation rules here.
    #      Examples:
    #        if not result.summary:  return {"status": "rejected", "feedback": "Summary is empty."}
    #        if len(result.code) < 10: return {"status": "rejected", "feedback": "Code too short."}
    #      Returning "rejected" triggers an automatic retry with the feedback as context.

    print("✅ [Validate] Passed.")
    return {}


def human_review_node(state: AgentState) -> dict:
    """
    Node 4 — Approve or reject the result (human-in-the-loop gate).

    When running WITHOUT --hitl (default):
      The graph never pauses here.  Any result with status="pending" is
      automatically approved so the workflow completes without input.

    When running WITH --hitl:
      LangGraph's interrupt_before=["human_review"] causes the graph to
      PAUSE before this node executes.  The CLI (run_with_hitl) then asks
      the user to approve or reject, updates the state externally via
      app.update_state(), and resumes the graph.  This node itself just
      returns {} because the state was already updated by the CLI.

    The status field drives the conditional edge after this node:
      "approved"  → save_node
      "rejected"  → back to agent_node (retry with feedback)
    """
    # This branch only runs in non-HITL mode (auto-approve).
    if state["status"] == "pending":
        return {"status": "approved"}
    # In HITL mode the CLI already set status to "approved" or "rejected"
    # before resuming, so we just pass through without changing anything.
    return {}


def save_node(state: AgentState) -> dict:
    """
    Node 5 — Persist the approved result and update long-term memory.

    Two things happen here:

    1. Write to disk (outputs/):
       The result is serialised to a JSON file with a human-readable name
       built from the first 50 chars of the task + a timestamp, e.g.:
         summarise_quarterly_report_20260605_143022.json
       These files are your audit trail — you can replay or inspect any run.

    2. Index into vector memory:
       The same result is chunked and embedded into the vector store so
       future runs on similar tasks can retrieve it as context (load_memory_node).
       num_chunks tells you how many embedding chunks were created.

    This node always sets status="approved" as a final confirmation so
    the graph state is clean when it reaches END.
    """
    result = state["result"]

    # ── 1. Save to outputs/ ──────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(exist_ok=True)
    # Slugify the task string to make a valid filename fragment.
    slug = state["task"][:50].lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"{slug}_{timestamp}.json"
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"📄 [Save] Output → {output_path}")

    # ── 2. Update vector memory ──────────────────────────────────────────
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
    """
    Conditional edge — decide what happens after human_review.

    LangGraph calls this function after human_review_node finishes and uses
    the returned string to pick the next node from the edges map:
      "save"  → save_node  (approved or retries exhausted)
      "agent" → agent_node (rejected, retry with feedback)

    The retry_count guard prevents an infinite rejection loop: if the agent
    has already tried MAX_RETRIES times we accept the best result we have
    rather than looping forever.  You can raise MAX_RETRIES at the top of
    this file if your tasks need more attempts.
    """
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
    """
    Construct and compile the LangGraph state machine.

    Node order:  START → load_memory → agent → validate → human_review → save → END
    Retry loops:
      • Auto-reject:  validate rejects  → human_review → route → agent  (up to MAX_RETRIES)
      • Human-reject: human sets rejected → route → agent               (up to MAX_RETRIES)

    interrupt_before=["human_review"] is what causes the graph to pause
    mid-run when --hitl is passed.  Without it the graph runs to END in
    one shot.

    MemorySaver is a simple in-process checkpointer that stores graph state
    in a dict keyed by thread_id.  This is what allows run_with_hitl to
    call app.ainvoke(None, ...) multiple times on the same thread and have
    LangGraph resume from where it left off.
    """
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
    """
    Run the full graph in one shot with no human intervention.

    The graph executes synchronously from START to END (still async under
    the hood for tool calls) and returns the final state.  The result is
    whatever the agent produced after passing validation.

    thread_id="agent-1" scopes the MemorySaver checkpoint to this run.
    Change it if you want separate checkpoint histories for parallel runs.
    """
    app = build_graph(use_hitl=False)
    config = {"configurable": {"thread_id": "agent-1"}}
    final = await app.ainvoke(_init_state(task), config=config)
    return final["result"]


async def run_with_hitl(task: str) -> AgentOutput:
    """
    Run the graph with a human approval gate after each agent attempt.

    Flow:
      1. graph.ainvoke() runs until it hits the interrupt before human_review
         and then pauses — control returns here.
      2. We read the current state with get_state() and show the result.
      3. The human types 'a' (approve) or 'r' (reject + feedback).
      4. We write the decision back into the graph state with update_state(),
         telling LangGraph to treat it as if human_review_node set those values.
      5. We resume the graph by calling ainvoke(None, ...) — passing None
         means "continue from the checkpoint, don't start over".
      6. Repeat until approved or until the graph reaches END naturally
         (snapshot.next == () means no more nodes to run).

    as_node="human_review" is important: it tells LangGraph which node's
    output we're simulating so the conditional edge after it fires correctly.
    """
    app = build_graph(use_hitl=True)
    config = {"configurable": {"thread_id": "agent-1"}}

    # First run — stops before human_review due to interrupt_before.
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
            # Inject "approved" into the graph state and resume to save_node.
            app.update_state(config, {"status": "approved"}, as_node="human_review")
            await app.ainvoke(None, config=config)
            break
        else:
            feedback = input("📝 Enter feedback: ").strip()
            # Inject "rejected" + feedback so route_after_review sends us back to agent.
            app.update_state(
                config,
                {"status": "rejected", "feedback": feedback},
                as_node="human_review",
            )
            await app.ainvoke(None, config=config)
            snapshot = app.get_state(config)
            # snapshot.next == () means the graph hit END (max retries exhausted).
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
