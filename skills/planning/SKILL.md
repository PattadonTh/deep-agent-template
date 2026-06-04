---
name: planning
description: Task decomposition discipline — before writing any code or calling any tool, decompose the task into an explicit ordered plan, identify unknowns, and write it to a file. Trigger on /planning and proactively whenever a task involves multiple steps, touches more than one file, requires tool sequencing, or the agent is about to act without a clear plan.
---

# Planning

Decompose before you act. A plan written before the first tool call saves more tokens than any optimization after.

## Recite this — verbatim, as the first thing in your first response

> **Mantra:**
>
> 1. **Understand before planning.** Read the relevant files and context first.
> 2. **Write the plan before acting.** No tool calls until the plan exists.
> 3. **One step at a time.** Complete and verify each step before moving to the next.
> 4. **Replan when reality diverges.** A stale plan is worse than no plan.

Then begin work.

---

## 1. Understand first

Before writing a single line or calling a tool, read what already exists.

- Read all files relevant to the task — do not assume structure from filenames alone.
- Identify: what already works, what is missing, what could conflict.
- If the task is ambiguous, resolve it now — not mid-execution.

Do not plan until you understand the current state.

## 2. Write the plan before acting

Write the plan to `/tmp/plan.md` before any action that modifies state.

Structure:

```
## Goal
[One sentence: what done looks like]

## Steps
- [ ] Step 1: [action] → [expected outcome]
- [ ] Step 2: [action] → [expected outcome]
...

## Unknowns
- [anything that might change the plan once discovered]
```

Rules:

- Every step must have a concrete, verifiable outcome — not "implement X" but "X returns Y when given Z".
- Steps must be ordered by dependency — no step should assume a later step is done.
- Unknowns are first-class. List them explicitly. They become the first steps.
- Do not add steps you are not sure are needed. A 5-step plan beats a 10-step plan with 5 phantom steps.

## 3. Execute one step at a time

Mark each step `[x]` when complete. Update `/tmp/plan.md` as you go.

- Complete one step fully before starting the next.
- After each step, verify the outcome matches what the plan predicted.
- If verification fails → do not proceed to the next step. Treat it as a bug and resolve it first.

## 4. Replan when reality diverges

A plan is a hypothesis. Update it when the hypothesis is wrong.

- If a step reveals new information that invalidates later steps → rewrite those steps before continuing.
- If scope grows mid-execution → add steps explicitly to the plan. Do not silently expand scope.
- If a step is no longer needed → remove it from the plan with a note explaining why.

Never silently deviate from the plan. Deviation without replanning means the plan is fiction.

---

## Operating rules

- Write the plan to `/tmp/plan.md` — always, not inline in the response.
- Recite the mantra **once** per planning session, in the first response.
- If the user says "skip the planning" → skip the recital and `/tmp/plan.md`, but still decompose mentally before acting.
- Never act on a multi-step task without a written plan. The plan is the contract between you and the task.
- The plan is for **you** to execute against — not a deliverable to show the user. Keep it terse and actionable.
