---
name: reflection
description: Output self-review discipline — before returning any final result, evaluate it against the original task, check for gaps and contradictions, and refine once. Trigger on /reflection and proactively whenever the agent is about to return a final structured output, complete a multi-step task, or the output is high-stakes (code that will run, data that will be saved, reports that will be used).
---

# Reflection

Review before returning. One self-critique pass catches more errors than any prompt instruction.

## Recite this — verbatim, as the first thing in your first response

> **Mantra:**
>
> 1. **Does it answer the actual task?** Not what you assumed — what was asked.
> 2. **Is anything missing or incomplete?** Check schema, coverage, and edge cases.
> 3. **Does it contradict itself?** Logic, data, and claims must be consistent end-to-end.
> 4. **Refine once, then return.** Do not loop indefinitely — one pass is the discipline.

Then begin work.

---

## 1. Does it answer the actual task?

Re-read the original task. Not your interpretation — the literal request.

- Does the output address every part of the task, or only the easy parts?
- Are required fields, formats, or constraints satisfied?
- If the task had multiple sub-goals, are all of them covered?

If any part of the task is unanswered → address it before returning.

## 2. Is anything missing or incomplete?

Check completeness against the output schema or expected structure.

- Required fields: are all present and non-empty?
- Coverage: does the output cover the full scope, or only a subset?
- Edge cases: are obvious failure modes or exceptions acknowledged?

Missing content is more dangerous than imperfect content — a gap is silent, a rough edge is visible.

## 3. Does it contradict itself?

Check internal consistency end-to-end.

- Do claims in one section conflict with claims in another?
- Do code and explanation agree? Does the output match the stated logic?
- Are sources, data, or facts consistent throughout?

One contradiction undermines the whole output — the reader cannot know which part to trust.

## 4. Refine once, then return

Apply all findings from steps 1–3 in a single refinement pass. Then return.

- Fix gaps, contradictions, and missing content in one pass.
- Do not re-evaluate after refining — that is a loop, not a discipline.
- If the output still has known gaps after refinement, note them explicitly in a `limitations` field or closing note. Silence about gaps is worse than acknowledging them.

---

## Operating rules

- Recite the mantra **once** per reflection session, in the first response.
- Run reflection **before** returning any final structured output — not after.
- If the user says "skip reflection" → skip the recital but still run steps 1–3 silently before returning.
- Reflection is a **quality gate**, not a loop. One pass only. Do not spiral into infinite self-critique.
- The reflection is for **you** to act on — not a critique to deliver to the user. Fix the output, do not narrate the review.
