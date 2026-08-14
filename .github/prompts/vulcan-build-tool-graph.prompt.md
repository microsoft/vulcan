---
mode: agent
description: Run VULCAN's graph-construction phase (build_graph, plan_sequences) with a cost estimate first and a usability check afterwards.
---

Follow the procedure in [`.claude/skills/build-tool-graph/SKILL.md`](../../.claude/skills/build-tool-graph/SKILL.md).

Read that file first and follow it step by step. It is the single source of truth for this
workflow — this prompt file exists only so the same procedure is reachable from Copilot
Chat, and it is deliberately thin so the two cannot drift apart.

The script it calls is plain Python and needs no agent:

```bash
python .claude/skills/build-tool-graph/scripts/*.py --env <config-name>
```
