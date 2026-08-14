---
mode: agent
description: Run VULCAN's task-generation and trajectory stages and verify every stage wrote to its documented path with the documented record shape.
---

Follow the procedure in [`.claude/skills/generate-training-data/SKILL.md`](../../.claude/skills/generate-training-data/SKILL.md).

Read that file first and follow it step by step. It is the single source of truth for this
workflow — this prompt file exists only so the same procedure is reachable from Copilot
Chat, and it is deliberately thin so the two cannot drift apart.

The script it calls is plain Python and needs no agent:

```bash
python .claude/skills/generate-training-data/scripts/*.py --env <config-name>
```
