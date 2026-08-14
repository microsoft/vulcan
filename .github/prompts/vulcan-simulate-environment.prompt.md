---
mode: agent
description: Build a VULCAN input catalog from shared tool schemas, run the environment-simulation phase, and review the generated environment class for state/behaviour mismatches, unusable initial states, arguments with no effect, non-determinism, and missing tool coverage.
---

Follow the procedure in [`.claude/skills/simulate-environment/SKILL.md`](../../.claude/skills/simulate-environment/SKILL.md).

Read that file first and follow it step by step. It is the single source of truth for this
workflow — this prompt file exists only so the same procedure is reachable from Copilot
Chat, and it is deliberately thin so the two cannot drift apart.

The script it calls is plain Python and needs no agent:

```bash
python .claude/skills/simulate-environment/scripts/*.py --env <config-name>
```
