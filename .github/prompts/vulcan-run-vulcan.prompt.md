---
mode: agent
description: Drive a complete VULCAN run from a conversation — tool schemas or an existing catalog through to the exported training set, stopping for confirmation before anything expensive.
---

Follow the procedure in [`.claude/skills/run-vulcan/SKILL.md`](../../.claude/skills/run-vulcan/SKILL.md).

Read that file first and follow it step by step. It is the single source of truth for this
workflow — this prompt file exists only so the same procedure is reachable from Copilot
Chat, and it is deliberately thin so the two cannot drift apart.

It calls the other three VULCAN skills in order. Their procedures are in the sibling
directories under `.claude/skills/`.
