---
name: Task or defect
about: The standard format for every issue in this repository
title: "area: what is wrong or missing, stated plainly"
labels: ''
assignees: ''
---

<!--
Keep these seven sections, in this order, for every issue. The title prefix is
the subsystem: runtime, planner, harness, observability, policy, session,
server, cli, docs, packaging.

The rule this repo runs on: a claim needs evidence. Quote the code or paste the
command output that shows the problem, rather than describing it from memory.
-->

## Summary

What is wrong or missing, in one short paragraph. Quote the offending code or the
real command output rather than paraphrasing it.

## Why this matters

What breaks, or what a user cannot do, as a consequence. Prefer a concrete
failure over an adjective.

## Where in the code

- `path/to/file.py:LINE` — what is there and why it is relevant

Include a command a reader can run to confirm the problem for themselves:

```bash
```

## What to change

1. …
2. …

State anything deliberately out of scope, so a pull request does not grow past
what was agreed here.

## How to verify

```bash
uv run pytest -q
uv run ruff check .
```

Note which new test proves the fix. This repo's convention is that a change
arrives with a test that **fails without it** — check that by reverting your
source edit and watching the new test go red.

## Acceptance criteria

- [ ] …
- [ ] `uv run pytest` stays green and `uv run ruff check .` is clean
- [ ] Any README or cookbook sentence this changes is updated in the same pull
      request (several are byte-compared against real output by the test suite)

## Skill level

Pick one and delete the other.

**good first issue** — say why it is well bounded, point at a sibling file that
shows the pattern to copy, and invite questions on the issue.

**experience required** — say which subsystems the change spans, what could
silently break, and ask for a design comment on the issue before any code is
written.
