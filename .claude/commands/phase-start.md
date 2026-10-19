Start Phase $ARGUMENTS in a fresh session.
1. Read docs/HANDOFF.md, LOG.md, docs/phase-status.md, docs/decisions.md, docs/contracts.md, and only the
   Phase $ARGUMENTS section of docs/plan.md. Run `git status`, `git log --oneline -10`, `pytest -q`, `ruff check .`.
2. Report in at most 8 lines: state found, whether HANDOFF.md matches reality (flag every mismatch), failing
   tests, and external state I must provide (Kaggle outputs, ngrok URL, labels).
3. Enter plan mode and produce: files to create/change, order of work, tests, risks, assumptions, and anything in
   the plan that is ambiguous, unrealistic or wrong. Ask questions before coding if needed.
4. Do not write code until I approve.
Reminder: no Claude/Anthropic attribution in any commit or PR (see CLAUDE.md).
