Finish Phase $ARGUMENTS (the next phase will start in a brand-new session with no memory of this one).
1. Run `ruff check .` and `pytest -q`; show real output. Do not finish while either fails.
2. Update LOG.md (every number with the command that regenerates it) and tick docs/phase-status.md only for items verified.
3. Rewrite docs/HANDOFF.md fully using its template: completed phase, repo state, runnable commands with expected
   output, decisions, honest open issues/shortcuts, external state not in git, and the exact first steps of the next phase.
4. Commit with a plain human-style message. NO Claude/Anthropic attribution of any kind. Then `git tag phase-$ARGUMENTS-done`. Do not push.
5. Print `git log --format='%an <%ae>%n%B' -3` and `git status --short` (must be clean) so I can confirm.
6. Finally print a ready-to-paste "next session prompt" (a single line starting with /phase-start N+1) plus a list of
   anything I must do manually before starting it.
