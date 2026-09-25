- Don't add claude or AI signatures in git commits

## Code comments

- Comments say what the code does, and why it is shaped that way when that is
  not obvious from reading it — an invariant it maintains, an ordering it
  depends on, a constraint it satisfies.
- Keep them terse.
- No bug archaeology: not what defect a line fixes, not what was tried and
  refuted, not which hypothesis turned out wrong.
- No measurements: no benchmark tables, no percentages, no machine names, no
  "measured N medians per arm". Numbers are true of one machine, one commit and
  one toolchain, and the code outlives all three.
- A switch or env var gets one line: what it changes, and whether it is safe to
  run with.
- That detail belongs in a git-bug issue.

## Commit messages

Keep them short. A subject line saying what changed, and a body only when the
change is not self-evident — a couple of sentences on what it does and why, not
a narrative.

The issue is where the detail lives: the reproducer, the measurements, what was
tried and refuted, the alternatives weighed. Cite the issue and close it there
rather than restating any of it in the commit. A reader who wants the story runs
`git-bug bug show <id>`.

Cite it as a trailer with the **full 64-character** id:

    git-bug: 75be07f093495e8b787475ecd3fe0832dd8491258e3bd328f2aba759d780416e

Never the short form. A git-bug short id looks exactly like a git short SHA, and
forges autolink any bare `[0-9a-f]{7,40}` token — so today it reads as a commit
to a human, and the day a real commit starts with those characters it becomes a
link to the wrong thing. The full id is SHA-256 and 64 characters, past the end
of that pattern, so it cannot be mistaken for a commit either way. Both forms
resolve in `git-bug bug show`.

Do not write `Closes <id>` or `Fixes <id>`: those verbs mean GitHub issues, and
these are not GitHub issues. Close the issue in git-bug and record the commit
SHA there — that direction is the stable one, because a commit SHA is a real
object the forge links correctly.

Avoid in a commit message anything that goes stale the same way it would in a
comment — call-site counts, file line numbers, benchmark percentages. Say what
changed; let the issue carry the evidence.

## Issue tracking

Use `git-bug` for local issue tracking. Issues are git objects under
`refs/bugs/*`, so they travel with the repository and `git status` stays clean.

Do not create planning documents, backlog files, or status `.md` files. File an
issue instead. Anything that would have become a long comment, a TODO, or a
paragraph of measurement history belongs in an issue.

**Filing.** Write the issue to a file and pass it with `-F`. The file is read
the way git reads a commit message: **first line is the title, then a blank
line, then the body.** `-t` is IGNORED when `-F` is given — pass a title that
way and it is silently dropped, the first body line becomes the title, and the
body loses that line.

```
printf '%s\n\n%s\n' "$title" "$body" > /tmp/issue.md
git-bug bug new -F /tmp/issue.md --non-interactive     # prints "<id> created"
git-bug bug label new <id> bug area:gc priority:high
```

For a one-liner, `git-bug bug new -t "<title>" -m "<body>" --non-interactive`
works and does honour `-t`.

**Reading and working.**

```
git-bug bug                                  # list (shows id, status, title)
git-bug bug --label area:gc --status open    # filter; repeat --label to AND
git-bug bug --by creation                    # sort
git-bug bug "some words"                     # full-text search
git-bug bug show <id>                        # full text
git-bug bug show <id> --field title|status|labels
git-bug bug show <id> --format json
git-bug bug comment new <id> -m "..." --non-interactive
git-bug bug status close <id>                # and `status open` to reopen
git-bug bug label rm <id> <label>
git-bug termui                               # browse; `git-bug webui` for a UI
```

The query language cannot parse a label containing a colon —
`git-bug bug label:area:gc` silently returns nothing. Always use the
`--label area:gc` flag form.

**Labels.** Give every issue a kind and an area. Kind is the same everywhere:
`bug`, `perf`, `debt`. Area is `area:<subsystem>`, named after the repository's
own structure rather than a fixed list — run `git-bug label` to see what a
project already uses and reuse those names instead of inventing near-duplicates.
Add `priority:high` for work that blocks something else.

**Syncing.** An ordinary `git push` does NOT carry `refs/bugs/*`. Use
`git-bug push [remote]` and `git-bug pull [remote]`, and only when asked, under
the same rule as pushing code.

**A fresh clone** needs an identity before it can write:
`git-bug user new -n "<name>" -e "<email>" --non-interactive`, then
`git-bug pull` to fetch existing issues.

**Closing.** Close an issue when the work lands, and say in the commit message
which issue it closes. Do not close one because it looks stale — reopen or
comment instead.
