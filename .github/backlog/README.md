# AI backlog agent

## Activate after reviewing this setup PR

1. Merge this workflow into the default branch. Only that branch's workflow is used.
2. In Settings → Secrets and variables → Actions, add `OPENAI_API_KEY` as a repository
   secret, using a dedicated OpenAI project with budget alerts. Do not put keys in issues.
3. In Settings → Actions → General, enable **Allow GitHub Actions to create and approve
   pull requests**. This agent creates PRs; it never approves or merges them.
4. Protect the default branch with required human reviews and required CI checks;
   exclude the agent from bypass rights. The workflow alone cannot enforce repository rules.
5. Create labels `ai-ready` and `ai-blocked`. Give a small, well-defined issue the
   `ai-ready` label. Only collaborators with write access should apply that label.

The GitHub Project is a view: use the **repository issue label**, not a Project
status or a draft Project item. A task must exist as an issue in this repository.

## Operation

Labeling an issue starts a run. Actions → AI backlog → Run workflow also accepts
an issue number (it must still carry `ai-ready`). With no number, it picks the
oldest eligible issue. An hourly fallback drains remaining eligible backlog;
GitHub concurrency permits only one active run and may replace pending runs.
Each run handles one issue. Remove the `ai-ready` label to stop future attempts;
cancel an active run in Actions to stop it immediately.

The stable branch `ai/issue-N` prevents duplicate work. Existing branches are
skipped, including after PR closure. To retry, inspect/close any existing PR and
delete its agent branch, remove `ai-blocked`, and start manually or reapply
`ai-ready`. Failed attempts are labeled `ai-blocked`, preventing hourly retries.
If branch publication succeeds but PR creation fails, inspect the branch and
create its PR manually, or delete it before retrying.

## Boundaries and validation

- Fixed commit references pin the Actions dependencies; update pins deliberately.
- Trusted write-access initiators only; one issue, 30-minute generation limit.
- Codex uses `workspace-write` and `drop-sudo`, read-only GitHub permissions,
  no persisted checkout credentials, and an API proxy. Issue text is untrusted.
- Only a fixed artifact uploader follows Codex in its job. No shell command or
  repository code runs after Codex on that runner. The patch is untrusted input.
- A fresh runner without API secrets or write credentials validates the patch,
  installs the base dependencies, and runs pytest. At least one non-skipped test
  must pass; existing placeholder skips do not count. Tests can still be weak or
  misleading: human review must assess coverage and research assumptions.
- The publisher revalidates the same immutable artifact against the same base
  commit, runs no proposed project code, and opens a **draft** PR. It aborts if
  the default branch moved. There is no merge or approval operation.
- Only regular non-executable files in `src/`, `tests/`, `scripts/`, `configs/`,
  and `README.md` may change. Limit: 20 files / 100 KB. CI, dependencies, hidden
  files, symlinks, datasets, and generated results require a human-led change.
- No dataset download or model training is requested. Reviewers must prevent
  changes that fabricate results or introduce leakage between company splits.

The independent verification job's token has read-only contents permission;
GitHub still supplies a job token to Actions. It has no publishing rights or
OpenAI key. Patch content never reaches a job with both project-code execution
and GitHub write access. Prompt instructions reduce risk but cannot guarantee
an agent's behavior; enforce merge rules separately.

Push events from `GITHUB_TOKEN` do not trigger new workflows. PR opened/updated
events can create workflow runs requiring a write-access user's approval; use
**Approve workflows to run** on the PR when shown. Verification here runs
explicitly before publication. See
https://docs.github.com/en/actions/concepts/security/github_token.

## Costs and troubleshooting

Generation uses the Codex Action's default model; OpenAI API usage is billed
separately. Use project budgets/alerts, check actual billing enforcement for
your account, and disable this workflow to pause it. The 30-minute timeout is
not a monetary cap. There is no live-run validation until the secret is present.

On failure, inspect the linked Actions run before retrying. Common causes:
missing API key, all tests skipped, unclear issue criteria, forbidden paths,
default branch movement, or GitHub disallowing PR creation. The agent's final
response is in the generation log. Patch artifacts expire after three days.

Reference: https://learn.chatgpt.com/docs/github-action
