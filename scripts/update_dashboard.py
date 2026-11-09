"""Refresh the README task chart using GitHub issues (Python standard library only)."""

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo


START = "<!-- task-dashboard:start -->"
END = "<!-- task-dashboard:end -->"


def count_tasks(issues, today):
    counts = {"done": 0, "due": 0, "later": 0}
    unscheduled = 0
    for issue in issues:
        labels = {label["name"] for label in issue.get("labels", [])}
        if "pull_request" in issue or "parent" in labels:
            continue
        if issue["state"] == "closed":
            counts["done"] += 1
            continue
        deadlines = []
        for label in labels:
            match = re.fullmatch(r"due:(\d{4}-\d{2}-\d{2})", label)
            if match:
                deadlines.append(date.fromisoformat(match[1]))
        if not deadlines:
            unscheduled += 1
        counts["later" if deadlines and min(deadlines) > today else "due"] += 1
    return counts, unscheduled


def render(issues, today):
    counts, unscheduled = count_tasks(issues, today)
    total = sum(counts.values())
    percentage = 100 * counts["done"] / total if total else 0
    return f"""{START}
## Project progress

**Finish by 22 November 2026 · {counts['done']}/{total} tasks complete ({percentage:.0f}%)**

```mermaid
%%{{init: {{"themeVariables": {{"pie1": "#2da44e", "pie2": "#fb8c00", "pie3": "#9ca3af"}}}}}}%%
pie showData
    title Project tasks
    "Done" : {counts['done']}
    "Due today or overdue" : {counts['due']}
    "Due later" : {counts['later']}
```

Counts exclude pull requests and parent issues to avoid counting work twice.
Open tasks with a due date on or before today appear in **Due today or overdue**.
Tasks without a deadline are also included there ({unscheduled} currently).
Deadlines use exact date labels such as `due:2026-11-09`.

Updated {today.isoformat()} (Europe/Oslo). Refreshes when issues change and daily.
Plan: data preparation by **15 Nov**, model training by **16 Nov**, experiments by
**20 Nov**, discussion and conclusions by **21 Nov**, final review by **22 Nov**.
{END}"""


def update_readme(path, dashboard):
    original = path.read_text()
    if START in original and END in original:
        before, rest = original.split(START, 1)
        _, after = rest.split(END, 1)
        updated = before + dashboard + after
    elif START in original or END in original:
        raise ValueError("README dashboard markers are incomplete")
    else:
        heading, rest = original.split("\n", 1)
        updated = heading + "\n\n" + dashboard + "\n" + rest
    if updated != original:
        path.write_text(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues-json", type=Path, help="Offline gh api --slurp output")
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parents[1] / "README.md")
    parser.add_argument("--today", type=date.fromisoformat)
    args = parser.parse_args()
    if args.issues_json:
        pages = json.loads(args.issues_json.read_text())
    else:
        repository = os.environ["GITHUB_REPOSITORY"]
        result = subprocess.run(
            ["gh", "api", f"repos/{repository}/issues?state=all&per_page=100", "--paginate", "--slurp"],
            check=True, capture_output=True, text=True,
        )
        pages = json.loads(result.stdout)
    issues = [issue for page in pages for issue in page]
    today = args.today or datetime.now(ZoneInfo("Europe/Oslo")).date()
    update_readme(args.readme, render(issues, today))


if __name__ == "__main__":
    main()
