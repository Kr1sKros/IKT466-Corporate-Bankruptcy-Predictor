import importlib.util
from datetime import date
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "dashboard", Path(__file__).resolve().parents[1] / "scripts/update_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def issue(state="open", *labels, **extra):
    return {"state": state, "labels": [{"name": label} for label in labels], **extra}


class DashboardTests(unittest.TestCase):
    def test_partition_excludes_parents_and_prs_and_keeps_overdue_visible(self):
        issues = [
            issue("closed", "due:2026-11-09"),
            issue("open", "due:2026-11-08"),
            issue("open", "due:2026-11-09"),
            issue("open", "due:2026-11-10"),
            issue("open"),
            issue("closed", "parent"),
            issue("open", pull_request={}),
        ]
        self.assertEqual(dashboard.count_tasks(issues, date(2026, 11, 9)),
                         ({"done": 1, "due": 3, "later": 1}, 1))

    def test_daily_rollover_and_year_boundary(self):
        issues = [issue("open", "due:2026-11-16")]
        self.assertEqual(dashboard.count_tasks(issues, date(2026, 11, 15))[0]["later"], 1)
        self.assertEqual(dashboard.count_tasks(issues, date(2026, 11, 16))[0]["due"], 1)
        self.assertEqual(dashboard.count_tasks([issue("open", "due:2027-01-01")],
                                             date(2026, 12, 31))[0]["later"], 1)
        self.assertEqual(dashboard.count_tasks([issue("open", "due:2027-01-01")],
                                             date(2027, 1, 1))[0]["due"], 1)

    def test_readme_update_is_idempotent_and_preserves_other_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text("# Project\n\nExisting content.\n")
            block = dashboard.render([], date(2026, 11, 9))
            dashboard.update_readme(path, block)
            first = path.read_text()
            dashboard.update_readme(path, block)
            self.assertEqual(path.read_text(), first)
            self.assertTrue(first.endswith("Existing content.\n"))
            self.assertEqual(first.count(dashboard.START), 1)


if __name__ == "__main__":
    unittest.main()
