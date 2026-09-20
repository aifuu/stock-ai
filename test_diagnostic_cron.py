#!/usr/bin/env python3
"""Unit tests for the 30-minute cadence on the two read-only diagnostics.

Read-only: this file only parses YAML under .github/workflows/ with
yaml.safe_load. It never imports or executes any workflow's own scripts.
"""
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

DIAGNOSTIC_WORKFLOWS = [
    "oos-signal-diagnostic.yml",
    "selection-audit.yml",
]

EXPECTED_CRON = "*/30 0-6 * * 1-5"


def _load_workflow(name):
    with open(WORKFLOWS_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_on_block(doc):
    if "on" in doc:
        return doc["on"]
    if True in doc:
        return doc[True]
    raise AssertionError("workflow has no 'on:' trigger block at all")


class DiagnosticWorkflowsUse30MinuteCron(unittest.TestCase):
    def test_schedule_trigger_uses_30_minute_cron(self):
        for name in DIAGNOSTIC_WORKFLOWS:
            with self.subTest(workflow=name):
                doc = _load_workflow(name)
                on_block = _get_on_block(doc)
                self.assertIn("schedule", on_block, msg=f"{name}: no schedule trigger")
                crons = [entry["cron"] for entry in on_block["schedule"]]
                self.assertIn(
                    EXPECTED_CRON, crons,
                    msg=f"{name}: expected cron {EXPECTED_CRON!r}, got {crons!r}",
                )

    def test_workflow_dispatch_preserved(self):
        for name in DIAGNOSTIC_WORKFLOWS:
            with self.subTest(workflow=name):
                doc = _load_workflow(name)
                on_block = _get_on_block(doc)
                self.assertIn("workflow_dispatch", on_block, msg=f"{name}: lost workflow_dispatch")

    def test_permissions_block_present(self):
        for name in DIAGNOSTIC_WORKFLOWS:
            with self.subTest(workflow=name):
                doc = _load_workflow(name)
                self.assertIn("permissions", doc, msg=f"{name}: no permissions: block")

    def test_concurrency_group_unique_to_this_workflow(self):
        groups = {}
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            doc = _load_workflow(path.name)
            concurrency = doc.get("concurrency")
            if not isinstance(concurrency, dict):
                continue
            group = concurrency.get("group")
            if not group or "${{" in str(group):
                continue
            groups.setdefault(group, []).append(path.name)

        for name in DIAGNOSTIC_WORKFLOWS:
            with self.subTest(workflow=name):
                doc = _load_workflow(name)
                concurrency = doc.get("concurrency")
                self.assertIsInstance(concurrency, dict, msg=f"{name}: no concurrency: block")
                group = concurrency.get("group")
                self.assertTrue(group, msg=f"{name}: concurrency block has no group")
                owners = groups.get(group, [])
                self.assertEqual(
                    owners, [name],
                    msg=f"{name}: concurrency group {group!r} is not unique to this workflow: {owners!r}",
                )

    def test_never_contains_git_push(self):
        for name in DIAGNOSTIC_WORKFLOWS:
            with self.subTest(workflow=name):
                text = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
                self.assertNotIn("git push", text, msg=f"{name}: unexpectedly contains 'git push' (must be read-only)")
                self.assertNotIn("git commit", text, msg=f"{name}: unexpectedly contains 'git commit' (must be read-only)")


if __name__ == "__main__":
    unittest.main()
