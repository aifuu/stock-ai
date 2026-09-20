#!/usr/bin/env python3
"""Unit tests for non-live workflow hygiene invariants (UNIT 2 hardening).

Read-only: this file only parses YAML under .github/workflows/ with
yaml.safe_load and does simple text scans of the `run:` blocks it finds.
It never imports or executes any workflow's own scripts, and in particular
never imports adversarial_strategy_validator.py, multi_oos_profit_validator.py,
multi_oos_profit_gate.py, adversarial_oos_diagnostic.py, build_strategy_policy.py,
stock_scan.py, or any walk_forward*.py module.

Invariants asserted:
  1. Every workflow that pushes to main has a top-level `concurrency:` block
     with a `group`, except workflows on DISABLED_ALLOWLIST (intentionally
     disabled by the user, out of scope for this hardening pass).
  2. No workflow silently swallows a push failure with `git push ... || true`
     (or `git push` piped through `|| true` in any spelling this scan can
     see) -- a push loop must fail the job (`exit 1`) when every retry fails.
  3. Every workflow file declares an explicit top-level `permissions:` block
     (least-privilege default instead of the implicit all-write GITHUB_TOKEN).
"""
import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# _apply_hold_validator.yml is intentionally disabled by the user
# (disabled_manually in GitHub Actions) and explicitly out of scope for this
# hardening unit ("Do NOT touch ... _apply_hold_validator.yml"). It still
# pushes to main with a single, non-retried `git push origin HEAD:main` and
# no concurrency group -- a known, accepted gap while it stays disabled.
DISABLED_ALLOWLIST = {
    "_apply_hold_validator.yml",
}

PUSH_RE = re.compile(r"git push\b")


def _load_workflow(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_workflow_paths():
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _run_blocks(doc):
    blocks = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps", []) or []:
            if "run" in step:
                blocks.append((job_name, step.get("name", "<unnamed>"), step["run"]))
    return blocks


def _pushes_to_main(doc):
    return any(PUSH_RE.search(run) for _, _, run in _run_blocks(doc))


class WorkflowYamlParses(unittest.TestCase):
    def test_every_workflow_yaml_safe_loads(self):
        paths = _all_workflow_paths()
        self.assertGreater(len(paths), 0, "no workflow files found")
        for path in paths:
            with self.subTest(workflow=path.name):
                doc = _load_workflow(path)
                self.assertIsInstance(doc, dict)


class PushersHaveConcurrencyGroup(unittest.TestCase):
    """Every workflow that git-pushes to main must serialize via concurrency:."""

    def test_pushing_workflows_declare_concurrency_group(self):
        checked_pushers = []
        for path in _all_workflow_paths():
            name = path.name
            doc = _load_workflow(path)
            if not _pushes_to_main(doc):
                continue
            checked_pushers.append(name)
            if name in DISABLED_ALLOWLIST:
                continue
            with self.subTest(workflow=name):
                concurrency = doc.get("concurrency")
                self.assertIsInstance(
                    concurrency, dict,
                    msg=f"{name}: pushes to main but has no top-level concurrency: block",
                )
                group = concurrency.get("group")
                self.assertTrue(
                    group,
                    msg=f"{name}: concurrency block has no group",
                )

        # Sanity: this test must actually exercise every workflow that pushes,
        # so a future refactor can't silently make it a no-op.
        self.assertIn("enforce-policy-signature.yml", checked_pushers)
        self.assertIn("apply_requested_hardening.yml", checked_pushers)
        self.assertIn("intraday-top3-backtest.yml", checked_pushers)
        self.assertIn("daily-movers-root-cause.yml", checked_pushers)
        self.assertIn("fold4-up-probability-diagnostic.yml", checked_pushers)
        self.assertIn("fold4-filter-diagnostic.yml", checked_pushers)
        self.assertIn("top10-ticker-performance-report.yml", checked_pushers)

    def test_concurrency_groups_do_not_collide_across_workflows(self):
        """A colliding group would serialize unrelated jobs behind each other."""
        groups = {}
        for path in _all_workflow_paths():
            doc = _load_workflow(path)
            concurrency = doc.get("concurrency")
            if not isinstance(concurrency, dict):
                continue
            group = concurrency.get("group")
            if not group or "${{" in str(group):
                # dynamic groups (e.g. matrix-based) are out of scope here
                continue
            groups.setdefault(group, []).append(path.name)
        collisions = {g: files for g, files in groups.items() if len(files) > 1}
        self.assertEqual(collisions, {}, msg=f"colliding concurrency groups: {collisions}")

    def test_ticker_performance_report_group_independent_of_live_jobs(self):
        """W4: must not share a group with the live paper-trading/research jobs."""
        doc = _load_workflow(WORKFLOWS_DIR / "top10-ticker-performance-report.yml")
        group = doc["concurrency"]["group"]
        live_groups = {
            _load_workflow(WORKFLOWS_DIR / "ai-stock-scan.yml")["concurrency"]["group"],
            _load_workflow(WORKFLOWS_DIR / "daily-model-retrain.yml")["concurrency"]["group"],
        }
        self.assertNotIn(group, live_groups)


class PushLoopsNeverSilentlySucceed(unittest.TestCase):
    """A push loop must fail the job when every retry fails, never `|| true`."""

    def test_no_push_swallows_failure_with_or_true(self):
        for path in _all_workflow_paths():
            name = path.name
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "git push" not in line:
                    continue
                with self.subTest(workflow=name, line=line.strip()):
                    self.assertNotIn(
                        "|| true", line,
                        msg=f"{name}: 'git push ... || true' silently ignores push failure: {line.strip()}",
                    )

    def test_pushing_workflows_use_a_retry_loop(self):
        """Every non-allowlisted pusher retries via a `for i in 1 2 3 4 5` loop
        (the standardized fetch + rebase --autostash + push pattern) rather
        than pushing exactly once."""
        retry_re = re.compile(r"for i in 1 2 3( 4 5)?;")
        for path in _all_workflow_paths():
            name = path.name
            if name in DISABLED_ALLOWLIST:
                continue
            doc = _load_workflow(path)
            if not _pushes_to_main(doc):
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(workflow=name):
                self.assertRegex(
                    text, retry_re,
                    msg=f"{name}: git push has no retry loop",
                )


class PermissionsBlockPresent(unittest.TestCase):
    """Every workflow declares an explicit least-privilege permissions: block."""

    def test_every_workflow_has_top_level_permissions(self):
        for path in _all_workflow_paths():
            name = path.name
            doc = _load_workflow(path)
            with self.subTest(workflow=name):
                self.assertIn(
                    "permissions", doc,
                    msg=f"{name}: no top-level permissions: block (implicit token scope)",
                )

    def test_legacy_disabled_and_adhoc_workflows_are_read_only_or_none(self):
        for name in (
            "adhoc-discord-notify.yml",
            "intraday-auto-entry.yml",
            "intraday-exit-monitor.yml",
            "intraday-monitor.yml",
        ):
            doc = _load_workflow(WORKFLOWS_DIR / name)
            with self.subTest(workflow=name):
                perms = doc["permissions"]
                contents = perms.get("contents") if isinstance(perms, dict) else perms
                self.assertIn(contents, ("read", "none"))


class NoUserInputInterpolatedIntoScriptBody(unittest.TestCase):
    """github.event.inputs.* must be passed via env:, never spliced into a
    Python/bash script body, to avoid script injection from a free-form
    workflow_dispatch text input."""

    def test_hardened_workflows_do_not_interpolate_event_inputs_in_run(self):
        for name in ("intraday-top3-backtest.yml",):
            path = WORKFLOWS_DIR / name
            doc = _load_workflow(path)
            for job_name, step_name, run in _run_blocks(doc):
                with self.subTest(workflow=name, job=job_name, step=step_name):
                    self.assertNotIn("github.event.inputs", run)


if __name__ == "__main__":
    unittest.main()
