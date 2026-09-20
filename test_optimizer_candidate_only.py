#!/usr/bin/env python3
"""Unit tests for profit-optimizer-validation.yml PLAN B (candidate-only output).

Background: profit-optimizer-validation.yml (currently disabled_manually) used
to write its selected policy directly into the LIVE, trading-facing files
(strategy_policy.json / strategy_policy_up.json / strategy_policy_down.json).
It now only ever writes "candidate" files under candidates/, and never
touches the live policy files or policy_manual_overrides.json. Promotion of a
candidate to live is a separate, explicit, human-approved action (see
candidates/README.md), not something any workflow does automatically.

Read-only: this file only parses YAML under .github/workflows/ with
yaml.safe_load and does text/regex scans of the `run:` blocks it finds. It
never imports or executes any workflow's own scripts, and in particular never
imports adversarial_strategy_validator.py, multi_oos_profit_validator.py,
multi_oos_profit_gate.py, adversarial_oos_diagnostic.py,
build_strategy_policy.py, stock_scan.py, or any walk_forward*.py module.
"""
import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "profit-optimizer-validation.yml"

LIVE_POLICY_FILES = (
    "strategy_policy.json",
    "strategy_policy_up.json",
    "strategy_policy_down.json",
    "policy_manual_overrides.json",
)


def _load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_steps(doc):
    """Yield (job_name, step_index, step) for every step in every job."""
    for job_name, job in (doc.get("jobs") or {}).items():
        for idx, step in enumerate(job.get("steps", []) or []):
            yield job_name, idx, step


def _run_blocks(doc):
    return [
        (job_name, step.get("name", "<unnamed>"), step["run"])
        for job_name, idx, step in _all_steps(doc)
        if "run" in step
    ]


class WorkflowYamlParses(unittest.TestCase):
    def test_workflow_yaml_safe_loads(self):
        doc = _load_workflow()
        self.assertIsInstance(doc, dict)


class BspPolicyFileAlwaysTargetsCandidatesDir(unittest.TestCase):
    """Every BSP_POLICY_FILE env value must live under candidates/."""

    def test_bsp_policy_file_values_start_with_candidates(self):
        doc = _load_workflow()
        found = []
        for job_name, idx, step in _all_steps(doc):
            env = step.get("env") or {}
            if "BSP_POLICY_FILE" not in env:
                continue
            value = str(env["BSP_POLICY_FILE"])
            found.append((job_name, step.get("name"), value))
            with self.subTest(job=job_name, step=step.get("name")):
                self.assertTrue(
                    value.startswith("candidates/"),
                    msg=f"BSP_POLICY_FILE={value!r} does not start with candidates/",
                )
        # Sanity: this must actually exercise both the main policy build and
        # the up/down trend-variant policy build, so a future refactor that
        # silently drops one of them can't make this test a no-op.
        self.assertGreaterEqual(len(found), 2, msg=f"expected >=2 BSP_POLICY_FILE steps, found {found}")

    def test_no_bsp_policy_file_env_points_at_a_live_policy_name(self):
        doc = _load_workflow()
        for job_name, idx, step in _all_steps(doc):
            env = step.get("env") or {}
            value = env.get("BSP_POLICY_FILE")
            if value is None:
                continue
            with self.subTest(job=job_name, step=step.get("name")):
                basename = str(value).rsplit("/", 1)[-1]
                self.assertNotIn(
                    basename, LIVE_POLICY_FILES,
                    msg=f"BSP_POLICY_FILE basename {basename!r} collides with a live policy file name",
                )


class GitAddNeverNamesOrGlobsALivePolicyFile(unittest.TestCase):
    """No `git add` anywhere in this workflow may stage a live policy file,
    whether by exact name or by a glob/wildcard that could match one."""

    GIT_ADD_RE = re.compile(r"git add\b([^\n]*)")
    # Conservative: any wildcard character in a git add argument list is
    # suspicious enough to fail this workflow's tests outright, since every
    # intentional path here is spelled out exactly (never globbed).
    WILDCARD_CHARS = set("*?[")

    def test_git_add_lines_never_touch_live_policy_files(self):
        doc = _load_workflow()
        run_blocks = _run_blocks(doc)
        self.assertTrue(run_blocks, "no run: blocks found")
        checked_git_add_lines = 0
        for job_name, step_name, run in run_blocks:
            for match in self.GIT_ADD_RE.finditer(run):
                checked_git_add_lines += 1
                args_text = match.group(1)
                with self.subTest(job=job_name, step=step_name, line=match.group(0).strip()):
                    tokens = args_text.split()
                    for token in tokens:
                        for live_name in LIVE_POLICY_FILES:
                            self.assertNotEqual(
                                token, live_name,
                                msg=f"{job_name}/{step_name}: `git add` names live policy file {live_name!r}",
                            )
                        if any(ch in token for ch in self.WILDCARD_CHARS):
                            self.fail(
                                f"{job_name}/{step_name}: `git add` uses a wildcard token "
                                f"{token!r} that could match a live policy file"
                            )
        self.assertGreater(checked_git_add_lines, 0, "no `git add` invocations found to check")

    def test_git_add_dash_u_is_gated_by_a_guard_step_first(self):
        """`git add -u` stages ALL modified tracked files repo-wide, so it can
        only be safe here because a guard step (see below) already asserted
        the live policy files have zero working-tree changes before it runs."""
        doc = _load_workflow()
        for job_name, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps", []) or []
            for idx, step in enumerate(steps):
                run = step.get("run", "")
                if "git add -u" not in run:
                    continue
                with self.subTest(job=job_name, step=step.get("name")):
                    preceding_names = [s.get("name", "") for s in steps[:idx]]
                    self.assertTrue(
                        any(name.startswith("Guard:") for name in preceding_names),
                        msg=f"{job_name}/{step.get('name')}: `git add -u` has no preceding Guard: step",
                    )


class GuardStepExistsBeforeEveryCommitStep(unittest.TestCase):
    """Every step whose name starts with 'Commit' must be immediately preceded
    (not necessarily adjacently, but earlier in the same job) by a step whose
    name starts with 'Guard:' that checks the live policy files."""

    def test_guard_step_precedes_every_commit_step(self):
        doc = _load_workflow()
        commit_steps_seen = 0
        for job_name, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps", []) or []
            for idx, step in enumerate(steps):
                name = step.get("name", "")
                if not name.startswith("Commit"):
                    continue
                commit_steps_seen += 1
                with self.subTest(job=job_name, step=name):
                    preceding_names = [s.get("name", "") for s in steps[:idx]]
                    guard_steps = [n for n in preceding_names if n.startswith("Guard:")]
                    self.assertTrue(
                        guard_steps,
                        msg=f"{job_name}/{name}: no Guard: step found before this commit step",
                    )
        self.assertGreaterEqual(commit_steps_seen, 2, "expected >=2 Commit steps (validate + trend variants)")

    def test_guard_step_checks_all_four_live_policy_files(self):
        doc = _load_workflow()
        guard_steps_seen = 0
        for job_name, idx, step in _all_steps(doc):
            name = step.get("name", "")
            if not name.startswith("Guard:"):
                continue
            guard_steps_seen += 1
            run = step.get("run", "")
            with self.subTest(job=job_name, step=name):
                self.assertIn("git status --porcelain", run)
                for live_name in LIVE_POLICY_FILES:
                    self.assertIn(
                        live_name, run,
                        msg=f"{job_name}/{name}: guard step does not check {live_name!r}",
                    )
                self.assertIn("exit 1", run, msg="guard step must fail the job on any detected change")
        self.assertGreaterEqual(guard_steps_seen, 2, "expected a Guard: step in both jobs")


class ComparisonStepExists(unittest.TestCase):
    """A read-only candidate-vs-live comparison step must exist in every job
    that builds a candidate policy, and must never perform signature
    operations or import repo modules."""

    FORBIDDEN_IMPORTS = (
        "adversarial_strategy_validator",
        "multi_oos_profit_validator",
        "multi_oos_profit_gate",
        "adversarial_oos_diagnostic",
        "build_strategy_policy",
        "stock_scan",
        "walk_forward",
    )

    def test_comparison_step_exists_in_every_job(self):
        doc = _load_workflow()
        comparison_steps_seen = 0
        for job_name, job in (doc.get("jobs") or {}).items():
            names = [s.get("name", "") for s in job.get("steps", []) or []]
            matches = [n for n in names if n.startswith("Compare")]
            with self.subTest(job=job_name):
                self.assertTrue(matches, msg=f"{job_name}: no comparison step found")
            comparison_steps_seen += len(matches)
        self.assertGreaterEqual(comparison_steps_seen, 2)

    def test_comparison_step_is_read_only(self):
        doc = _load_workflow()
        for job_name, idx, step in _all_steps(doc):
            name = step.get("name", "")
            if not name.startswith("Compare"):
                continue
            run = step.get("run", "")
            with self.subTest(job=job_name, step=name):
                self.assertNotIn("hmac", run.lower())
                self.assertNotIn("approval_signature", run)
                self.assertNotIn("POLICY_SIGNING_SECRET", run)
                for forbidden in self.FORBIDDEN_IMPORTS:
                    self.assertNotIn(
                        f"import {forbidden}", run,
                        msg=f"{job_name}/{name}: comparison step imports forbidden module {forbidden}",
                    )
                self.assertNotIn("git add", run)
                self.assertNotIn("git commit", run)
                self.assertNotIn("git push", run)

    def test_comparison_step_writes_to_candidates_dir(self):
        doc = _load_workflow()
        for job_name, idx, step in _all_steps(doc):
            name = step.get("name", "")
            if not name.startswith("Compare"):
                continue
            env = step.get("env") or {}
            with self.subTest(job=job_name, step=name):
                out_file = str(env.get("CMP_OUT_FILE", ""))
                self.assertTrue(
                    out_file.startswith("candidates/"),
                    msg=f"CMP_OUT_FILE={out_file!r} does not start with candidates/",
                )


class WorkflowTriggersAndConcurrencyUnchanged(unittest.TestCase):
    """This unit must not change when/how the workflow runs, only what it
    writes: same triggers, same concurrency group, still disabled only via
    the GitHub UI (no push trigger added)."""

    def test_has_workflow_dispatch_trigger(self):
        doc = _load_workflow()
        on_block = doc.get("on") if "on" in doc else doc.get(True)
        self.assertIsInstance(on_block, dict)
        self.assertIn("workflow_dispatch", on_block)

    def test_has_schedule_trigger(self):
        doc = _load_workflow()
        on_block = doc.get("on") if "on" in doc else doc.get(True)
        self.assertIn("schedule", on_block)
        schedule = on_block["schedule"]
        self.assertIsInstance(schedule, list)
        self.assertTrue(any(isinstance(s, dict) and s.get("cron") == "0 9 * * 6" for s in schedule))

    def test_no_push_trigger(self):
        doc = _load_workflow()
        on_block = doc.get("on") if "on" in doc else doc.get(True)
        self.assertNotIn("push", on_block)

    def test_concurrency_group_unchanged(self):
        doc = _load_workflow()
        concurrency = doc.get("concurrency")
        self.assertIsInstance(concurrency, dict)
        self.assertEqual(concurrency.get("group"), "stock-ai-profit-optimizer")
        self.assertEqual(concurrency.get("cancel-in-progress"), False)

    def test_permissions_block_unchanged(self):
        doc = _load_workflow()
        self.assertEqual(doc.get("permissions"), {"contents": "write"})


class WorkflowStillEditsOnlyExpectedFiles(unittest.TestCase):
    """Guard against scope creep: this workflow's jobs are still exactly
    `validate` and `validate_trend_variants`, and it still never mentions
    the forbidden modules anywhere (not just in the comparison step)."""

    FORBIDDEN_MODULES_TO_NEVER_IMPORT = (
        "multi_oos_profit_validator",
    )

    def test_job_names_unchanged(self):
        doc = _load_workflow()
        self.assertEqual(set(doc.get("jobs", {}).keys()), {"validate", "validate_trend_variants"})

    def test_never_imports_multi_oos_profit_validator(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        for forbidden in self.FORBIDDEN_MODULES_TO_NEVER_IMPORT:
            self.assertNotIn(f"import {forbidden}", text)


if __name__ == "__main__":
    unittest.main()
