#!/usr/bin/env python3
"""Unit tests for .github/workflows/*.yml push-trigger branch filters.

Background: a workflow that fires on `push:` with no `branches:` filter but
whose job checks out `ref: main` will run against main's code for a push to
*any* branch that happens to touch its watched `paths:` (the same bug class
as the refresh-walk-forward-candidates.yml incident). These tests assert
every such workflow restricts its push trigger to `branches: [main]`, and
that adding the filter did not drop any existing `paths:` filter.

Read-only: this file only parses YAML under .github/workflows/ with
yaml.safe_load. It never imports or executes any workflow's own scripts,
and in particular never imports build_strategy_policy.py,
adversarial_strategy_validator.py, multi_oos_profit_validator.py,
multi_oos_profit_gate.py, adversarial_oos_diagnostic.py, stock_scan.py, or
any walk_forward*.py module.
"""
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Workflows that are intentionally disabled / out of scope for the
# branches:[main] requirement, computed honestly rather than assumed:
# - _apply_hold_validator.yml: manually disabled, already has branches:[main]
# - profit-optimizer-validation.yml: manually disabled, has no push trigger
DISABLED_ALLOWLIST = {
    "_apply_hold_validator.yml",
    "profit-optimizer-validation.yml",
}

# Workflows whose job step explicitly does `actions/checkout` with
# `ref: main`, so a push-triggered run on any other branch would still run
# against main's code unless the push trigger itself is branch-filtered.
# (fold4-up-probability-root-cause.yml is deliberately NOT in this set: its
# checkout step has no explicit `ref:` and defaults to the triggering
# ref/SHA, so it isn't technically this bug class -- it still gets
# branches:[main] below because it was explicitly named in scope.)
EXPECTED_REF_MAIN_WORKFLOWS = {
    "enforce-policy-signature.yml",
    "fold4-filter-diagnostic.yml",
    "policy-metadata-check.yml",
    "_apply_hold_validator.yml",
    # Already fixed prior to this change (refresh-walk-forward-candidates.yml
    # is the workflow from the original Refresh incident this fix mirrors).
    "refresh-walk-forward-candidates.yml",
    "sync-ticker-universe.yml",
}

# The four workflows this fix targets, with the `paths:` list each one had
# before the branches:[main] filter was added (must be preserved verbatim).
EXPECTED_PATHS = {
    "enforce-policy-signature.yml": [
        ".github/workflows/enforce-policy-signature.yml",
    ],
    "fold4-filter-diagnostic.yml": [
        "fold4_oos_filter_trace.py",
        ".github/workflows/fold4-filter-diagnostic.yml",
    ],
    "policy-metadata-check.yml": [
        "strategy_policy.json",
        "strategy_policy_up.json",
        "strategy_policy_down.json",
        "policy_manual_overrides.json",
        "check_policy_metadata.py",
        ".github/workflows/policy-metadata-check.yml",
    ],
    "fold4-up-probability-root-cause.yml": [
        "walk_forward_exact_pipeline_trace.py",
        ".github/workflows/fold4-up-probability-root-cause.yml",
    ],
}


def _load_workflow(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_on_block(doc):
    """PyYAML 1.1 parses the bare scalar key `on` as the boolean True.

    Handle both spellings so this works regardless of PyYAML version /
    whether the workflow author quoted the key.
    """
    if "on" in doc:
        return doc["on"]
    if True in doc:
        return doc[True]
    raise AssertionError("workflow has no 'on:' trigger block at all")


def _checks_out_ref_main(workflow_path):
    text = workflow_path.read_text(encoding="utf-8")
    return "ref: main" in text or "ref: 'main'" in text or 'ref: "main"' in text


def _all_workflow_paths():
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


class WorkflowYamlParses(unittest.TestCase):
    """Every workflow file must be valid YAML with a parseable on: block."""

    def test_every_workflow_yaml_safe_loads(self):
        paths = _all_workflow_paths()
        self.assertGreater(len(paths), 0, "no workflow files found")
        for path in paths:
            with self.subTest(workflow=path.name):
                doc = _load_workflow(path)
                self.assertIsInstance(doc, dict)


class PushTriggersRestrictToMainWhenCheckingOutMain(unittest.TestCase):
    """Core regression test for the branch-filter bug class."""

    def test_push_trigger_with_checkout_ref_main_has_branches_main(self):
        checked = []
        for path in _all_workflow_paths():
            name = path.name
            doc = _load_workflow(path)
            on_block = _get_on_block(doc)

            has_push = isinstance(on_block, dict) and "push" in on_block
            if not has_push:
                continue

            checks_out_main = _checks_out_ref_main(path)
            if not checks_out_main:
                continue

            checked.append(name)
            if name in DISABLED_ALLOWLIST:
                continue

            push_block = on_block["push"] or {}
            self.assertIsInstance(
                push_block,
                dict,
                msg=f"{name}: push trigger has no keys to hold branches:",
            )
            branches = push_block.get("branches")
            self.assertEqual(
                branches,
                ["main"],
                msg=f"{name}: push trigger checks out ref: main but lacks "
                f"branches: [main] (got {branches!r}); a push to any other "
                f"branch touching this workflow's paths would still run it "
                f"against main",
            )

        # Sanity: make sure this test actually exercised the workflows we
        # know about, so a future refactor can't silently make it a no-op.
        self.assertEqual(set(checked), EXPECTED_REF_MAIN_WORKFLOWS)

    def test_apply_hold_validator_already_has_branches_main(self):
        path = WORKFLOWS_DIR / "_apply_hold_validator.yml"
        doc = _load_workflow(path)
        on_block = _get_on_block(doc)
        self.assertEqual(on_block["push"]["branches"], ["main"])

    def test_profit_optimizer_validation_has_no_push_trigger(self):
        path = WORKFLOWS_DIR / "profit-optimizer-validation.yml"
        doc = _load_workflow(path)
        on_block = _get_on_block(doc)
        self.assertNotIn("push", on_block)


class PathsFiltersPreserved(unittest.TestCase):
    """Adding branches:[main] must not drop the existing paths: filter."""

    def test_paths_filters_preserved_for_fixed_workflows(self):
        for name, expected_paths in EXPECTED_PATHS.items():
            with self.subTest(workflow=name):
                path = WORKFLOWS_DIR / name
                doc = _load_workflow(path)
                on_block = _get_on_block(doc)
                push_block = on_block["push"]
                self.assertEqual(push_block.get("paths"), expected_paths)
                self.assertEqual(push_block.get("branches"), ["main"])


class OnKeyYamlBooleanQuirkHandled(unittest.TestCase):
    """Regression test for the yaml.safe_load('on') -> True quirk itself."""

    def test_bare_on_key_parses_as_boolean_true(self):
        doc = yaml.safe_load("on:\n  push:\n    branches: [main]\n")
        self.assertNotIn("on", doc)
        self.assertIn(True, doc)
        self.assertEqual(_get_on_block(doc), {"push": {"branches": ["main"]}})

    def test_quoted_on_key_parses_as_string(self):
        doc = yaml.safe_load('"on":\n  push:\n    branches: [main]\n')
        self.assertIn("on", doc)
        self.assertNotIn(True, doc)
        self.assertEqual(_get_on_block(doc), {"push": {"branches": ["main"]}})


if __name__ == "__main__":
    unittest.main()
