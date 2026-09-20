#!/usr/bin/env python3
"""Unit tests for the safe_state recovery-file persistence fix (UNIT 3).

Read-only / text-based: this file only parses YAML under .github/workflows/
with yaml.safe_load and does simple text scans of the `run:` blocks it
finds. It never imports or executes any workflow's own scripts, and in
particular never imports adversarial_strategy_validator.py,
multi_oos_profit_validator.py, multi_oos_profit_gate.py,
adversarial_oos_diagnostic.py, build_strategy_policy.py, stock_scan.py, or
any walk_forward*.py module.

Background: safe_state.safe_append_history() writes new rows for a
corrupt history CSV to `<name>.recovery.csv` (see safe_state.py's
`_recovery_path`), while quarantining the unreadable original as
`<name>.corrupt-<UTC ts>`. `.gitignore` ignores `*.bak` and `*.corrupt-*`
but deliberately NOT `*.recovery.csv`. Every workflow that persists paper
trading state does so via a fixed-name `git add` list
(`if [ -f "$f" ]; then git add "$f"; fi`), so a recovery file must be
listed by its exact name or it is silently dropped when the runner ends.
"""
import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

AI_STOCK_SCAN = WORKFLOWS_DIR / "ai-stock-scan.yml"
NIKKEI_PAPER = WORKFLOWS_DIR / "nikkei-macd-dip-paper.yml"

# History files actually written via safe_state.safe_append_history() and
# persisted by a workflow today (cross-checked against the source .py files):
#   profit_top10_paper.py:      HISTORY_FILE = 'profit_top10_paper_history.csv'
#   nikkei_macd_dip_paper.py:   history_file = 'nikkei_macd_dip_paper_history_{top,mid,bottom}.csv'
# daily_directional_top1.py's directional_paper_history.csv is written via
# safe_append_history too, but is NOT persisted by any workflow today (no
# `git add`/HISTORY_FILE reference in any .github/workflows/*.yml) -- that is
# a pre-existing gap, out of scope for this fix, and intentionally not
# asserted here as "must have a recovery entry".
HISTORY_TO_RECOVERY = {
    "profit_top10_paper_history.csv": "profit_top10_paper_history.recovery.csv",
    "nikkei_macd_dip_paper_history_top.csv": "nikkei_macd_dip_paper_history_top.recovery.csv",
    "nikkei_macd_dip_paper_history_mid.csv": "nikkei_macd_dip_paper_history_mid.recovery.csv",
    "nikkei_macd_dip_paper_history_bottom.csv": "nikkei_macd_dip_paper_history_bottom.recovery.csv",
}

# nikkei_macd_dip_paper_history.csv (no suffix) is a legacy filename with no
# corresponding `history_file` entry in nikkei_macd_dip_paper.py's CONFIGS --
# it is never written by safe_append_history today, so it must NOT gain a
# recovery sibling (there is nothing for one to recover).
NO_RECOVERY_EXPECTED = {"nikkei_macd_dip_paper_history.csv"}

# Fixed-name `if [ -f "$f" ]; then git add "$f"; fi` persistence loops only
# (the shell style the task requires new entries to match) -- NOT the
# `upload-artifact` debug path: lists, which are unrelated to git history.
GIT_ADD_GUARD_RE = re.compile(r'if \[ -f "\$f" \]; then git add "\$f"; fi')


def _load_workflow(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_blocks(doc):
    blocks = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps", []) or []:
            if "run" in step:
                blocks.append((job_name, step.get("name", "<unnamed>"), step["run"]))
    return blocks


def _for_f_in_lists(run_text):
    """Extract the space/backslash-newline separated file list of every
    `for f in ...; do` loop in a run: block (handles both the single-line
    and the `for f in \\\n  a b \\\n  c\\\n; do` continuation styles)."""
    lists = []
    for m in re.finditer(r"for f in\s+(.*?)\s*;\s*do", run_text, re.S):
        raw = m.group(1).replace("\\\n", " ")
        lists.append(raw.split())
    return lists


class RecoveryFilesAreListedInPersistenceSteps(unittest.TestCase):
    """Every workflow that persists a safe_state history CSV must also
    persist its exact `.recovery.csv` sibling, in the same `if [ -f "$f" ]`
    step style."""

    def test_ai_stock_scan_am_and_pm_persist_recovery_file(self):
        doc = _load_workflow(AI_STOCK_SCAN)
        blocks = _run_blocks(doc)
        loop_blocks = [
            (job, name, run) for job, name, run in blocks
            if "profit_top10_paper_history.csv" in run and "for f in" in run
        ]
        # Both AM and PM session loops must be checked (this cannot become a
        # silent no-op if a future refactor removes one of them).
        self.assertEqual(
            len(loop_blocks), 2,
            msg=f"expected exactly 2 persistence loops (AM+PM) in {AI_STOCK_SCAN.name}, found {len(loop_blocks)}",
        )
        for job, name, run in loop_blocks:
            with self.subTest(job=job, step=name):
                file_lists = _for_f_in_lists(run)
                self.assertEqual(len(file_lists), 1, msg="expected exactly one for-f-in list per loop block")
                files = file_lists[0]
                self.assertIn("profit_top10_paper_history.csv", files)
                self.assertIn(
                    "profit_top10_paper_history.recovery.csv", files,
                    msg=f"{AI_STOCK_SCAN.name} [{job}/{name}]: recovery file missing from persistence list",
                )

    def test_nikkei_macd_dip_paper_persists_all_three_recovery_files(self):
        doc = _load_workflow(NIKKEI_PAPER)
        blocks = _run_blocks(doc)
        loop_blocks = [
            (job, name, run) for job, name, run in blocks
            if "nikkei_macd_dip_paper_history_top.csv" in run and "for f in" in run
        ]
        self.assertEqual(len(loop_blocks), 1)
        _, _, run = loop_blocks[0]
        file_lists = _for_f_in_lists(run)
        self.assertEqual(len(file_lists), 1)
        files = file_lists[0]
        for history_file, recovery_file in HISTORY_TO_RECOVERY.items():
            if history_file.startswith("nikkei_macd_dip_paper_history"):
                with self.subTest(history_file=history_file):
                    self.assertIn(history_file, files)
                    self.assertIn(
                        recovery_file, files,
                        msg=f"{NIKKEI_PAPER.name}: {recovery_file} missing from persistence list",
                    )

    def test_legacy_unsuffixed_nikkei_history_gets_no_spurious_recovery_entry(self):
        """nikkei_macd_dip_paper_history.csv (no suffix) is never written by
        safe_append_history, so it must not gain a `.recovery.csv` sibling
        that could never legitimately be produced."""
        text = NIKKEI_PAPER.read_text(encoding="utf-8")
        self.assertNotIn("nikkei_macd_dip_paper_history.recovery.csv", text)

    def test_recovery_entries_keep_the_if_exists_guard_style(self):
        """The new entries must live inside the same `if [ -f "$f" ]; then
        git add "$f"; fi` loop as their history siblings, not a bespoke
        unconditional `git add` (which would fail the step if the file
        doesn't exist yet)."""
        for path in (AI_STOCK_SCAN, NIKKEI_PAPER):
            text = path.read_text(encoding="utf-8")
            with self.subTest(workflow=path.name):
                self.assertRegex(text, GIT_ADD_GUARD_RE)
                # every recovery filename that does appear must appear inside
                # a for-f-in list, never as a standalone `git add <recovery>`
                for recovery_file in HISTORY_TO_RECOVERY.values():
                    if recovery_file in text:
                        self.assertNotRegex(
                            text, re.compile(r'git add "?' + re.escape(recovery_file) + r'"?(?!\s*\\?\n?\s*\w)'),
                            msg=f"{path.name}: {recovery_file} must not be added outside the for-f-in guard loop",
                        )


class PersistenceListsUseNoGlobs(unittest.TestCase):
    """A glob (e.g. `*.csv`, `${name}*`) in a fixed-name persistence list
    could sweep up `.bak`/`.corrupt-*` files that must never be committed."""

    GLOB_CHARS = re.compile(r"[*?\[\]]")

    def test_no_persistence_for_f_in_list_contains_a_glob(self):
        for path in (AI_STOCK_SCAN, NIKKEI_PAPER):
            doc = _load_workflow(path)
            for job, name, run in _run_blocks(doc):
                for files in _for_f_in_lists(run):
                    with self.subTest(workflow=path.name, job=job, step=name):
                        for f in files:
                            self.assertFalse(
                                self.GLOB_CHARS.search(f),
                                msg=f"{path.name} [{job}/{name}]: glob-like token '{f}' in persistence list",
                            )

    def test_no_persistence_list_entry_is_a_bak_or_corrupt_file(self):
        for path in (AI_STOCK_SCAN, NIKKEI_PAPER):
            doc = _load_workflow(path)
            for job, name, run in _run_blocks(doc):
                for files in _for_f_in_lists(run):
                    with self.subTest(workflow=path.name, job=job, step=name):
                        for f in files:
                            self.assertFalse(f.endswith(".bak"), msg=f"{f} is a .bak file")
                            self.assertNotIn(".corrupt-", f, msg=f"{f} looks like a .corrupt-* quarantine file")


class GitignoreStillExemptsRecoveryFiles(unittest.TestCase):
    """Sanity check on the precondition this fix relies on: .gitignore must
    keep ignoring .bak/.corrupt-* but NOT *.recovery.csv."""

    def test_gitignore_ignores_bak_and_corrupt_but_not_recovery(self):
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        lines = {line.strip() for line in text.splitlines()}
        self.assertIn("*.bak", lines)
        self.assertIn("*.corrupt-*", lines)
        self.assertNotIn("*.recovery.csv", lines)
        self.assertNotIn("*.recovery*", lines)


class SafeStateRecoveryPathNamingMatchesWorkflowEntries(unittest.TestCase):
    """safe_state._recovery_path()'s naming contract (<name>.csv ->
    <name>.recovery.csv) must match exactly what the workflows now stage."""

    def test_recovery_path_naming_matches_added_workflow_entries(self):
        import safe_state

        for history_file, recovery_file in HISTORY_TO_RECOVERY.items():
            with self.subTest(history_file=history_file):
                self.assertEqual(safe_state._recovery_path(history_file), recovery_file)


if __name__ == "__main__":
    unittest.main()
