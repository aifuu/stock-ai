#!/usr/bin/env python3
"""Unit tests for the train_data.csv -> train_data.csv.gz git-storage change.

stock_scan.py is not imported directly: importing it executes a large amount
of module-level code (network downloads, feature computation, and even a
call to load_training_data() itself). Instead, load_training_data() is
extracted from the file's AST and exec'd in an isolated namespace with a
stubbed TRAIN_FILE/FEATURES, exactly like the "AST extraction" approach
already used elsewhere in this repo's tooling. This never imports, and never
calls, any of the forbidden signature/validator modules.

These tests never modify any file inside the repository; all CSV/gzip
fixtures are written under a TemporaryDirectory.
"""
import ast
import gzip
import os
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
STOCK_SCAN_PY = REPO_ROOT / "stock_scan.py"


def _extract_function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found in {path}")


def _load_isolated_load_training_data(features):
    """Exec just load_training_data() from stock_scan.py with stubbed
    TRAIN_FILE/FEATURES globals, never importing the real module."""
    source = _extract_function_source(STOCK_SCAN_PY, "load_training_data")
    namespace = {
        "os": os,
        "pd": pd,
        "TRAIN_FILE": "train_data.csv",
        "FEATURES": features,
    }
    exec(compile(source, "<load_training_data extract>", "exec"), namespace)
    return namespace["load_training_data"]


class LoadTrainingDataGzFallback(unittest.TestCase):
    """load_training_data() must transparently fall back to train_data.csv.gz
    when the plain CSV is absent, and prefer the plain file when both exist
    (the in-job case right after daily_model_retrain.py writes it fresh)."""

    FEATURES = ["feat_a", "feat_b"]

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._prev_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        self.addCleanup(os.chdir, self._prev_cwd)
        self.load_training_data = _load_isolated_load_training_data(self.FEATURES)

    def _make_df(self, marker):
        return pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "target": [0, 1, 2],
            "marker": [marker, marker, marker],
        })

    def test_falls_back_to_gz_when_plain_missing(self):
        self._make_df("gz").to_csv("train_data.csv.gz", index=False, compression="gzip")
        X, y = self.load_training_data()
        self.assertIsNotNone(X)
        self.assertEqual(len(X), 3)
        self.assertEqual(list(y), [0, 1, 2])

    def test_prefers_plain_csv_when_both_exist(self):
        self._make_df("plain").to_csv("train_data.csv", index=False)
        self._make_df("gz").to_csv("train_data.csv.gz", index=False, compression="gzip")
        X, y = self.load_training_data()
        # The plain, freshly-regenerated file (written by daily_model_retrain.py
        # earlier in the same job) must win over the committed .gz snapshot.
        self.assertTrue((pd.read_csv("train_data.csv")["marker"] == "plain").all())
        self.assertEqual(len(X), 3)

    def test_returns_none_when_neither_file_exists(self):
        X, y = self.load_training_data()
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_returns_none_on_missing_required_columns(self):
        pd.DataFrame({"feat_a": [1.0], "target": [0]}).to_csv("train_data.csv", index=False)
        X, y = self.load_training_data()
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_returns_none_on_empty_csv(self):
        pd.DataFrame({"feat_a": [], "feat_b": [], "target": []}).to_csv(
            "train_data.csv", index=False
        )
        X, y = self.load_training_data()
        self.assertIsNone(X)
        self.assertIsNone(y)


class GzipRoundTripIdenticalDataFrame(unittest.TestCase):
    """Local simulation: gzip round trip must yield a byte-for-byte identical
    DataFrame back, confirming pandas' transparent compression inference on
    the .gz extension (no format/column changes required anywhere)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_gz_round_trip_matches_plain_round_trip(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50, freq="D").astype(str),
            "ticker": [f"T{i % 7}" for i in range(50)],
            "close": [100.0 + i * 0.37 for i in range(50)],
            "target": [i % 3 for i in range(50)],
        })
        plain_path = Path(self.tmpdir.name) / "train_data.csv"
        gz_path = Path(self.tmpdir.name) / "train_data.csv.gz"

        df.to_csv(plain_path, index=False, encoding="utf-8-sig")
        with open(plain_path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
            dst.write(src.read())

        roundtrip = pd.read_csv(gz_path, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(roundtrip, df)

    def test_gz_file_is_smaller_than_plain_for_repetitive_data(self):
        df = pd.DataFrame({"a": [1] * 5000, "b": ["repeated-value"] * 5000})
        plain_path = Path(self.tmpdir.name) / "d.csv"
        gz_path = Path(self.tmpdir.name) / "d.csv.gz"
        df.to_csv(plain_path, index=False)
        with open(plain_path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
            dst.write(src.read())
        self.assertLess(gz_path.stat().st_size, plain_path.stat().st_size)


class WorkflowAndGitignoreInvariants(unittest.TestCase):
    """Static regression guard over the actual committed workflow/.gitignore
    content (no execution): catches accidental removal of the artifact
    upload, the "never commit train_data.csv[.gz]" invariant, the
    walk_forward_all_candidates.csv.gz size guard, or the push-trigger
    branch scope fix."""

    def _read(self, relpath):
        return (REPO_ROOT / relpath).read_text(encoding="utf-8")

    def _yaml(self, relpath):
        import yaml

        return yaml.safe_load(self._read(relpath))

    def test_gitignore_excludes_plain_and_gz_train_data_csv(self):
        gitignore = self._read(".gitignore")
        lines = [l.strip() for l in gitignore.splitlines()]
        self.assertIn("train_data.csv", lines)
        self.assertIn("train_data.csv.gz", lines)

    def _assert_uploads_train_data_as_artifact(self, wf_yaml, retention_days=30):
        steps = wf_yaml["jobs"][next(iter(wf_yaml["jobs"]))]["steps"]
        upload_steps = [
            s
            for s in steps
            if s.get("uses", "").startswith("actions/upload-artifact@")
            and "train_data.csv" in str(s.get("with", {}).get("path", ""))
        ]
        self.assertEqual(
            len(upload_steps), 1, "expected exactly one train_data.csv artifact upload step"
        )
        self.assertEqual(upload_steps[0]["with"]["path"], "train_data.csv")
        self.assertEqual(upload_steps[0]["with"]["retention-days"], retention_days)

    def _assert_never_commits_train_data(self, wf_text):
        # train_data.csv (plain or gz) must never be `git add`-ed anywhere:
        # only defensive `git rm --cached` of a possibly still-tracked copy.
        self.assertNotIn("git add train_data.csv", wf_text)
        self.assertIn("git rm --cached --ignore-unmatch train_data.csv train_data.csv.gz", wf_text)
        self.assertNotIn("gzip -k -9 -f train_data.csv", wf_text)

    def test_daily_retrain_workflow_uploads_artifact_and_never_commits_train_data(self):
        wf_text = self._read(".github/workflows/daily-model-retrain.yml")
        self._assert_never_commits_train_data(wf_text)
        self._assert_uploads_train_data_as_artifact(self._yaml(".github/workflows/daily-model-retrain.yml"))

    def test_refresh_candidates_workflow_uploads_artifact_and_never_commits_train_data(self):
        wf_text = self._read(".github/workflows/refresh-walk-forward-candidates.yml")
        self._assert_never_commits_train_data(wf_text)
        self._assert_uploads_train_data_as_artifact(
            self._yaml(".github/workflows/refresh-walk-forward-candidates.yml")
        )

    def test_refresh_candidates_still_commits_walk_forward_gz_with_staged_warnings(self):
        # walk_forward_all_candidates.csv.gz is unaffected by this change: it
        # stays git-tracked, and now carries staged 80/90/95MB annotations on
        # top of the pre-existing 90/97MB text warning + hard skip.
        wf = self._read(".github/workflows/refresh-walk-forward-candidates.yml")
        self.assertIn("git add walk_forward_all_candidates.csv.gz", wf)
        self.assertIn("gzip -k -9 -f walk_forward_all_candidates.csv", wf)
        self.assertIn('::warning::walk_forward_all_candidates.csv.gz size ${GZ_MB}MB >= 80MB', wf)
        self.assertIn('::warning::walk_forward_all_candidates.csv.gz size ${GZ_MB}MB >= 90MB', wf)
        self.assertIn('::error::walk_forward_all_candidates.csv.gz size ${GZ_MB}MB >= 95MB', wf)
        self.assertIn("97000000", wf)

    def test_refresh_candidates_push_trigger_is_scoped_to_main(self):
        # Regression guard for the incident this fix responds to: an
        # unscoped `push:` trigger + a hardcoded `ref: main` checkout means
        # any branch editing this file fires a real (non-dry-run) job
        # against main. See root-cause note in this workflow's `on:` block.
        wf = self._read(".github/workflows/refresh-walk-forward-candidates.yml")
        push_section = wf.split("push:", 1)[1].split("schedule:", 1)[0]
        self.assertIn("branches: [main]", push_section)

    def test_no_workflow_or_python_file_reads_a_committed_train_data_copy(self):
        # The only file that ever reads train_data.csv/.gz on a fresh
        # checkout (without regenerating it first in the same job) is
        # stock_scan.py's load_training_data(), which is never imported or
        # executed by any workflow (only py_compile'd / ast-parsed) and
        # already degrades gracefully (returns None, None) when neither
        # file exists. daily_directional_top1.py defines TRAIN_FILE but
        # never reads it in the production (load_model/main) path.
        read_pattern = re.compile(r"read_csv\([^)]*(TRAIN_FILE|train_data\.csv|train_path)")
        consumers = []
        for py_file in sorted(REPO_ROOT.glob("*.py")):
            if py_file.name == Path(__file__).name:
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if read_pattern.search(text):
                consumers.append(py_file.name)
        self.assertEqual(consumers, ["stock_scan.py"])


if __name__ == "__main__":
    unittest.main()
