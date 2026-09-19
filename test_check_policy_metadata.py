#!/usr/bin/env python3
"""Unit tests for check_policy_metadata.py.

Read-only tool: these tests never modify strategy_policy.json /
strategy_policy_up.json / strategy_policy_down.json in the repo, and never
touch approval_signature.
"""
import json
import tempfile
import unittest
from pathlib import Path

import check_policy_metadata as cpm

REPO_ROOT = Path(__file__).resolve().parent


class RealRepoFiles(unittest.TestCase):
    """(a)(b) Check the tool against the repository's own current policy files."""

    def test_main_policy_detects_score_mismatch(self):
        status, warnings = cpm.check_file(REPO_ROOT / "strategy_policy.json")
        self.assertEqual(status, "mismatch")
        joined = " ".join(warnings)
        self.assertIn("min_score_for_buy", joined)
        self.assertIn("80", joined)
        self.assertIn("40", joined)

    def test_up_policy_current_state(self):
        path = REPO_ROOT / "strategy_policy_up.json"
        if not path.exists():
            self.skipTest("strategy_policy_up.json not present")
        status, warnings = cpm.check_file(path)
        # Documents current reality rather than assuming it; current policy
        # (up_threshold=20, min_score_for_buy=70, nikkei_filter=false,
        # atr_tp_multiplier=3.0, atr_sl_multiplier=1.0, hold_days=3) matches
        # strategy_name "UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3".
        self.assertEqual(status, "ok", msg=f"warnings={warnings}")
        self.assertEqual(warnings, [])


class MatchingPolicy(unittest.TestCase):
    """(c) A self-consistent policy file must be reported as no-mismatch."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write(self, name, payload):
        path = Path(self.tmpdir.name) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_consistent_policy_is_ok(self):
        payload = {
            "status": "APPROVED",
            "up_threshold": 20,
            "min_score_for_buy": 80,
            "nikkei_filter": True,
            "atr_tp_multiplier": 4.0,
            "atr_sl_multiplier": 1.0,
            "hold_days": 1,
            "strategy_name": "UP20_SCORE80_NIKKEION_TP4.0_SL1.0_H1",
        }
        path = self._write("consistent_policy.json", payload)
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "ok")
        self.assertEqual(warnings, [])

    def test_numeric_formatting_variance_is_absorbed(self):
        # "4.0" in the name vs plain int 4 in the field must NOT be flagged.
        payload = {
            "up_threshold": 20,
            "min_score_for_buy": 80,
            "nikkei_filter": False,
            "atr_tp_multiplier": 4,
            "atr_sl_multiplier": 1,
            "hold_days": 1,
            "strategy_name": "UP20_SCORE80_NIKKEIOFF_TP4.0_SL1.0_H1",
        }
        path = self._write("formatting_variance_policy.json", payload)
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "ok")
        self.assertEqual(warnings, [])


class RobustnessAgainstBadInput(unittest.TestCase):
    """(d) Broken JSON / unparsable name / missing file must never raise."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_missing_file_does_not_raise(self):
        path = Path(self.tmpdir.name) / "does_not_exist.json"
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "missing")
        self.assertEqual(warnings, [])

    def test_broken_json_does_not_raise(self):
        path = Path(self.tmpdir.name) / "broken.json"
        path.write_text('{"strategy_name": "UP20_SCORE80_NIKKEION_TP4.0_SL1.0_H1", garbage!!', encoding="utf-8")
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "invalid_json")
        self.assertEqual(len(warnings), 1)

    def test_unparsable_strategy_name_does_not_raise(self):
        path = Path(self.tmpdir.name) / "weird_name.json"
        path.write_text(json.dumps({"strategy_name": "TOTALLY_CUSTOM_NAME"}), encoding="utf-8")
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "unknown_format")
        self.assertEqual(len(warnings), 1)
        self.assertIn("形式不明", warnings[0])

    def test_missing_strategy_name_does_not_raise(self):
        path = Path(self.tmpdir.name) / "no_name.json"
        path.write_text(json.dumps({"up_threshold": 20}), encoding="utf-8")
        status, warnings = cpm.check_file(path)
        self.assertEqual(status, "no_name")
        self.assertEqual(len(warnings), 1)

    def test_main_entrypoint_never_raises_and_returns_zero(self):
        # Full main() over the real repo files must also exit 0 regardless
        # of whether it finds mismatches.
        exit_code = cpm.main()
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
