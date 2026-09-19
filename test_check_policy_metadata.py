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


class ManualOverrideClassification(unittest.TestCase):
    """(a)-(e) policy_manual_overrides.json driven warning/notice split.

    Never touches the repo's real strategy_policy*.json or
    policy_manual_overrides.json; everything here runs against files
    written into a temporary directory.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def _write_policy(self, name="strategy_policy.json"):
        payload = {
            "up_threshold": 20,
            "min_score_for_buy": 40,
            "nikkei_filter": True,
            "atr_tp_multiplier": 4.0,
            "atr_sl_multiplier": 1.0,
            "hold_days": 1,
            "strategy_name": "UP20_SCORE80_NIKKEION_TP4.0_SL1.0_H1",
        }
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_overrides(self, overrides_obj):
        path = self.root / "policy_manual_overrides.json"
        if isinstance(overrides_obj, str):
            path.write_text(overrides_obj, encoding="utf-8")
        else:
            path.write_text(json.dumps(overrides_obj), encoding="utf-8")
        return path

    def _mismatches_for(self, policy_path):
        info = cpm._analyze(policy_path)
        self.assertEqual(info["status"], "mismatch")
        return info["mismatches"]

    def test_a_recorded_override_matching_both_values_is_not_a_warning(self):
        policy_path = self._write_policy()
        self._write_overrides({
            "overrides": [{
                "policy_file": "strategy_policy.json",
                "field": "min_score_for_buy",
                "validated_value": 80,
                "live_value": 40,
                "reason": "test",
                "date": "2026-09-16",
                "validated": False,
            }]
        })
        overrides = cpm.load_manual_overrides(self.root)
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(warnings, [])
        self.assertEqual(len(notices), 1)
        self.assertIn("記録済みの手動上書き", notices[0])

    def test_b_no_recorded_entry_is_a_warning(self):
        policy_path = self._write_policy()
        self._write_overrides({"overrides": []})
        overrides = cpm.load_manual_overrides(self.root)
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("min_score_for_buy", warnings[0])

    def test_c_recorded_live_value_mismatches_actual_value_is_a_warning(self):
        policy_path = self._write_policy()  # actual min_score_for_buy = 40
        self._write_overrides({
            "overrides": [{
                "policy_file": "strategy_policy.json",
                "field": "min_score_for_buy",
                "validated_value": 80,
                "live_value": 50,  # does not match actual (40)
                "reason": "test",
            }]
        })
        overrides = cpm.load_manual_overrides(self.root)
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)

    def test_d_recorded_validated_value_mismatches_name_value_is_a_warning(self):
        policy_path = self._write_policy()  # name-side SCORE80
        self._write_overrides({
            "overrides": [{
                "policy_file": "strategy_policy.json",
                "field": "min_score_for_buy",
                "validated_value": 90,  # does not match name-side (80)
                "live_value": 40,
                "reason": "test",
            }]
        })
        overrides = cpm.load_manual_overrides(self.root)
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)

    def test_e_missing_overrides_file_is_treated_as_no_record(self):
        policy_path = self._write_policy()
        # no policy_manual_overrides.json written at all
        overrides = cpm.load_manual_overrides(self.root)
        self.assertEqual(overrides, [])
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)

    def test_e_broken_json_overrides_file_is_treated_as_no_record(self):
        policy_path = self._write_policy()
        self._write_overrides('{"overrides": [ this is not valid json')
        overrides = cpm.load_manual_overrides(self.root)
        self.assertEqual(overrides, [])
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)

    def test_e_malformed_shape_overrides_file_is_treated_as_no_record(self):
        policy_path = self._write_policy()
        # "overrides" is not a list, and entries below are missing required keys.
        self._write_overrides({"overrides": {"not": "a list"}})
        overrides = cpm.load_manual_overrides(self.root)
        self.assertEqual(overrides, [])
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(notices, [])
        self.assertEqual(len(warnings), 1)

    def test_e_malformed_entries_are_skipped_not_raised(self):
        policy_path = self._write_policy()
        self._write_overrides({
            "overrides": [
                {"policy_file": "strategy_policy.json"},  # missing field/values
                "not-a-dict",
                {"policy_file": "strategy_policy.json", "field": "min_score_for_buy",
                 "validated_value": 80, "live_value": 40, "reason": "ok"},
            ]
        })
        overrides = cpm.load_manual_overrides(self.root)
        self.assertEqual(len(overrides), 1)
        mismatches = self._mismatches_for(policy_path)
        warnings, notices = cpm.classify_mismatches("strategy_policy.json", mismatches, overrides)
        self.assertEqual(warnings, [])
        self.assertEqual(len(notices), 1)

    def test_load_manual_overrides_never_writes_and_never_raises(self):
        # Directory without the file at all: still must not raise.
        empty_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(empty_dir, ignore_errors=True))
        overrides = cpm.load_manual_overrides(empty_dir)
        self.assertEqual(overrides, [])


class ManualOverrideEndToEnd(unittest.TestCase):
    """(f) main() over the real repo must still exit 0 and stay a no-op
    against the repo's actual policy files (read-only)."""

    def test_main_over_real_repo_still_exits_zero(self):
        self.assertEqual(cpm.main(), 0)

    def test_real_repo_override_file_is_well_formed_and_matches(self):
        overrides = cpm.load_manual_overrides(REPO_ROOT)
        self.assertEqual(len(overrides), 1)
        entry = overrides[0]
        self.assertEqual(entry["policy_file"], "strategy_policy.json")
        self.assertEqual(entry["field"], "min_score_for_buy")

        info = cpm._analyze(REPO_ROOT / "strategy_policy.json")
        self.assertEqual(info["status"], "mismatch")
        warnings, notices = cpm.classify_mismatches(
            "strategy_policy.json", info["mismatches"], overrides
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(notices), 1)


if __name__ == "__main__":
    unittest.main()
