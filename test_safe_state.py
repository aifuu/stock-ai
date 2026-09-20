import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

import safe_state


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="safe_state_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


class AtomicWriteJsonTests(TmpDirMixin, unittest.TestCase):
    def test_writes_and_reads_back(self):
        safe_state.atomic_write_json("s.json", {"a": 1})
        with open("s.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})

    def test_no_bak_on_first_write(self):
        safe_state.atomic_write_json("s.json", {"a": 1})
        self.assertFalse(os.path.exists("s.json.bak"))

    def test_bak_holds_previous_generation(self):
        safe_state.atomic_write_json("s.json", {"a": 1})
        safe_state.atomic_write_json("s.json", {"a": 2})
        with open("s.json.bak", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})
        with open("s.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 2})

    def test_no_tmp_file_left_after_success(self):
        safe_state.atomic_write_json("s.json", {"a": 1})
        self.assertFalse(os.path.exists("s.json.tmp"))

    def test_interrupted_replace_leaves_target_untouched(self):
        safe_state.atomic_write_json("s.json", {"a": 1})
        real_replace = os.replace

        def boom(src, dst):
            raise OSError("simulated interruption during os.replace")

        os.replace = boom
        try:
            with self.assertRaises(OSError):
                safe_state.atomic_write_json("s.json", {"a": 2})
        finally:
            os.replace = real_replace
        # rename never happened: target file is exactly the old good content,
        # never partially written / never truncated.
        with open("s.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})
        # the new content sits only in the (uncommitted) tmp file.
        with open("s.json.tmp", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 2})

    def test_ensure_ascii_false_indent_2_matches_prior_format(self):
        safe_state.atomic_write_json("s.json", {"a": "日本語"})
        with open("s.json", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("日本語", text)
        self.assertIn("\n", text)  # indent=2 produces multi-line output


class LoadJsonStateTests(TmpDirMixin, unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(safe_state.load_json_state("nope.json"))

    def test_valid_file_returns_data(self):
        with open("s.json", "w", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        self.assertEqual(safe_state.load_json_state("s.json"), {"a": 1})

    def test_corrupt_json_falls_back_to_bak(self):
        with open("s.json.bak", "w", encoding="utf-8") as f:
            json.dump({"a": "good"}, f)
        with open("s.json", "w", encoding="utf-8") as f:
            f.write("{not valid json")
        notified = []
        result = safe_state.load_json_state("s.json", notify=notified.append, label="s.json")
        self.assertEqual(result, {"a": "good"})
        self.assertEqual(len(notified), 1)
        self.assertIn("bak", notified[0].lower())

    def test_truncated_json_falls_back_to_bak(self):
        with open("s.json.bak", "w", encoding="utf-8") as f:
            json.dump({"a": "good"}, f)
        with open("s.json", "w", encoding="utf-8") as f:
            f.write('{"a": "trun')
        self.assertEqual(safe_state.load_json_state("s.json"), {"a": "good"})

    def test_empty_file_falls_back_to_bak(self):
        with open("s.json.bak", "w", encoding="utf-8") as f:
            json.dump({"a": "good"}, f)
        open("s.json", "w").close()
        self.assertEqual(safe_state.load_json_state("s.json"), {"a": "good"})

    def test_invalid_utf8_falls_back_to_bak(self):
        with open("s.json.bak", "w", encoding="utf-8") as f:
            json.dump({"a": "good"}, f)
        with open("s.json", "wb") as f:
            f.write(b"\xff\xfe\x00\x01not utf8")
        self.assertEqual(safe_state.load_json_state("s.json"), {"a": "good"})

    def test_both_corrupt_raises_and_quarantines(self):
        with open("s.json", "w", encoding="utf-8") as f:
            f.write("{not valid")
        with open("s.json.bak", "w", encoding="utf-8") as f:
            f.write("{also not valid")
        notified = []
        with self.assertRaises(safe_state.StateCorruptError):
            safe_state.load_json_state("s.json", notify=notified.append, label="s.json")
        self.assertEqual(len(notified), 1)
        quarantines = [f for f in os.listdir(".") if f.startswith("s.json.corrupt-")]
        self.assertEqual(len(quarantines), 1)
        with open(quarantines[0], "w" if False else "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not valid")
        # originals are left exactly as they were -- never overwritten.
        with open("s.json", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not valid")
        with open("s.json.bak", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{also not valid")

    def test_no_bak_and_corrupt_raises_and_quarantines(self):
        with open("s.json", "w", encoding="utf-8") as f:
            f.write("{not valid")
        notified = []
        with self.assertRaises(safe_state.StateCorruptError):
            safe_state.load_json_state("s.json", notify=notified.append, label="s.json")
        self.assertEqual(len(notified), 1)
        quarantines = [f for f in os.listdir(".") if f.startswith("s.json.corrupt-")]
        self.assertEqual(len(quarantines), 1)

    def test_notify_failure_never_propagates(self):
        with open("s.json", "w", encoding="utf-8") as f:
            f.write("{not valid")

        def broken_notify(_msg):
            raise RuntimeError("discord is down")

        with self.assertRaises(safe_state.StateCorruptError):
            safe_state.load_json_state("s.json", notify=broken_notify, label="s.json")

    def test_wrong_type_treated_as_corrupt_with_validate(self):
        with open("s.json", "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)
        with self.assertRaises(safe_state.StateCorruptError):
            safe_state.load_json_state("s.json", validate=lambda d: isinstance(d, dict))

    def test_wrong_type_recovers_from_valid_bak(self):
        with open("s.json.bak", "w", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        with open("s.json", "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)
        result = safe_state.load_json_state("s.json", validate=lambda d: isinstance(d, dict))
        self.assertEqual(result, {"a": 1})


class SafeAppendHistoryTests(TmpDirMixin, unittest.TestCase):
    def test_creates_new_file(self):
        safe_state.safe_append_history("h.csv", {"a": 1, "b": 2})
        df = pd.read_csv("h.csv")
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(len(df), 1)

    def test_appends_to_existing_file_same_columns_and_order(self):
        safe_state.safe_append_history("h.csv", {"a": 1, "b": 2})
        safe_state.safe_append_history("h.csv", {"a": 3, "b": 4})
        df = pd.read_csv("h.csv")
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(df["a"].tolist(), [1, 3])

    def test_healthy_output_matches_legacy_concat_to_csv(self):
        row1, row2 = {"a": 1, "b": "x"}, {"a": 2, "b": "y"}
        safe_state.safe_append_history("h.csv", row1)
        safe_state.safe_append_history("h.csv", row2)
        with open("h.csv", "rb") as f:
            got = f.read()

        # legacy behavior being replaced: try: concat(read_csv(old), new)
        # except: pass; then unconditionally to_csv(...).
        df = pd.DataFrame([row1])
        df.to_csv("legacy.csv", index=False, encoding="utf-8-sig")
        df = pd.concat([pd.read_csv("legacy.csv"), pd.DataFrame([row2])], ignore_index=True)
        df.to_csv("legacy.csv", index=False, encoding="utf-8-sig")
        with open("legacy.csv", "rb") as f:
            want = f.read()
        self.assertEqual(got, want)

    def test_corrupt_existing_file_is_not_overwritten(self):
        with open("h.csv", "w", encoding="utf-8") as f:
            f.write("this,is,not\nvalid,csv,\"unterminated")
        with open("h.csv", "rb") as f:
            original = f.read()
        notified = []
        safe_state.safe_append_history("h.csv", {"a": 1}, notify=notified.append, label="h.csv")
        # original left untouched
        with open("h.csv", "rb") as f:
            self.assertEqual(f.read(), original)
        self.assertEqual(len(notified), 1)
        quarantines = [f for f in os.listdir(".") if f.startswith("h.csv.corrupt-")]
        self.assertEqual(len(quarantines), 1)
        recovery = pd.read_csv("h.recovery.csv")
        self.assertEqual(recovery["a"].tolist(), [1])

    def test_recovery_file_appends_without_duplicating_header(self):
        with open("h.csv", "w", encoding="utf-8") as f:
            f.write("this,is,not\nvalid,csv,\"unterminated")
        safe_state.safe_append_history("h.csv", {"a": 1})
        safe_state.safe_append_history("h.csv", {"a": 2})
        with open("h.recovery.csv", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, ["a", "1", "2"])

    def test_empty_existing_file_is_treated_as_corrupt_not_overwritten(self):
        open("h.csv", "w").close()
        with open("h.csv", "rb") as f:
            original = f.read()
        safe_state.safe_append_history("h.csv", {"a": 1})
        with open("h.csv", "rb") as f:
            self.assertEqual(f.read(), original)
        self.assertTrue(os.path.exists("h.recovery.csv"))


if __name__ == "__main__":
    unittest.main()
