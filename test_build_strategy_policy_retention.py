"""build_strategy_policy.pyの「現行戦略の最低保持日数判定」に対する単体テスト。

build_strategy_policy.pyはモジュール直下でCSV読み込み・署名生成・
strategy_policy.json書き込みまで一気に実行するスクリプトであり、
importするだけで実行されてしまう(AI_POLICY_SIGNING_SECRET未設定なら
即SystemExitするが、設定されていればテストファイルであっても実際に
署名付きpolicyを生成しかねない)。そのため、このテストはAI_POLICY_SIGNING_SECRET
を一切設定せず、build_strategy_policy.pyをモジュールとしてimportもしない。
ASTでbuild_effective_strategy_name()とevaluate_min_hold_retention()の2つの
純粋関数の定義だけを取り出してexecし、その関数だけを検証する。
署名(canonical_policy_payload/policy_signature/POLICY_SIGNING_SECRET)には
一切触れない。
"""
import ast
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "build_strategy_policy.py"
FUNCTION_NAMES = ("build_effective_strategy_name", "evaluate_min_hold_retention")


def _load_pure_functions():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE_PATH))
    wanted = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
    ]
    found_names = {node.name for node in wanted}
    missing = set(FUNCTION_NAMES) - found_names
    assert not missing, f"build_strategy_policy.pyに関数が見つかりません: {missing}"

    module_ast = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace = {"datetime": datetime}
    exec(compile(module_ast, filename=str(MODULE_PATH), mode="exec"), namespace)
    return namespace["build_effective_strategy_name"], namespace["evaluate_min_hold_retention"]


build_effective_strategy_name, evaluate_min_hold_retention = _load_pure_functions()


# 現行のstrategy_policy.json相当(名前はSCORE80、実際の値はSCORE40)。
# ファイル自体は読み込まず、値をそのままここに書き写す(policy_manual_overrides.jsonに
# validated_value=80/live_value=40/validated=falseとして記録済みの状態と対応)。
LIVE_POLICY_NAME_VS_VALUE_MISMATCH = {
    "status": "APPROVED",
    "updated_at": "2026-09-16T06:39:35",
    "up_threshold": 20,
    "min_score_for_buy": 40,
    "nikkei_filter": True,
    "atr_tp_multiplier": 4.0,
    "atr_sl_multiplier": 1.0,
    "hold_days": 1,
    "strategy_name": "UP20_SCORE80_NIKKEION_TP4.0_SL1.0_H1",
}

# strategy_policy_up.json相当(名前と実際の値が一致している正常ケース)。
CONSISTENT_POLICY = {
    "status": "APPROVED",
    "updated_at": "2026-09-16T08:12:55",
    "up_threshold": 20,
    "min_score_for_buy": 70,
    "nikkei_filter": False,
    "atr_tp_multiplier": 3.0,
    "atr_sl_multiplier": 1.0,
    "hold_days": 3,
    "strategy_name": "UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3",
}


class BuildEffectiveStrategyNameTests(unittest.TestCase):
    def test_matches_validator_format(self):
        name = build_effective_strategy_name(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        self.assertEqual(name, "UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1")

    def test_consistent_policy_matches_recorded_name(self):
        name = build_effective_strategy_name(CONSISTENT_POLICY)
        self.assertEqual(name, CONSISTENT_POLICY["strategy_name"])

    def test_missing_field_returns_none(self):
        broken = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        del broken["atr_tp_multiplier"]
        self.assertIsNone(build_effective_strategy_name(broken))

    def test_type_invalid_field_returns_none(self):
        broken = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        broken["atr_tp_multiplier"] = "not-a-number"
        self.assertIsNone(build_effective_strategy_name(broken))

    def test_numeric_formatting_variance(self):
        # up_thresholdがfloat、atr_tp_multiplierがint相当(4)で保存されていても、
        # 候補側の書式(整数UP/SCORE/HOLD、小数TP/SL "x.0")に正規化されること。
        variant = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        variant["up_threshold"] = 20.0
        variant["atr_tp_multiplier"] = 4
        variant["atr_sl_multiplier"] = 1
        variant["hold_days"] = 1.0
        name = build_effective_strategy_name(variant)
        self.assertEqual(name, "UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1")

    def test_matches_actual_candidate_csv_format(self):
        # (g) 実際の候補CSVの名前の書式と、組み立てた実効名の書式が一致することを実データで確認。
        csv_path = REPO_ROOT / "adversarial_final_candidates.csv"
        df = pd.read_csv(csv_path, nrows=50)
        self.assertGreater(len(df), 0, "adversarial_final_candidates.csvが空です")
        for _, row in df.iterrows():
            policy = {
                "up_threshold": row["up"],
                "min_score_for_buy": row["score"],
                "nikkei_filter": bool(row["nikkei"]),
                "atr_tp_multiplier": row["tp"],
                "atr_sl_multiplier": row["sl"],
                "hold_days": row["hold"],
            }
            built = build_effective_strategy_name(policy)
            self.assertEqual(
                built, row["strategy"],
                f"実効名の組み立てが候補CSVの書式と一致しません: {built!r} != {row['strategy']!r}",
            )


class EvaluateMinHoldRetentionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.fromisoformat("2026-09-19T00:00:00")

    def test_a_score40_candidate_protects_live_policy_despite_score80_name(self):
        approved = pd.Series(["UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1", "UP30_SCORE80_NIKKEION_TP4.0_SL1.0_H1"])
        result = evaluate_min_hold_retention(
            approved, LIVE_POLICY_NAME_VS_VALUE_MISMATCH, self.now, 30,
            policy_file_name="strategy_policy.json",
            manual_overrides=[{
                "policy_file": "strategy_policy.json",
                "field": "min_score_for_buy",
                "validated_value": 80,
                "live_value": 40,
            }],
        )
        self.assertTrue(result["keep"])
        self.assertIn("UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1", result["reason"])
        self.assertTrue(any("食い違って" in line for line in result["log_lines"]))
        self.assertTrue(any("policy_manual_overrides.json" in line for line in result["log_lines"]))

    def test_b_score80_only_candidate_does_not_falsely_protect(self):
        approved = pd.Series(["UP30_SCORE80_NIKKEION_TP4.0_SL1.0_H1", "UP25_SCORE80_NIKKEION_TP4.0_SL1.0_H1"])
        result = evaluate_min_hold_retention(
            approved, LIVE_POLICY_NAME_VS_VALUE_MISMATCH, self.now, 30,
            policy_file_name="strategy_policy.json",
        )
        self.assertFalse(result["keep"])
        self.assertIsNone(result["reason"])

    def test_c_consistent_policy_matches_legacy_name_based_behavior(self):
        # 名前と実際の値が一致するケースでは、実効名==strategy_nameとなるため
        # 旧ロジック(strategy_name直接比較)と完全に同じ判定になるはず。
        approved = pd.Series(["UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3", "UP99_SCOREXX"])
        legacy_still_qualifies = bool(
            (approved == CONSISTENT_POLICY["strategy_name"]).any()
        )
        result = evaluate_min_hold_retention(
            approved, CONSISTENT_POLICY, self.now, 30,
            policy_file_name="strategy_policy_up.json",
        )
        self.assertEqual(result["keep"], legacy_still_qualifies)
        self.assertTrue(result["keep"])
        self.assertEqual(result["log_lines"], [])  # 食い違いがないのでログなし

        approved_no_match = pd.Series(["UP99_SCOREXX"])
        legacy_no_match = bool((approved_no_match == CONSISTENT_POLICY["strategy_name"]).any())
        result_no_match = evaluate_min_hold_retention(
            approved_no_match, CONSISTENT_POLICY, self.now, 30,
            policy_file_name="strategy_policy_up.json",
        )
        self.assertEqual(result_no_match["keep"], legacy_no_match)
        self.assertFalse(result_no_match["keep"])

    def test_d_unbuildable_effective_name_falls_back_to_strategy_name_without_raising(self):
        broken = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        del broken["hold_days"]
        approved = pd.Series([broken["strategy_name"]])
        result = evaluate_min_hold_retention(
            approved, broken, self.now, 30, policy_file_name="strategy_policy.json",
        )
        self.assertTrue(result["keep"])
        self.assertIn(broken["strategy_name"], result["reason"])
        self.assertTrue(any("組み立てられなかった" in line for line in result["log_lines"]))

        # フィールドの型が不正な場合も例外を送出しない。
        broken_type = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        broken_type["nikkei_filter"] = object()  # bool()できてしまうため atr 側を壊す
        broken_type["atr_sl_multiplier"] = {"not": "a number"}
        try:
            result2 = evaluate_min_hold_retention(
                approved, broken_type, self.now, 30, policy_file_name="strategy_policy.json",
            )
        except Exception as exc:  # pragma: no cover - テストの主張そのもの
            self.fail(f"型不正な入力で例外が送出されました: {exc}")
        self.assertIsInstance(result2["keep"], bool)

    def test_e_past_min_hold_days_is_not_protected(self):
        old_policy = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        old_policy["updated_at"] = (self.now - timedelta(days=31)).isoformat(timespec="seconds")
        approved = pd.Series(["UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1"])
        result = evaluate_min_hold_retention(
            approved, old_policy, self.now, 30, policy_file_name="strategy_policy.json",
        )
        self.assertFalse(result["keep"])
        self.assertIsNone(result["reason"])

    def test_e_not_approved_status_is_not_protected(self):
        default_policy = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        default_policy["status"] = "DEFAULT"
        approved = pd.Series(["UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1"])
        result = evaluate_min_hold_retention(
            approved, default_policy, self.now, 30, policy_file_name="strategy_policy.json",
        )
        self.assertFalse(result["keep"])

    def test_f_numeric_formatting_variance_still_matches_candidate(self):
        variant = dict(LIVE_POLICY_NAME_VS_VALUE_MISMATCH)
        variant["atr_tp_multiplier"] = 4  # int instead of 4.0
        variant["up_threshold"] = 20.0  # float instead of int
        approved = pd.Series(["UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1"])
        result = evaluate_min_hold_retention(
            approved, variant, self.now, 30, policy_file_name="strategy_policy.json",
        )
        self.assertTrue(result["keep"])


if __name__ == "__main__":
    unittest.main()
