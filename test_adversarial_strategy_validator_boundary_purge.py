"""adversarial_strategy_validator.pyのフェーズ境界(OOS会計の境界)purgeに対する単体テスト。

adversarial_strategy_validator.pyはモジュール直下で候補CSVの読み込み・
yfinanceでの価格ダウンロード・グリッドサーチ・Discord送信までを一気に実行する
スクリプトであり、importするだけで実行されてしまう(かつネットワークアクセスも
発生する)。そのため、このテストは一切importせず、build_strategy_policyの
既存テスト(test_build_strategy_policy_retention.py)と同じ方法で、ASTから
対象の純粋関数の定義だけを取り出してexecし、その関数だけを検証する。
候補CSV・価格データ・Discord送信・グリッドサーチ本体には一切触れない。

対象関数:
- purge_embargo: 既存のpurge/embargo境界計算(変更なし、回帰確認用)。
- _apply_phase_boundary_purge: 今回追加した、決済日(exit_date)がフェーズ終端
  (phase_end)を越える取引を除外(purge)する新関数。
"""
import ast
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "adversarial_strategy_validator.py"
FUNCTION_NAMES = ("purge_embargo", "_apply_phase_boundary_purge")


def _load_pure_functions():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE_PATH))
    wanted = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
    ]
    found_names = {node.name for node in wanted}
    missing = set(FUNCTION_NAMES) - found_names
    assert not missing, f"adversarial_strategy_validator.pyに関数が見つかりません: {missing}"

    module_ast = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace = {}
    exec(compile(module_ast, filename=str(MODULE_PATH), mode="exec"), namespace)
    return namespace["purge_embargo"], namespace["_apply_phase_boundary_purge"]


purge_embargo, _apply_phase_boundary_purge = _load_pure_functions()


def _ts(s):
    return pd.Timestamp(s)


def _row(entry, exit_, ticker="7203.T"):
    return {
        "date": exit_,
        "entry_date": _ts(entry),
        "exit_date": _ts(exit_),
        "ticker": ticker,
        "result": "WIN",
        "return": 1.0,
    }


class PurgeEmbargoRegressionTests(unittest.TestCase):
    # purge_embargo自体は今回変更していないため、既存挙動の回帰確認のみ行う。
    def test_purges_tail_and_embargoes_head(self):
        before = [_ts(f"2026-01-{d:02d}") for d in range(1, 11)]
        after = [_ts(f"2026-02-{d:02d}") for d in range(1, 11)]
        new_before, new_after = purge_embargo(before, after, purge_days=3, embargo_days=2)
        self.assertEqual(new_before, before[:-3])
        self.assertEqual(new_after, after[2:])

    def test_noop_when_days_zero(self):
        before = [_ts("2026-01-01"), _ts("2026-01-02")]
        after = [_ts("2026-02-01"), _ts("2026-02-02")]
        new_before, new_after = purge_embargo(before, after, purge_days=0, embargo_days=0)
        self.assertEqual(new_before, before)
        self.assertEqual(new_after, after)


class ApplyPhaseBoundaryPurgeTests(unittest.TestCase):
    def setUp(self):
        self.phase_end = _ts("2026-03-31")

    def test_boundary_trade_exiting_after_phase_end_is_purged(self):
        # entry_dateはフェーズ内(phase_end当日)だが、hold_days分の保有で
        # 決済(exit_date)がphase_endを越えるケース。
        rows = [_row("2026-03-31", "2026-04-02")]
        out = _apply_phase_boundary_purge(rows, self.phase_end, legacy_overlap=False)
        self.assertEqual(out, [])

    def test_in_boundary_trade_is_unchanged(self):
        # 決済がphase_end以前(当日含む)の取引はそのまま残る。
        r_before = _row("2026-03-20", "2026-03-25")
        r_on_boundary = _row("2026-03-30", "2026-03-31")
        rows = [r_before, r_on_boundary]
        out = _apply_phase_boundary_purge(rows, self.phase_end, legacy_overlap=False)
        self.assertEqual(out, rows)

    def test_mixed_rows_only_boundary_crossers_are_removed(self):
        keep1 = _row("2026-03-10", "2026-03-12", ticker="A")
        drop = _row("2026-03-29", "2026-04-01", ticker="B")
        keep2 = _row("2026-03-31", "2026-03-31", ticker="C")
        out = _apply_phase_boundary_purge([keep1, drop, keep2], self.phase_end, legacy_overlap=False)
        self.assertEqual(out, [keep1, keep2])

    def test_legacy_overlap_is_never_purged_even_past_boundary(self):
        # WF_LEGACY_OVERLAP=1(legacy_overlap=True)の挙動は変更しない。
        rows = [_row("2026-03-31", "2026-04-10")]
        out = _apply_phase_boundary_purge(rows, self.phase_end, legacy_overlap=True)
        self.assertEqual(out, rows)

    def test_phase_end_none_is_noop_for_backward_compat(self):
        rows = [_row("2026-03-31", "2026-04-10")]
        out = _apply_phase_boundary_purge(rows, None, legacy_overlap=False)
        self.assertEqual(out, rows)

    def test_empty_rows_returns_empty(self):
        self.assertEqual(_apply_phase_boundary_purge([], self.phase_end, legacy_overlap=False), [])


if __name__ == "__main__":
    unittest.main()
