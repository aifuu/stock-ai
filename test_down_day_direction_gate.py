"""A案(下落日は新規エントリーを空売りのみに固定する方向ゲート)の単体テスト。

背景(2026-10-09の事象): daily_decision.jsonのtrendは'down'だったが、日経レジームは
強気(kairi25+2.62%/ret5+1.06%)だったため、run_profit_loop._market_regime()の
従来ロジック(強気→買いのみ)に従って両トラック(live/daytrade)とも買い、下落日に
買って損失を出した。

このテストが確認する仕様(run_profit_loop.py参照):
  - 当日のdaily_decision.jsonのtrendが'down'の日は、日経レジームに関わらず新規
    エントリーを空売りのみに固定する(effective direction gate)。
  - SHORT候補が無い日は新規エントリーなし(買いへのフォールバックは絶対にしない)。
  - UP日(down_day=False)の挙動はフィルター・スコアリングとも一切変更しない
    (origin/mainの実装と比較してbyte-for-byteに近い一致を確認)。
  - daily_decision.jsonが当日分無し/壊れている場合はネットワークに依存せず既存の
    地合いロジックへフォールバックする(クラッシュしない)。
  - 急落ブレーキ(新規買いのみ停止)のロジック自体は変更していない。

テストは全てchdirした一時ディレクトリ内で実行し、実repoのdaily_decision.json
(cwd相対)を一切読まない。
"""
import json
import os
import subprocess
import tempfile
import types
import unittest
from datetime import datetime
from unittest.mock import patch

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402

requests.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: requests.post"))

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: yfinance.download"))

import daily_decision as dd  # noqa: E402
import run_profit_loop as loop  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))


class TmpCwdMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()


def _write_decision(trend, date="2026-10-09", path=None):
    d = {"schema_version": 1, "date": date, "trend": trend}
    with open(path or dd.DECISION_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f)


def _cand(ticker, direction, price=3000.0, score=80.0, up=60.0, down=10.0):
    return {
        "ticker": ticker, "direction": direction, "price": price,
        "tp": price * (1.02 if direction == "BUY" else 0.98),
        "sl": price * (0.99 if direction == "BUY" else 1.01),
        "score": score, "up_probability": up, "down_probability": down,
        "flat_probability": max(0.0, 100.0 - up - down),
    }


# =========================================================================
# daily_decision.todays_trend(): 読み取り専用ヘルパー(ネットワーク無し)
# =========================================================================

class TodaysTrendHelper(TmpCwdMixin, unittest.TestCase):
    def test_returns_trend_for_todays_file(self):
        _write_decision("down", date="2026-10-09")
        with patch.object(dd, "_now", return_value=datetime(2026, 10, 9)):
            self.assertEqual(dd.todays_trend(), "down")

    def test_returns_none_when_file_missing(self):
        self.assertIsNone(dd.todays_trend())

    def test_returns_none_when_date_does_not_match_today(self):
        _write_decision("down", date="2020-01-01")
        with patch.object(dd, "_now", return_value=datetime(2026, 10, 9)):
            self.assertIsNone(dd.todays_trend())

    def test_returns_none_without_crashing_on_corrupt_json(self):
        with open(dd.DECISION_FILE, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(dd.todays_trend())


# =========================================================================
# run_profit_loop._down_day_decision(): daily_decisionを読んでdown_dayを判定
# =========================================================================

class DownDayDecisionHelper(TmpCwdMixin, unittest.TestCase):
    def test_true_on_down_trend(self):
        with patch.object(dd, "todays_trend", return_value="down"):
            self.assertTrue(loop._down_day_decision())

    def test_false_on_up_trend(self):
        with patch.object(dd, "todays_trend", return_value="up"):
            self.assertFalse(loop._down_day_decision())

    def test_false_when_missing_no_crash(self):
        with patch.object(dd, "todays_trend", return_value=None):
            self.assertFalse(loop._down_day_decision())

    def test_false_when_todays_trend_raises_no_crash(self):
        with patch.object(dd, "todays_trend", side_effect=RuntimeError("boom")):
            self.assertFalse(loop._down_day_decision())


# =========================================================================
# profit_priority(): down_day gate across regimes, and no-BUY-fallback
# =========================================================================

class ProfitPriorityDownDayGate(unittest.TestCase):
    def _pool(self):
        return [
            _cand("7203.T", "BUY", score=95.0, up=80.0, down=5.0),
            _cand("8035.T", "SHORT", score=90.0, up=5.0, down=80.0),
            _cand("9984.T", "BUY", score=85.0, up=75.0, down=5.0),
            _cand("6758.T", "SHORT", score=80.0, up=5.0, down=75.0),
        ]

    def _ranked(self, regime, down_day):
        with patch.object(loop, "_market_regime", return_value=(regime, 1.0, 1.0)), \
             patch.object(loop, "_down_day_decision", return_value=down_day), \
             patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
            return loop.profit_priority(self._pool())

    def test_down_day_bullish_regime_short_only(self):
        ranked = self._ranked("bullish", True)
        self.assertTrue(ranked)
        self.assertTrue(all(c["direction"] == "SHORT" for c in ranked))
        self.assertEqual(ranked[0]["ticker"], "8035.T")  # 高scoreのSHORT
        for c in ranked:
            self.assertEqual(c["market_regime_raw"], "bullish")
            self.assertEqual(c["direction_gate"], "short_only_down_day")

    def test_down_day_bearish_regime_short_only(self):
        ranked = self._ranked("bearish", True)
        self.assertTrue(all(c["direction"] == "SHORT" for c in ranked))
        for c in ranked:
            self.assertEqual(c["direction_gate"], "short_only_down_day")

    def test_down_day_neutral_regime_short_only(self):
        ranked = self._ranked("neutral", True)
        self.assertTrue(all(c["direction"] == "SHORT" for c in ranked))
        for c in ranked:
            self.assertEqual(c["direction_gate"], "short_only_down_day")

    def test_down_day_with_zero_short_candidates_returns_empty_never_buy(self):
        pool = [_cand("7203.T", "BUY", score=95.0, up=80.0, down=5.0),
                _cand("9984.T", "BUY", score=85.0, up=75.0, down=5.0)]
        with patch.object(loop, "_market_regime", return_value=("bullish", 1.0, 1.0)), \
             patch.object(loop, "_down_day_decision", return_value=True), \
             patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
            ranked = loop.profit_priority(pool)
        self.assertEqual(ranked, [])

    def test_up_day_bullish_regime_unaffected(self):
        ranked = self._ranked("bullish", False)
        self.assertTrue(all(c["direction"] == "BUY" for c in ranked))
        for c in ranked:
            self.assertEqual(c["direction_gate"], "regime_bullish")

    def test_up_day_bearish_regime_unaffected(self):
        ranked = self._ranked("bearish", False)
        self.assertTrue(all(c["direction"] == "SHORT" for c in ranked))
        for c in ranked:
            self.assertEqual(c["direction_gate"], "regime_bearish")

    def test_up_day_neutral_regime_unaffected_both_directions_present(self):
        ranked = self._ranked("neutral", False)
        self.assertEqual({c["direction"] for c in ranked}, {"BUY", "SHORT"})
        for c in ranked:
            self.assertEqual(c["direction_gate"], "regime_neutral")


# =========================================================================
# UP日の挙動がorigin/main(このブランチの変更前)と一致することの確認。
# origin/mainのrun_profit_loop.pyをgit show経由で読み込み、末尾の
# app.scan=... 等のモジュール副作用(本物のprofit_top10_paper.scanを書き換えて
# しまう)だけを切り落として実行し、profit_priority()を直接比較する。
# =========================================================================

def _load_origin_main_profit_priority():
    src = subprocess.run(
        ["git", "show", "origin/main:run_profit_loop.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    cut = src.index("_original_scan = app.scan")
    src = src[:cut]
    mod = types.ModuleType("run_profit_loop_origin_main_reference")
    exec(compile(src, "run_profit_loop_origin_main.py", "exec"), mod.__dict__)
    return mod


class UpDayByteIdenticalToOriginMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.origin_mod = _load_origin_main_profit_priority()
        except Exception as exc:  # pragma: no cover - environment without git history
            cls.origin_mod = None
            cls.skip_reason = str(exc)

    def _pool(self):
        return [
            _cand("7203.T", "BUY", score=95.0, up=80.0, down=5.0),
            _cand("8035.T", "SHORT", score=90.0, up=5.0, down=80.0),
            _cand("9984.T", "BUY", score=85.0, up=75.0, down=5.0),
            _cand("6758.T", "SHORT", score=80.0, up=5.0, down=75.0),
            _cand("1234.T", "BUY", score=55.0, up=42.0, down=30.0),
        ]

    def _compare(self, regime):
        if self.origin_mod is None:
            self.skipTest(f"origin/main reference unavailable: {self.skip_reason}")
        pool = self._pool()
        with patch.object(loop, "_market_regime", return_value=(regime, 1.0, 1.0)), \
             patch.object(loop, "_down_day_decision", return_value=False), \
             patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
            new_ranked = loop.profit_priority(pool)
        with patch.object(self.origin_mod, "_market_regime", return_value=(regime, 1.0, 1.0)), \
             patch.object(self.origin_mod, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
            old_ranked = self.origin_mod.profit_priority(pool)
        new_keys = [k for k in (new_ranked[0].keys() if new_ranked else [])
                    if k not in ("market_regime_raw", "direction_gate")]
        self.assertEqual(len(new_ranked), len(old_ranked))
        for new_item, old_item in zip(new_ranked, old_ranked):
            for key in new_keys:
                self.assertEqual(new_item.get(key), old_item.get(key), key)

    def test_bullish(self):
        self._compare("bullish")

    def test_bearish(self):
        self._compare("bearish")

    def test_neutral(self):
        self._compare("neutral")


# =========================================================================
# open_top1_only(): down_dayゲートとの一貫性、及びブレーキ(買いのみ停止)との共存
# =========================================================================

class OpenTop1OnlyDownDayGate(unittest.TestCase):
    def _pool(self):
        return [
            _cand("7203.T", "BUY", score=95.0, up=80.0, down=5.0),
            _cand("8035.T", "SHORT", score=90.0, up=5.0, down=80.0),
        ]

    def _open(self, regime, down_day, candidates=None):
        opened_cands = []

        def fake_open(state, policy, cands, today):
            state["positions"].append(dict(cands[0]))
            opened_cands.append(cands[0])
            return list(cands)

        orig_open = loop._original_open
        loop._original_open = fake_open
        try:
            with patch.object(loop, "_market_regime", return_value=(regime, 1.0, 1.0)), \
                 patch.object(loop, "_down_day_decision", return_value=down_day), \
                 patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
                state = {"positions": [], "trades_today": 0, "trades_by_ticker_today": {},
                         "last_exit_by_ticker": {}}
                opened = loop.open_top1_only(state, {}, candidates or self._pool(), "2026-10-09")
        finally:
            loop._original_open = orig_open
        return opened

    def test_down_day_bullish_regime_opens_short(self):
        opened = self._open("bullish", True)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["direction"], "SHORT")
        self.assertEqual(opened[0]["direction_gate"], "short_only_down_day")

    def test_down_day_with_only_buy_candidates_opens_nothing(self):
        opened = self._open("bullish", True, candidates=[_cand("7203.T", "BUY", score=95.0, up=80.0, down=5.0)])
        self.assertEqual(opened, [])

    def test_up_day_bullish_regime_opens_buy_unchanged(self):
        opened = self._open("bullish", False)
        self.assertEqual(opened[0]["direction"], "BUY")


# =========================================================================
# 急落ブレーキ(新規買いのみ停止)のロジック自体は変更していないことの回帰確認。
# =========================================================================

class CrashBrakeStillBuyOnly(unittest.TestCase):
    def test_drop_buys_if_braked_only_drops_buy(self):
        decision = {"intraday_crash_brake": True}
        candidates = [_cand("7203.T", "BUY"), _cand("8035.T", "SHORT")]
        kept = dd.drop_buys_if_braked(decision, candidates)
        self.assertEqual([c["ticker"] for c in kept], ["8035.T"])

    def test_no_brake_keeps_both(self):
        decision = {"intraday_crash_brake": False}
        candidates = [_cand("7203.T", "BUY"), _cand("8035.T", "SHORT")]
        kept = dd.drop_buys_if_braked(decision, candidates)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
