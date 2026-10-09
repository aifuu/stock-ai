"""DOWN日フォールバック + 急落ブレーキ(新規買いのみ停止)の統合テスト。

オーナー決定(2026-10-08):
  A. DOWN日にstrategy_policy_down.jsonが無ければ、2026-10-08以前と同じく
     strategy_policy.jsonで実売買する(policy_fallback=True)。方向は従来どおり
     日経レジームフィルター(弱気→空売りのみ/強気→買いのみ/中立→両方)が決める。
  B. 急落ブレーキ(前日確定終値比 <= -1.5%、当日中は解除しない)は新規の「買い」だけを
     止める。新規空売りはレジームフィルターに従って継続、決済は常に継続。
  C. liveとdaytradeは同じ日のどのtickでも同じdaily policyファイルを使う。

daily_decision.pyは実物(一時ディレクトリ内の実policyファイルのコピーとハッシュ)を使い、
ネットワーク(先物データ・Discord・yfinance)と承認署名の検証(秘密鍵が無いため)だけ差し替える。
"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402

requests.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: requests.post"))

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: yfinance.download"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import daytrade_tp5000_sl8000_paper as dt  # noqa: E402
import profit_top10_paper as live  # noqa: E402
import run_profit_loop as loop  # noqa: E402
import daily_decision as dd  # noqa: E402
import futures_trend as ft  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")
REPO = os.path.dirname(os.path.abspath(__file__))
DAY = datetime(2026, 10, 9, 8, 30, tzinfo=TZ)  # 金曜(営業日)


def _futures_bars(trend="down", last_confirmed_day="2026-10-08", n=40):
    idx = pd.bdate_range(end=last_confirmed_day, periods=n)
    base = np.linspace(70000, 60000, n) if trend == "down" else np.linspace(60000, 70000, n)
    return pd.DataFrame({"Close": base}, index=idx)


def _real_policy_thresholds(path):
    """承認署名の検証には秘密鍵が要るため、load_policyの代わりに実ファイルの閾値を使う。"""
    with open(os.path.join(REPO, path), encoding="utf-8") as f:
        p = json.load(f)
    return {"status": p["status"], "up_threshold": float(p["up_threshold"]),
            "min_score_for_buy": float(p["min_score_for_buy"]),
            "nikkei_filter": str(p["nikkei_filter"]).lower() in ("true", "1"),
            "atr_tp_multiplier": float(p["atr_tp_multiplier"]),
            "atr_sl_multiplier": float(p["atr_sl_multiplier"]), "hold_days": int(p["hold_days"])}


def _cand(ticker, direction, price, score, up, down):
    return {"ticker": ticker, "company": ticker, "direction": direction, "price": price,
            "tp": price * (1.02 if direction == "BUY" else 0.98),
            "sl": price * (0.99 if direction == "BUY" else 1.01),
            "score": score, "up_probability": up, "down_probability": down,
            "flat_probability": 5.0, "data_date": "2026-10-08", "buy_reason": "test",
            "expected_value_pct": 1.0}


# BUYが最上位、空売りが2位(中立レジームならBUYが選ばれる並び)。
POOL = [
    _cand("7203.T", "BUY", 3000.0, 95.0, 80.0, 5.0),
    _cand("8035.T", "SHORT", 4000.0, 90.0, 5.0, 80.0),
    _cand("9984.T", "BUY", 5000.0, 85.0, 75.0, 5.0),
    _cand("6758.T", "SHORT", 2500.0, 80.0, 5.0, 75.0),
]


class _FakeDT(datetime):
    current = DAY

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz is None else cls.current.astimezone(tz)


class DownDayAndBrakeIntegration(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = tempfile.mkdtemp()
        for f in ("strategy_policy.json", "strategy_policy_up.json"):
            shutil.copy(os.path.join(REPO, f), os.path.join(self.dir, f))
        os.chdir(self.dir)
        self.loaded_policy_files = []
        self.price_ratio = 0.995  # 前日確定終値比(ブレーキ判定用の場中価格)
        self.regime = "bearish"
        self._patches = [
            patch.object(live, "datetime", _FakeDT),
            patch.object(dd, "datetime", _FakeDT),
            patch.object(ft, "_download", side_effect=lambda *a, **k: _futures_bars(self.trend)),
            patch.object(ft, "intraday_price",
                         side_effect=lambda *a, **k: self._base() * self.price_ratio),
            patch.object(live, "is_tse_trading_day", return_value=True),
            patch.object(dt, "is_tse_trading_day", return_value=True),
            patch.object(live, "load_policy", side_effect=self._load_policy),
            patch.object(live, "scan", side_effect=lambda policy: loop.scan_candidates_fixed(policy)),
            patch.object(loop, "_original_scan", side_effect=lambda policy: ([dict(c) for c in POOL], 225)),
            patch.object(live, "open_positions", side_effect=loop.open_top1_only),
            patch.object(live, "mark_and_close", return_value=[]),
            patch.object(loop, "_market_regime", side_effect=lambda: (self.regime, -1.0, -1.0)),
            patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}),
            patch.object(live, "discord_send"),
            patch.object(live, "discord_progress"),
            patch.object(dt, "discord_send"),
        ]
        self.trend = "down"
        self.mocks = {}
        for p in self._patches:
            m = p.start()
            self.mocks[getattr(p, "attribute", None)] = m

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        os.chdir(self._cwd)
        shutil.rmtree(self.dir, ignore_errors=True)

    # --- helpers -------------------------------------------------------
    def _base(self):
        return float(_futures_bars(self.trend)["Close"].iloc[-1])

    def _load_policy(self, policy_file=None):
        self.loaded_policy_files.append(policy_file)
        return _real_policy_thresholds(policy_file or live.POLICY_FILE)

    def _at(self, hour, minute):
        _FakeDT.current = DAY.replace(hour=hour, minute=minute)
        return _FakeDT.current

    def _live_tick(self, hour, minute):
        now = self._at(hour, minute)
        live._run()
        # 本番と同じく、liveのtickがこのiterationのTOP10キャッシュを書いた状態にする
        with open(dt.fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"timestamp": now.timestamp(), "raw": [dict(c) for c in POOL], "scanned": 225}, f)
        with open(live.STATE_FILE, encoding="utf-8") as f:
            return json.load(f)

    def _daytrade_entry(self, now):
        state = dt.default_state()
        dt._try_entry(state, now, now.strftime("%Y-%m-%d"))
        return state

    def _decision(self):
        with open(dd.DECISION_FILE, encoding="utf-8") as f:
            return json.load(f)

    # --- tests ---------------------------------------------------------
    def test_down_day_fallback_live_opens_short_on_bearish_regime(self):
        dd.ensure_decision(self._at(8, 30))
        d = self._decision()
        self.assertEqual(d["trend"], "down")
        self.assertEqual(d["policy_file"], "strategy_policy.json")
        self.assertTrue(d["policy_fallback"])
        self.assertTrue(d["entry_allowed"])

        state = self._live_tick(10, 0)
        self.assertEqual(self.loaded_policy_files[-1], "strategy_policy.json")
        self.assertEqual(len(state["positions"]), 1)
        pos = state["positions"][0]
        self.assertEqual(pos["direction"], "SHORT")
        self.assertEqual(pos["ticker"], "8035.T")  # 弱気→空売りのみ、最上位の空売り
        self.assertEqual(pos["market_regime_raw"], "bearish")
        self.assertEqual(pos["direction_gate"], "short_only_down_day")
        self.assertFalse(os.path.exists(dd.SHADOW_FILE))  # 実売買日はシャドー無し

    def test_daytrade_same_tick_uses_same_policy_and_same_top10_with_short_allowed(self):
        dd.ensure_decision(self._at(8, 30))
        live_state = self._live_tick(10, 0)
        now = _FakeDT.current
        dt_state = self._daytrade_entry(now)
        pending = dt_state["pending"]
        self.assertIsNotNone(pending, "DOWN日でもdaytradeは実エントリー判断をする")
        self.assertEqual(pending["policy_file"], "strategy_policy.json")
        self.assertEqual(pending["policy_file"], self.loaded_policy_files[0])
        self.assertEqual(pending["policy_hash"], self._decision()["policy_hash"])
        # 同じTOP10(弱気レジームで空売りのみ)の先頭=liveと同じ銘柄・方向
        top10, _ = loop.scan_candidates_fixed(_real_policy_thresholds("strategy_policy.json"))
        self.assertEqual(pending["ticker"], top10[0]["ticker"])
        self.assertEqual(pending["direction"], "SHORT")
        self.assertEqual((pending["ticker"], pending["direction"]),
                         (live_state["positions"][0]["ticker"], live_state["positions"][0]["direction"]))

    def test_same_daily_policy_file_on_every_tick_for_both_tracks(self):
        self.regime = "neutral"
        dd.ensure_decision(self._at(8, 30))
        dt_files = []
        ticks = [(9, 55), (10, 30), (11, 0), (13, 0), (14, 30)]
        for i, (h, m) in enumerate(ticks):
            if (h, m) == (11, 0):
                self.price_ratio = 0.98  # 場中急落 → ブレーキ(policyは変えない)
            if (h, m) == (13, 0):
                self.trend = "up"  # 場中に先物データが変わっても当日の判断は作り直さない
                self.price_ratio = 1.01
            self._live_tick(h, m)
            f, src = dt.choose_policy_file_reusing_live_tick(_FakeDT.current)
            dt_files.append(f)
        live_files = set(self.loaded_policy_files)
        self.assertEqual(live_files, {"strategy_policy.json"})
        self.assertEqual(set(dt_files), {"strategy_policy.json"})
        self.assertEqual(len(self.loaded_policy_files), len(ticks))
        self.assertTrue(self._decision()["intraday_crash_brake"])  # sticky

    def test_crash_brake_blocks_buys_on_both_tracks_but_allows_short(self):
        # 中立レジームでも下落日ゲートにより新規は空売りのみ(A案)。この時点で既に
        # 買いはTOP10に出ないため、ブレーキ自身のbrake_buyスキップは発生しない
        # (クラッシュブレーキのロジック自体は変更していない)。
        self.regime = "neutral"
        dd.ensure_decision(self._at(8, 30))
        self.price_ratio = 0.984  # -1.6%
        live_state = self._live_tick(10, 0)
        self.assertTrue(self._decision()["intraday_crash_brake"])
        self.assertEqual([p["direction"] for p in live_state["positions"]], ["SHORT"])
        self.assertEqual(live_state["positions"][0]["ticker"], "8035.T")

        dt_state = self._daytrade_entry(_FakeDT.current)
        pending = dt_state["pending"]
        self.assertEqual(pending["direction"], "SHORT")
        self.assertEqual(pending["ticker"], "8035.T")
        self.assertNotIn("7203.T", pending["skipped"])

        # 反発してもその日は解除されず、次のtickでもBUYは建たない
        self.price_ratio = 1.02
        live_state = self._live_tick(11, 0)
        self.assertTrue(all(p["direction"] == "SHORT" for p in live_state["positions"]))
        self.assertTrue(self._decision()["intraday_crash_brake"])

    def test_without_brake_neutral_regime_opens_top_buy(self):
        # 対照(UP日・ブレーキ無し): 下落日ゲートが働かない日なら、従来どおり
        # レジーム中立で並び最上位(BUY)を両トラックとも選ぶ。
        self.trend = "up"
        self.regime = "neutral"
        dd.ensure_decision(self._at(8, 30))
        live_state = self._live_tick(10, 0)
        self.assertEqual(live_state["positions"][0]["direction"], "BUY")
        self.assertEqual(live_state["positions"][0]["direction_gate"], "regime_neutral")
        pending = self._daytrade_entry(_FakeDT.current)["pending"]
        self.assertEqual((pending["ticker"], pending["direction"]), ("7203.T", "BUY"))

    def test_down_day_short_only_despite_bullish_regime(self):
        # A案(2026-10-09の事象そのもの): DOWN日は日経レジームが強気でも新規は
        # 空売りのみに固定する。ブレーキ(新規買いのみ停止)も、元々買いが
        # 候補に出ないのでここでは効かない(クラッシュブレーキのロジック自体は不変)。
        self.regime = "bullish"
        dd.ensure_decision(self._at(8, 30))
        self.price_ratio = 0.98  # ブレーキも発動させておく(買いが候補に出ないことの確認)
        live_state = self._live_tick(10, 0)
        self.assertEqual([p["direction"] for p in live_state["positions"]], ["SHORT"])
        self.assertEqual(live_state["positions"][0]["ticker"], "8035.T")
        self.assertEqual(live_state["positions"][0]["market_regime_raw"], "bullish")
        self.assertEqual(live_state["positions"][0]["direction_gate"], "short_only_down_day")
        self.assertTrue(self._decision()["intraday_crash_brake"])

        dt_state = self._daytrade_entry(_FakeDT.current)
        pending = dt_state["pending"]
        self.assertEqual(pending["direction"], "SHORT")
        self.assertEqual(pending["ticker"], "8035.T")
        self.assertEqual(pending["direction_gate"], "short_only_down_day")
        # 強気レジームでも買いはそもそもTOP10に出ないため、ブレーキのbrake_buy
        # スキップ理由は発生しない(7203.T/9984.Tはskippedに現れない)。
        self.assertNotIn("7203.T", pending["skipped"])
        self.assertNotIn("9984.T", pending["skipped"])

    def test_replay_2026_10_09_bullish_regime_down_day_both_tracks_choose_short(self):
        """2026-10-09の再現: daily_decisionはDOWNだが日経レジームは強気
        (kairi25+2.62%/ret5+1.06%相当)。実際の候補に近い確率帯(BUY約34-36%上昇/
        33-35%下落、SHORTはpolicy適合)で、両トラックとも空売りのみを選ぶことを確認する。
        """
        self.regime = "bullish"
        realistic_pool = [
            _cand("8766.T", "BUY", 3200.0, 72.0, 35.0, 33.0),
            _cand("4704.T", "BUY", 2800.0, 68.0, 34.5, 34.0),
            _cand("9501.T", "SHORT", 1500.0, 75.0, 25.0, 55.0),
            _cand("6502.T", "SHORT", 2100.0, 70.0, 22.0, 50.0),
        ]
        with patch.object(loop, "_original_scan", side_effect=lambda policy: ([dict(c) for c in realistic_pool], 225)):
            dd.ensure_decision(self._at(8, 30))
            now = self._at(10, 0)
            live._run()
            # liveのtickと同じく、このiterationのTOP10キャッシュをrealistic_poolで書く
            # (_live_tickヘルパーはモジュール定数POOLを書くため、ここでは使わない)。
            with open(dt.fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"timestamp": now.timestamp(), "raw": [dict(c) for c in realistic_pool], "scanned": 225}, f)
            with open(live.STATE_FILE, encoding="utf-8") as f:
                live_state = json.load(f)
            dt_state = self._daytrade_entry(_FakeDT.current)

        self.assertEqual(len(live_state["positions"]), 1)
        self.assertEqual(live_state["positions"][0]["direction"], "SHORT")
        self.assertEqual(live_state["positions"][0]["ticker"], "9501.T")  # SHORTの最高scoreが選ばれる
        self.assertEqual(live_state["positions"][0]["direction_gate"], "short_only_down_day")

        pending = dt_state["pending"]
        self.assertEqual(pending["direction"], "SHORT")
        self.assertEqual(pending["ticker"], "9501.T")
        self.assertEqual(pending["direction_gate"], "short_only_down_day")

    def test_exits_continue_while_brake_active(self):
        dd.ensure_decision(self._at(8, 30))
        self.price_ratio = 0.98
        self._live_tick(10, 0)
        self.mocks["mark_and_close"].assert_called()  # liveの決済処理は毎tick実行
        # daytrade: 保有中ポジションの決済評価はブレーキ中も行われる
        state = dt.default_state()
        state["positions"] = [{"ticker": "7203.T", "direction": "BUY", "entry_date": "2026-10-09"}]
        dt.save_state(state)
        with patch.object(dt, "_evaluate_exit", return_value=["exit"]) as ev, \
             patch.object(dt, "_update_daily_summary"):
            dt.run(now=self._at(10, 30))
        ev.assert_called_once()
        self.assertTrue(dd.todays_crash_brake(_FakeDT.current))

    def test_policy_changed_after_decision_blocks_new_entries_on_fallback_path(self):
        dd.ensure_decision(self._at(8, 30))
        with open("strategy_policy.json", "a", encoding="utf-8") as f:
            f.write(" ")
        live_state = self._live_tick(10, 0)
        self.assertEqual(live_state["positions"], [])
        f, src = dt.choose_policy_file_reusing_live_tick(_FakeDT.current)
        self.assertIsNone(f)
        self.assertEqual(src, "blocked:policy_changed_since_decision")

    def test_brake_resets_next_day(self):
        dd.ensure_decision(self._at(8, 30))
        self.price_ratio = 0.98
        self._live_tick(10, 0)
        self.assertTrue(self._decision()["intraday_crash_brake"])
        global DAY
        saved = DAY
        try:
            DAY = datetime(2026, 10, 13, 8, 30, tzinfo=TZ)  # 翌営業日(10/12は祝日)
            self.price_ratio = 1.0
            self.regime = "neutral"
            with patch.object(ft, "_download",
                              side_effect=lambda *a, **k: _futures_bars("down", "2026-10-09")):
                d = dd.ensure_decision(self._at(8, 30))
                self.assertEqual(d["date"], "2026-10-13")
                self.assertFalse(d["intraday_crash_brake"])
                self.assertEqual(dd.entry_status(d), (True, None))
                self.assertFalse(dd.buy_blocked_by_crash_brake(d))
        finally:
            DAY = saved


class RealRepoFilesLoad(unittest.TestCase):
    """origin/mainにある実ファイルが新コードで読み込めること。"""

    def test_real_policy_files_hash_prefixes_unchanged(self):
        self.assertEqual(dd.policy_hash(os.path.join(REPO, "strategy_policy.json")), "cf106bdc4c56")
        self.assertEqual(dd.policy_hash(os.path.join(REPO, "strategy_policy_up.json")), "cfe6da4cc960")
        self.assertFalse(os.path.exists(os.path.join(REPO, "strategy_policy_down.json")))

    def test_real_state_files_load_through_the_real_loaders(self):
        tmp = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            for name in (live.STATE_FILE, dt.STATE_FILE):
                src = os.path.join(REPO, name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(tmp, name))
            os.chdir(tmp)
            with patch.object(live, "discord_send", side_effect=AssertionError("corrupt live state")), \
                 patch.object(dt, "discord_send", side_effect=AssertionError("corrupt daytrade state")):
                s_live = live.load_state()
                s_dt = dt.load_state()
            self.assertIsInstance(s_live["positions"], list)
            self.assertIsInstance(s_dt["positions"], list)
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_real_state_and_decision_files_load(self):
        for name in ("daily_decision.json", "profit_top10_paper_state.json",
                     "daytrade_tp5000_sl8000_state.json"):
            path = os.path.join(REPO, name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            if name == "daily_decision.json":
                dd.entry_status(data)
                dd.buy_blocked_by_crash_brake(data)


if __name__ == "__main__":
    unittest.main()
