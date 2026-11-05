"""TOP1建値バグ修正(2026-11)のテスト。

バグ: profit_top10_paper.scan()はdownload_5m(t)['Close'].iloc[-1]で候補を
値付けしていたが、yfinanceの5分足は実時刻から15-20分程度遅延するため、
09:0x台のtickでは最終バーが前営業日の最終バーになっている場合があり、
その値段で「約定した」と記録してしまっていた(2026-11-05の285A.T: 記録上
10,640円で約定したことになっていたが、実際の初値は09:05 Open 9,820円で、
10,640円はその日一度も付いていない。詳細はtop1_entry_fill_correction.csv)。

修正: daytrade_tp5000_sl8000_paper.pyのFill rule v2(decision_next_bar_open_v2)
と同じ原則で、TOP1も「判断(decision)」と「約定(fill)」を分離する:
  - run_profit_loop.open_top1_only()はTOP1を選ぶが即座には発注せず、
    state['pending_entry']として保存する(選定ロジック自体は変更しない)。
  - run_profit_loop.fill_pending_entry()が、決定時刻をceilした次の5分足バーが
    実際に出現したtickでそのバーのOpenで約定させる(そのバー自体が欠けていても
    当日のそれ以降最初のバーを使う。前営業日のバーは絶対に使わない)。
  - profit_top10_paper.mark_and_close()は、約定バー(fill_bar_time)より前の
    当日バーでTP/SL判定をしない(約定前の値動きで決済してしまわないため)。

ネットワークは一切使わない(download_5m/discord_send/日経レジーム取得は
すべてモック、DISCORD_WEBHOOKは未設定にしてdiscord_sendを完全no-opにする)。
"""
import copy
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402

requests.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: requests.post attempted"))

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: yfinance.download attempted"))

import pandas as pd  # noqa: E402

import profit_top10_paper as app  # noqa: E402
import run_profit_loop as loop  # noqa: E402
import safe_state  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")


class _FakeDT(datetime):
    """open_top1_only()が呼ぶdatetime.now(app.TZ)を決定的な時刻に固定する
    (他のrun_profit_loopテストと同じ手法)。"""

    current = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz is None else cls.current.astimezone(tz)


def _cand(ticker="285A.T", direction="SHORT", price=10640.0, tp=10000.0, sl=10850.0, **extra):
    base = {
        "ticker": ticker, "company": ticker, "direction": direction, "price": price,
        "tp": tp, "sl": sl, "score": 80.0, "up_probability": 10.0, "down_probability": 85.0,
        "flat_probability": 5.0, "data_date": "2026-11-05", "buy_reason": "test",
        "expected_value_pct": 1.0,
    }
    base.update(extra)
    return base


def _state(capital=2_000_000.0, **extra):
    s = {
        "capital": capital, "peak": capital, "max_dd": 0.0, "positions": [],
        "trades_today": 0, "trades_by_ticker_today": {}, "last_exit_by_ticker": {},
        "pending_entry": None, "cancelled_entries": [],
    }
    s.update(extra)
    return s


def _bars(rows):
    """rows = [(timestamp_str, open, high, low, close), ...]"""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def _decide(state, candidates, now, regime="neutral", down_day=False, policy=None, today="2026-11-05"):
    """open_top1_only()を、ネットワークに依存する部分(日経レジーム判定・
    down_day判定・feedback weight)を固定した状態で呼ぶ。"""
    _FakeDT.current = now
    with patch.object(loop, "datetime", _FakeDT), \
         patch.object(loop, "_market_regime", return_value=(regime, 1.0, 1.0)), \
         patch.object(loop, "_down_day_decision", return_value=down_day), \
         patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}):
        return loop.open_top1_only(state, policy or {}, candidates, today)


def _filled_position(ticker="7203.T", direction="BUY", entry_date="2026-11-05", entry_price=1000.0,
                      tp=1100.0, sl=950.0, fill_bar_time="2026-11-05T09:10:00", shares=100, **extra):
    base = {
        "ticker": ticker, "company": ticker, "direction": direction, "entry_date": entry_date,
        "entry_time": "09:10", "entry_price": entry_price, "tp": tp, "sl": sl, "shares": shares,
        "invested_amount": entry_price * shares, "score": 70.0, "up_probability": 60.0,
        "down_probability": 10.0, "expected_value_pct": 1.0, "buy_reason": "test",
        "current_price": entry_price, "fill_method": "decision_next_bar_open_top1_v1",
        "decision_time": "2026-11-05T09:01:00", "fill_bar_time": fill_bar_time,
        "decision_price": entry_price - 20.0,
    }
    base.update(extra)
    return base


class TopOneNextBarFillTests(unittest.TestCase):

    # (a) 285A.T再現: 前営業日最終バー(10,640円)は使わず、09:05バーのOpen(9,820円)
    # で約定する。tp/slは(fill-decision)分だけ平行移動される。
    def test_a_285a_reproduction_uses_next_bar_open_not_stale_bar(self):
        state = _state()
        decision_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        opened = _decide(state, [_cand()], decision_now, regime="bearish", down_day=False)
        self.assertEqual(opened, [])
        self.assertEqual(state["positions"], [])
        pending = state["pending_entry"]
        self.assertEqual(pending["ticker"], "285A.T")
        self.assertEqual(pending["direction"], "SHORT")
        self.assertEqual(pending["decision_price"], 10640.0)
        self.assertEqual(pending["fill_bar_time"], "2026-11-05T09:05:00")
        capital_before = state["capital"]

        # 09:01時点: まだ前営業日(11/04 15:20)の最終バーしか無い → 使ってはいけない
        stale_only = _bars([("2026-11-04 15:20", 10640.0, 10650.0, 10630.0, 10640.0)])
        with patch.object(app, "download_5m", return_value=stale_only):
            filled = loop.fill_pending_entry(state, {}, decision_now, True, False)
        self.assertEqual(filled, [])
        self.assertEqual(state["positions"], [])
        self.assertIsNotNone(state["pending_entry"], "前営業日のバーだけでは約定してはいけない")
        self.assertEqual(state["capital"], capital_before, "約定していない間は資金が動かない")

        # 09:25時点: 09:05バーが登場(前営業日バーも依然データに含まれる)
        now2 = datetime(2026, 11, 5, 9, 25, tzinfo=TZ)
        both = _bars([
            ("2026-11-04 15:20", 10640.0, 10650.0, 10630.0, 10640.0),
            ("2026-11-05 09:05", 9820.0, 10000.0, 9700.0, 9900.0),
            ("2026-11-05 09:10", 9900.0, 9950.0, 9850.0, 9870.0),
        ])
        with patch.object(app, "download_5m", return_value=both):
            filled = loop.fill_pending_entry(state, {}, now2, True, False)
        self.assertEqual(len(filled), 1)
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(len(state["positions"]), 1)
        pos = state["positions"][0]
        self.assertEqual(pos["entry_price"], 9820.0, "10,640円(前営業日分)ではなく09:05 Openで約定する")
        self.assertEqual(pos["entry_time"], "09:05")
        self.assertEqual(pos["fill_method"], "decision_next_bar_open_top1_v1")
        self.assertEqual(pos["decision_price"], 10640.0)
        shift = 9820.0 - 10640.0
        self.assertAlmostEqual(pos["tp"], 10000.0 + shift)
        self.assertAlmostEqual(pos["sl"], 10850.0 + shift)
        self.assertGreater(pos["shares"], 0)
        self.assertAlmostEqual(pos["invested_amount"], pos["shares"] * 9820.0)

    # (b) 前営業日バーしか無い間はひたすら保留(キャンセルも約定もしない)
    def test_b_only_previous_day_bars_available_stays_pending(self):
        state = _state()
        decision_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        _decide(state, [_cand()], decision_now)
        stale_only = _bars([("2026-11-04 15:20", 10640.0, 10650.0, 10630.0, 10640.0)])
        for minute in (5, 10, 15, 20):
            now = datetime(2026, 11, 5, 9, minute, tzinfo=TZ)
            with patch.object(app, "download_5m", return_value=stale_only):
                filled = loop.fill_pending_entry(state, {}, now, True, False)
            self.assertEqual(filled, [])
            self.assertIsNotNone(state["pending_entry"])
        self.assertEqual(state["positions"], [])

    # (c) 約定予定バーそのものが存在しなくても、それ以降最初に実在するバーを使う
    def test_c_missing_opening_bar_uses_first_later_bar(self):
        state = _state()
        decision_now = datetime(2026, 11, 5, 9, 0, tzinfo=TZ)  # ちょうど境界 → fill_bar_time=09:00
        _decide(state, [_cand(ticker="999.T")], decision_now)
        self.assertEqual(state["pending_entry"]["fill_bar_time"], "2026-11-05T09:00:00")

        # 09:00バー自体が無く、最初のバーは09:05
        bars = _bars([
            ("2026-11-05 09:05", 500.0, 520.0, 495.0, 510.0),
            ("2026-11-05 09:10", 510.0, 515.0, 505.0, 512.0),
        ])
        now = datetime(2026, 11, 5, 9, 25, tzinfo=TZ)
        with patch.object(app, "download_5m", return_value=bars):
            filled = loop.fill_pending_entry(state, {}, now, True, False)
        self.assertEqual(len(filled), 1)
        self.assertEqual(state["positions"][0]["entry_price"], 500.0)
        self.assertEqual(state["positions"][0]["entry_time"], "09:05")

    # (d) mark_and_close(): 約定バー(fill_bar_time)より前の当日バーはTP/SL判定の
    # 対象外。約定バー自体は既存のSL優先ロジックでそのまま判定される。
    def test_d_bars_before_fill_bar_time_never_trigger_exit(self):
        now = datetime(2026, 11, 5, 9, 20, tzinfo=TZ)
        # before-fill-bar(09:05)のLow=900はSL=950を割っているが無視されなければならない。
        # 約定バー自体(09:10)のLow=940がSLを割るので、ここで決済される。
        pos = _filled_position(direction="BUY", entry_price=1000.0, tp=1100.0, sl=950.0,
                                fill_bar_time="2026-11-05T09:10:00")
        state = _state(positions=[pos])
        bars = _bars([
            ("2026-11-05 09:05", 1000.0, 1005.0, 900.0, 990.0),  # 約定前。SL=950割れだが無視すべき
            ("2026-11-05 09:10", 995.0, 1000.0, 940.0, 960.0),   # 約定バー自体。SLを割る→ここで決済
            ("2026-11-05 09:15", 960.0, 965.0, 955.0, 958.0),
        ])
        captured = []
        with patch.object(app, "download_5m", return_value=bars), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(state, now, {"hold_days": 1})
        self.assertEqual(len(captured), 1, "約定前バーのSL割れでは決済されない")
        self.assertEqual(captured[0]["result"], "SL")
        self.assertEqual(captured[0]["exit_price"], 950.0)
        self.assertEqual(state["positions"], [])

    def test_d_bars_before_fill_bar_time_ignored_then_later_bar_triggers(self):
        now = datetime(2026, 11, 5, 9, 20, tzinfo=TZ)
        # 約定バー自体(09:10)はTP/SLどちらも割らない。その後の09:15で初めてSLを割る。
        pos = _filled_position(direction="BUY", entry_price=1000.0, tp=1100.0, sl=950.0,
                                fill_bar_time="2026-11-05T09:10:00")
        state = _state(positions=[pos])
        bars = _bars([
            ("2026-11-05 09:05", 1000.0, 1005.0, 900.0, 990.0),  # 約定前。SL割れだが無視すべき
            ("2026-11-05 09:10", 995.0, 1005.0, 970.0, 980.0),   # 約定バー自体。ヒットなし
            ("2026-11-05 09:15", 980.0, 985.0, 940.0, 945.0),    # ここで初めてSLを割る
        ])
        captured = []
        with patch.object(app, "download_5m", return_value=bars), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(state, now, {"hold_days": 1})
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["result"], "SL")
        self.assertEqual(captured[0]["exit_price"], 950.0)

    # (e) ペンディング中は新しい判断をしない(1枠を保持)
    def test_e_pending_blocks_second_decision(self):
        state = _state()
        _decide(state, [_cand(ticker="AAA.T")], datetime(2026, 11, 5, 9, 1, tzinfo=TZ))
        self.assertEqual(state["pending_entry"]["ticker"], "AAA.T")
        opened2 = _decide(state, [_cand(ticker="BBB.T")], datetime(2026, 11, 5, 9, 6, tzinfo=TZ))
        self.assertEqual(opened2, [])
        self.assertEqual(state["pending_entry"]["ticker"], "AAA.T", "保留中は新しい判断をしない")

    # (f) 日付が変わったらキャンセル
    def test_f_new_day_cancels_pending(self):
        state = _state(pending_entry={
            "ticker": "285A.T", "direction": "SHORT", "decision_date": "2026-11-04",
            "decision_time": "2026-11-04T09:01:00", "fill_bar_time": "2026-11-04T09:05:00",
            "decision_price": 10640.0, "tp": 10000.0, "sl": 10850.0,
        })
        now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        filled = loop.fill_pending_entry(state, {}, now, True, False)
        self.assertEqual(filled, [])
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(len(state["cancelled_entries"]), 1)
        self.assertEqual(state["cancelled_entries"][0]["reason"], "CANCELLED_NO_FILL_NEW_DAY")

    # (g) entry_ok=Falseなら方向を問わずキャンセル。買いブレーキはBUYだけキャンセルし、SHORTは残す。
    def test_g_entry_blocked_cancels_regardless_of_direction(self):
        state = _state(pending_entry={
            "ticker": "7203.T", "direction": "SHORT", "decision_date": "2026-11-05",
            "decision_time": "2026-11-05T09:01:00", "fill_bar_time": "2026-11-05T09:05:00",
            "decision_price": 3000.0, "tp": 2900.0, "sl": 3050.0,
        })
        now = datetime(2026, 11, 5, 9, 10, tzinfo=TZ)
        filled = loop.fill_pending_entry(state, {}, now, False, False)
        self.assertEqual(filled, [])
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(state["cancelled_entries"][0]["reason"], "CANCELLED_NO_FILL_ENTRY_BLOCKED")

    def test_g_crash_brake_cancels_buy_but_keeps_short_pending(self):
        buy_state = _state(pending_entry={
            "ticker": "7203.T", "direction": "BUY", "decision_date": "2026-11-05",
            "decision_time": "2026-11-05T09:01:00", "fill_bar_time": "2026-11-05T09:05:00",
            "decision_price": 3000.0, "tp": 3100.0, "sl": 2950.0,
        })
        now = datetime(2026, 11, 5, 9, 10, tzinfo=TZ)
        filled = loop.fill_pending_entry(buy_state, {}, now, True, True)
        self.assertEqual(filled, [])
        self.assertIsNone(buy_state["pending_entry"])
        self.assertEqual(buy_state["cancelled_entries"][0]["reason"], "CANCELLED_NO_FILL_CRASH_BRAKE")

        short_state = _state(pending_entry={
            "ticker": "8035.T", "direction": "SHORT", "decision_date": "2026-11-05",
            "decision_time": "2026-11-05T09:01:00", "fill_bar_time": "2026-11-05T09:05:00",
            "decision_price": 4000.0, "tp": 3900.0, "sl": 4050.0,
        })
        with patch.object(app, "download_5m", return_value=None):
            filled = loop.fill_pending_entry(short_state, {}, now, True, True)
        self.assertEqual(filled, [])
        self.assertIsNotNone(short_state["pending_entry"], "買いブレーキはSHORTをキャンセルしない")
        self.assertEqual(short_state["cancelled_entries"], [])

    # (h) 約定価格では買えない(資金不足)場合はキャンセルする
    def test_h_unaffordable_at_fill_price_cancels(self):
        state = _state(capital=100_000.0, pending_entry={
            "ticker": "999.T", "direction": "BUY", "decision_date": "2026-11-05",
            "decision_time": "2026-11-05T09:01:00", "fill_bar_time": "2026-11-05T09:05:00",
            "decision_price": 500.0, "tp": 550.0, "sl": 480.0,
            "score": 70.0, "up_probability": 60.0, "down_probability": 10.0,
            "expected_value_pct": 1.0, "company": "999.T", "buy_reason": "test",
        })
        now = datetime(2026, 11, 5, 9, 10, tzinfo=TZ)
        # 決定時は100株=5万円で買えたが、約定バーで大きくギャップアップし
        # 100株=20万円になった(資金10万円では1単元も買えない)。
        bars = _bars([("2026-11-05 09:05", 2000.0, 2050.0, 1980.0, 2000.0)])
        with patch.object(app, "download_5m", return_value=bars):
            filled = loop.fill_pending_entry(state, {}, now, True, False)
        self.assertEqual(filled, [])
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(state["positions"], [])
        self.assertEqual(state["cancelled_entries"][0]["reason"], "CANCELLED_NO_FILL_NOT_OPENED")

    # (i) 旧stateファイル(新キー無し)は読み込め、fill_bar_timeを持たない既存
    # ポジション(旧経路)の決済挙動は一切変わらない。
    def test_i_legacy_state_without_new_keys_loads_with_defaults(self):
        tmp = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            legacy = {"capital": 900_000.0, "peak": 1_000_000.0, "max_dd": 10.0,
                      "positions": [], "trade_count_date": "2026-11-04", "trades_today": 2,
                      "trades_by_ticker_today": {"7203.T": 1}}
            safe_state.atomic_write_json(app.STATE_FILE, legacy)
            with patch.object(app, "discord_send", side_effect=AssertionError("must not notify on clean load")):
                loaded = app.load_state()
            self.assertIsNone(loaded["pending_entry"])
            self.assertEqual(loaded["cancelled_entries"], [])
            self.assertEqual(loaded["capital"], 900_000.0)
        finally:
            os.chdir(cwd)

    def test_i_legacy_position_without_fill_bar_time_unaffected_by_fill_skip_logic(self):
        now = datetime(2026, 11, 5, 9, 20, tzinfo=TZ)
        legacy_pos = _filled_position(direction="BUY", entry_price=1000.0, tp=1100.0, sl=950.0)
        del legacy_pos["fill_method"], legacy_pos["decision_time"], legacy_pos["fill_bar_time"], legacy_pos["decision_price"]
        state = _state(positions=[legacy_pos])
        # 09:05バーのLow=900はSLを割っており、fill_bar_timeが無い(=旧経路)ので
        # 従来どおり09:00以降の最初のバーからそのまま判定されるべき。
        bars = _bars([("2026-11-05 09:05", 1000.0, 1005.0, 900.0, 990.0)])
        captured = []
        with patch.object(app, "download_5m", return_value=bars), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(state, now, {"hold_days": 1})
        self.assertEqual(len(captured), 1, "fill_bar_timeが無い旧ポジションは09:05バーから判定される(従来どおり)")
        self.assertEqual(captured[0]["result"], "SL")

    # (j) キャンセルはtrades_todayを消費しない
    def test_j_cancel_does_not_consume_trades_today(self):
        state = _state(trades_today=3, pending_entry={
            "ticker": "285A.T", "direction": "SHORT", "decision_date": "2026-11-04",
            "decision_time": "2026-11-04T09:01:00", "fill_bar_time": "2026-11-04T09:05:00",
            "decision_price": 10640.0, "tp": 10000.0, "sl": 10850.0,
        })
        now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        loop.fill_pending_entry(state, {}, now, True, False)
        self.assertEqual(state["trades_today"], 3)

    # (k) 決済履行の行に新しい列(fill_method/decision_time/fill_bar_time/decision_price)が入る
    def test_k_history_row_carries_new_fill_columns(self):
        now = datetime(2026, 11, 5, 9, 20, tzinfo=TZ)
        pos = _filled_position(direction="BUY", entry_price=1000.0, tp=1100.0, sl=950.0,
                                fill_bar_time="2026-11-05T09:10:00",
                                decision_time="2026-11-05T09:01:00", decision_price=980.0)
        state = _state(positions=[pos])
        bars = _bars([("2026-11-05 09:10", 995.0, 1000.0, 940.0, 960.0)])
        captured = []
        with patch.object(app, "download_5m", return_value=bars), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(state, now, {"hold_days": 1})
        self.assertEqual(len(captured), 1)
        row = captured[0]
        self.assertEqual(row["fill_method"], "decision_next_bar_open_top1_v1")
        self.assertEqual(row["decision_time"], "2026-11-05T09:01:00")
        self.assertEqual(row["fill_bar_time"], "2026-11-05T09:10:00")
        self.assertEqual(row["decision_price"], 980.0)

    # (l) pending_entryを持つstateはsave/loadを経由しても壊れない
    def test_l_state_round_trips_through_save_load_with_pending_entry(self):
        tmp = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            pending = {
                "ticker": "285A.T", "direction": "SHORT", "decision_date": "2026-11-05",
                "decision_time": "2026-11-05T09:01:00", "fill_bar_time": "2026-11-05T09:05:00",
                "decision_price": 10640.0, "tp": 10000.0, "sl": 10850.0,
            }
            state = _state(pending_entry=copy.deepcopy(pending))
            app.save_state(state)
            with patch.object(app, "discord_send", side_effect=AssertionError("must not notify on clean load")):
                loaded = app.load_state()
            self.assertEqual(loaded["pending_entry"], pending)
            self.assertEqual(loaded["cancelled_entries"], [])
        finally:
            os.chdir(cwd)

    # (m) 連鎖キャンセルバグ再現(2026-11-05実運用): 285A.T保有中(資産の大半を
    # 使用、残り現金61,664円)に割安な7211.T(SHORT 380円)が「100株分の金額だけ
    # なら残り現金で買える」ように見えても、実際のサイジング(open_positions、
    # 総資産基準)では資金不足で開かれない。修正前はペンディング作成→1tick後に
    # CANCELLED_NO_FILL_NOT_OPENEDで取消→即座に次のペンディング作成…の無限
    # ループになり、8tickで15件のDiscord通知・7件のcancelled_entriesを生んだ
    # (f561a05より前は無言で見送られていたケース)。修正後はf561a05より前と同じ
    # 「無言で見送り」に戻る: ペンディングを作らず、Discord通知・取消ともに0件、
    # stateは一切変化しない。
    def test_m_same_tick_pending_chain_reproduction_8_ticks_silent(self):
        held = {
            "ticker": "285A.T", "direction": "SHORT", "entry_date": "2026-11-05",
            "entry_time": "09:05", "entry_price": 10640.0, "tp": 10000.0, "sl": 10850.0,
            "shares": 100, "invested_amount": 1_064_000.0, "score": 80.0,
            "up_probability": 10.0, "down_probability": 85.0, "expected_value_pct": 1.0,
            "company": "285A.T", "buy_reason": "test", "current_price": 10640.0,
            "allocation": 1.0,
        }
        state = _state(capital=1_125_664.22, peak=1_125_664.22, positions=[held],
                        trades_today=1, trades_by_ticker_today={"285A.T": 1})
        before = copy.deepcopy(state)
        candidate = _cand(ticker="7211.T", direction="SHORT", price=380.0, tp=350.0, sl=400.0)

        with patch.object(app, "discord_send", side_effect=AssertionError("Discord通知は0件のはず")):
            for i in range(8):
                now = datetime(2026, 11, 5, 9, 1 + i, tzinfo=TZ)
                filled = loop.fill_pending_entry(state, {}, now, True, False)
                self.assertEqual(filled, [])
                opened = _decide(state, [candidate], now, down_day=True)
                self.assertEqual(opened, [])

        self.assertIsNone(state["pending_entry"])
        self.assertEqual(state["cancelled_entries"], [])
        self.assertEqual(state["positions"], [held])
        self.assertEqual(state, before, "資金不足で開けない候補はstateを一切変化させないはず")

    # (n) リスクポリシーによる新規停止中(日次損失上限)もペンディングを作らない。
    # フラットポジション(保有0件)でも、open_positions側(position_allowed経由)が
    # 拒否する場合は同様にサイレントに見送るべき。
    def test_n_risk_policy_daily_loss_stop_blocks_pending_flat_position(self):
        state = _state(capital=900_000.0, daily_start_capital=1_000_000.0,
                        peak=1_000_000.0, positions=[])
        candidate = _cand(ticker="7203.T", direction="BUY", price=1000.0, tp=1100.0, sl=950.0)
        with patch.object(app, "discord_send", side_effect=AssertionError("Discord通知は0件のはず")):
            opened = _decide(state, [candidate], datetime(2026, 11, 5, 9, 1, tzinfo=TZ),
                              regime="bullish", down_day=False)
        self.assertEqual(opened, [])
        self.assertIsNone(state["pending_entry"])

    # (o) 資金に余裕がある通常ケース: ペンディングは1件だけ作られDiscord通知も
    # 1件だけ。次バーOpenで約定し、約定した同一tick内でopen_top1_onlyを再度
    # 呼んでも(残り現金がほぼ無くなっているため)安い2件目のペンディングは
    # 作られない。
    def test_o_free_cash_normal_case_creates_pending_fills_and_no_second_pending_same_tick(self):
        state = _state(capital=1_000_000.0, peak=1_000_000.0, positions=[])
        candidate1 = _cand(ticker="AAA.T", direction="BUY", price=1000.0, tp=1100.0, sl=950.0)
        discord_calls = []
        decide_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        with patch.object(app, "discord_send", side_effect=lambda msg: discord_calls.append(msg)):
            opened = _decide(state, [candidate1], decide_now, regime="bullish", down_day=False)
        self.assertEqual(opened, [])
        self.assertIsNotNone(state["pending_entry"])
        self.assertEqual(len(discord_calls), 1)

        fill_now = datetime(2026, 11, 5, 9, 25, tzinfo=TZ)
        bars = _bars([("2026-11-05 09:05", 1005.0, 1010.0, 995.0, 1000.0)])
        with patch.object(app, "download_5m", return_value=bars):
            filled = loop.fill_pending_entry(state, {}, fill_now, True, False)
        self.assertEqual(len(filled), 1)
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(len(state["positions"]), 1)

        candidate2 = _cand(ticker="BBB.T", direction="BUY", price=50.0, tp=55.0, sl=47.0)
        with patch.object(app, "discord_send", side_effect=AssertionError("2件目のDiscord通知は無いはず")):
            opened2 = _decide(state, [candidate2], fill_now, regime="bullish", down_day=False)
        self.assertEqual(opened2, [])
        self.assertIsNone(state["pending_entry"])
        self.assertEqual(len(state["positions"]), 1)

    # (p) 小さい既存ポジションがあっても、総資産基準のサイジングで実際に入る余地が
    # ある場合(ジャスト収まる計算)はペンディングを作り、通常どおり約定する
    # (旧挙動の保持)。
    def test_p_small_held_position_second_position_genuinely_fits_still_fills(self):
        held = {"ticker": "ZZZ.T", "invested_amount": 500.0}
        state = _state(capital=1_000_000.0, peak=1_000_000.0, positions=[held])
        candidate = _cand(ticker="CCC.T", direction="BUY", price=333.0, tp=360.0, sl=320.0)
        decide_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        with patch.object(app, "discord_send", side_effect=lambda msg: None):
            opened = _decide(state, [candidate], decide_now, regime="bullish", down_day=False)
        self.assertEqual(opened, [])
        self.assertIsNotNone(
            state["pending_entry"],
            "総資産基準でも実際に入る余地がある場合はペンディングを作るべき(旧挙動)",
        )

        fill_now = datetime(2026, 11, 5, 9, 25, tzinfo=TZ)
        bars = _bars([("2026-11-05 09:05", 333.0, 340.0, 325.0, 335.0)])
        with patch.object(app, "download_5m", return_value=bars):
            filled = loop.fill_pending_entry(state, {}, fill_now, True, False)
        self.assertEqual(len(filled), 1, "資金に余地がある場合の約定(旧挙動)は維持されるべき")
        self.assertEqual(state["positions"][-1]["ticker"], "CCC.T")

    # (q) 発注シミュレーション(probe)自体が例外を投げても、そのtickを落とさず
    # 従来どおりペンディングを作成する(実際の判定はfill_pending_entry側に委ねる)。
    def test_q_probe_exception_does_not_block_pending_or_raise(self):
        state = _state(capital=1_000_000.0, peak=1_000_000.0, positions=[])
        candidate = _cand(ticker="DDD.T", direction="BUY", price=1000.0, tp=1100.0, sl=950.0)
        decide_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        with patch.object(loop, "_original_open", side_effect=RuntimeError("probe boom")), \
             patch.object(app, "discord_send", side_effect=lambda msg: None):
            opened = _decide(state, [candidate], decide_now, regime="bullish", down_day=False)
        self.assertEqual(opened, [])
        self.assertIsNotNone(
            state["pending_entry"],
            "probe例外はtickを落とさずペンディング作成を継続すべき",
        )
        self.assertEqual(state["pending_entry"]["ticker"], "DDD.T")

        fill_now = datetime(2026, 11, 5, 9, 25, tzinfo=TZ)
        bars = _bars([("2026-11-05 09:05", 1000.0, 1010.0, 990.0, 1000.0)])
        with patch.object(app, "download_5m", return_value=bars):
            filled = loop.fill_pending_entry(state, {}, fill_now, True, False)
        self.assertEqual(len(filled), 1, "probe失敗後も実際の約定(_original_open)は正常に行われるべき")

    # (r) probeはstate実体のコピー上で行われ、実際のstate(trades_today・
    # positions・trades_by_ticker_today)を変化させない。
    def test_r_probe_does_not_mutate_real_state(self):
        state = _state(capital=1_000_000.0, peak=1_000_000.0, positions=[],
                        trades_today=2, trades_by_ticker_today={"EEE.T": 1})
        candidate = _cand(ticker="FFF.T", direction="BUY", price=1000.0, tp=1100.0, sl=950.0)
        before_positions = copy.deepcopy(state["positions"])
        before_trades_today = state["trades_today"]
        before_trades_by_ticker = copy.deepcopy(state["trades_by_ticker_today"])
        decide_now = datetime(2026, 11, 5, 9, 1, tzinfo=TZ)
        with patch.object(app, "discord_send", side_effect=lambda msg: None):
            _decide(state, [candidate], decide_now, regime="bullish", down_day=False)
        self.assertIsNotNone(state["pending_entry"])
        self.assertEqual(state["positions"], before_positions)
        self.assertEqual(state["trades_today"], before_trades_today)
        self.assertEqual(state["trades_by_ticker_today"], before_trades_by_ticker)


if __name__ == "__main__":
    unittest.main()
