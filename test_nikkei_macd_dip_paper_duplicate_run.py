"""nikkei_macd_dip_paper.pyの「重複run(2回のworkflow_dispatchが接近して来た
場合や手動runとの重複)でも二重エントリー・二重決済が起きない」ことの回帰
テスト(2026-09-29の監査で要求された検証)。

背景: nikkei-macd-dip-paper.ymlのscheduleトリガーは実運用で4.5〜7.8時間遅延
するため撤廃し、クラウド側ルーチンが本来の5チェックポイント(JST 09:35/
11:15/12:40/14:00/15:15)でworkflow_dispatchする運用に一本化した。この変更に
伴い「2回のdispatchが接近して来た場合(例: 1分差)や手動runとの重複」で
二重にポジションを建ててしまわないかを確認する必要がある。

このワークフローのconcurrencyブロック(group: stock-ai-nikkei-macd-dip-paper,
cancel-in-progress: false)はワークフロー全体(checkout〜state永続化コミット
まで)に対して設定されているため、GitHub Actions側で同一グループのrunは
並走せず必ず直列実行される(2つ目のrunは1つ目が完了してから初めて
checkoutする)。つまり「重複run」は実運用上、常に
  1) run A: 最新mainをcheckout → 判定・売買 → state commit & push
  2) run B: (run A完了後に)最新main(=run Aのcommitを含む)をcheckout →
     判定・売買 → state commit & push
という「同じディレクトリへの状態ファイルを介した逐次呼び出し」に帰着する。
このテストはこれを模擬し、_run()を同一の一時ディレクトリに対して複数回
連続呼び出しして、position/last_exit_dateの状態ガードだけで二重エントリー・
二重決済が防げていることを検証する。

ネットワークI/O(make_nikkei/find_best_candidate/get_current_price)と
時刻(datetime.now)はモックし、実際のyfinance呼び出し・実時刻には依存しない。
adversarial_strategy_validator.py等の禁止モジュールはimportしない。
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime

import pandas as pd

import nikkei_macd_dip_paper as app


class _FixedNow:
    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


# 平日・祝日でない日(2026-09-29は火曜)の場中時刻。is_trading_window()が
# Trueになる必要がある。
FIXED_NOW = datetime(2026, 9, 29, 10, 0, tzinfo=app.TZ)
TODAY_STR = FIXED_NOW.strftime("%Y-%m-%d")

FIXED_SIGNAL = {
    "today_macd": -5.0,
    "today_thr": -3.0,
    "pct_rank": 5.0,
    "signal": True,
    "nikkei_date": "2026-09-26",
}

FIXED_CANDIDATE = {
    "ticker": "TEST.T",
    "company": "テスト",
    "score": 80.0,
    "up_probability": 60.0,
    "down_probability": 20.0,
    "price": 1000.0,
    "atr": 50.0,
}

ENTRY_PRICE = 1000.0


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="nikkei_macd_dip_dup_run_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


class MockNetworkMixin:
    """make_nikkei/nikkei_macd_signal/find_best_candidate/get_current_price
    をネットワーク非依存の固定値に差し替える。"""

    def setUp(self):
        super().setUp()
        self._orig_datetime = app.datetime
        self._orig_make_nikkei = app.make_nikkei
        self._orig_signal = app.nikkei_macd_signal
        self._orig_find_best = app.find_best_candidate
        self._orig_get_price = app.get_current_price

        app.datetime = _FixedNow(FIXED_NOW)
        app.make_nikkei = lambda: pd.DataFrame(
            {"macd": [0.0] * 10},
            index=pd.bdate_range(end="2026-09-25", periods=10),
        )
        app.nikkei_macd_signal = lambda nikkei: dict(FIXED_SIGNAL)
        app.find_best_candidate = lambda nikkei: (dict(FIXED_CANDIDATE), 225, None)
        self._current_price = ENTRY_PRICE
        app.get_current_price = lambda ticker, fallback_close=None: self._current_price

    def tearDown(self):
        app.datetime = self._orig_datetime
        app.make_nikkei = self._orig_make_nikkei
        app.nikkei_macd_signal = self._orig_signal
        app.find_best_candidate = self._orig_find_best
        app.get_current_price = self._orig_get_price
        super().tearDown()

    def set_current_price(self, price):
        self._current_price = price


class DuplicateRunDoesNotDoubleEnterTests(MockNetworkMixin, TmpDirMixin, unittest.TestCase):
    def test_immediate_duplicate_dispatch_does_not_double_enter(self):
        # run A: 最初の(dispatchされた)run。シグナル成立・候補ありのため
        # 3パターンとも新規エントリーするはず。
        app._run()

        states_after_run_a = {cfg["name"]: app.load_state(cfg) for cfg in app.CONFIGS}
        for cfg in app.CONFIGS:
            name = cfg["name"]
            with self.subTest(cfg=name):
                self.assertIsNotNone(states_after_run_a[name]["position"], f"{name}: run Aでエントリーされていない")
                self.assertEqual(states_after_run_a[name]["trades_total"], 1)

        # run B: 直後に来たduplicate dispatch(concurrencyで直列化された後、
        # run Aがcommitした最新state(position保有中)をcheckoutして読む状況を
        # 模擬)。同一プロセス内で状態ファイルはrun Aが書いたものがそのまま
        # ディスクに残っているため、_run()を再度呼ぶだけで「次のjobが最新
        # mainをcheckoutして読む」のと同じ効果になる。
        app._run()

        states_after_run_b = {cfg["name"]: app.load_state(cfg) for cfg in app.CONFIGS}
        for cfg in app.CONFIGS:
            name = cfg["name"]
            with self.subTest(cfg=name):
                # 二重エントリーしていれば trades_total が2になるはずだが、
                # position保有中はneed_entryから除外されるため1のまま。
                self.assertEqual(states_after_run_b[name]["trades_total"], 1, f"{name}: run Bで二重エントリーが発生した")
                self.assertEqual(
                    states_after_run_b[name]["position"]["entry_date"],
                    states_after_run_a[name]["position"]["entry_date"],
                )
                self.assertEqual(
                    states_after_run_b[name]["position"]["entry_price"],
                    states_after_run_a[name]["position"]["entry_price"],
                )

    def test_duplicate_run_after_same_day_exit_does_not_reenter(self):
        # run A: 新規エントリー。
        app._run()
        for cfg in app.CONFIGS:
            self.assertIsNotNone(app.load_state(cfg)["position"])

        # run B: 現在価格が全パターンのTPを上回る(top/mid/bottom全てのTP=
        # entry+atr*tp_mult、最大はbottomのentry+50*3.5=1175)ため、同一run内で
        # 決済される。決済直後にlast_exit_date=todayとなり、同一run内の
        # need_entry判定からも除外されるため、決済したそのrunで即再エントリー
        # はしない。
        self.set_current_price(2000.0)
        app._run()
        states_after_run_b = {cfg["name"]: app.load_state(cfg) for cfg in app.CONFIGS}
        for cfg in app.CONFIGS:
            name = cfg["name"]
            with self.subTest(cfg=name):
                self.assertIsNone(states_after_run_b[name]["position"], f"{name}: run Bで決済されていない")
                self.assertEqual(states_after_run_b[name]["last_exit_date"], TODAY_STR)
                self.assertEqual(states_after_run_b[name]["trades_total"], 1)

        # run C: さらに後続の重複run(例えば手動runが重なった場合)。
        # 当日は既に決済済み(last_exit_date==today)のため、シグナルが
        # 成立していても再エントリーしない。
        self.set_current_price(ENTRY_PRICE)
        app._run()
        states_after_run_c = {cfg["name"]: app.load_state(cfg) for cfg in app.CONFIGS}
        for cfg in app.CONFIGS:
            name = cfg["name"]
            with self.subTest(cfg=name):
                self.assertIsNone(states_after_run_c[name]["position"], f"{name}: run Cで同日中に再エントリーされた")
                self.assertEqual(states_after_run_c[name]["trades_total"], 1)


class TradingWindowGateIsJstAwareTests(unittest.TestCase):
    """(c): is_trading_window()がJST基準で判定していることの確認。

    is_trading_window(now)自体はnow.time()をそのまま比較するだけで、tzを
    内部で変換はしない。「JST基準になる」のは、呼び出し側(_run())が
    TZ=ZoneInfo("Asia/Tokyo")でdatetime.now(TZ)を作ってから渡しているため
    (GitHub Actionsランナーは通常UTC)。そのため、(1)TZ定数が実際に
    Asia/Tokyoであること、(2)_run()がnaiveなdatetime.now()ではなく
    datetime.now(TZ)を使っていること、をソースから直接確認し、あわせて
    UTC真夜中(JST朝の場中)のような、tzを無視すると誤判定になる絶対時刻でも
    JST wall-clockに変換した値を渡せば正しく判定されることを確認する。
    """

    def test_tz_constant_is_asia_tokyo(self):
        self.assertEqual(str(app.TZ), "Asia/Tokyo")

    def test_run_uses_tz_aware_now_not_naive_local_time(self):
        import inspect

        src = inspect.getsource(app._run)
        self.assertIn("datetime.now(TZ)", src, "_run()はnaiveなdatetime.now()ではなくdatetime.now(TZ)を使うべき")

    def test_utc_midnight_which_is_jst_morning_market_hours_is_open(self):
        # UTC 2026-09-29 00:30 = JST 2026-09-29 09:30(場中)。GitHub Actions
        # ランナーの実クロックはUTCのため、TZ変換を行わずnow.time()をUTCの
        # まま比較していたら誤って場中外と判定してしまうケースに相当する。
        from datetime import timezone

        utc_midnight_30 = datetime(2026, 9, 29, 0, 30, tzinfo=timezone.utc)
        jst_equivalent = utc_midnight_30.astimezone(app.TZ)
        self.assertEqual(jst_equivalent.strftime("%H:%M"), "09:30")
        self.assertTrue(app.is_trading_window(jst_equivalent))
        # 変換前のUTC時刻をそのまま(tzだけ剥がさず)渡すと、wall-clockは
        # 00:30のままなので場中外と判定される(=変換の有無で結果が変わる
        # ことの確認)。
        self.assertFalse(app.is_trading_window(utc_midnight_30))

    def test_holiday_is_rejected_even_within_clock_hours(self):
        # 2026-01-01(元日、祝日)はTSE休場日のため、時刻だけ場中でも
        # is_trading_window()はFalseになるはず。
        new_years_day_10am_jst = datetime(2026, 1, 1, 10, 0, tzinfo=app.TZ)
        self.assertFalse(app.is_trading_window(new_years_day_10am_jst))

    def test_weekend_is_rejected(self):
        # 2026-10-03は土曜。
        saturday_10am_jst = datetime(2026, 10, 3, 10, 0, tzinfo=app.TZ)
        self.assertFalse(app.is_trading_window(saturday_10am_jst))


if __name__ == "__main__":
    unittest.main()
