"""未寄り付き(本日約定なし)銘柄を前日終値で約定させないことの回帰テスト。

2026-11-06 2914.T(日本たばこ産業): 終日売り気配のまま場中に一度も値が付かず、
大引けでストップ安(6,269円)比例配分。nikkei_macd_dip_paper.py は09:35の
判定で fast_info.last_price(=前日終値7,769円)を「現在価格」としてmid/bottom
でエントリーしていた(実際には約定し得ない価格)。yfinance 1.4.1の実データ
では、この日の2914.Tの当日5分足は0本(値の付かない足は返らない)だった。

検証対象:
  - nikkei_macd_dip_paper: today_traded_close()/get_current_price()と_run()の
    エントリー・決済の両経路
  - all_candidates_paper: traded_today()とbuild_new_positions()の除外
  - daytrade_tp5000_sl8000_paper: 約定予定バーのOpenがNaNでも落ちない

ネットワーク(yfinance/Discord)はすべてモックし、実データには依存しない。
"""
import os
import shutil
import tempfile
import types
import unittest
from datetime import datetime
from unittest.mock import patch

os.environ.pop("DISCORD_WEBHOOK", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import all_candidates_paper as acp  # noqa: E402
import daytrade_tp5000_sl8000_paper as dt  # noqa: E402
import nikkei_macd_dip_paper as macd  # noqa: E402
import profit_top10_paper as live_p10  # noqa: E402

TZ = macd.TZ

# 2026-11-06(金)は東証営業日。09:35はMACDトラックの1回目のチェックポイント。
NOW_0935 = datetime(2026, 11, 6, 9, 35, tzinfo=TZ)
PREV_CLOSE = 7769.0


def _bars_5m(day, closes, start="09:00", volume=1000, tz=True):
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="5min")
    if tz:
        idx = idx.tz_localize(TZ)
    vols = volume if isinstance(volume, list) else [volume] * len(closes)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": vols},
        index=idx,
    )


def _yesterday_only():
    """2914.T 2026-11-06の再現: 前営業日(11/05)の足だけがあり当日足は0本。"""
    return _bars_5m("2026-11-05", [7748.0, 7750.0], start="15:15")


class _FakeFastInfo:
    def __init__(self, last_price):
        self.last_price = last_price

    def __getitem__(self, key):
        return self.last_price


def _fake_yf(download_df=None, last_price=PREV_CLOSE, download_exc=None, fast_exc=None):
    def download(*_a, **_k):
        if download_exc is not None:
            raise download_exc
        return download_df

    class _Ticker:
        def __init__(self, _ticker):
            if fast_exc is not None:
                raise fast_exc
            self.fast_info = _FakeFastInfo(last_price)

    return types.SimpleNamespace(download=download, Ticker=_Ticker)


# =====================================================================
# nikkei_macd_dip_paper: today_traded_close / get_current_price
# =====================================================================

class MacdTodayTradedClose(unittest.TestCase):
    def setUp(self):
        self._orig_yf = macd.yf

    def tearDown(self):
        macd.yf = self._orig_yf

    def test_yesterday_bars_only_returns_none(self):
        macd.yf = _fake_yf(_yesterday_only())
        self.assertIsNone(macd.today_traded_close("2914.T", now=NOW_0935))

    def test_empty_or_none_download_returns_none(self):
        for df in (None, pd.DataFrame()):
            with self.subTest(df=type(df)):
                macd.yf = _fake_yf(df)
                self.assertIsNone(macd.today_traded_close("2914.T", now=NOW_0935))

    def test_download_exception_returns_none(self):
        macd.yf = _fake_yf(download_exc=RuntimeError("network down"))
        self.assertIsNone(macd.today_traded_close("2914.T", now=NOW_0935))

    def test_today_bars_with_zero_volume_or_nan_close_return_none(self):
        today = _bars_5m("2026-11-06", [7769.0, 7769.0], volume=0)
        nan_bars = _bars_5m("2026-11-06", [np.nan, np.nan])
        for df in (pd.concat([_yesterday_only(), today]), pd.concat([_yesterday_only(), nan_bars])):
            with self.subTest():
                macd.yf = _fake_yf(df)
                self.assertIsNone(macd.today_traded_close("2914.T", now=NOW_0935))

    def test_returns_last_traded_today_close(self):
        today = _bars_5m("2026-11-06", [6500.0, 6400.0, 6300.0], volume=[100, 200, 0])
        macd.yf = _fake_yf(pd.concat([_yesterday_only(), today]))
        # 出来高0の最後の足(6300)は使わず、最後に約定のあった6400
        self.assertEqual(macd.today_traded_close("2914.T", now=NOW_0935), 6400.0)

    def test_utc_naive_index_is_converted_to_jst_date(self):
        # 2026-11-06 00:00 UTC = 09:00 JST(当日)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-06 00:00"), pd.Timestamp("2026-11-06 00:05")])
        df = pd.DataFrame({"Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0],
                           "Close": [1000.0, 1001.0], "Volume": [10, 10]}, index=idx)
        macd.yf = _fake_yf(df)
        self.assertEqual(macd.today_traded_close("X.T", now=NOW_0935), 1001.0)

    def test_multiindex_columns_are_handled(self):
        today = _bars_5m("2026-11-06", [1000.0, 1010.0])
        today.columns = pd.MultiIndex.from_product([today.columns, ["X.T"]])
        macd.yf = _fake_yf(today)
        self.assertEqual(macd.today_traded_close("X.T", now=NOW_0935), 1010.0)


class MacdGetCurrentPrice(unittest.TestCase):
    def setUp(self):
        self._orig_yf = macd.yf

    def tearDown(self):
        macd.yf = self._orig_yf

    def test_untraded_today_returns_none_even_though_fast_info_has_prev_close(self):
        # 2914.T 2026-11-06そのもの: fast_info.last_priceは前日終値7,769を返す
        macd.yf = _fake_yf(_yesterday_only(), last_price=PREV_CLOSE)
        self.assertIsNone(macd.get_current_price("2914.T", fallback_close=PREV_CLOSE, now=NOW_0935))

    def test_traded_today_uses_fast_info_last_price(self):
        today = _bars_5m("2026-11-06", [1000.0, 1010.0])
        macd.yf = _fake_yf(today, last_price=1012.0)
        self.assertEqual(macd.get_current_price("X.T", now=NOW_0935), 1012.0)

    def test_traded_today_falls_back_to_today_close_when_fast_info_fails(self):
        today = _bars_5m("2026-11-06", [1000.0, 1010.0])
        for kwargs in ({"fast_exc": RuntimeError("x")}, {"last_price": float("nan")}, {"last_price": 0.0}):
            with self.subTest(kwargs=kwargs):
                macd.yf = _fake_yf(today, **kwargs)
                self.assertEqual(macd.get_current_price("X.T", fallback_close=999.0, now=NOW_0935), 1010.0)

    def test_never_falls_back_to_fallback_close(self):
        macd.yf = _fake_yf(None, fast_exc=RuntimeError("x"))
        self.assertIsNone(macd.get_current_price("X.T", fallback_close=PREV_CLOSE, now=NOW_0935))


# =====================================================================
# nikkei_macd_dip_paper: _run()(エントリー/決済の両経路)
# =====================================================================

class _FixedNow:
    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


SIGNAL = {"today_macd": -969.4, "today_thr": -900.0, "pct_rank": 2.38, "signal": True,
          "nikkei_date": "2026-11-05"}
JT_CANDIDATE = {"ticker": "2914.T", "company": "日本たばこ産業", "score": 75.8,
                "up_probability": 51.6, "down_probability": 26.6, "price": PREV_CLOSE, "atr": 165.6}


class MacdRunDoesNotTradeUntradedStock(unittest.TestCase):
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="macd_untraded_")
        os.chdir(self.tmp)
        self._saved = {k: getattr(macd, k) for k in
                       ("datetime", "make_nikkei", "nikkei_macd_signal", "find_best_candidate", "yf", "WEBHOOK_URL")}
        macd.WEBHOOK_URL = None
        macd.datetime = _FixedNow(NOW_0935)
        macd.make_nikkei = lambda: pd.DataFrame({"macd": [0.0] * 10},
                                                index=pd.bdate_range(end="2026-11-05", periods=10))
        macd.nikkei_macd_signal = lambda nikkei: dict(SIGNAL)
        macd.find_best_candidate = lambda nikkei: (dict(JT_CANDIDATE), 225, None)
        self.sent = []
        self._send_patch = patch.object(macd, "send", side_effect=lambda m: self.sent.append(str(m)))
        self._send_patch.start()

    def tearDown(self):
        self._send_patch.stop()
        for k, v in self._saved.items():
            setattr(macd, k, v)
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _states(self):
        return {cfg["name"]: macd.load_state(cfg) for cfg in macd.CONFIGS}

    def test_jt_20261106_replay_no_entry_in_any_config(self):
        macd.yf = _fake_yf(_yesterday_only(), last_price=PREV_CLOSE)
        macd._run()
        for name, s in self._states().items():
            with self.subTest(cfg=name):
                self.assertIsNone(s["position"])
                self.assertEqual(s["trades_total"], 0)
        self.assertTrue(any("未寄り付き" in m and "2914.T" in m for m in self.sent))

    def test_entry_happens_on_later_run_once_stock_has_traded(self):
        macd.yf = _fake_yf(_yesterday_only(), last_price=PREV_CLOSE)
        macd._run()
        later = datetime(2026, 11, 6, 11, 15, tzinfo=TZ)
        macd.datetime = _FixedNow(later)
        today = _bars_5m("2026-11-06", [6300.0, 6280.0], start="10:30")
        macd.yf = _fake_yf(pd.concat([_yesterday_only(), today]), last_price=6275.0)
        macd._run()
        for name, s in self._states().items():
            with self.subTest(cfg=name):
                self.assertIsNotNone(s["position"])
                self.assertEqual(s["position"]["entry_price"], 6275.0)
                self.assertEqual(s["trades_total"], 1)

    def test_held_position_is_not_closed_at_stale_price_when_untraded(self):
        # 保有中の銘柄が当日未寄り付き: 前日終値(=entry)でHOLD_LIMIT等の決済をしない
        for cfg in macd.CONFIGS:
            s = macd.default_state(cfg)
            s["position"] = {
                "ticker": "2914.T", "company": "日本たばこ産業", "entry_date": "2026-11-02",
                "entry_price": PREV_CLOSE, "shares": 100, "invested_amount": PREV_CLOSE * 100,
                "tp": PREV_CLOSE - 1.0, "sl": 1.0, "atr": 165.6, "score": 75.8,
                "up_probability": 51.6, "down_probability": 26.6,
            }
            macd.save_state(cfg, s)
        macd.yf = _fake_yf(_yesterday_only(), last_price=PREV_CLOSE)
        macd._run()
        for name, s in self._states().items():
            with self.subTest(cfg=name):
                self.assertIsNotNone(s["position"], f"{name}: 約定し得ない価格で決済された")
                self.assertIsNone(s.get("last_exit_date"))
            self.assertFalse(os.path.exists(f"nikkei_macd_dip_paper_history_{name}.csv"))


# =====================================================================
# all_candidates_paper: traded_today / build_new_positions
# =====================================================================

def _naive_jst(df):
    """acp.download_5mが返す形(JSTのtz-naive index)に揃える。"""
    df = df.copy()
    df.index = df.index.tz_convert(TZ).tz_localize(None)
    return df


class AllCandidatesTradedToday(unittest.TestCase):
    NOW_1605 = datetime(2026, 11, 6, 16, 5, tzinfo=TZ)

    def test_false_for_yesterday_only_none_and_zero_volume(self):
        cases = {
            "yesterday_only": _naive_jst(_yesterday_only()),
            "none": None,
            "empty": pd.DataFrame(),
            "zero_volume": _naive_jst(_bars_5m("2026-11-06", [7769.0], volume=0)),
            "nan_close": _naive_jst(_bars_5m("2026-11-06", [np.nan])),
        }
        for name, df in cases.items():
            with self.subTest(case=name):
                self.assertFalse(acp.traded_today("2914.T", self.NOW_1605, download_5m_fn=lambda t, d=df: d))

    def test_true_when_today_has_a_traded_bar(self):
        df = _naive_jst(pd.concat([_yesterday_only(), _bars_5m("2026-11-06", [1612.0], start="15:20")]))
        self.assertTrue(acp.traded_today("9005.T", self.NOW_1605, download_5m_fn=lambda t: df))


class AllCandidatesBuildNewPositionsSkipsUntraded(unittest.TestCase):
    def _cand(self, ticker, price):
        return {"ticker": ticker, "direction": "SHORT", "price": price, "tp": price * 0.9, "sl": price * 1.05,
                "score": 70.0, "up_probability": 20.0, "down_probability": 40.0, "data_date": "2026-11-06"}

    def test_untraded_ticker_skipped_and_other_ranks_unchanged(self):
        cands = [self._cand("2914.T", 7750.0), self._cand("9005.T", 1612.0), self._cand("8795.T", 4611.0)]
        policy = {"nikkei_filter": False, "hold_days": 1}
        positions = acp.build_new_positions(set(), cands, "2026-11-06", policy, "f.json", "h", False,
                                            untraded_tickers={"2914.T"})
        self.assertEqual([p["ticker"] for p in positions], ["9005.T", "8795.T"])
        self.assertEqual([p["rank"] for p in positions], [2, 3])

    def test_default_keeps_all(self):
        cands = [self._cand("2914.T", 7750.0), self._cand("9005.T", 1612.0)]
        policy = {"nikkei_filter": False, "hold_days": 1}
        positions = acp.build_new_positions(set(), cands, "2026-11-06", policy, "f.json", "h", False)
        self.assertEqual(len(positions), 2)


# =====================================================================
# daytrade: 約定予定バーのOpenがNaNでもtickが落ちない
# =====================================================================

class DaytradeNanOpenFillBar(unittest.TestCase):
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        dt._DAILY_BARS_CACHE.clear()
        self._send_patch = patch.object(dt, "discord_send")
        self._send_patch.start()

    def tearDown(self):
        self._send_patch.stop()
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()

    def _pending(self):
        decision_time = pd.Timestamp("2026-11-06 09:56:00")
        return {
            "ticker": "2914.T", "direction": "SHORT", "decision_date": "2026-11-06",
            "decision_time": decision_time.isoformat(),
            "fill_bar_time": dt._ceil_bar_time(decision_time).isoformat(),
            "score": 70.0, "up_probability": 20.0, "down_probability": 40.0, "market_regime": "bearish",
            "top10_rank": 1, "policy_file": "strategy_policy.json", "policy_source": "x",
            "policy_hash": "abc", "model_id": None, "model_version": "v", "budget": 1_000_000.0,
            "entry_window_start": "09:55",
        }

    def _nan_bar(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-06 10:00:00")])
        return pd.DataFrame({"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
                             "Volume": [0]}, index=idx)

    def test_nan_open_keeps_pending_without_exception(self):
        state = dt.default_state()
        state["pending"] = self._pending()
        with patch.object(live_p10, "download_5m", return_value=self._nan_bar()):
            msgs = dt._check_pending_fill(state, datetime(2026, 11, 6, 10, 16, tzinfo=TZ), "2026-11-06")
        self.assertEqual(msgs, [])
        self.assertIsNotNone(state["pending"])
        self.assertEqual(state["positions"], [])

    def test_nan_open_cancelled_at_forced_exit_time(self):
        state = dt.default_state()
        state["pending"] = self._pending()
        with patch.object(live_p10, "download_5m", return_value=self._nan_bar()):
            msgs = dt._check_pending_fill(state, datetime(2026, 11, 6, 15, 20, tzinfo=TZ), "2026-11-06")
        self.assertEqual(msgs, [])
        self.assertIsNone(state["pending"])
        self.assertEqual(state["positions"], [])

    def test_no_today_bar_at_all_stays_pending(self):
        # 2914.T型(当日足0本): 既存仕様どおり約定せずペンディング維持
        state = dt.default_state()
        state["pending"] = self._pending()
        with patch.object(live_p10, "download_5m", return_value=_naive_jst(_yesterday_only())):
            msgs = dt._check_pending_fill(state, datetime(2026, 11, 6, 10, 16, tzinfo=TZ), "2026-11-06")
        self.assertEqual(msgs, [])
        self.assertIsNotNone(state["pending"])
        self.assertEqual(state["positions"], [])


if __name__ == "__main__":
    unittest.main()
