"""live_model_policy_validation.py のテスト。

このファイルは実ネットワーク呼び出し(yfinance/gh CLI)を一切行わない。
daily_directional_top1.download(/profit_top10_paper.download経由も含む)・
load_model・download_5mをすべてモンキーパッチし、決定論的な合成OHLCVデータと
スタブモデルで検証する。

最重要テスト(PoolParityWithLiveScanTests): live_model_policy_validation.
build_candidate_pool_for_date()のok条件・ランキングが、profit_top10_paper.scan()
(実コード、モンキーパッチしたdownload/load_model/download_5mのみ差し替え)と
完全に一致することを、合成データで直接比較して証明する。
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import daily_directional_top1 as trader
import daily_model_retrain as dmr
import live_model_policy_validation as m
import profit_top10_paper as live_p10

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_pristine_profit_top10_paper():
    """profit_top10_paper.pyをsys.modulesキャッシュとは別の、独立した新しい
    モジュールインスタンスとして読み込む。

    ★背景: run_profit_loop.py はimport時にモジュールレベルの副作用として
    `app.scan = scan_candidates_fixed`(app=profit_top10_paper)を実行し、
    キャッシュ済みのprofit_top10_paper.scanをプロセス全体で恒久的に
    差し替える(関数シグネチャも変わる)。他のテストファイル
    (test_run_profit_loop_cooldown.py)がこのモジュールをimportすると、
    pytest全体のセッション内でlive_p10.scanが「本当のscan()」ではなく
    なってしまい、どのテストファイルが先に収集されるか(アルファベット順等)
    に依存して本テストが落ちたり通ったりする(収集順フラキネス)。
    importlib経由で別インスタンスとして読み直すことで、この汚染と無関係に
    常に本物のscan()関数本体を取得する(かつ他のテストファイルの状態を
    一切変更しない)。
    """
    spec = importlib.util.spec_from_file_location(
        "profit_top10_paper_pristine_for_tests", os.path.join(REPO_ROOT, "profit_top10_paper.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRISTINE_P10 = _load_pristine_profit_top10_paper()

FEATURES = trader.FEATURES
FAKE_TICKERS = ["AAAA.T", "BBBB.T", "CCCC.T", "DDDD.T", "EEEE.T", "FFFF.T", "ZZZZ.T"]
N_DAYS = 280
END_DATE = "2026-09-30"


# =====================================================================
# 合成データ生成(決定論的)
# =====================================================================

def _date_index(n=N_DAYS, end=END_DATE):
    return pd.bdate_range(end=end, periods=n)


def _make_ohlcv(seed, n=N_DAYS, end=END_DATE, start_price=1000.0, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = _date_index(n, end)
    rets = drift + rng.normal(0, 0.012, n)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    volume = (1_000_000 + rng.normal(0, 50_000, n)).clip(min=100_000)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx,
    )


def _make_trending_ohlcv(n=N_DAYS, end=END_DATE, start_price=1000.0):
    """強い上昇トレンド(高RSI)になるよう、ほぼノイズ無しで増加させる銘柄
    (ZZZZ.T専用)。directional_score/stub modelの両方で一貫してBUY最有力候補に
    なることを期待する。6日に1回だけ小さな下げ日を混ぜる(down_day_volume_bias等
    の「下げ日ローリング」特徴量が、直近10日以内に下げ日が1つも無くNaNのままに
    なる=銘柄が丸ごと候補から脱落する、という合成データ特有の罠を避けるため)。"""
    idx = _date_index(n, end)
    t = np.arange(n)
    rets = np.where(t % 6 == 5, -0.0015, 0.0025)
    close = start_price * np.cumprod(1.0 + rets)
    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = prev_close * 1.0005
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = np.full(n, 1_200_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx,
    )


def _make_index_close(n=N_DAYS, end=END_DATE, start_price=30000.0, drift=0.0002, seed=999):
    rng = np.random.default_rng(seed)
    idx = _date_index(n, end)
    rets = drift + rng.normal(0, 0.006, n)
    close = start_price * np.cumprod(1.0 + rets)
    high = close * 1.002
    low = close * 0.998
    open_ = close * 1.0005
    volume = np.full(n, 500_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx,
    )


def _build_fixture_frames(tickers=FAKE_TICKERS):
    # ★注意: Python組み込みのhash(str)はプロセスごとにランダム化される
    # (PYTHONHASHSEED)ため、シードに使うと実行のたびに合成データ自体が
    # 変わってしまい、再現性が無くなる(=テストがときどき理由不明で
    # 失敗/成功する)。固定の列挙インデックスだけをシードにする。
    frames = {t: _make_ohlcv(seed=10 + i) for i, t in enumerate(tickers) if t != "ZZZZ.T"}
    if "ZZZZ.T" in tickers:
        frames["ZZZZ.T"] = _make_trending_ohlcv()
    nikkei_src = _make_index_close(seed=111)
    topix_src = _make_index_close(seed=222, start_price=2000.0)
    futures_src = _make_index_close(seed=333, start_price=30000.0)
    return frames, nikkei_src, topix_src, futures_src


class StubModel:
    """rsi特徴量だけに依存する決定論的なpredict_proba。1行呼び出し(live scan())
    と複数行バッチ呼び出し(本モジュールのbuild_candidate_pool_for_date)の両方で、
    行ごとに完全に同じ結果を返す(=行の並び順・バッチサイズに依存しない)。"""

    classes_ = np.array([0, 1, 2])
    feature_names_in_ = np.array(FEATURES)

    def predict_proba(self, X):
        rsi = pd.Series(X["rsi"]).to_numpy(dtype=float)
        up = np.clip((rsi - 30.0) / 70.0, 0.03, 0.94)
        down = np.clip(1.0 - up, 0.03, 0.94)
        scale = 0.92 / np.maximum(up + down, 1e-9)
        up2, down2 = up * scale, down * scale
        flat2 = 1.0 - up2 - down2
        return np.stack([down2, flat2, up2], axis=1)


def _patched_download(frames, nikkei_src, topix_src, futures_src):
    def _download(ticker, period="3y"):
        if ticker == "^N225":
            return nikkei_src.copy()
        if ticker == trader.TOPIX_PROXY:
            return topix_src.copy()
        if ticker == "NIY=F":
            return futures_src.copy()
        if ticker in frames:
            return frames[ticker].copy()
        return None
    return _download


class FixtureMixin:
    """合成データ・スタブモデルでtrader/profit_top10_paperのネットワーク依存
    関数をすべてモンキーパッチする。"""

    def setUp(self):
        self.frames, self.nikkei_src, self.topix_src, self.futures_src = _build_fixture_frames()
        self.model = StubModel()
        self._download = _patched_download(self.frames, self.nikkei_src, self.topix_src, self.futures_src)

        trader._FUTURES_FEATURE_CACHE = None
        trader._TOPIX_FEATURE_CACHE = None

        self._patches = [
            patch.object(trader, "download", side_effect=self._download),
            patch.object(live_p10, "download", side_effect=self._download),
            patch.object(live_p10, "TICKERS", FAKE_TICKERS),
            patch.object(live_p10, "load_model", return_value=self.model),
            patch.object(live_p10, "download_5m", return_value=None),
            # PRISTINE_P10はlive_p10と同じソースから読み込んだ別インスタンス
            # (run_profit_loop.pyのscan差し替えから隔離されている)なので、
            # 同じモック群を独立して当てる。
            patch.object(PRISTINE_P10, "download", side_effect=self._download),
            patch.object(PRISTINE_P10, "TICKERS", FAKE_TICKERS),
            patch.object(PRISTINE_P10, "load_model", return_value=self.model),
            patch.object(PRISTINE_P10, "download_5m", return_value=None),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

        self.nikkei = trader.make_nikkei()
        self.ticker_frames = {}
        for t in FAKE_TICKERS:
            x = dmr.build_ticker_frame(t, self.nikkei, futures_df=None)
            self.assertIsNotNone(x, msg=f"{t}: フィクスチャ生成に失敗")
            self.ticker_frames[t] = x
        all_dates = sorted(set().union(*[set(x.index) for x in self.ticker_frames.values()]))
        self.all_dates_sorted = pd.DatetimeIndex(all_dates)
        self.nikkei_ff = self.nikkei.reindex(all_dates).ffill()
        self.last_date = self.all_dates_sorted[-1]

    def base_policy(self, **overrides):
        p = {
            "up_threshold": 20.0, "min_score_for_buy": 0.0, "nikkei_filter": False,
            "atr_tp_multiplier": 4.0, "atr_sl_multiplier": 1.0, "hold_days": 1,
        }
        p.update(overrides)
        return p


# =====================================================================
# 最重要: build_candidate_pool_for_date() と live scan() の完全一致
# =====================================================================

class PoolParityWithLiveScanTests(FixtureMixin, unittest.TestCase):
    def test_pool_matches_live_scan_when_candidates_pass(self):
        policy = self.base_policy(up_threshold=20.0, min_score_for_buy=0.0, nikkei_filter=False)
        live_pool, scanned = PRISTINE_P10.scan(policy, limit=None)
        self.assertGreater(scanned, 0)

        mine_pool, cand, fallback = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        self.assertTrue(cand, "このフィクスチャ・閾値ではok条件通過候補が出るはず")
        self.assertEqual(len(live_pool), len(mine_pool))
        for live_item, mine_item in zip(live_pool, mine_pool):
            self.assertEqual(live_item["ticker"], mine_item["ticker"])
            self.assertEqual(live_item["direction"], mine_item["direction"])
            self.assertAlmostEqual(live_item["score"], mine_item["score"], places=9)
            self.assertAlmostEqual(live_item["tp"], mine_item["tp"], places=6)
            self.assertAlmostEqual(live_item["sl"], mine_item["sl"], places=6)
            self.assertAlmostEqual(live_item["expected_value_pct"], mine_item["expected_value_pct"], places=9)
            self.assertAlmostEqual(live_item["up_probability"], mine_item["up_probability"], places=9)
            self.assertAlmostEqual(live_item["down_probability"], mine_item["down_probability"], places=9)

    def test_pool_falls_back_to_unfiltered_ranking_when_nothing_passes(self):
        # up_threshold99.9%は誰も通らない → scan()自身の'pool = cand or fallback'により
        # フィルター無視の参考順位にフォールバックする、という実挙動を再現できているか。
        policy = self.base_policy(up_threshold=99.9, min_score_for_buy=99.9, nikkei_filter=False)
        live_pool, scanned = PRISTINE_P10.scan(policy, limit=None)
        mine_pool, cand, fallback = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        self.assertEqual(cand, [])
        self.assertEqual(len(live_pool), len(mine_pool))
        self.assertEqual(len(live_pool), scanned * 2)
        for live_item, mine_item in zip(live_pool, mine_pool):
            self.assertEqual(live_item["ticker"], mine_item["ticker"])
            self.assertEqual(live_item["direction"], mine_item["direction"])

    def test_pool_respects_nikkei_filter_like_live_scan(self):
        policy = self.base_policy(up_threshold=20.0, min_score_for_buy=0.0, nikkei_filter=True)
        live_pool, _ = PRISTINE_P10.scan(policy, limit=None)
        mine_pool, _, _ = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        self.assertEqual([(c["ticker"], c["direction"]) for c in live_pool],
                          [(c["ticker"], c["direction"]) for c in mine_pool])

    def test_short_disabled_matches_live_scan(self):
        policy = self.base_policy(up_threshold=20.0, min_score_for_buy=0.0)
        with patch.object(PRISTINE_P10, "SHORT_ENABLED", False):
            live_pool, _ = PRISTINE_P10.scan(policy, limit=None)
            mine_pool, _, _ = m.build_candidate_pool_for_date(
                self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy, short_enabled=False,
            )
        self.assertTrue(all(c["direction"] == "BUY" for c in live_pool))
        self.assertEqual([(c["ticker"], c["direction"]) for c in live_pool],
                          [(c["ticker"], c["direction"]) for c in mine_pool])

    def test_trending_ticker_is_a_buy_candidate(self):
        policy = self.base_policy(up_threshold=20.0, min_score_for_buy=0.0)
        pool, cand, _ = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        zzzz = [c for c in cand if c["ticker"] == "ZZZZ.T"]
        self.assertTrue(zzzz, "強い上昇トレンド銘柄はok条件を通過するはず")
        self.assertEqual(zzzz[0]["direction"], "BUY")


# =====================================================================
# policy選択(futures_trendに応じたfrozen policy切替)
# =====================================================================

class PolicySelectionTests(unittest.TestCase):
    def test_up_trend_selects_up_policy_file(self):
        self.assertEqual(m.live_policy_file_for_trend("up"), live_p10.POLICY_FILE_UP)

    def test_down_trend_falls_back_to_normal_since_down_file_absent(self):
        self.assertFalse(os.path.exists(live_p10.POLICY_FILE_DOWN))
        self.assertEqual(m.live_policy_file_for_trend("down"), live_p10.POLICY_FILE)

    def test_frozen_policy_for_up_trend_is_frozen_up(self):
        frozen_file, policy = m.frozen_policy_for_trend("up")
        self.assertEqual(frozen_file, "all_candidates_frozen_policy_up.json")
        self.assertEqual(policy["hold_days"], 3)

    def test_frozen_policy_for_down_trend_is_frozen_normal(self):
        frozen_file, policy = m.frozen_policy_for_trend("down")
        self.assertEqual(frozen_file, "all_candidates_frozen_policy.json")
        self.assertEqual(policy["hold_days"], 1)

    def test_purge_margin_is_twice_the_larger_hold_days(self):
        self.assertEqual(m.purge_margin_days(), 6)

    def test_trend_for_date_uses_prior_value_when_date_missing(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-10")])
        series = pd.DataFrame({"trend": ["down", "up"]}, index=idx)
        self.assertEqual(m.trend_for_date(series, pd.Timestamp("2026-01-07")), "down")
        self.assertEqual(m.trend_for_date(series, pd.Timestamp("2026-01-10")), "up")

    def test_trend_for_date_before_series_start_uses_default(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
        series = pd.DataFrame({"trend": ["up"]}, index=idx)
        self.assertEqual(m.trend_for_date(series, pd.Timestamp("2025-01-01"), default="down"), "down")

    def test_trend_for_date_empty_series_uses_default(self):
        self.assertEqual(m.trend_for_date(pd.DataFrame(), pd.Timestamp("2026-01-01"), default="up"), "up")


# =====================================================================
# エントリー規約
# =====================================================================

class ResolveEntryTests(FixtureMixin, unittest.TestCase):
    def test_same_close_uses_decision_day_close_and_original_tp_sl(self):
        policy = self.base_policy()
        pool, cand, _ = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        c = pool[0]
        entry = m.resolve_entry(c, self.ticker_frames, self.last_date, self.all_dates_sorted, "same_close", policy)
        self.assertEqual(entry["entry_date"], self.last_date)
        self.assertAlmostEqual(entry["entry_price"], c["price"], places=6)
        self.assertAlmostEqual(entry["tp"], c["tp"], places=6)
        self.assertFalse(entry["include_entry_bar"])

    def test_next_open_uses_next_bar_open_and_recomputes_tp_sl(self):
        policy = self.base_policy()
        mid_date = self.all_dates_sorted[-10]
        pool, cand, _ = m.build_candidate_pool_for_date(
            mid_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        c = pool[0]
        entry = m.resolve_entry(c, self.ticker_frames, mid_date, self.all_dates_sorted, "next_open", policy)
        xf = self.ticker_frames[c["ticker"]]
        expected_next = self.all_dates_sorted[self.all_dates_sorted.get_loc(mid_date) + 1]
        self.assertEqual(entry["entry_date"], expected_next)
        self.assertAlmostEqual(entry["entry_price"], float(xf.loc[expected_next, "Open"]), places=6)
        self.assertTrue(entry["include_entry_bar"])
        expected_tp, expected_sl = live_p10._tp_sl(entry["entry_price"], c["direction"], c["atr_abs"], policy)
        self.assertAlmostEqual(entry["tp"], expected_tp, places=6)
        self.assertAlmostEqual(entry["sl"], expected_sl, places=6)

    def test_next_open_returns_none_at_end_of_data(self):
        policy = self.base_policy()
        pool, cand, _ = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        entry = m.resolve_entry(pool[0], self.ticker_frames, self.last_date, self.all_dates_sorted, "next_open", policy)
        self.assertIsNone(entry)

    def test_same_close_returns_none_if_ticker_missing_that_date(self):
        policy = self.base_policy()
        fake_candidate = {"ticker": "NOPE.T", "direction": "BUY", "price": 100.0, "tp": 110.0,
                           "sl": 95.0, "atr_abs": 5.0}
        entry = m.resolve_entry(fake_candidate, self.ticker_frames, self.last_date, self.all_dates_sorted,
                                 "same_close", policy)
        self.assertIsNone(entry)


# =====================================================================
# 決済シミュレーション(日足ベース)
# =====================================================================

class SimulateTradeExitTests(unittest.TestCase):
    def _xf(self, rows):
        idx = pd.bdate_range("2026-01-05", periods=len(rows))
        return pd.DataFrame(rows, index=idx)

    def test_tp_hit(self):
        xf = self._xf([
            {"High": 105, "Low": 95, "Close": 100},
            {"High": 120, "Low": 108, "Close": 115},
        ])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=115.0, sl=90.0, direction="BUY",
                                   hold_days=5, fee_rate_pct=0.0, include_entry_bar=False)
        self.assertEqual(r["reason"], "TP")
        self.assertAlmostEqual(r["exit_price"], 115.0)
        self.assertAlmostEqual(r["return_pct"], 15.0, places=6)

    def test_sl_hit(self):
        xf = self._xf([
            {"High": 105, "Low": 95, "Close": 100},
            {"High": 108, "Low": 85, "Close": 90},
        ])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=120.0, sl=90.0, direction="BUY",
                                   hold_days=5, fee_rate_pct=0.0, include_entry_bar=False)
        self.assertEqual(r["reason"], "SL")
        self.assertAlmostEqual(r["exit_price"], 90.0)

    def test_both_touched_same_bar_ties_to_sl(self):
        xf = self._xf([
            {"High": 105, "Low": 95, "Close": 100},
            {"High": 125, "Low": 85, "Close": 100},
        ])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=120.0, sl=90.0, direction="BUY",
                                   hold_days=5, fee_rate_pct=0.0, include_entry_bar=False)
        self.assertEqual(r["reason"], "SL_BOTH")
        self.assertAlmostEqual(r["exit_price"], 90.0)

    def test_hold_days_limit_exits_at_close(self):
        xf = self._xf([
            {"High": 102, "Low": 99, "Close": 101},
            {"High": 103, "Low": 100, "Close": 102},
            {"High": 104, "Low": 101, "Close": 103},
        ])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=150.0, sl=50.0, direction="BUY",
                                   hold_days=2, fee_rate_pct=0.0, include_entry_bar=False)
        self.assertEqual(r["reason"], "TIME")
        self.assertAlmostEqual(r["exit_price"], 103.0)
        self.assertEqual(r["days_held"], 2)

    def test_forced_eos_when_data_runs_out(self):
        xf = self._xf([{"High": 102, "Low": 99, "Close": 101}])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=150.0, sl=50.0, direction="BUY",
                                   hold_days=10, fee_rate_pct=0.0, include_entry_bar=True)
        self.assertEqual(r["reason"], "FORCED_EOS")
        self.assertAlmostEqual(r["exit_price"], 101.0)

    def test_fee_reduces_return(self):
        xf = self._xf([{"High": 120, "Low": 95, "Close": 100}])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=110.0, sl=90.0, direction="BUY",
                                   hold_days=5, fee_rate_pct=0.5, include_entry_bar=True)
        self.assertEqual(r["reason"], "TP")
        self.assertAlmostEqual(r["return_pct"], 10.0 - 0.5, places=6)

    def test_short_direction_tp_sl_logic(self):
        xf = self._xf([
            {"High": 105, "Low": 95, "Close": 100},
            {"High": 102, "Low": 80, "Close": 85},
        ])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=85.0, sl=110.0, direction="SHORT",
                                   hold_days=5, fee_rate_pct=0.0, include_entry_bar=False)
        self.assertEqual(r["reason"], "TP")
        self.assertAlmostEqual(r["return_pct"], (100.0 / 85.0 - 1.0) * 100.0, places=6)

    def test_include_entry_bar_allows_same_day_exit(self):
        xf = self._xf([{"High": 130, "Low": 95, "Close": 110}])
        r = m.simulate_trade_exit(xf, xf.index[0], 100.0, tp=120.0, sl=90.0, direction="BUY",
                                   hold_days=5, fee_rate_pct=0.0, include_entry_bar=True)
        self.assertEqual(r["reason"], "TP")
        self.assertEqual(r["days_held"], 1)


# =====================================================================
# 指標計算
# =====================================================================

class ComputeFullMetricsTests(unittest.TestCase):
    def test_empty_trades(self):
        r = m.compute_full_metrics([])
        self.assertEqual(r["trades"], 0)
        self.assertEqual(r["pf"], 0.0)

    def test_pf_and_win_rate(self):
        trades = [
            {"return_pct": 10.0, "exit_date": "2026-01-05"},
            {"return_pct": -5.0, "exit_date": "2026-01-06"},
            {"return_pct": 10.0, "exit_date": "2026-01-07"},
        ]
        r = m.compute_full_metrics(trades)
        self.assertEqual(r["trades"], 3)
        self.assertAlmostEqual(r["pf"], 20.0 / 5.0, places=6)
        self.assertAlmostEqual(r["win_rate"], 200.0 / 3.0, places=6)
        self.assertAlmostEqual(r["avg_return_pct"], 5.0, places=6)

    def test_all_losses_pf_is_zero_not_crash(self):
        trades = [{"return_pct": -5.0, "exit_date": "2026-01-05"}, {"return_pct": -3.0, "exit_date": "2026-01-06"}]
        r = m.compute_full_metrics(trades)
        self.assertEqual(r["pf"], 0.0)

    def test_all_wins_pf_is_inf(self):
        trades = [{"return_pct": 5.0, "exit_date": "2026-01-05"}, {"return_pct": 3.0, "exit_date": "2026-01-06"}]
        r = m.compute_full_metrics(trades)
        self.assertEqual(r["pf"], float("inf"))

    def test_max_drawdown(self):
        trades = [
            {"return_pct": 10.0, "exit_date": "2026-01-05"},
            {"return_pct": -50.0, "exit_date": "2026-01-06"},
            {"return_pct": 10.0, "exit_date": "2026-01-07"},
        ]
        r = m.compute_full_metrics(trades)
        # capital: 1.10 -> peak 1.10 -> *0.5=0.55 (dd=-50%) -> *1.1=0.605
        self.assertAlmostEqual(r["max_dd_pct"], -50.0, places=3)

    def test_longest_losing_streak(self):
        trades = [
            {"return_pct": -1.0, "exit_date": "2026-01-01"},
            {"return_pct": -1.0, "exit_date": "2026-01-02"},
            {"return_pct": 1.0, "exit_date": "2026-01-03"},
            {"return_pct": -1.0, "exit_date": "2026-01-04"},
            {"return_pct": -1.0, "exit_date": "2026-01-05"},
            {"return_pct": -1.0, "exit_date": "2026-01-06"},
        ]
        r = m.compute_full_metrics(trades)
        self.assertEqual(r["longest_losing_streak"], 3)

    def test_monthly_aggregation(self):
        trades = [
            {"return_pct": 5.0, "exit_date": "2026-01-05"},
            {"return_pct": 5.0, "exit_date": "2026-01-20"},
            {"return_pct": -20.0, "exit_date": "2026-02-10"},
        ]
        r = m.compute_full_metrics(trades)
        self.assertAlmostEqual(r["avg_month_return_pct"], (10.0 + (-20.0)) / 2.0, places=6)
        self.assertAlmostEqual(r["pct_months_positive"], 50.0, places=6)


class ComputeEqualWeightMetricsTests(unittest.TestCase):
    def test_empty(self):
        r = m.compute_equal_weight_metrics([])
        self.assertEqual(r["trades"], 0)

    def test_basic_pnl(self):
        trades = [
            {"return_pct": 10.0, "pnl_jpy": 1000.0, "exit_date": "2026-01-05"},
            {"return_pct": -5.0, "pnl_jpy": -500.0, "exit_date": "2026-01-06"},
        ]
        r = m.compute_equal_weight_metrics(trades)
        self.assertEqual(r["trades"], 2)
        self.assertAlmostEqual(r["pf"], 1000.0 / 500.0, places=6)
        self.assertAlmostEqual(r["total_pnl_jpy"], 500.0, places=6)
        self.assertAlmostEqual(r["win_rate"], 50.0, places=6)


class EquityCurveTests(unittest.TestCase):
    def test_compounding_and_drawdown(self):
        trades = [
            {"return_pct": 10.0, "exit_date": pd.Timestamp("2026-01-05")},
            {"return_pct": -50.0, "exit_date": pd.Timestamp("2026-01-06")},
        ]
        curve, final_capital, max_dd = m.equity_curve_from_trades(trades, initial_capital=1_000_000.0)
        self.assertAlmostEqual(final_capital, 1_000_000.0 * 1.10 * 0.50, places=2)
        self.assertAlmostEqual(max_dd, -50.0, places=3)
        self.assertEqual(len(curve), 2)


# =====================================================================
# フォールド構築
# =====================================================================

class UsableDatesAndFoldsTests(unittest.TestCase):
    def _coverage_frames(self, n_tickers=60, n_days=400, warmup=100):
        idx = pd.bdate_range("2024-01-01", periods=n_days)
        frames = {}
        for i in range(n_tickers):
            df = pd.DataFrame({f: np.linspace(1, 2, n_days) for f in FEATURES}, index=idx)
            if i % 3 == 0:
                df.iloc[:warmup] = np.nan
            frames[f"T{i}.T"] = df
        return frames, idx, warmup

    def test_usable_dates_respects_coverage_threshold(self):
        frames, idx, warmup = self._coverage_frames(n_tickers=60, warmup=100)
        good, all_dates = m.usable_dates(frames, min_coverage_tickers=50)
        self.assertEqual(len(all_dates), 400)
        self.assertTrue((good >= idx[warmup]).all())

    def test_build_folds_count_and_coverage(self):
        dates = pd.bdate_range("2024-01-01", periods=500)
        with patch.object(m, "purge_margin_days", return_value=6):
            folds = m.build_folds(dates, n_folds=4, recent_days=60)
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[-1]["name"], "recent60")
        self.assertEqual(folds[-1]["oos_end"], dates[-1])
        # 最初の4フォールドが全期間をちょうど分割し、重複も欠落もない
        self.assertEqual(folds[0]["oos_start"], dates[0])
        self.assertEqual(folds[3]["oos_end"], dates[-1])
        for k in range(3):
            self.assertEqual(folds[k]["oos_end"] + pd.Timedelta(days=0),
                              dates[dates.get_loc(folds[k + 1]["oos_start"]) - 1])

    def test_build_folds_train_cutoff_uses_purge_margin(self):
        dates = pd.bdate_range("2024-01-01", periods=200)
        with patch.object(m, "purge_margin_days", return_value=6):
            folds = m.build_folds(dates, n_folds=4, recent_days=60)
        for fold in folds:
            self.assertEqual(fold["train_cutoff"], fold["oos_start"] - pd.Timedelta(days=6))

    def test_build_folds_too_few_dates_raises(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        with self.assertRaises(RuntimeError):
            m.build_folds(dates, n_folds=4, recent_days=60)

    def test_fold_oos_dates_filters_inclusive_range(self):
        dates = list(pd.bdate_range("2024-01-01", periods=10))
        fold = {"oos_start": dates[2], "oos_end": dates[5]}
        result = m.fold_oos_dates(dates, fold)
        self.assertEqual(result, dates[2:6])


# =====================================================================
# TOP1 / equal-weight シミュレーションのスモークテスト
# =====================================================================

class SimulationSmokeTests(FixtureMixin, unittest.TestCase):
    def test_top1_track_single_position_at_a_time(self):
        oos_dates = list(self.all_dates_sorted[-40:])
        trades = m.simulate_top1_track(
            oos_dates, self.ticker_frames, self.nikkei_ff, self.model,
            pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted),
            "next_open", self.all_dates_sorted,
        )
        for a, b in zip(trades, trades[1:]):
            self.assertLessEqual(a["exit_date"], b["entry_date"])
        for t in trades:
            self.assertIn(t["reason"], ("TP", "SL", "SL_BOTH", "TIME", "FORCED_EOS"))

    def test_equal_weight_buckets_are_nested(self):
        oos_dates = list(self.all_dates_sorted[-30:])
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        buckets = m.simulate_equal_weight_buckets(
            oos_dates, self.ticker_frames, self.nikkei_ff, self.model, trend_series, "same_close",
            self.all_dates_sorted,
        )
        self.assertLessEqual(len(buckets["TOP1"]), len(buckets["TOP3"]))
        self.assertLessEqual(len(buckets["TOP3"]), len(buckets["TOP5"]))
        self.assertLessEqual(len(buckets["TOP5"]), len(buckets["ALL"]))


# =====================================================================
# fidelity checks
# =====================================================================

class FidelityCheckTests(FixtureMixin, unittest.TestCase):
    def test_live_top1_matches_when_expected_ticker_is_top(self):
        # ZZZZ.T単独のuniverseにして、他の合成銘柄とのランキング競合(乱数依存で
        # 不安定になりうる)を排除し、「top1==指定した期待値ならmatched=True」
        # というロジックそのものだけを確定的に検証する。
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        solo_frames = {"ZZZZ.T": self.ticker_frames["ZZZZ.T"]}
        result = m.fidelity_check_live_top1(
            solo_frames, self.nikkei_ff, trend_series, self.last_date, model=self.model,
            expected_ticker="ZZZZ.T", expected_direction="BUY",
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["top1_ticker"], "ZZZZ.T")

    def test_live_top1_reports_rank_when_not_matched(self):
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        result = m.fidelity_check_live_top1(
            self.ticker_frames, self.nikkei_ff, trend_series, self.last_date, model=self.model,
            expected_ticker="NOPE.T", expected_direction="BUY",
        )
        self.assertFalse(result["matched"])
        self.assertIn("expected_rank_in_pool", result)

    def test_live_top1_skips_when_no_model(self):
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        with patch.object(trader, "load_model", return_value=None):
            result = m.fidelity_check_live_top1(
                self.ticker_frames, self.nikkei_ff, trend_series, self.last_date, model=None,
            )
        self.assertIn("SKIP", result["status"])

    def test_research_track_match_rate(self):
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        # fidelity_check_research_track()自身が使うのと同じ(frozen)policyでないと
        # cand集合が食い違う(frozen upはmin_score_for_buy=70で、base_policy()の
        # 緩い閾値=0とは別物)ため、ここでも同じpolicy解決を使う。
        trend = m.trend_for_date(trend_series, self.last_date)
        _, policy = m.frozen_policy_for_trend(trend)
        pool, cand, _ = m.build_candidate_pool_for_date(
            self.last_date, self.ticker_frames, self.nikkei_ff, self.model, policy,
        )
        self.assertGreaterEqual(len(cand), 2, "このフィクスチャではfrozen policyでも2件以上通るはず")
        recorded_rows = [{"ticker": c["ticker"], "direction": c["direction"]} for c in cand[:2]]
        recorded_rows.append({"ticker": "NOPE.T", "direction": "BUY"})
        recorded_df = pd.DataFrame(recorded_rows)
        result = m.fidelity_check_research_track(
            self.ticker_frames, self.nikkei_ff, trend_series, self.last_date, recorded_df, model=self.model,
        )
        self.assertEqual(result["recorded_count"], 3)
        self.assertEqual(result["intersection"], 2)
        self.assertAlmostEqual(result["match_rate_vs_recorded"], 2 / 3, places=6)


class DiagnosticVsWalkForwardTests(FixtureMixin, unittest.TestCase):
    def test_skip_when_columns_missing(self):
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        wf_df = pd.DataFrame({"foo": [1, 2, 3]})
        result = m.diagnostic_vs_walk_forward(self.ticker_frames, self.nikkei_ff, trend_series, self.model, wf_df)
        self.assertIn("SKIP", result["status"])

    def test_detects_columns_and_computes_overlap(self):
        trend_series = pd.DataFrame({"trend": ["up"] * len(self.all_dates_sorted)}, index=self.all_dates_sorted)
        policy = self.base_policy()
        recent_dates = self.all_dates_sorted[-5:]
        rows = []
        for d in recent_dates:
            pool, cand, _ = m.build_candidate_pool_for_date(d, self.ticker_frames, self.nikkei_ff, self.model, policy)
            buy = [c for c in pool if c["direction"] == "BUY"]
            for c in buy:
                rows.append({"date": str(d.date()), "ticker": c["ticker"],
                              "up_probability": c["up_probability"], "score": c["score"]})
        wf_df = pd.DataFrame(rows)
        result = m.diagnostic_vs_walk_forward(self.ticker_frames, self.nikkei_ff, trend_series, self.model, wf_df)
        self.assertEqual(result["columns_found"]["date"], "date")
        self.assertEqual(result["columns_found"]["ticker"], "ticker")
        self.assertGreaterEqual(result["overlapping_dates"], 1)
        # 完全一致データを突っ込んでいるので相関は1.0に近いはず
        if "spearman_score" in result:
            self.assertGreater(result["spearman_score"], 0.99)


# =====================================================================
# release資産ダウンロード(subprocess経由、ネットワークなし)
# =====================================================================

class DownloadReleaseAssetTests(unittest.TestCase):
    def test_success(self):
        fake = type("P", (), {"returncode": 0, "stderr": ""})()
        with patch("subprocess.run", return_value=fake):
            ok, err = m.download_release_asset("aifuu/stock-ai", "tag", "asset.csv.gz", "/tmp/out.csv.gz")
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_failure_returns_stderr(self):
        fake = type("P", (), {"returncode": 1, "stderr": "release not found"})()
        with patch("subprocess.run", return_value=fake):
            ok, err = m.download_release_asset("aifuu/stock-ai", "tag", "asset.csv.gz", "/tmp/out.csv.gz")
        self.assertFalse(ok)
        self.assertIn("release not found", err)

    def test_exception_is_caught(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("gh not installed")):
            ok, err = m.download_release_asset("aifuu/stock-ai", "tag", "asset.csv.gz", "/tmp/out.csv.gz")
        self.assertFalse(ok)
        self.assertIn("gh not installed", err)


# =====================================================================
# run_validation() エンドツーエンド(結合テスト、小さいuniverse・短い
# フォールドで高速に実行する)。全体配線(fold構築→学習→シミュレーション→
# 指標集計)の回帰を検知するためのもの。
# =====================================================================

class RunValidationIntegrationTests(unittest.TestCase):
    N_DAYS_BIG = 500
    FAKE_TICKERS_SMALL = ["AAAA.T", "BBBB.T", "ZZZZ.T"]

    def setUp(self):
        frames = {
            tk: _make_ohlcv(seed=10 + i, n=self.N_DAYS_BIG, end=END_DATE)
            for i, tk in enumerate(self.FAKE_TICKERS_SMALL) if tk != "ZZZZ.T"
        }
        frames["ZZZZ.T"] = _make_trending_ohlcv(n=self.N_DAYS_BIG, end=END_DATE)
        nikkei_src = _make_index_close(n=self.N_DAYS_BIG, end=END_DATE, seed=111)
        topix_src = _make_index_close(n=self.N_DAYS_BIG, end=END_DATE, seed=222, start_price=2000.0)
        futures_src = _make_index_close(n=self.N_DAYS_BIG, end=END_DATE, seed=333, start_price=30000.0)
        self.model = StubModel()
        dl = _patched_download(frames, nikkei_src, topix_src, futures_src)

        trader._FUTURES_FEATURE_CACHE = None
        trader._TOPIX_FEATURE_CACHE = None

        def fake_trend_series(start, end):
            idx = pd.date_range(start, end, freq="D")
            return pd.DataFrame({"trend": ["up"] * len(idx), "source": ["fake"] * len(idx)}, index=idx)

        import futures_trend
        self._patches = [
            patch.object(trader, "download", side_effect=dl),
            patch.object(live_p10, "download", side_effect=dl),
            patch.object(live_p10, "TICKERS", self.FAKE_TICKERS_SMALL),
            patch.object(live_p10, "load_model", return_value=self.model),
            patch.object(live_p10, "download_5m", return_value=None),
            patch.object(futures_trend, "historical_trend_series", side_effect=fake_trend_series),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_compounding_top1_and_equal_weight_top1_are_distinct_keys(self):
        # これは実際にlive_model_policy_validation.pyのバグとして発生した
        # ケースの回帰テスト: BUCKETS=("TOP1","ALL","TOP3","TOP5")(等金額バケット名)
        # をそのままconv_resultのキーに使うと、複利・単一ポジションのTOP1
        # (compute_full_metrics、max_dd_pct等を持つ)を、等金額日次コホートの
        # TOP1(compute_equal_weight_metrics、total_pnl_jpy等を持つ)が
        # 上書きしてしまっていた。
        results, *_ = m.run_validation(
            tickers=self.FAKE_TICKERS_SMALL, n_folds=1, recent_days=15, min_coverage_tickers=2,
            progress=lambda s: None,
        )
        for conv, buckets in results["overall"].items():
            self.assertIn("TOP1", buckets)
            self.assertIn("EW_TOP1", buckets)
            self.assertIn("max_dd_pct", buckets["TOP1"], "複利TOP1はmax_dd_pctを持つはず")
            self.assertNotIn("max_dd_pct", buckets["EW_TOP1"], "等金額TOP1は別の指標セットのはず")
            self.assertIn("total_pnl_jpy", buckets["EW_TOP1"])
            for b in ("EW_ALL", "EW_TOP3", "EW_TOP5"):
                self.assertIn(b, buckets)

    def test_run_validation_result_structure(self):
        results, ticker_frames, nikkei_ff, trend_series, all_dates_sorted = m.run_validation(
            tickers=self.FAKE_TICKERS_SMALL, n_folds=1, recent_days=15, min_coverage_tickers=2,
            progress=lambda s: None,
        )
        self.assertIn("meta", results)
        self.assertEqual(results["meta"]["universe_size"], 3)
        self.assertGreater(results["meta"]["usable_dates"], 0)
        self.assertGreaterEqual(len(results["folds"]), 1)
        self.assertEqual(set(ticker_frames.keys()), set(self.FAKE_TICKERS_SMALL))
        self.assertEqual(len(nikkei_ff), len(all_dates_sorted))
        for fold in results["folds"]:
            for conv in m.ENTRY_CONVENTIONS:
                self.assertIn(conv, fold["conventions"])

    def test_too_few_folds_for_data_raises_clear_error(self):
        with self.assertRaises(RuntimeError):
            m.run_validation(
                tickers=self.FAKE_TICKERS_SMALL, n_folds=100, recent_days=15, min_coverage_tickers=2,
                progress=lambda s: None,
            )


if __name__ == "__main__":
    unittest.main()
