"""Part B: all_candidates_paper.py のランキングをliveのTOP1ルールに整合させた
変更(compute_live_ranking/live_rank/regime_eligible/rank_method)のテスト。

liveのTOP1選定ルールは run_profit_loop.open_top1_only() (内部で
run_profit_loop.profit_priority() を呼ぶ)。このテストは:
  1. compute_live_ranking()のlive_rank==1がopen_top1_only()の実際の選択と
     一致すること(fixtureベース、open_top1_only自身の約定処理は
     _original_openを差し替えてバイパスする)
  2. レジーム方向フィルタ(regime_eligible)が正しく付与されること
  3. rank_methodが無い/異なる既存(レガシー)行はev_rankでバケット分けされ、
     rank_method='live_profit_priority_v2'の新しい行はlive_rankでバケット
     分けされること(後方互換)
  4. 現在コミット済みの本番state(GitHub Release、読み取り専用でcurl -Lで
     取得)が、この変更後も壊れずに扱えること(欠けている列は後方互換で
     レガシー扱いになる)

run_profit_loop._market_regime / _load_feedback_weights はネットワーク
(日経の取得)に依存するため、決定的にするために毎回patchする。
"""
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import all_candidates_paper as acp
import run_profit_loop as loop


def _candidate(ticker, direction, score, up, down, price=3000.0):
    tp = price * (1.05 if direction == "BUY" else 0.95)
    sl = price * (0.97 if direction == "BUY" else 1.03)
    return {
        "ticker": ticker, "direction": direction, "price": price, "tp": tp, "sl": sl,
        "score": score, "up_probability": up, "down_probability": down,
        "flat_probability": max(0.0, 100.0 - up - down), "data_date": "2026-09-25",
    }


class _RegimeAndFeedbackPatched:
    """run_profit_loop.profit_priority()が呼ぶ_market_regime/_load_feedback_weights
    を決定的な値に固定し、ネットワーク呼び出しを一切発生させない。"""

    def __init__(self, regime="neutral"):
        self.regime = regime

    def __enter__(self):
        self._p1 = patch.object(loop, "_market_regime", return_value=(self.regime, 1.0, 1.0))
        self._p2 = patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0})
        self._p1.start()
        self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop()
        self._p2.stop()


class LiveRankMatchesOpenTop1Only(unittest.TestCase):
    def _qualified(self):
        return [
            _candidate("7203.T", "BUY", score=60.0, up=55.0, down=10.0),
            _candidate("9984.T", "BUY", score=90.0, up=70.0, down=5.0),
            _candidate("8035.T", "BUY", score=40.0, up=45.0, down=20.0),
        ]

    def _open_top1_choice(self, qualified):
        """liveのopen_top1_only()が実際に選ぶticker/directionを返す。
        _original_open(実約定処理)はbypassし、選ばれたcandidateそのものを
        そのまま返すだけのfakeに差し替える(資金/ポジション管理は本テストの
        関心外)。"""
        def _fake_open(state, policy, cands, today):
            state["positions"].append(dict(cands[0]))
            return list(cands)

        orig_open = loop._original_open
        loop._original_open = _fake_open
        try:
            state = {"positions": [], "trades_today": 0, "trades_by_ticker_today": {},
                      "last_exit_by_ticker": {}}
            opened = loop.open_top1_only(state, {}, qualified, "2026-09-25")
        finally:
            loop._original_open = orig_open
        self.assertEqual(len(opened), 1)
        return opened[0]["ticker"], opened[0]["direction"]

    def test_live_rank_1_matches_open_top1_only_neutral_regime(self):
        qualified = self._qualified()
        with _RegimeAndFeedbackPatched("neutral"):
            rank_info = acp.compute_live_ranking(qualified)
            top1_ticker, top1_direction = self._open_top1_choice(qualified)

        rank1_keys = [k for k, v in rank_info.items() if v["live_rank"] == 1]
        self.assertEqual(len(rank1_keys), 1)
        self.assertEqual(rank1_keys[0], (top1_ticker, top1_direction))

    def test_live_rank_1_matches_open_top1_only_bullish_regime(self):
        qualified = self._qualified() + [
            _candidate("6758.T", "SHORT", score=95.0, up=5.0, down=80.0),
        ]
        with _RegimeAndFeedbackPatched("bullish"):
            rank_info = acp.compute_live_ranking(qualified)
            top1_ticker, top1_direction = self._open_top1_choice(qualified)

        self.assertEqual(top1_direction, "BUY")  # bullish regime must pick a BUY
        rank1_keys = [k for k, v in rank_info.items() if v["live_rank"] == 1]
        self.assertEqual(rank1_keys[0], (top1_ticker, top1_direction))
        # the bullish regime's lone SHORT candidate must never be chosen/ranked
        self.assertFalse(rank_info[("6758.T", "SHORT")]["regime_eligible"])
        self.assertIsNone(rank_info[("6758.T", "SHORT")]["live_rank"])


class RegimeEligibleFlagMatchesLiveDirectionFilter(unittest.TestCase):
    def test_bearish_regime_keeps_short_drops_buy(self):
        qualified = [
            _candidate("7203.T", "BUY", score=60.0, up=55.0, down=10.0),
            _candidate("9984.T", "SHORT", score=60.0, up=10.0, down=55.0),
        ]
        with _RegimeAndFeedbackPatched("bearish"):
            rank_info = acp.compute_live_ranking(qualified)
        self.assertFalse(rank_info[("7203.T", "BUY")]["regime_eligible"])
        self.assertIsNone(rank_info[("7203.T", "BUY")]["live_rank"])
        self.assertTrue(rank_info[("9984.T", "SHORT")]["regime_eligible"])
        self.assertEqual(rank_info[("9984.T", "SHORT")]["live_rank"], 1)
        for info in rank_info.values():
            self.assertEqual(info["market_regime"], "bearish")

    def test_bullish_regime_keeps_buy_drops_short(self):
        qualified = [
            _candidate("7203.T", "BUY", score=60.0, up=55.0, down=10.0),
            _candidate("9984.T", "SHORT", score=60.0, up=10.0, down=55.0),
        ]
        with _RegimeAndFeedbackPatched("bullish"):
            rank_info = acp.compute_live_ranking(qualified)
        self.assertTrue(rank_info[("7203.T", "BUY")]["regime_eligible"])
        self.assertFalse(rank_info[("9984.T", "SHORT")]["regime_eligible"])

    def test_empty_qualified_list_never_calls_market_regime(self):
        # モジュールdocstring/compute_live_ranking: 空リストならネットワーク
        # 呼び出し(日経レジーム取得)を一切行わない。
        with patch.object(loop, "_market_regime", side_effect=AssertionError("must not be called")):
            rank_info = acp.compute_live_ranking([])
        self.assertEqual(rank_info, {})


class LegacyRowsSummarizeWithEvRankNewRowsWithLiveRank(unittest.TestCase):
    def test_bucket_uses_live_rank_only_for_v2_rows(self):
        legacy_rows = [
            # 2026-09-01: レガシー行(rank_methodなし) -> ev_rank('rank')でバケット分け
            {"trade_id": "legacy1", "date": "2026-09-01", "ticker": "A", "direction": "BUY",
             "rank": 1, "exit_price": 3100.0, "return_pct": 1.0},
            {"trade_id": "legacy2", "date": "2026-09-01", "ticker": "B", "direction": "BUY",
             "rank": 2, "exit_price": 3100.0, "return_pct": 2.0},
        ]
        v2_rows = [
            # 2026-09-25: 新しい行(rank_method='live_profit_priority_v2')。
            # ev_rank('rank')とlive_rankを意図的に逆転させ、TOP1が
            # live_rankで判定されていることを明確にする。
            {"trade_id": "v2a", "date": "2026-09-25", "ticker": "C", "direction": "BUY",
             "rank": 2, "live_rank": 1, "regime_eligible": True,
             "rank_method": "live_profit_priority_v2", "exit_price": 3100.0, "return_pct": 3.0},
            {"trade_id": "v2b", "date": "2026-09-25", "ticker": "D", "direction": "BUY",
             "rank": 1, "live_rank": 2, "regime_eligible": True,
             "rank_method": "live_profit_priority_v2", "exit_price": 3100.0, "return_pct": 4.0},
            {"trade_id": "v2c", "date": "2026-09-25", "ticker": "E", "direction": "SHORT",
             "rank": 3, "live_rank": None, "regime_eligible": False,
             "rank_method": "live_profit_priority_v2", "exit_price": 3100.0, "return_pct": -1.0},
        ]
        df = acp.rows_to_dataframe(legacy_rows + v2_rows)
        daily = acp.compute_daily_summary(df)

        legacy_top1 = daily[(daily["date"] == "2026-09-01") & (daily["bucket"] == "TOP1")].iloc[0]
        self.assertEqual(int(legacy_top1["candidate_count"]), 1)  # ticker A (ev_rank==1)

        v2_top1 = daily[(daily["date"] == "2026-09-25") & (daily["bucket"] == "TOP1")].iloc[0]
        self.assertEqual(int(v2_top1["candidate_count"]), 1)  # ticker C (live_rank==1, NOT D)
        self.assertAlmostEqual(float(v2_top1["avg_return_pct"]), 3.0)

        v2_all = daily[(daily["date"] == "2026-09-25") & (daily["bucket"] == "ALL")].iloc[0]
        self.assertEqual(int(v2_all["candidate_count"]), 3)  # ALL includes the regime-ineligible row too

    def test_missing_live_rank_columns_entirely_falls_back_to_ev_rank(self):
        # 古い月次アセット(この変更より前)はlive_rank/rank_method列自体が
        # 存在しない。rows_to_dataframeを経由せず生DataFrameで再現する。
        df = pd.DataFrame([
            {"date": "2026-08-01", "rank": 1, "exit_price": 3100.0, "return_pct": 1.0},
            {"date": "2026-08-01", "rank": 2, "exit_price": 3100.0, "return_pct": 2.0},
        ])
        daily = acp.compute_daily_summary(df)
        top1 = daily[(daily["date"] == "2026-08-01") & (daily["bucket"] == "TOP1")].iloc[0]
        self.assertEqual(int(top1["candidate_count"]), 1)
        self.assertAlmostEqual(float(top1["avg_return_pct"]), 1.0)


class BackwardCompatibilityWithRealCurrentState(unittest.TestCase):
    """本番state(all-candidates-paper-state リリースアセット)を読み取り専用で
    ダウンロードし、この変更後もpositionsが壊れずに扱えることを確認する。
    ネットワークが使えない環境ではskipする(読み取り専用の確認用であり、
    CI本番のverificationではここで実データを検証する)。
    """

    def test_real_state_positions_are_legacy_and_still_load(self):
        url = (
            f"{acp.RELEASE_BASE_URL}/{acp.RELEASE_TAG_STATE}/{acp.STATE_ASSET_NAME}"
        )
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            proc = subprocess.run(
                ["curl", "-sS", "-L", "--max-time", "20", "-w", "%{http_code}",
                 "-o", tmp.name, url],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0 or proc.stdout.strip() != "200":
                self.skipTest(f"real state asset unreachable in this environment (rc={proc.returncode}, http={proc.stdout!r})")
            import json
            with open(tmp.name, encoding="utf-8") as f:
                state = json.load(f)

        self.assertTrue(acp._validate_state(state))
        positions = state.get("positions", [])
        self.assertGreater(len(positions), 0, "expected at least one real position to validate against")

        # 2026-10-02(commit 614f18a)以降、研究トラックは新しい行に
        # rank_method='live_profit_priority_v2'を書き込む(設計通り)。
        # それより前の行はrank_method列自体を持たない(レガシー)。両方が
        # 混在していてよいが、v2以外の値は想定しない。
        for p in positions:
            rank_method = p.get("rank_method")
            self.assertIn(rank_method, (None, "", acp.RANK_METHOD_LIVE_V2))

        df = acp.rows_to_dataframe(positions)
        legacy_mask = df["rank_method"].isna()
        v2_mask = df["rank_method"] == acp.RANK_METHOD_LIVE_V2
        self.assertTrue((legacy_mask | v2_mask).all())
        self.assertEqual(int(legacy_mask.sum()) + int(v2_mask.sum()), len(df))

        # _effective_rank/_bucket_frameが例外を出さず、rank_methodごとに
        # live_rank(v2)/ev_rank(レガシー)へ正しくフォールバックすること。
        effective_rank = acp._effective_rank(df)
        for bucket in acp.BUCKETS:
            if bucket == "ALL":
                continue
            n = int(bucket[len("TOP"):])
            b = acp._bucket_frame(df, bucket)
            self.assertTrue((effective_rank.loc[b.index] <= n).all())


if __name__ == "__main__":
    unittest.main()
