#!/usr/bin/env python3
"""Fast paper-trading entrypoint.

Every 5-minute run does one cheap broad-market prefilter, then runs the full
AI/5-minute analysis only on the best candidates. Progressive LEVEL1-6 reuses
the same detailed candidate pool, so it never repeats yfinance downloads.

If the research workflow has not produced a compatible directional model yet,
a paper-only heuristic model is used instead of silently producing zero trades.
This fallback is never used for live orders.
"""
import math

import numpy as np
import pandas as pd
import yfinance as yf

import run_profit_loop as loop

DETAIL_UNIVERSE = 50
_original_scan = loop._original_scan
_cache = {"result": None}


class PaperFallbackDirectionalModel:
    """Paper-only DOWN/FLAT/UP model used only when the real model is unavailable."""
    feature_names_in_ = np.array(loop.app.FEATURES)
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        rows = pd.DataFrame(X)
        out = []
        for _, r in rows.iterrows():
            def num(name, default=0.0):
                try:
                    v = float(r.get(name, default))
                    return v if np.isfinite(v) else default
                except Exception:
                    return default

            momentum = (num("momentum_score") - 50.0) / 25.0
            trend = num("trend_alignment") - 1.5
            ret5 = max(-3.0, min(3.0, num("ret5") / 3.0))
            macd_scale = max(abs(num("ma25")) * 0.003, 1e-9)
            macd_bias = max(-3.0, min(3.0, (num("macd") - num("signal")) / macd_scale))
            rsi_bias = max(-2.0, min(2.0, (num("rsi", 50.0) - 50.0) / 20.0))
            directional = 0.55 * momentum + 0.65 * trend + 0.35 * ret5 + 0.25 * macd_bias - 0.15 * rsi_bias
            strength = min(3.0, abs(directional))
            flat = max(0.08, 0.42 - 0.10 * strength)
            up_raw = math.exp(max(-5.0, min(5.0, directional)))
            down_raw = math.exp(max(-5.0, min(5.0, -directional)))
            total = up_raw + down_raw + flat
            out.append([down_raw / total, flat / total, up_raw / total])
        return np.asarray(out, dtype=float)


_original_load_model = loop.app.load_model


def _load_model_for_paper():
    try:
        model = _original_load_model()
        if model is not None:
            print("✅ directional AI model loaded")
            return model
    except Exception as exc:
        print(f"⚠️ directional AIモデル取得失敗: {exc}")
    print("🟠 PAPER FALLBACK: directional_model.pkl が未準備/不一致 → 紙取引専用モデルで継続")
    return PaperFallbackDirectionalModel()


loop.app.load_model = _load_model_for_paper


def _prefilter_universe(tickers):
    """Batch-download daily bars once and select the most active/trending names."""
    if not tickers:
        return []
    print(f"⚡ PAPER PREFILTER: {len(tickers)}銘柄を日足バッチ取得 → TOP{DETAIL_UNIVERSE}")
    try:
        data = yf.download(
            tickers,
            period="60d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        print(f"⚠️ 日足バッチ取得失敗: {exc} → 元ユニバースをそのまま使用")
        return list(tickers)

    rows = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].copy()
            else:
                df = data.copy()
            if df.empty or "Close" not in df or "Volume" not in df:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            volume = pd.to_numeric(df["Volume"], errors="coerce").dropna()
            if len(close) < 26 or len(volume) < 20:
                continue
            price = float(close.iloc[-1])
            ma25 = float(close.rolling(25).mean().iloc[-1])
            ret5 = float(close.iloc[-1] / close.iloc[-6] - 1.0)
            avg20 = float(volume.tail(20).mean())
            recent5 = float(volume.tail(5).mean())
            if price <= 0 or avg20 <= 0:
                continue
            vol_ratio = recent5 / avg20
            kairi = abs(price / ma25 - 1.0) if ma25 > 0 else 0.0
            momentum = abs(ret5)
            activity = max(0.0, min(vol_ratio, 5.0))
            score = 0.45 * math.log1p(activity) + 0.35 * min(momentum * 10.0, 2.0) + 0.20 * min(kairi * 10.0, 2.0)
            rows.append((score, vol_ratio, ticker))
        except Exception:
            continue

    rows.sort(reverse=True)
    selected = [ticker for _, _, ticker in rows[:DETAIL_UNIVERSE]]
    if not selected:
        return list(tickers)
    print(f"✅ PREFILTER選抜: {len(selected)}/{len(tickers)}銘柄")
    print("   " + ", ".join(selected[:10]) + (" ..." if len(selected) > 10 else ""))
    return selected


def cached_scan(policy):
    if _cache["result"] is None:
        base_policy = dict(policy)
        base_policy["up_threshold"] = 0.0
        base_policy["min_score_for_buy"] = 0.0
        base_policy["nikkei_filter"] = False

        original_tickers = list(loop.app.TICKERS)
        detail_tickers = _prefilter_universe(original_tickers)
        loop.app.TICKERS = detail_tickers
        try:
            print(f"🔬 DETAIL SCAN: {len(detail_tickers)}銘柄だけ5分足＋AI詳細分析")
            _cache["result"] = _original_scan(base_policy)
        finally:
            loop.app.TICKERS = original_tickers
    else:
        print("♻️ PAPER FAST CACHE: 既取得候補プールをLEVEL1-6で再利用")
    return _cache["result"]


loop._original_scan = cached_scan

if __name__ == "__main__":
    loop.app.scan_candidates = loop.scan_candidates_progressive
    loop.app.close_positions = loop.close_positions_with_cooldown
    loop.app.open_positions = loop.open_top1_only
    loop.app.main()
