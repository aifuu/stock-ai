#!/usr/bin/env python3
"""Fast paper-trading entrypoint with paper-only continuity fallbacks."""
import argparse
import math

import numpy as np
import pandas as pd
import yfinance as yf

import run_profit_loop as loop
import daily_directional_top1 as directional

DETAIL_UNIVERSE = 50
# Keep the real/base scanner separately.  Do not overwrite run_profit_loop._original_scan
# because scan_candidates_progressive() resolves that module-global name at runtime.
_base_scan = loop._original_scan
_cache = {"result": None}


class PaperFallbackDirectionalModel:
    """Paper-only DOWN/FLAT/UP fallback used when the real model is unavailable."""
    feature_names_in_ = np.array(directional.FEATURES)
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
            directional_score = 0.55 * momentum + 0.65 * trend + 0.35 * ret5 + 0.25 * macd_bias - 0.15 * rsi_bias
            strength = min(3.0, abs(directional_score))
            flat = max(0.08, 0.42 - 0.10 * strength)
            up_raw = math.exp(max(-5.0, min(5.0, directional_score)))
            down_raw = math.exp(max(-5.0, min(5.0, -directional_score)))
            total = up_raw + down_raw + flat
            out.append([down_raw / total, flat / total, up_raw / total])
        return np.asarray(out, dtype=float)


_original_load_model = loop.app.load_model
_original_features = loop.app.features


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


def _features_for_paper(df, nikkei, futures_df=None):
    x = _original_features(df, nikkei, futures_df)
    if x is None:
        return x
    futures_cols = ["future_return", "future_ma5", "future_rsi", "future_gap"]
    if any(c not in x.columns or x[c].isna().all() for c in futures_cols):
        n = nikkei.reindex(x.index).ffill()
        x["future_return"] = n["ret5_raw"].fillna(0.0)
        x["future_ma5"] = 0.0
        x["future_rsi"] = n["rsi"].fillna(50.0)
        x["future_gap"] = n["ret5_raw"].fillna(0.0)
        print("🟡 PAPER FUTURES FALLBACK: NIY=F欠損 → 日経現物由来の代替特徴量で継続")
    return x


loop.app.load_model = _load_model_for_paper
loop.app.features = _features_for_paper


def _prefilter_universe(tickers):
    if not tickers:
        return []
    print(f"⚡ PAPER PREFILTER: {len(tickers)}銘柄を日足バッチ取得 → TOP{DETAIL_UNIVERSE}")
    try:
        data = yf.download(tickers, period="60d", interval="1d", auto_adjust=False, progress=False, threads=True, group_by="ticker")
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
            price = float(close.iloc[-1]); ma25 = float(close.rolling(25).mean().iloc[-1])
            ret5 = float(close.iloc[-1] / close.iloc[-6] - 1.0); avg20 = float(volume.tail(20).mean()); recent5 = float(volume.tail(5).mean())
            if price <= 0 or avg20 <= 0:
                continue
            vol_ratio = recent5 / avg20; kairi = abs(price / ma25 - 1.0) if ma25 > 0 else 0.0; momentum = abs(ret5); activity = max(0.0, min(vol_ratio, 5.0))
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
    """Run the base scanner once on a fast TOP50 prefilter and cache its raw pool."""
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
            _cache["result"] = _base_scan(base_policy)
        finally:
            loop.app.TICKERS = original_tickers
    else:
        print("♻️ PAPER FAST CACHE: 既取得候補プールをLEVEL1-6で再利用")
    return _cache["result"]


def scan_progressive_with_prefilter(policy):
    """Use the cached TOP50 detail pool as the base of the existing progressive LEVEL1-6 logic."""
    previous = loop._original_scan
    loop._original_scan = cached_scan
    try:
        return loop.scan_candidates_progressive(policy)
    finally:
        loop._original_scan = previous


def run_analysis_only():
    """09:30前の分析・再評価だけを実行し、ポジションは絶対に開かない。"""
    policy = loop.app.load_policy()
    candidates, _ = scan_progressive_with_prefilter(policy)
    print("========================================")
    print("PRE-09:30 ANALYSIS ONLY / NO TRADE")
    print("========================================")
    if not candidates:
        print("⚠️ 候補なし。取引は実行せず、次回スケジュールで再評価します。")
        return 0
    for i, c in enumerate(candidates[:10], 1):
        print(
            f"TOP{i}: {c.get('company', c.get('ticker',''))} "
            f"({c.get('ticker','')}) | {str(c.get('direction','BUY')).upper()} | "
            f"score={float(c.get('score',0) or 0):.1f} | "
            f"UP={float(c.get('up_probability',0) or 0):.1f}% | "
            f"DOWN={float(c.get('down_probability',0) or 0):.1f}% | "
            f"EV={float(c.get('profit_ev_pct',0) or 0):+.2f}%"
        )
    print("🛑 PRE-09:30: 売買執行なし。09:30以降に最終TOP1選定→ペーパートレードします。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true", help="09:30前の分析のみ。売買は一切しない")
    args = parser.parse_args()
    if args.analysis_only:
        raise SystemExit(run_analysis_only())
    loop.app.scan = scan_progressive_with_prefilter
    loop.app.mark_and_close = loop.close_positions_with_cooldown
    loop.app.open_positions = loop.open_top1_only
    loop.app.main()
