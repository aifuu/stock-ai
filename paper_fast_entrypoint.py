#!/usr/bin/env python3
"""Fast paper-trading entrypoint with paper-only continuity fallbacks.

The expensive universe/detail scan is cached to disk for a few minutes so
separate Python processes in the same GitHub Actions job (TOP10 + multi-hold)
can reuse the exact same candidate pool instead of fetching/scoring twice.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

import run_profit_loop as loop
import daily_directional_top1 as directional

_base_scan = loop._original_scan
_cache = {"result": None}
PAPER_MODEL_TYPE = "validated_model"
PAPER_FEATURE_SOURCE = "futures"

# Cross-process reuse inside one CI job only.
SCAN_CACHE_FILE = "scan_candidates_cache.json"
SCAN_CACHE_TTL_SECONDS = 300


def _json_safe(obj):
    """Recursively convert numpy/pandas values to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _load_disk_scan_cache():
    """Load a recent scan cache; corrupt/expired cache is simply ignored."""
    try:
        if not os.path.exists(SCAN_CACHE_FILE):
            return None
        with open(SCAN_CACHE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        age = time.time() - float(payload.get("timestamp", 0))
        if age < 0 or age > SCAN_CACHE_TTL_SECONDS:
            return None
        raw = payload.get("raw")
        if raw is None:
            return None
        return raw, payload.get("scanned")
    except Exception as exc:
        print(f"⚠️ scan cache読み込み失敗（新規スキャンを実行）: {exc}")
        return None


def _save_disk_scan_cache(result):
    """Best-effort atomic cache write; cache failure never stops trading."""
    try:
        raw, scanned = result
        payload = {
            "timestamp": time.time(),
            "raw": _json_safe(raw),
            "scanned": _json_safe(scanned),
        }
        tmp = SCAN_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, SCAN_CACHE_FILE)
    except Exception as exc:
        print(f"⚠️ scan cache保存失敗（続行します、次回は再スキャン）: {exc}")


class PaperFallbackDirectionalModel:
    """Paper-only DOWN/FLAT/UP fallback used when the real model is unavailable."""

    # 実際に読む列だけを明示。directional.FEATURES全体を流用すると、新しく増えた特徴量
    # (例: vs_topix_1d_pt)がNaNの銘柄をprofit_top10_paper.scan()のdropna(subset=cols)が
    # 丸ごと弾いてしまう(このフォールバックモデルは以下7列しか使わない)。
    feature_names_in_ = np.array(["momentum_score", "trend_alignment", "ret5", "ma25", "macd", "signal", "rsi"])
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
            ds = 0.55 * momentum + 0.65 * trend + 0.35 * ret5 + 0.25 * macd_bias - 0.15 * rsi_bias
            strength = min(3.0, abs(ds))
            flat = max(0.08, 0.42 - 0.10 * strength)
            up_raw = math.exp(max(-5.0, min(5.0, ds)))
            down_raw = math.exp(max(-5.0, min(5.0, -ds)))
            total = up_raw + down_raw + flat
            out.append([down_raw / total, flat / total, up_raw / total])
        return np.asarray(out, dtype=float)


_original_load_model = loop.app.load_model
_original_features = loop.app.features
_original_download = loop.app.download
_download_cache = {}


def _batch_download_all(tickers, period="3y"):
    """225銘柄をyf.download()で1回だけ一括取得し、daily_directional_top1.download()と
    同じ後処理(MultiIndexのフラット化)を揃えてキャッシュに格納する。
    出来高偏重プレフィルター(TOP50絞り込み)を廃止し、全銘柄を本物のAIモデルに
    かけられるようにするための置き換え。
    """
    print(f"📦 BATCH DOWNLOAD: {len(tickers)}銘柄を一括取得（プレフィルターなし・全銘柄をAI詳細分析）")
    try:
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        print(f"⚠️ 一括取得失敗: {exc} → 個別取得にフォールバック")
        return {}

    cache = {}
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].copy()
            else:
                df = data.copy()
            if df is None or df.empty:
                continue
            df = df.dropna(how="all")
            if df.empty:
                continue
            cache[ticker] = df
        except Exception:
            continue
    print(f"✅ BATCH DOWNLOAD完了: {len(cache)}/{len(tickers)}銘柄取得成功（未取得分のみ個別取得にフォールバック）")
    return cache


def _download_with_batch_cache(ticker, period="3y"):
    if period == "3y":
        cached = _download_cache.get(ticker)
        if cached is not None:
            return cached
    return _original_download(ticker, period=period)


def _load_model_for_paper():
    global PAPER_MODEL_TYPE
    try:
        model = _original_load_model()
        if model is not None:
            PAPER_MODEL_TYPE = "validated_model"
            print("✅ directional AI model loaded | provenance=validated_model")
            return model
    except Exception as exc:
        print(f"⚠️ directional AIモデル取得失敗: {exc}")
    PAPER_MODEL_TYPE = "fallback"
    print("🟠 PAPER FALLBACK: directional_model.pkl が未準備/不一致 → 紙取引専用モデルで継続 | provenance=fallback")
    return PaperFallbackDirectionalModel()


def _features_for_paper(df, nikkei, futures_df=None):
    global PAPER_FEATURE_SOURCE
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
        PAPER_FEATURE_SOURCE = "cash_proxy"
        print("🟡 PAPER FUTURES FALLBACK: NK=F欠損 → 日経現物由来の代替特徴量で継続 | feature_source=cash_proxy")
    else:
        PAPER_FEATURE_SOURCE = "futures"
    return x


loop.app.load_model = _load_model_for_paper
loop.app.features = _features_for_paper
loop.app.download = _download_with_batch_cache


def _append_history_with_provenance(row):
    enriched = dict(row)
    ticker = str(enriched.get("ticker", ""))
    entry_date = str(enriched.get("entry_date", ""))
    entry_time = str(enriched.get("entry_time", ""))
    state = getattr(loop, "_ACTIVE_CLOSE_STATE", None)
    match = None
    if state is not None:
        for p in state.get("positions", []):
            if (
                str(p.get("ticker", "")) == ticker
                and str(p.get("entry_date", "")) == entry_date
                and str(p.get("entry_time", "")) == entry_time
            ):
                match = p
                break
    if match is not None:
        enriched.update({
            "selection_mode": match.get("selection_mode", "legacy_unknown"),
            "selection_level": match.get("selection_level", ""),
            "market_regime": match.get("market_regime", "unknown"),
            "profit_ev_pct": match.get("profit_ev_pct", ""),
            "top10_rank": match.get("top10_rank", ""),
            "regime_preferred": match.get("regime_preferred", ""),
        })
    else:
        enriched.setdefault("selection_mode", "legacy_unknown")
        enriched.setdefault("selection_level", "")
        enriched.setdefault("market_regime", "unknown")
        enriched.setdefault("profit_ev_pct", "")
        enriched.setdefault("top10_rank", "")
        enriched.setdefault("regime_preferred", "")
    enriched["model_type"] = PAPER_MODEL_TYPE
    enriched["feature_source"] = PAPER_FEATURE_SOURCE
    enriched["validation_eligible"] = bool(
        enriched.get("selection_mode") in ("normal", "progressive_level")
        and PAPER_MODEL_TYPE == "validated_model"
        and PAPER_FEATURE_SOURCE == "futures"
    )
    return loop._ORIGINAL_APPEND_HISTORY(enriched)


if not hasattr(loop, "_ORIGINAL_APPEND_HISTORY"):
    loop._ORIGINAL_APPEND_HISTORY = loop.app.append_history
loop.app.append_history = _append_history_with_provenance


def _close_with_provenance(state, now, policy):
    loop._ACTIVE_CLOSE_STATE = state
    try:
        return loop.close_positions_with_cooldown(state, now, policy)
    finally:
        loop._ACTIVE_CLOSE_STATE = None


def cached_scan(policy):
    if _cache["result"] is not None:
        print("♻️ PAPER FAST CACHE: 既取得候補プールをLEVEL1-6で再利用")
        return _cache["result"]

    disk_cached = _load_disk_scan_cache()
    if disk_cached is not None:
        print("♻️ PAPER FAST CACHE(disk): 同一ジョブ内の直前スキャン結果を再利用（二重スキャン回避）")
        _cache["result"] = disk_cached
        return _cache["result"]

    base_policy = dict(policy)
    # First build a broad candidate pool. Progressive levels below apply the
    # real thresholds; these relaxed values only prevent early filtering in
    # the expensive base scan.
    base_policy["up_threshold"] = 0.0
    base_policy["min_score_for_buy"] = 0.0
    base_policy["nikkei_filter"] = False
    tickers = list(loop.app.TICKERS)
    _download_cache.clear()
    _download_cache.update(_batch_download_all(tickers))
    print(f"🔬 FULL SCAN: {len(tickers)}銘柄全てにAI詳細分析（出来高偏重プレフィルターは廃止・全銘柄が対象）")
    _cache["result"] = _base_scan(base_policy)
    _save_disk_scan_cache(_cache["result"])
    return _cache["result"]


def scan_progressive_with_prefilter(policy):
    previous = loop._original_scan
    loop._original_scan = cached_scan
    try:
        return loop.scan_candidates_progressive(policy)
    finally:
        loop._original_scan = previous


def run_analysis_only():
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
            f"TOP{i}: {c.get('company', c.get('ticker', ''))} ({c.get('ticker', '')}) | "
            f"{str(c.get('direction', 'BUY')).upper()} | "
            f"score={float(c.get('score', 0) or 0):.1f} | "
            f"UP={float(c.get('up_probability', 0) or 0):.1f}% | "
            f"DOWN={float(c.get('down_probability', 0) or 0):.1f}% | "
            f"EV={float(c.get('profit_ev_pct', 0) or 0):+.2f}%"
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
    loop.app.mark_and_close = _close_with_provenance
    loop.app.open_positions = loop.open_top1_only
    loop.app.main()
