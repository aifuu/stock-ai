import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from daily_directional_top1 import (
    TICKERS,
    NAMES,
    download,
    make_nikkei,
    load_model,
    features,
    atr,
    directional_score,
)

POLICY_FILE = "strategy_policy.json"
TZ = ZoneInfo("Asia/Tokyo")


def load_policy_for_diagnostic():
    defaults = {
        "up_threshold": 50.0,
        "min_score_for_buy": 60.0,
        "nikkei_filter": False,
    }
    if not os.path.exists(POLICY_FILE):
        return defaults
    try:
        with open(POLICY_FILE, encoding="utf-8") as f:
            p = json.load(f)
        defaults["up_threshold"] = float(p.get("up_threshold", defaults["up_threshold"]))
        defaults["min_score_for_buy"] = float(p.get("min_score_for_buy", defaults["min_score_for_buy"]))
        defaults["nikkei_filter"] = str(p.get("nikkei_filter", False)).lower() in ("true", "1", "yes", "on")
        defaults["status"] = str(p.get("status", "UNKNOWN"))
        defaults["updated_at"] = p.get("updated_at")
    except Exception as exc:
        print(f"policy読み込み失敗: {exc}")
    return defaults


def main():
    now = datetime.now(TZ)
    policy = load_policy_for_diagnostic()
    nikkei = make_nikkei()
    model = load_model()
    if nikkei is None or model is None:
        raise RuntimeError("日経データまたはAIモデルを取得できませんでした")

    counts = {
        "universe": len(TICKERS),
        "download_ok": 0,
        "min150_ok": 0,
        "features_ok": 0,
        "predict_ok": 0,
        "classes_ok": 0,
        "atr_ok": 0,
        "up_threshold_ok": 0,
        "up_gt_down_ok": 0,
        "flat_ok": 0,
        "score_ok": 0,
        "nikkei_ok": 0,
        "final": 0,
    }
    reasons = {k: [] for k in (
        "download", "min150", "features", "predict", "classes", "atr",
        "up_threshold", "up_gt_down", "flat", "score", "nikkei"
    )}
    scored = []
    feature_cols = list(getattr(model, "feature_names_in_", []))

    for ticker in TICKERS:
        df = download(ticker)
        if df is None or len(df) < 150:
            reasons["download" if df is None else "min150"].append(ticker)
            continue
        counts["download_ok"] += 1
        counts["min150_ok"] += 1

        try:
            x = features(df, nikkei).dropna(subset=feature_cols)
        except Exception:
            reasons["features"].append(ticker)
            continue
        if x.empty:
            reasons["features"].append(ticker)
            continue
        counts["features_ok"] += 1

        try:
            last = x.iloc[-1]
            probs = model.predict_proba(x.iloc[-1:])[0]
            classes = list(model.classes_)
            if not all(c in classes for c in (0, 1, 2)):
                reasons["classes"].append(ticker)
                continue
            counts["predict_ok"] += 1
            counts["classes_ok"] += 1
            down = float(probs[classes.index(0)])
            up = float(probs[classes.index(2)])
            flat = float(probs[classes.index(1)])
            long_s, _ = directional_score(last, up, down)
            score = float(long_s)
        except Exception as exc:
            reasons["predict"].append(f"{ticker}({type(exc).__name__})")
            continue

        try:
            a = float(atr(df).iloc[-1])
            if not np.isfinite(a) or a <= 0:
                reasons["atr"].append(ticker)
                continue
            counts["atr_ok"] += 1
        except Exception:
            reasons["atr"].append(ticker)
            continue

        up_ok = up * 100 >= policy["up_threshold"]
        gt_ok = up > down
        flat_ok = flat < 0.50
        score_ok = score >= policy["min_score_for_buy"]
        nikkei_ok = True
        if policy["nikkei_filter"]:
            try:
                nlast = nikkei.reindex(x.index).ffill().iloc[-1]
                nikkei_ok = float(nlast["kairi25"]) > 0 and float(nlast["ret5"]) > 0
            except Exception:
                nikkei_ok = False

        if up_ok:
            counts["up_threshold_ok"] += 1
        else:
            reasons["up_threshold"].append(ticker)
        if gt_ok:
            counts["up_gt_down_ok"] += 1
        else:
            reasons["up_gt_down"].append(ticker)
        if flat_ok:
            counts["flat_ok"] += 1
        else:
            reasons["flat"].append(ticker)
        if score_ok:
            counts["score_ok"] += 1
        else:
            reasons["score"].append(ticker)
        if nikkei_ok:
            counts["nikkei_ok"] += 1
        else:
            reasons["nikkei"].append(ticker)

        if up_ok and gt_ok and flat_ok and score_ok and nikkei_ok:
            counts["final"] += 1
            scored.append((score, up * 100, ticker, NAMES.get(ticker, ticker)))

    scored.sort(reverse=True)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔎 100銘柄 OOSシグナル段階別診断")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📅 {now:%Y-%m-%d %H:%M} JST")
    print(f"Policy: {policy.get('status', 'UNKNOWN')} / UP >= {policy['up_threshold']:.1f}% / Score >= {policy['min_score_for_buy']:.1f} / 日経Filter={policy['nikkei_filter']}")
    print("")
    labels = [
        ("universe", "① 対象銘柄"),
        ("download_ok", "② データ取得成功"),
        ("min150_ok", "③ 150本以上"),
        ("features_ok", "④ 特徴量OK"),
        ("predict_ok", "⑤ AI予測OK"),
        ("classes_ok", "⑥ 3クラス確認"),
        ("atr_ok", "⑦ ATR OK"),
        ("up_threshold_ok", "⑧ UP確率条件OK"),
        ("up_gt_down_ok", "⑨ UP > DOWN"),
        ("flat_ok", "⑩ Flat < 50%"),
        ("score_ok", "⑪ Score条件OK"),
        ("nikkei_ok", "⑫ 日経条件OK"),
        ("final", "🔥 最終OOSシグナル"),
    ]
    for key, label in labels:
        print(f"{label:<24} {counts[key]:>3}/{counts['universe']}")

    print("")
    print("【除外数】")
    for key, label in [
        ("download", "データ取得失敗"), ("min150", "150本未満"), ("features", "特徴量NG"),
        ("predict", "AI予測例外"), ("classes", "3クラス不一致"), ("atr", "ATR NG"),
        ("up_threshold", "UP確率不足"), ("up_gt_down", "UP <= DOWN"), ("flat", "Flat >= 50%"),
        ("score", "Score不足"), ("nikkei", "日経条件NG")
    ]:
        print(f"{label:<24} {len(reasons[key]):>3}")

    print("")
    print("【最終候補】")
    if scored:
        for score, up, ticker, name in scored[:10]:
            print(f"  {ticker:<10} UP={up:5.1f}% Score={score:5.1f} {name}")
    else:
        print("  0件")
        worst = sorted(((len(v), k) for k, v in reasons.items()), reverse=True)
        print(f"  最大の除外要因: {worst[0][1]} = {worst[0][0]}件")

    print("")
    print("【除外銘柄（各段階）】")
    for key, items in reasons.items():
        if items:
            print(f"  {key}: {', '.join(items[:20])}{' ...' if len(items) > 20 else ''}")

    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if webhook:
        import requests
        lines = [
            "🔎 **100銘柄 OOSシグナル診断**",
            f"📅 {now:%Y-%m-%d %H:%M} JST",
            f"対象 {counts['universe']} → データ {counts['download_ok']} → 特徴量 {counts['features_ok']} → AI {counts['predict_ok']} → ATR {counts['atr_ok']} → 最終 **{counts['final']}**",
            f"UP条件 {counts['up_threshold_ok']} / UP>DOWN {counts['up_gt_down_ok']} / Flat<50 {counts['flat_ok']} / Score条件 {counts['score_ok']} / 日経 {counts['nikkei_ok']}",
        ]
        if scored:
            lines.append("候補: " + " / ".join(f"{t} UP{u:.1f}% S{s:.1f}" for s, u, t, _ in scored[:5]))
        else:
            top = sorted(((len(v), k) for k, v in reasons.items()), reverse=True)[:3]
            lines.append("⚠️ 最終0件。主な除外: " + ", ".join(f"{k}={n}" for n, k in top))
        try:
            r = requests.post(webhook, json={"content": "\n".join(lines)[:1950]}, timeout=30)
            r.raise_for_status()
            print("✅ Discord診断通知送信成功")
        except Exception as exc:
            print(f"⚠️ Discord診断通知失敗: {exc}")


if __name__ == "__main__":
    main()
