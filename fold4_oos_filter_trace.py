import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CANDIDATE_FILE = Path("walk_forward_all_candidates.csv")
OUT_FILE = Path("fold4_oos_filter_trace.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
OOS_DAYS = int(os.getenv("WF_MULTI_OOS_DAYS", "239"))
TOP_N = int(os.getenv("WF_TOP_N", "10"))
UP = 45
SCORE = 50
NIKKEI = False
TP = 4.0
SL = 1.0
HOLD = 7


def download(ticker):
    try:
        x = yf.download(
            ticker,
            start=(START_DATE - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
            end=(END_DATE + pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=False, progress=False, threads=False,
        )
        if x is None or x.empty:
            return None
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x.index = pd.to_datetime(x.index)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        return x
    except Exception as e:
        print(f"⚠ download {ticker}: {e}")
        return None


def main():
    if not CANDIDATE_FILE.exists():
        raise SystemExit("❌ walk_forward_all_candidates.csv がありません")

    c = pd.read_csv(CANDIDATE_FILE)
    required = ["date", "ticker", "score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "nikkei_uptrend"]
    missing = [x for x in required if x not in c.columns]
    if missing:
        raise SystemExit("❌ 候補CSVの不足列: " + ", ".join(missing))

    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    for col in ["score", "up_prob", "flat_prob", "down_prob", "price", "take_profit"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c["nikkei_uptrend"] = c["nikkei_uptrend"].astype(str).str.lower().isin(["true", "1", "yes"])
    c = c.dropna(subset=required)
    c = c[(c["date"] >= START_DATE) & (c["date"] <= END_DATE)].copy()

    dates = sorted(c["date"].drop_duplicates().tolist())
    if len(dates) <= OOS_DAYS:
        raise SystemExit(f"❌ 営業日不足: {len(dates)} <= {OOS_DAYS}")

    # multi_oos_profit_gate.py の Fold 4 と同じ「最後のOOS_DAYS営業日」。
    oos_dates = dates[-OOS_DAYS:]
    raw = c[c["date"].isin(oos_dates)].copy()

    print("=" * 90)
    print("🔎 FOLD 4 OOS FILTER TRACE")
    print("目的: UP45_SCORE50_NIKKEIOFF_TP4.0_SL1.0_H7 がどこで減るかを完全可視化")
    print("重要: ゲート条件・探索閾値・policyは変更しない。診断のみ。")
    print("=" * 90)
    print(f"OOS期間: {oos_dates[0].date()} -> {oos_dates[-1].date()} ({len(oos_dates)}営業日)")

    n0 = len(raw)
    x = raw[raw["up_prob"] >= UP].copy(); n1 = len(x)
    x = x[x["up_prob"] > x["down_prob"]].copy(); n2 = len(x)
    x = x[x["flat_prob"] < 50].copy(); n3 = len(x)
    x = x[x["score"] >= SCORE].copy(); n4 = len(x)

    # NIKKEI OFFはフィルターを適用しない。ここを明示する。
    n5 = len(x)

    # multi_oos_profit_validator.py と同じ profit_rank / TOP_N ロジック。
    x["profit_ev"] = (
        (x["up_prob"] / 100.0) * ((x["take_profit"] - x["price"]) / x["price"] * 100.0)
        - (x["down_prob"] / 100.0) * ((x["price"] - x["price"] * 0.0) / x["price"] * 0.0)
    )
    # 上記の stop_loss を使う本体式に合わせる（候補CSVにはstop_lossがある）。
    if "stop_loss" in raw.columns:
        x["stop_loss"] = pd.to_numeric(x["stop_loss"], errors="coerce")
        x["profit_ev"] = (
            (x["up_prob"] / 100.0) * ((x["take_profit"] - x["price"]) / x["price"] * 100.0)
            - (x["down_prob"] / 100.0) * ((x["price"] - x["stop_loss"]) / x["price"] * 100.0)
        )
    x["profit_rank"] = (0.70 * x["score"] + 0.30 * x["profit_ev"].clip(-5, 5) * 10).clip(0, 100)
    selected = (
        x.sort_values(["date", "profit_rank", "score", "up_prob"], ascending=[True, False, False, False])
        .groupby("date", group_keys=False).head(TOP_N).copy()
    )
    n6 = len(selected)

    prices = {}
    for ticker in selected["ticker"].drop_duplicates().tolist():
        p = download(ticker)
        if p is not None and not p.empty:
            prices[ticker] = p
        time.sleep(0.03)

    resolved = []
    missing_price = []
    no_future = []
    for _, r in selected.iterrows():
        ticker = r["ticker"]
        p = prices.get(ticker)
        if p is None:
            missing_price.append(ticker)
            continue
        future = p[p.index > r["date"]].head(HOLD)
        if future.empty:
            no_future.append(ticker)
            continue
        atr_ratio = max(0.01, min(20.0, ((float(r["take_profit"]) / float(r["price"]) - 1.0) / 3.0) * 100.0))
        take = float(r["price"]) * (1.0 + atr_ratio / 100.0 * TP)
        stop = float(r["price"]) * (1.0 - atr_ratio / 100.0 * SL)
        resolved.append((ticker, r["date"], take, stop))

    n7 = len(resolved)

    rows = [
        {"stage": "RAW_OOS", "count": n0, "excluded_from_previous": 0},
        {"stage": f"UP>={UP}", "count": n1, "excluded_from_previous": n0 - n1},
        {"stage": "UP>DOWN", "count": n2, "excluded_from_previous": n1 - n2},
        {"stage": "FLAT<50", "count": n3, "excluded_from_previous": n2 - n3},
        {"stage": f"SCORE>={SCORE}", "count": n4, "excluded_from_previous": n3 - n4},
        {"stage": "NIKKEI_OFF(no_filter)", "count": n5, "excluded_from_previous": 0},
        {"stage": f"TOP_N<={TOP_N}_per_day", "count": n6, "excluded_from_previous": n5 - n6},
        {"stage": f"TP_SL_resolved(TP{TP}_SL{SL}_H{HOLD})", "count": n7, "excluded_from_previous": n6 - n7},
    ]
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    for r in rows:
        print(f"{r['stage']:<32} {r['count']:>7,d}   (-{r['excluded_from_previous']:,})")
    print("=" * 90)
    print(f"FINAL TP/SL resolved: {n7}")
    print(f"価格データ欠落: {len(missing_price)}")
    print(f"将来価格なし: {len(no_future)}")
    print(f"診断CSV: {OUT_FILE}")
    print("※NIKKEI OFFは条件なし。ONの場合のみnikkei_uptrend=Trueを適用します。")


if __name__ == "__main__":
    main()
