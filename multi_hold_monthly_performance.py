#!/usr/bin/env python3
"""Monthly performance aggregation for the 1d/3d/5d multi-hold paper trader.

Reads multi_hold_paper_history.csv and writes multi_hold_monthly_performance.csv.
The history is treated as an append-only CLOSED-trade ledger.  Each hold bucket
is replayed independently from 1.0 equity in exit_date order.

Important: the ALL row is an aggregate trade-statistics view.  It is NOT treated
as a single executable portfolio because the 1d/3d/5d buckets are independent
capital buckets and can contain the same daily TOP1 at the same time.
"""
from pathlib import Path
import os
import pandas as pd

HISTORY_FILE = "multi_hold_paper_history.csv"
OUTPUT_FILE = "multi_hold_monthly_performance.csv"
HOLDS = (1, 3, 5)

# 月間+5%目標(元本100万円なら+5万円/月)。勝率ではなく月次収益率で評価する。
MONTHLY_TARGET_PCT = 5.0
# 1d/3d/5dバケットはそれぞれequity=1.0から独立再生される正規化リターンのため、
# 円換算はAI_INITIAL_CAPITAL(既定100万円)を想定元本として使う近似値。
NOTIONAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))

COLUMNS = [
    "month", "hold_days", "trades", "profit_factor", "monthly_return_pct",
    "monthly_profit_jpy", "monthly_plus5_achieved",
    "cumulative_return_pct", "max_drawdown_pct",
]


def _profit_factor(returns_pct):
    gains = sum(float(r) for r in returns_pct if float(r) > 0)
    losses = -sum(float(r) for r in returns_pct if float(r) < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _compound_curve(returns_pct):
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    curve = []
    for raw in returns_pct:
        r = float(raw)
        equity *= 1.0 + r / 100.0
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append(equity)
    return curve, max_dd


def _bucket_rows(df, label):
    if df.empty:
        return []

    df = df.sort_values(["exit_date", "ticker"], kind="stable").copy()
    df["month"] = df["exit_date"].dt.strftime("%Y-%m")

    # Full-history equity for the cumulative-return column.
    full_curve, _ = _compound_curve(df["return_pct"].tolist())
    df["cum_equity"] = full_curve

    rows = []
    for month, g in df.groupby("month", sort=True):
        returns = g["return_pct"].astype(float).tolist()
        trades = len(returns)
        pf = _profit_factor(returns)
        monthly_curve, monthly_dd = _compound_curve(returns)
        monthly_return = (monthly_curve[-1] - 1.0) * 100.0 if monthly_curve else 0.0
        cumulative_return = (float(g["cum_equity"].iloc[-1]) - 1.0) * 100.0

        rows.append({
            "month": month,
            "hold_days": label,
            "trades": trades,
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "monthly_return_pct": round(monthly_return, 3),
            "monthly_profit_jpy": round(monthly_return / 100.0 * NOTIONAL_CAPITAL, 0),
            "monthly_plus5_achieved": monthly_return >= MONTHLY_TARGET_PCT,
            "cumulative_return_pct": round(cumulative_return, 3),
            "max_drawdown_pct": round(monthly_dd, 3),
        })
    return rows


def _load_closed_history():
    path = Path(HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty or "exit_date" not in df.columns or "return_pct" not in df.columns:
        return pd.DataFrame()

    # Normalize types before filtering.  This prevents CSV string/int mismatches
    # from silently dropping 1d/3d/5d records.
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    if "hold_days" in df.columns:
        df["hold_days"] = pd.to_numeric(df["hold_days"], errors="coerce").astype("Int64")
    else:
        df["hold_days"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    df = df.dropna(subset=["exit_date", "return_pct", "hold_days"]).copy()
    df["hold_days"] = df["hold_days"].astype(int)
    df = df[df["hold_days"].isin(HOLDS)]

    # Only CLOSED ledger rows belong in performance.  If status is present,
    # explicitly exclude any open/pending records.
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("CLOSED")]

    # Protect the append-only ledger from accidental duplicate rows when a CI
    # retry replays the same close.  Prefer a stable trade identity when fields
    # exist; otherwise keep all rows because return rows are still legitimate.
    identity = [c for c in ("entry_date", "entry_time", "ticker", "hold_days", "exit_date") if c in df.columns]
    if identity:
        df = df.drop_duplicates(subset=identity, keep="last")

    return df.sort_values(["exit_date", "ticker"], kind="stable").reset_index(drop=True)


def main():
    df = _load_closed_history()

    if df.empty:
        print(f"ℹ️ {HISTORY_FILE} に有効な決済レコードがありません → ヘッダのみ出力")
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return 0

    rows = []
    for h in HOLDS:
        rows.extend(_bucket_rows(df[df["hold_days"] == h], str(h)))

    # Aggregate ALL view.  It is intentionally a statistics view rather than
    # a fake single portfolio, because each hold bucket has separate capital.
    rows.extend(_bucket_rows(df, "ALL"))

    out = pd.DataFrame(rows, columns=COLUMNS)
    hold_order = {"1": 1, "3": 2, "5": 3, "ALL": 4}
    out["_hold_order"] = out["hold_days"].map(hold_order).fillna(99)
    out = out.sort_values(["month", "_hold_order"], kind="stable").drop(columns="_hold_order")
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    latest_month = out["month"].max()
    latest = out[(out["month"] == latest_month) & (out["hold_days"] == "ALL")]
    if not latest.empty:
        r = latest.iloc[0]
        sign = "プラス" if float(r["monthly_return_pct"]) >= 0 else "マイナス"
        plus5_text = "✅達成" if bool(r["monthly_plus5_achieved"]) else "未達成"
        print(
            f"✅ {latest_month} 月次(ALL): {float(r['monthly_return_pct']):+.2f}% ({sign}) "
            f"| 利益額 ¥{float(r['monthly_profit_jpy']):+,.0f} | 月間+5%目標 {plus5_text} | PF {r['profit_factor']} "
            f"| 最大DD {float(r['max_drawdown_pct']):.2f}% | 取引数 {int(r['trades'])}"
        )

    print(f"✅ {OUTPUT_FILE} を更新 ({len(out)}行) | 決済レコード {len(df)}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
