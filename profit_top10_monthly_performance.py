import os
import pandas as pd
import numpy as np

HISTORY_FILE = "profit_top10_paper_history.csv"
OUTPUT_FILE = "profit_top10_monthly_performance.csv"


def max_drawdown_pct(equity):
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    return float(-dd.min())


def main():
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=[
            "month", "trades", "wins", "win_rate_pct", "profit_factor",
            "monthly_return_pct", "cumulative_return_pct", "max_drawdown_pct"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("TOP10月次: 履歴なし")
        return

    df = pd.read_csv(HISTORY_FILE)
    if df.empty or "exit_date" not in df.columns or "total_assets" not in df.columns:
        print("TOP10月次: 集計対象なし")
        return

    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["total_assets"] = pd.to_numeric(df["total_assets"], errors="coerce")
    df["pnl"] = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["exit_date", "total_assets"]).sort_values("exit_date")
    if df.empty:
        print("TOP10月次: 集計対象なし")
        return

    df["month"] = df["exit_date"].dt.to_period("M").astype(str)
    rows = []
    previous_assets = None
    cumulative_base = float(df.iloc[0]["total_assets"] - df.iloc[0]["pnl"])
    if cumulative_base <= 0:
        cumulative_base = float(df.iloc[0]["total_assets"])

    for month, g in df.groupby("month", sort=True):
        assets = g["total_assets"].astype(float)
        pnl = g["pnl"].astype(float)
        wins = int((pnl > 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

        start_assets = previous_assets if previous_assets is not None else cumulative_base
        end_assets = float(assets.iloc[-1])
        monthly_return = (end_assets / start_assets - 1.0) * 100.0 if start_assets else 0.0
        cumulative_return = (end_assets / cumulative_base - 1.0) * 100.0 if cumulative_base else 0.0

        # 決済時点の total_assets は共有資産の実残高なので、return_pct の単純複利は使わない。
        # 月内の決済後資産推移から最大DDを算出する。
        equity = pd.Series([start_assets] + assets.tolist(), dtype=float)
        dd = max_drawdown_pct(equity)

        rows.append({
            "month": month,
            "trades": int(len(g)),
            "wins": wins,
            "win_rate_pct": round(wins / len(g) * 100.0, 2) if len(g) else 0.0,
            "profit_factor": round(pf, 3) if np.isfinite(pf) else "inf",
            "monthly_return_pct": round(monthly_return, 3),
            "cumulative_return_pct": round(cumulative_return, 3),
            "max_drawdown_pct": round(dd, 3),
        })
        previous_assets = end_assets

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("month    trades  wins  win_rate_pct  profit_factor  monthly_return_pct  cumulative_return_pct  max_drawdown_pct")
    for _, r in out.iterrows():
        print(f"{r['month']} {int(r['trades']):7d} {int(r['wins']):5d} {float(r['win_rate_pct']):13.2f} {r['profit_factor']:14} {float(r['monthly_return_pct']):19.3f} {float(r['cumulative_return_pct']):22.3f} {float(r['max_drawdown_pct']):18.3f}")


if __name__ == "__main__":
    main()
