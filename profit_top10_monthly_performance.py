#!/usr/bin/env python3
"""Monthly performance aggregation for the TOP10 profit-loop paper trader.

Reads profit_top10_paper_history.csv and writes profit_top10_monthly_performance.csv.

Unlike multi_hold_paper (independent 1d/3d/5d capital buckets, each replayed from an
equity of 1.0), profit_top10_paper shares ONE capital pool across up to TOP_N=10
concurrent positions. Re-deriving equity by naively compounding each trade's own
return_pct would double count overlapping exposure. Instead this script uses the
`total_assets` column, which mark_and_close() already records as the real portfolio
capital at the moment of each close - the correct ground truth for TOP10's shared pool.
"""
from pathlib import Path
import pandas as pd

HISTORY_FILE = "profit_top10_paper_history.csv"
OUTPUT_FILE = "profit_top10_monthly_performance.csv"

COLUMNS = [
    "month", "trades", "wins", "win_rate_pct",
    "profit_factor", "monthly_return_pct", "cumulative_return_pct",
    "max_drawdown_pct",
]


def _profit_factor(pnls):
    gains = sum(float(p) for p in pnls if float(p) > 0)
    losses = -sum(float(p) for p in pnls if float(p) < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _load_closed_history():
    path = Path(HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"exit_date", "pnl", "total_assets"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    if "exit_time" not in df.columns:
        df["exit_time"] = ""
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["total_assets"] = pd.to_numeric(df["total_assets"], errors="coerce")
    df = df.dropna(subset=["exit_date", "pnl", "total_assets"]).copy()

    # 決済日時の実際の順序で並べ直す(CSVの追記順やCIリトライで前後することがあるため)。
    df = df.sort_values(["exit_date", "exit_time"], kind="stable").reset_index(drop=True)

    # append-onlyレジャーへのCIリトライによる重複行を防ぐ。
    identity = [c for c in ("entry_date", "entry_time", "ticker", "exit_date", "exit_time") if c in df.columns]
    if identity:
        df = df.drop_duplicates(subset=identity, keep="last")
        df = df.sort_values(["exit_date", "exit_time"], kind="stable").reset_index(drop=True)

    return df


def main():
    df = _load_closed_history()

    if df.empty:
        print(f"ℹ️ {HISTORY_FILE} に有効な決済レコードがありません → ヘッダのみ出力")
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return 0

    df["month"] = df["exit_date"].dt.strftime("%Y-%m")

    # 最初の決済直前の資産を起点(start_capital)とする。total_assetsは決済「後」の値なので
    # 最初の行のpnlを差し引いて逆算する。
    start_capital = float(df["total_assets"].iloc[0]) - float(df["pnl"].iloc[0])
    if start_capital <= 0:
        start_capital = float(df["total_assets"].iloc[0])

    rows = []
    peak = start_capital
    prev_month_end = start_capital
    for month, g in df.groupby("month", sort=True):
        pnls = g["pnl"].tolist()
        trades = len(pnls)
        wins = sum(p > 0 for p in pnls)
        win_rate = wins / trades * 100.0 if trades else 0.0
        pf = _profit_factor(pnls)
        month_end = float(g["total_assets"].iloc[-1])
        monthly_return = (month_end / prev_month_end - 1.0) * 100.0 if prev_month_end > 0 else 0.0
        cumulative_return = (month_end / start_capital - 1.0) * 100.0 if start_capital > 0 else 0.0

        month_dd = 0.0
        for v in g["total_assets"]:
            peak = max(peak, float(v))
            dd = (peak - float(v)) / peak * 100.0 if peak > 0 else 0.0
            month_dd = max(month_dd, dd)

        rows.append({
            "month": month,
            "trades": trades,
            "wins": wins,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "monthly_return_pct": round(monthly_return, 3),
            "cumulative_return_pct": round(cumulative_return, 3),
            "max_drawdown_pct": round(month_dd, 3),
        })
        prev_month_end = month_end

    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    latest = out.iloc[-1]
    sign = "プラス" if float(latest["monthly_return_pct"]) >= 0 else "マイナス"
    print(
        f"✅ {latest['month']} 月次(TOP10): {float(latest['monthly_return_pct']):+.2f}% ({sign}) "
        f"| 勝率 {float(latest['win_rate_pct']):.1f}% | PF {latest['profit_factor']} "
        f"| 最大DD {float(latest['max_drawdown_pct']):.2f}% | 取引数 {int(latest['trades'])}"
    )
    print(f"✅ {OUTPUT_FILE} を更新 ({len(out)}行) | 決済レコード {len(df)}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
