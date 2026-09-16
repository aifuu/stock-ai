#!/usr/bin/env python3
"""TOP10ペーパートレードが実際に選んだ銘柄の、銘柄別・事後パフォーマンス分析。

profit_top10_paper.py の実売買ログ(profit_top10_paper_history.csv)だけを読み、
銘柄(ticker)別に取引回数・勝率・平均リターン・累積pnl・決済理由の内訳を集計する。
既存の代替候補(trade_feedback_engine.py=グループ単位集計のみ、
daily_movers_root_cause.py=選定パスと意図的に独立、
profit_top10_monthly_performance.py=ポートフォリオ全体の月次集計のみ、
selection_audit.py=選定時点のスコア監査)はいずれも銘柄別の事後損益を出さないため、
このスクリプトはそのギャップを埋める観察専用のレポートである。
本番の選定/執行ロジック(profit_top10_paper.py, strategy_policy.json,
daily_directional_top1.py)には一切書き込まない。読むのはHISTORY_FILEのみ、
書くのはOUTPUT_FILEのみ。
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HISTORY_FILE = "profit_top10_paper_history.csv"
OUTPUT_FILE = "top10_ticker_performance.csv"
JST = ZoneInfo("Asia/Tokyo")

# 直近何日を「recent」窓とみなすか(環境変数で調整可)。
RECENT_LOOKBACK_DAYS = int(os.getenv("TICKER_PERF_LOOKBACK_DAYS", "30"))
# この取引回数未満の銘柄は「参考程度」として明示的にフラグを立てる。
MIN_TRADES_FOR_RELIABLE = int(os.getenv("TICKER_PERF_MIN_TRADES", "5"))
# ベスト/ワーストランキングに表示する件数。
RANKING_TOP_N = int(os.getenv("TICKER_PERF_RANKING_TOP_N", "5"))
# 履歴全体の決済件数がこれ未満なら、レポート全体に「サンプル不足」注記を出す。
GLOBAL_MIN_SAMPLE_WARN = int(os.getenv("TICKER_PERF_GLOBAL_MIN_SAMPLE_WARN", "20"))

COLUMNS = [
    "as_of", "period", "ticker", "company", "trades", "win_rate_pct",
    "avg_return_pct", "total_pnl", "tp_count", "sl_count",
    "other_result_breakdown", "low_sample",
]


def discord_send(message: str) -> bool:
    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook:
        print("ℹ️ DISCORD_WEBHOOK 未設定のため通知スキップ")
        return False
    try:
        import requests
        r = requests.post(webhook, json={"content": message[:1950]}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Discord通知失敗: {e}")
        return False


def _load_closed_history() -> pd.DataFrame:
    path = Path(HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠ {HISTORY_FILE} 読み込み失敗: {e}")
        return pd.DataFrame()

    required = {"exit_date", "ticker", "pnl", "return_pct", "result"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df = df.dropna(subset=["exit_date", "ticker", "pnl", "return_pct"]).copy()
    if "company" not in df.columns:
        df["company"] = df["ticker"]
    df["company"] = df["company"].fillna(df["ticker"])
    if "exit_time" not in df.columns:
        df["exit_time"] = ""

    # append-onlyレジャーへのCIリトライによる重複行を防ぐ(profit_top10_monthly_performance.py と同じ対策)。
    identity = [c for c in ("entry_date", "entry_time", "ticker", "exit_date", "exit_time") if c in df.columns]
    if identity:
        df = df.drop_duplicates(subset=identity, keep="last")

    return df.sort_values(["exit_date", "exit_time"], kind="stable").reset_index(drop=True)


def _other_result_breakdown(results: pd.Series) -> str:
    others = results[~results.isin(["TP", "SL"])]
    if others.empty:
        return ""
    counts = others.value_counts()
    return ";".join(f"{reason}:{n}" for reason, n in counts.items())


def _aggregate(df: pd.DataFrame, period_label: str, as_of: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for ticker, g in df.groupby("ticker", dropna=False):
        n = len(g)
        results = g["result"].astype(str)
        rows.append({
            "as_of": as_of,
            "period": period_label,
            "ticker": ticker,
            "company": str(g["company"].iloc[-1]),
            "trades": n,
            "win_rate_pct": round(float((g["pnl"] > 0).mean() * 100.0), 2),
            "avg_return_pct": round(float(g["return_pct"].mean()), 3),
            "total_pnl": round(float(g["pnl"].sum()), 2),
            "tp_count": int((results == "TP").sum()),
            "sl_count": int((results == "SL").sum()),
            "other_result_breakdown": _other_result_breakdown(results),
            "low_sample": n < MIN_TRADES_FOR_RELIABLE,
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def _format_ranking_lines(period_df: pd.DataFrame, title: str) -> list[str]:
    if period_df.empty:
        return [f"{title}: 該当データなし"]
    lines = [title]
    best = period_df.head(RANKING_TOP_N)
    worst = period_df.tail(RANKING_TOP_N).iloc[::-1]
    lines.append("  [ベスト]")
    for _, r in best.iterrows():
        flag = "※参考(サンプル不足)" if r["low_sample"] else ""
        lines.append(
            f"    {r['ticker']} {r['company']}: 累積pnl {r['total_pnl']:+,.0f} | "
            f"取引{int(r['trades'])}件 勝率{r['win_rate_pct']:.0f}% 平均{r['avg_return_pct']:+.2f}% {flag}"
        )
    if len(period_df) > 1:
        lines.append("  [ワースト]")
        for _, r in worst.iterrows():
            flag = "※参考(サンプル不足)" if r["low_sample"] else ""
            lines.append(
                f"    {r['ticker']} {r['company']}: 累積pnl {r['total_pnl']:+,.0f} | "
                f"取引{int(r['trades'])}件 勝率{r['win_rate_pct']:.0f}% 平均{r['avg_return_pct']:+.2f}% {flag}"
            )
    return lines


def main() -> int:
    now = datetime.now(JST)
    as_of = now.strftime("%Y-%m-%d")
    df = _load_closed_history()

    if df.empty:
        print(f"ℹ️ {HISTORY_FILE} に有効な決済レコードがありません → ヘッダのみ出力")
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return 0

    total_trades = len(df)
    cutoff = pd.Timestamp(now.astimezone(JST).replace(tzinfo=None)) - pd.Timedelta(days=RECENT_LOOKBACK_DAYS)
    recent_df = df[df["exit_date"] >= cutoff]

    all_agg = _aggregate(df, "all", as_of)
    recent_agg = _aggregate(recent_df, f"recent_{RECENT_LOOKBACK_DAYS}d", as_of)

    out = pd.concat([all_agg, recent_agg], ignore_index=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    low_sample_note = (
        f"⚠️ 現在の履歴は決済{total_trades}件のみ(1銘柄あたり最小{MIN_TRADES_FOR_RELIABLE}件未満は"
        f"low_sample=Trueで区別)。以下は参考情報であり、統計的な信頼性はまだありません。"
        if total_trades < GLOBAL_MIN_SAMPLE_WARN else ""
    )

    console_lines = ["=" * 90, f"📊 TOP10銘柄別・事後パフォーマンス分析 | as_of={as_of} | 全決済{total_trades}件"]
    if low_sample_note:
        console_lines.append(low_sample_note)
    console_lines.append("=" * 90)
    console_lines += _format_ranking_lines(all_agg, f"【全期間】銘柄別ランキング(累積pnl順・{len(all_agg)}銘柄)")
    console_lines += _format_ranking_lines(recent_agg, f"【直近{RECENT_LOOKBACK_DAYS}日】銘柄別ランキング(累積pnl順・{len(recent_agg)}銘柄)")
    console_text = "\n".join(console_lines)
    print(console_text)
    print(f"✅ {OUTPUT_FILE} を更新 ({len(out)}行)")

    discord_lines = [f"📊 TOP10銘柄別・事後パフォーマンス分析 ({as_of})", f"全決済{total_trades}件"]
    if low_sample_note:
        discord_lines.append(low_sample_note)
    discord_lines.append("")
    discord_lines += _format_ranking_lines(all_agg, f"【全期間】(累積pnl順・{len(all_agg)}銘柄)")
    discord_lines.append("")
    discord_lines += _format_ranking_lines(recent_agg, f"【直近{RECENT_LOOKBACK_DAYS}日】(累積pnl順・{len(recent_agg)}銘柄)")
    discord_send("\n".join(discord_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
