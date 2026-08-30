import os
from pathlib import Path

import numpy as np
import pandas as pd

# ★修正(2026-08): 旧パイプライン(adversarial_strategy_validator.py)時代の
# 単一OOS前提のゲート("oos_pf_ratio"列)を参照せず、現行の
# multi_oos_profit_validator.py(4-fold集計)の出力を直接診断する。
# ゲート条件は現行multi_oos_profit_validator.pyと同一の6条件:
#   1. fold数  2. 陽性fold数  3. 合計取引数
#   4. 複利リターン  5. 月次プラス率  6. 最大DD
# 閾値・合否ロジック自体は変更しない。

INPUT = Path("adversarial_multi_oos_folds.csv")
OUTPUT = Path("adversarial_oos_diagnostics.csv")
NEAR = Path("adversarial_oos_near_miss.csv")

FOLDS = int(os.getenv("WF_OOS_FOLDS", "4"))
MIN_TOTAL_TRADES = int(os.getenv("WF_MIN_TOTAL_OOS_TRADES", "20"))
MIN_MONTHLY = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55")) * 100
MAX_DD = float(os.getenv("WF_MAX_OOS_DD", "35"))
MIN_POSITIVE_FOLDS = int(os.getenv("WF_MIN_POSITIVE_FOLDS", "3"))

if not INPUT.exists() or INPUT.stat().st_size == 0:
    print(f"⚠ {INPUT} がありません(または空)。診断をスキップ")
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

try:
    fold_table = pd.read_csv(INPUT)
except pd.errors.EmptyDataError:
    print(f"⚠ {INPUT} に列データがありません(全fold・全戦略が候補0件)。診断をスキップ")
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

if fold_table.empty:
    pd.DataFrame().to_csv(OUTPUT, index=False)
    print("⚠ OOS診断対象が0件(fold結果自体が空)")
    raise SystemExit(0)

required = [
    "strategy", "fold", "oos_signals", "oos_pf", "oos_avg_return",
    "oos_monthly_positive_ratio", "oos_dd", "oos_compound_return",
]
missing = [c for c in required if c not in fold_table.columns]
if missing:
    print("⚠ 現行multi-OOS結果に必要列がありません: " + ", ".join(missing))
    print("利用可能列: " + ", ".join(fold_table.columns))
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

rows = []
for name, g in fold_table.groupby("strategy"):
    folds_seen = len(g)
    positive = int((pd.to_numeric(g.oos_compound_return, errors="coerce") > 0).sum())
    total = int(pd.to_numeric(g.oos_signals, errors="coerce").fillna(0).sum())
    pfv = pd.to_numeric(g.oos_pf, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    avg = pd.to_numeric(g.oos_avg_return, errors="coerce")
    compound = float(((1 + avg.fillna(0) / 100).prod() - 1) * 100)
    monthly = float(pd.to_numeric(g.oos_monthly_positive_ratio, errors="coerce").mean())
    worst = float(pd.to_numeric(g.oos_dd, errors="coerce").min())

    gate_folds = folds_seen >= FOLDS
    gate_positive = positive >= MIN_POSITIVE_FOLDS
    gate_trades = total >= MIN_TOTAL_TRADES
    gate_compound = compound > 0
    gate_monthly = monthly >= MIN_MONTHLY
    gate_dd = worst >= -MAX_DD

    reasons = []
    if not gate_folds:
        reasons.append(f"fold数不足({folds_seen}/{FOLDS})")
    if not gate_positive:
        reasons.append(f"陽性fold数不足({positive}/{FOLDS}、必要{MIN_POSITIVE_FOLDS})")
    if not gate_trades:
        reasons.append(f"合計取引数<{MIN_TOTAL_TRADES}({total})")
    if not gate_compound:
        reasons.append(f"複利リターン<=0({compound:+.2f}%)")
    if not gate_monthly:
        reasons.append(f"月間プラス率<{MIN_MONTHLY:.1f}%({monthly:.1f}%)")
    if not gate_dd:
        reasons.append(f"DD>{MAX_DD:.1f}%({worst:.2f}%)")

    gates_passed = sum([
        gate_folds,
        gate_positive,
        gate_trades,
        gate_compound,
        gate_monthly,
        gate_dd,
    ])

    rows.append({
        "strategy": name,
        "folds_seen": folds_seen,
        "positive_oos_folds": positive,
        "oos_signals_total": total,
        "oos_pf_mean": float(pfv.mean()) if len(pfv) else 0.0,
        "oos_pf_min": float(pfv.min()) if len(pfv) else 0.0,
        "oos_avg_return_mean": float(avg.mean()),
        "oos_monthly_positive_ratio_mean": monthly,
        "oos_worst_dd": worst,
        "oos_compound_return": compound,
        "gates_passed": int(gates_passed),
        "gates_total": 6,
        "failure_reasons": " / ".join(reasons) if reasons else "PASS",
    })

x = pd.DataFrame(rows).sort_values(
    [
        "gates_passed",
        "oos_compound_return",
        "oos_monthly_positive_ratio_mean",
        "oos_pf_mean",
        "oos_signals_total",
    ],
    ascending=[False, False, False, False, False],
).reset_index(drop=True)

x.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
x.head(20).to_csv(NEAR, index=False, encoding="utf-8-sig")

print("=" * 80)
print("🔎 OOS GATE DIAGNOSTIC(4-fold集計後、現行ゲート基準)")
print(f"候補戦略数: {len(x)}")
print(f"全6ゲート通過: {int((x.gates_passed == 6).sum())}")
for _, r in x.head(10).iterrows():
    print(
        f"{r['strategy']} | 通過={int(r['gates_passed'])}/6 | "
        f"陽性fold={int(r['positive_oos_folds'])}/{FOLDS} "
        f"合計件数={int(r['oos_signals_total'])} "
        f"PF={r['oos_pf_mean']:.2f} "
        f"月間+={r['oos_monthly_positive_ratio_mean']:.1f}% "
        f"DD={r['oos_worst_dd']:.2f}% "
        f"複利={r['oos_compound_return']:+.2f}% | "
        f"FAIL: {r['failure_reasons']}"
    )

webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
if webhook and not x.empty:
    try:
        import requests

        lines = [
            "🔎 AI OOS GATE DIAGNOSTIC(4-fold集計)",
            f"候補={len(x)} 全6ゲート通過={int((x.gates_passed == 6).sum())}",
        ]
        for _, r in x.head(5).iterrows():
            lines.append(
                f"{r['strategy']} | {int(r['gates_passed'])}/6 | "
                f"陽性fold{int(r['positive_oos_folds'])}/{FOLDS} "
                f"件数{int(r['oos_signals_total'])} "
                f"PF{r['oos_pf_mean']:.2f} "
                f"月間+{r['oos_monthly_positive_ratio_mean']:.1f}% "
                f"DD{r['oos_worst_dd']:.1f}%\n"
                f"NG: {r['failure_reasons']}"
            )
        requests.post(
            webhook,
            json={"content": "\n".join(lines)[:1900]},
            timeout=30,
        ).raise_for_status()
        print("✅ Discord診断通知送信成功")
    except Exception as e:
        print("Discord診断通知エラー:", e)
