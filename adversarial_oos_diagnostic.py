import os
from pathlib import Path
import pandas as pd

INPUT = Path("adversarial_oos_results.csv")
OUTPUT = Path("adversarial_oos_diagnostics.csv")
NEAR = Path("adversarial_oos_near_miss.csv")
MIN_OOS_TRADES = int(os.getenv("WF_MIN_OOS_TRADES", "20"))
MIN_OOS_PF = float(os.getenv("WF_MIN_OOS_PF", "1.0"))
MIN_OOS_AVG_RETURN = float(os.getenv("WF_MIN_OOS_AVG_RETURN", "0.0"))
MIN_MONTHLY_POSITIVE_RATIO = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55")) * 100.0
MAX_OOS_DD = float(os.getenv("WF_MAX_OOS_DD", "35.0"))
MIN_POSITIVE_FOLDS = int(os.getenv("WF_MIN_POSITIVE_FOLDS", "3"))

if not INPUT.exists():
    print("⚠ adversarial_oos_results.csv がありません。診断をスキップ")
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

x = pd.read_csv(INPUT)
if x.empty:
    pd.DataFrame().to_csv(OUTPUT, index=False)
    print("⚠ OOS診断対象が0件")
    raise SystemExit(0)

# multi_oos_profit_validator.py の現行 aggregate 列名に正規化
required = ["oos_signals", "oos_pf_mean", "oos_avg_return_mean", "oos_monthly_positive_ratio_mean", "oos_worst_dd", "oos_compound_return"]
missing = [c for c in required if c not in x.columns]
if missing:
    print("⚠ 現行OOS結果に必要列がありません: " + ", ".join(missing))
    print("利用可能列: " + ", ".join(x.columns))
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

for c in required + ["positive_oos_folds", "oos_pf_ratio"]:
    if c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

x["oos_pf"] = x["oos_pf_mean"]
x["oos_avg_return"] = x["oos_avg_return_mean"]
x["oos_monthly_positive_ratio"] = x["oos_monthly_positive_ratio_mean"]
x["oos_dd"] = x["oos_worst_dd"]

x["gate_trades"] = x["oos_signals"] >= MIN_OOS_TRADES
x["gate_pf"] = x["oos_pf"] >= MIN_OOS_PF
x["gate_avg_return"] = x["oos_avg_return"] > MIN_OOS_AVG_RETURN
x["gate_monthly_positive"] = x["oos_monthly_positive_ratio"] >= MIN_MONTHLY_POSITIVE_RATIO
x["gate_dd"] = x["oos_dd"] >= -MAX_OOS_DD
x["gate_compound"] = x["oos_compound_return"] > 0

if "positive_oos_folds" in x.columns:
    x["gate_positive_folds"] = x["positive_oos_folds"] >= MIN_POSITIVE_FOLDS
else:
    x["gate_positive_folds"] = True

# 旧 oos_pf_ratio は現行 aggregate に存在しないため捏造しない
if "oos_pf_ratio" in x.columns:
    x["gate_pf_ratio"] = x["oos_pf_ratio"] >= float(os.getenv("WF_MIN_OOS_TO_VALIDATION_PF", "0.60"))
    pf_ratio_note = "現行結果のoos_pf_ratioを使用"
else:
    x["gate_pf_ratio"] = True
    x["oos_pf_ratio"] = float("nan")
    pf_ratio_note = "判定不能（現行aggregateにoos_pf_ratioなし）"

x["oos_gates_passed"] = x[["gate_trades", "gate_pf", "gate_avg_return", "gate_monthly_positive", "gate_dd", "gate_compound", "gate_positive_folds"]].sum(axis=1)
x["oos_gates_total"] = 7
x["oos_gates_failed"] = x["oos_gates_total"] - x["oos_gates_passed"]


def reasons(r):
    out = []
    if not r.gate_trades: out.append(f"件数<{MIN_OOS_TRADES}")
    if not r.gate_pf: out.append(f"PF<{MIN_OOS_PF}")
    if not r.gate_avg_return: out.append("平均利益<=0")
    if not r.gate_pf_ratio: out.append("OOS/Val PF比が基準未満")
    if not r.gate_monthly_positive: out.append(f"月間プラス率<{MIN_MONTHLY_POSITIVE_RATIO:.1f}%")
    if not r.gate_dd: out.append(f"DD>{MAX_OOS_DD:.1f}%")
    if not r.gate_compound: out.append("複利リターン<=0")
    if not r.gate_positive_folds: out.append(f"プラスOOS Fold数<{MIN_POSITIVE_FOLDS}")
    return " / ".join(out) if out else "PASS"

x["oos_failure_reasons"] = x.apply(reasons, axis=1)
x["pf_ratio_diagnostic"] = pf_ratio_note
x = x.sort_values(["oos_gates_passed", "oos_compound_return", "oos_monthly_positive_ratio", "oos_pf", "oos_signals"], ascending=False).reset_index(drop=True)
x.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
x.head(20).to_csv(NEAR, index=False, encoding="utf-8-sig")

print("=" * 80)
print("🔎 OOS GATE DIAGNOSTIC（現行 multi-OOS 対応）")
print(f"候補数: {len(x)}")
print(f"主要7ゲート全通過: {int((x.oos_gates_failed == 0).sum())}")
print(f"PF比診断: {pf_ratio_note}")
for _, r in x.head(10).iterrows():
    folds = int(r["positive_oos_folds"]) if "positive_oos_folds" in r and pd.notna(r["positive_oos_folds"]) else "-"
    print(f"{r['strategy']} | 通過={int(r['oos_gates_passed'])}/7 | 件数={int(r['oos_signals'])} PF={r['oos_pf']:.2f} 平均={r['oos_avg_return']:+.3f}% 月間+={r['oos_monthly_positive_ratio']:.1f}% DD={r['oos_dd']:.2f}% 複利={r['oos_compound_return']:+.2f}% プラスFold={folds} | FAIL: {r['oos_failure_reasons']}")

webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
if webhook:
    try:
        import requests
        lines = ["🔎 AI OOS GATE DIAGNOSTIC", f"候補={len(x)} 主要7ゲート全通過={int((x.oos_gates_failed == 0).sum())}"]
        for _, r in x.head(5).iterrows():
            lines.append(f"{r['strategy']} | {int(r['oos_gates_passed'])}/7 | 件数{int(r['oos_signals'])} PF{r['oos_pf']:.2f} 平均{r['oos_avg_return']:+.3f}% 月間+{r['oos_monthly_positive_ratio']:.1f}% DD{r['oos_dd']:.1f}%\nNG: {r['oos_failure_reasons']}")
        requests.post(webhook, json={"content": "\n".join(lines)[:1900]}, timeout=30).raise_for_status()
        print("✅ Discord診断通知送信成功")
    except Exception as e:
        print("Discord診断通知エラー:", e)
