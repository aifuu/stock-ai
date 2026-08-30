import os
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("adversarial_oos_results.csv")
OUTPUT = Path("adversarial_oos_diagnostics.csv")
NEAR = Path("adversarial_oos_near_miss.csv")

MIN_OOS_TRADES = int(os.getenv("WF_MIN_OOS_TRADES", "20"))
MIN_OOS_PF = float(os.getenv("WF_MIN_OOS_PF", "1.0"))
MIN_OOS_AVG_RETURN = float(os.getenv("WF_MIN_OOS_AVG_RETURN", "0.0"))
MIN_OOS_TO_VALIDATION_PF = float(os.getenv("WF_MIN_OOS_TO_VALIDATION_PF", "0.60"))
MIN_MONTHLY_POSITIVE_RATIO = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55")) * 100.0
MAX_OOS_DD = float(os.getenv("WF_MAX_OOS_DD", "35.0"))

if not INPUT.exists():
    print("⚠ adversarial_oos_results.csv がありません。診断をスキップ")
    pd.DataFrame().to_csv(OUTPUT, index=False)
    raise SystemExit(0)

x = pd.read_csv(INPUT)
if x.empty:
    pd.DataFrame().to_csv(OUTPUT, index=False)
    print("⚠ OOS診断対象が0件")
    raise SystemExit(0)

num_cols = [
    "oos_signals", "oos_pf", "oos_avg_return", "oos_pf_ratio",
    "oos_monthly_positive_ratio", "oos_dd", "oos_compound_return",
    "validation_pf"
]
for c in num_cols:
    if c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

x["gate_trades"] = x["oos_signals"] >= MIN_OOS_TRADES
x["gate_pf"] = x["oos_pf"] >= MIN_OOS_PF
x["gate_avg_return"] = x["oos_avg_return"] > MIN_OOS_AVG_RETURN
x["gate_pf_ratio"] = x["oos_pf_ratio"] >= MIN_OOS_TO_VALIDATION_PF
x["gate_monthly_positive"] = x["oos_monthly_positive_ratio"] >= MIN_MONTHLY_POSITIVE_RATIO
x["gate_dd"] = x["oos_dd"] >= -MAX_OOS_DD
x["gate_compound"] = x["oos_compound_return"] > 0

gates = [
    "gate_trades", "gate_pf", "gate_avg_return", "gate_pf_ratio",
    "gate_monthly_positive", "gate_dd", "gate_compound"
]
x["oos_gates_passed"] = x[gates].sum(axis=1)
x["oos_gates_failed"] = len(gates) - x["oos_gates_passed"]

def reasons(r):
    out = []
    if not r.gate_trades:
        out.append(f"件数<{MIN_OOS_TRADES}")
    if not r.gate_pf:
        out.append(f"PF<{MIN_OOS_PF}")
    if not r.gate_avg_return:
        out.append("平均利益<=0")
    if not r.gate_pf_ratio:
        out.append(f"OOS/Val PF比<{MIN_OOS_TO_VALIDATION_PF:.2f}")
    if not r.gate_monthly_positive:
        out.append(f"月間プラス率<{MIN_MONTHLY_POSITIVE_RATIO:.1f}%")
    if not r.gate_dd:
        out.append(f"DD>{MAX_OOS_DD:.1f}%")
    if not r.gate_compound:
        out.append("複利リターン<=0")
    return " / ".join(out) if out else "PASS"

x["oos_failure_reasons"] = x.apply(reasons, axis=1)
x = x.sort_values(
    ["oos_gates_passed", "oos_compound_return", "oos_monthly_positive_ratio", "oos_pf", "oos_signals"],
    ascending=[False, False, False, False, False],
).reset_index(drop=True)
x.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
x.head(20).to_csv(NEAR, index=False, encoding="utf-8-sig")

print("=" * 80)
print("🔎 OOS GATE DIAGNOSTIC")
print(f"候補数: {len(x)}")
print(f"全7ゲート通過: {int((x.oos_gates_passed == 7).sum())}
")
for _, r in x.head(10).iterrows():
    print(
        f"{r['strategy']} | 通過={int(r['oos_gates_passed'])}/7 | "
        f"件数={int(r['oos_signals'])} PF={r['oos_pf']:.2f} "
        f"平均={r['oos_avg_return']:+.3f}% "
        f"月間+={r['oos_monthly_positive_ratio']:.1f}% "
        f"DD={r['oos_dd']:.2f}% "
        f"複利={r['oos_compound_return']:+.2f}% | "
        f"FAIL: {r['oos_failure_reasons']}"
    )

webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
if webhook:
    try:
        import requests
        lines = ["🔎 AI OOS GATE DIAGNOSTIC", f"候補={len(x)} 全7ゲート={int((x.oos_gates_passed == 7).sum())}"]
        for _, r in x.head(5).iterrows():
            lines.append(
                f"{r['strategy']} | {int(r['oos_gates_passed'])}/7 | "
                f"件数{int(r['oos_signals'])} PF{r['oos_pf']:.2f} "
                f"月間+{r['oos_monthly_positive_ratio']:.1f}% DD{r['oos_dd']:.1f}%\n"
                f"  NG: {r['oos_failure_reasons']}"
            )
        requests.post(webhook, json={"content": "\n".join(lines)[:1900]}, timeout=30)
    except Exception as e:
        print("Discord診断通知エラー:", e)
