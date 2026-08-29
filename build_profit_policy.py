import json
import os
from datetime import datetime
import pandas as pd

INPUT_FILE = "adversarial_final_candidates.csv"
POLICY_FILE = "strategy_policy.json"
MIN_MONTHLY_POSITIVE_RATIO = 55.0
MIN_OOS_PF = 1.0
MIN_OOS_AVG_RETURN = 0.0
MAX_OOS_DD = 35.0

DEFAULT_POLICY = {
    "status": "DEFAULT",
    "updated_at": None,
    "up_threshold": 50,
    "min_score_for_buy": 60,
    "nikkei_filter": False,
    "atr_tp_multiplier": 3.0,
    "atr_sl_multiplier": 1.5,
    "hold_days": 5,
    "source": "default",
}


def safe_float(v, default=0.0):
    try:
        x = float(v)
        return default if pd.isna(x) else x
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def safe_bool(v, default=False):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return default


def load_old():
    if not os.path.exists(POLICY_FILE):
        return DEFAULT_POLICY.copy()
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            old = json.load(f)
        out = DEFAULT_POLICY.copy()
        out.update(old)
        return out
    except Exception:
        return DEFAULT_POLICY.copy()


def keep(reason):
    print("🟡", reason)
    print("strategy_policy.jsonは変更しません")
    raise SystemExit(0)


if not os.path.exists(INPUT_FILE):
    keep(f"{INPUT_FILE} がありません")
if os.path.getsize(INPUT_FILE) == 0:
    keep(f"{INPUT_FILE} が空です")

try:
    df = pd.read_csv(INPUT_FILE)
except pd.errors.EmptyDataError:
    keep(f"{INPUT_FILE} にデータがありません")

if df.empty:
    keep("候補が0件です")

required = [
    "final_status", "up_threshold", "score_threshold", "nikkei_filter",
    "tp_multiplier", "sl_multiplier", "hold_days", "validation_signals",
    "validation_win_rate", "validation_avg_return", "validation_pf",
    "validation_dd", "oos_signals", "oos_win_rate", "oos_avg_return",
    "oos_pf", "oos_dd", "oos_validation_pf_ratio",
    "oos_monthly_positive_ratio", "oos_compound_return",
    "oos_compound_final_capital", "oos_expected_value",
    "oos_avg_month_return", "oos_worst_month_return",
]
missing = [c for c in required if c not in df.columns]
if missing:
    print("❌ 候補CSVの不足列:")
    for c in missing:
        print(" -", c)
    raise SystemExit(1)

for c in required:
    if c not in ("final_status", "nikkei_filter"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["nikkei_filter"] = df["nikkei_filter"].apply(safe_bool)

approved = df[df["final_status"].astype(str).str.upper().eq("PASS")].copy()
if approved.empty:
    keep("PASS候補なし")

approved = approved[
    (approved["validation_signals"] >= 30)
    & (approved["validation_pf"] >= 1.0)
    & (approved["validation_avg_return"] > 0)
    & (approved["validation_dd"] >= -35)
    & (approved["oos_signals"] >= 20)
    & (approved["oos_pf"] >= MIN_OOS_PF)
    & (approved["oos_avg_return"] > MIN_OOS_AVG_RETURN)
    & (approved["oos_validation_pf_ratio"] >= 0.60)
    & (approved["oos_monthly_positive_ratio"] >= MIN_MONTHLY_POSITIVE_RATIO)
    & (approved["oos_dd"] >= -MAX_OOS_DD)
    & (approved["oos_compound_return"] > 0)
].copy()

if approved.empty:
    keep("月間プラス率・OOS複利・PF・DD条件を満たす候補なし")

approved["profit_objective"] = (
    approved["oos_monthly_positive_ratio"] * 0.40
    + approved["oos_compound_return"].clip(-100, 1000) * 0.30
    + approved["oos_pf"].clip(0, 8) * 5.0 * 0.15
    + approved["oos_avg_return"].clip(-5, 5) * 10.0 * 0.10
    + approved["oos_expected_value"].clip(-5, 5) * 10.0 * 0.05
)

approved = approved.sort_values(
    ["profit_objective", "oos_monthly_positive_ratio", "oos_compound_return", "oos_pf", "oos_expected_value"],
    ascending=[False, False, False, False, False],
).reset_index(drop=True)
best = approved.iloc[0]
old = load_old()

new = dict(old)
new.update({
    "status": "APPROVED",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "up_threshold": safe_int(best["up_threshold"]),
    "min_score_for_buy": safe_int(best["score_threshold"]),
    "nikkei_filter": safe_bool(best["nikkei_filter"]),
    "atr_tp_multiplier": safe_float(best["tp_multiplier"]),
    "atr_sl_multiplier": safe_float(best["sl_multiplier"]),
    "hold_days": safe_int(best["hold_days"]),
    "validation_signals": safe_int(best["validation_signals"]),
    "validation_win_rate": safe_float(best["validation_win_rate"]),
    "validation_avg_return": safe_float(best["validation_avg_return"]),
    "validation_pf": safe_float(best["validation_pf"]),
    "validation_dd": safe_float(best["validation_dd"]),
    "validation_monthly_positive_ratio": safe_float(best.get("validation_monthly_positive_ratio", 0)),
    "oos_signals": safe_int(best["oos_signals"]),
    "oos_win_rate": safe_float(best["oos_win_rate"]),
    "oos_avg_return": safe_float(best["oos_avg_return"]),
    "oos_pf": safe_float(best["oos_pf"]),
    "oos_dd": safe_float(best["oos_dd"]),
    "oos_validation_pf_ratio": safe_float(best["oos_validation_pf_ratio"]),
    "oos_monthly_positive_ratio": safe_float(best["oos_monthly_positive_ratio"]),
    "oos_compound_return": safe_float(best["oos_compound_return"]),
    "oos_compound_final_capital": safe_float(best["oos_compound_final_capital"], 1000000),
    "oos_expected_value": safe_float(best["oos_expected_value"]),
    "oos_avg_month_return": safe_float(best["oos_avg_month_return"]),
    "oos_worst_month_return": safe_float(best["oos_worst_month_return"]),
    "profit_objective": safe_float(best["profit_objective"]),
    "strategy_name": str(best.get("strategy", "")),
    "source": "profit_optimizer_oos",
})

with open(POLICY_FILE, "w", encoding="utf-8") as f:
    json.dump(new, f, ensure_ascii=False, indent=2)

print("=" * 70)
print("✅ PROFIT-FIRST POLICY UPDATED")
print("戦略:", new["strategy_name"])
print("月間プラス率:", f"{new['oos_monthly_positive_ratio']:.1f}%")
print("OOS複利:", f"{new['oos_compound_return']:+.2f}%")
print("OOS最終資産:", f"¥{new['oos_compound_final_capital']:,.0f}")
print("期待利益:", f"{new['oos_expected_value']:+.3f}%")
print("PF:", f"{new['oos_pf']:.2f}")
print("最大DD:", f"{new['oos_dd']:.2f}%")
print("勝率（参考）:", f"{new['oos_win_rate']:.1f}%")
