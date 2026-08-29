import hashlib
import hmac
import json
import os
from datetime import datetime

import pandas as pd

INPUT_FILE = "adversarial_final_candidates.csv"
POLICY_FILE = "strategy_policy.json"

MIN_OOS_TRADES = 20
MIN_OOS_PF = 1.00
MIN_OOS_AVG_RETURN = 0.00
MIN_OOS_TO_VALIDATION_PF = 0.60
MAX_VALIDATION_DD = 30.0
MIN_VALIDATION_TRADES = 50
MIN_VALIDATION_PF = 1.00
MIN_VALIDATION_AVG_RETURN = 0.00
MIN_MC_BANKRUPTCY_PROB = 5.0
MAX_MC_DD90 = 30.0

DEFAULT_POLICY = {
    "status": "DEFAULT", "updated_at": None,
    "up_threshold": 50, "min_score_for_buy": 60,
    "nikkei_filter": False, "atr_tp_multiplier": 3.0,
    "atr_sl_multiplier": 1.5, "hold_days": 5,
    "validation_signals": 0, "validation_win_rate": 0.0,
    "validation_avg_return": 0.0, "validation_pf": 0.0,
    "validation_dd": 0.0, "oos_signals": 0, "oos_win_rate": 0.0,
    "oos_avg_return": 0.0, "oos_pf": 0.0, "oos_dd": 0.0,
    "oos_validation_pf_ratio": 0.0, "mc_sizing": 0.005,
    "mc_10y_probability": 0.0, "mc_15y_probability": 0.0,
    "mc_20y_probability": 0.0, "mc_bankruptcy_probability": 0.0,
    "mc_p90_max_dd": 0.0, "strategy_name": "", "source": "default"
}


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if pd.isna(value) else value
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return default


def load_existing_policy():
    if not os.path.exists(POLICY_FILE):
        return DEFAULT_POLICY.copy()
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            policy = json.load(f)
        if not isinstance(policy, dict):
            raise ValueError("policyがdictではありません")
        merged = DEFAULT_POLICY.copy()
        merged.update(policy)
        return merged
    except Exception as e:
        print("⚠ 既存policy読み込み失敗:", e)
        return DEFAULT_POLICY.copy()


def keep_existing_policy(reason):
    print("")
    print("🟡", reason)
    print("strategy_policy.jsonは変更しません")
    raise SystemExit(0)


def canonical_policy_payload(policy):
    """Return the exact policy fields covered by the approval signature."""
    covered = {k: policy.get(k) for k in (
        "status", "updated_at", "up_threshold", "min_score_for_buy",
        "nikkei_filter", "atr_tp_multiplier", "atr_sl_multiplier", "hold_days",
        "validation_signals", "validation_win_rate", "validation_avg_return",
        "validation_pf", "validation_dd", "oos_signals", "oos_win_rate",
        "oos_avg_return", "oos_pf", "oos_dd", "oos_validation_pf_ratio",
        "mc_sizing", "mc_10y_probability", "mc_15y_probability",
        "mc_20y_probability", "mc_bankruptcy_probability", "mc_p90_max_dd",
        "strategy_name", "source",
    )}
    return json.dumps(covered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def policy_signature(policy, secret):
    return hmac.new(
        secret.encode("utf-8"),
        canonical_policy_payload(policy).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


POLICY_SIGNING_SECRET = os.getenv("AI_POLICY_SIGNING_SECRET", "").strip()
if not POLICY_SIGNING_SECRET:
    keep_existing_policy("AI_POLICY_SIGNING_SECRET が未設定のため、安全のためAPPROVED policyを生成しません")

# =========================================================
# 入力CSV確認
# =========================================================
if not os.path.exists(INPUT_FILE):
    keep_existing_policy(f"{INPUT_FILE} がありません")

if os.path.getsize(INPUT_FILE) == 0:
    keep_existing_policy(f"{INPUT_FILE} が空です（0バイト）")

try:
    df = pd.read_csv(INPUT_FILE)
except pd.errors.EmptyDataError:
    keep_existing_policy(f"{INPUT_FILE} にデータがありません")
except Exception as e:
    print("❌ CSV読み込み失敗:", e)
    raise SystemExit(1)

if df.empty:
    keep_existing_policy("候補が0件です")

# adversarial_strategy_validator.py の実際の出力名は
# oos_val_pf_ratio。内部policy名は oos_validation_pf_ratio なので統一する。
if "oos_validation_pf_ratio" not in df.columns and "oos_val_pf_ratio" in df.columns:
    df["oos_validation_pf_ratio"] = df["oos_val_pf_ratio"]

required_columns = [
    "final_status", "up_threshold", "score_threshold", "nikkei_filter",
    "tp_multiplier", "sl_multiplier", "hold_days", "validation_signals",
    "validation_win_rate", "validation_avg_return", "validation_pf",
    "validation_dd", "oos_signals", "oos_win_rate", "oos_avg_return",
    "oos_pf", "oos_dd", "oos_validation_pf_ratio",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("❌ strategy候補CSVに不足列があります:")
    for col in missing:
        print(" -", col)
    raise SystemExit(1)

numeric_columns = [
    "up_threshold", "score_threshold", "tp_multiplier", "sl_multiplier",
    "hold_days", "validation_signals", "validation_win_rate",
    "validation_avg_return", "validation_pf", "validation_dd",
    "oos_signals", "oos_win_rate", "oos_avg_return", "oos_pf", "oos_dd",
    "oos_validation_pf_ratio",
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["nikkei_filter"] = df["nikkei_filter"].apply(safe_bool)

mc_columns = ["sizing", "prob_10y", "prob_15y", "prob_20y", "bankruptcy_prob", "p90_max_dd"]
mc_columns_exist = all(col in df.columns for col in mc_columns)
if not mc_columns_exist:
    # ★修正(2026-08): Monte Carlo未実施時は警告だけで続行せず、
    # 既存policyを維持して自動採用を停止する（フェイルクローズ）。
    keep_existing_policy(f"{INPUT_FILE} にMonte Carlo列がありません。Monte Carlo未検証のため自動採用しません")

approved = df[df["final_status"].astype(str).str.upper().eq("PASS")].copy()
if approved.empty:
    keep_existing_policy("PASS候補なし")

approved = approved[
    (approved["validation_signals"] >= MIN_VALIDATION_TRADES)
    & (approved["validation_pf"] >= MIN_VALIDATION_PF)
    & (approved["validation_avg_return"] > MIN_VALIDATION_AVG_RETURN)
    & (approved["validation_dd"].abs() <= MAX_VALIDATION_DD)
]

approved = approved[
    (approved["oos_signals"] >= MIN_OOS_TRADES)
    & (approved["oos_pf"] >= MIN_OOS_PF)
    & (approved["oos_avg_return"] > MIN_OOS_AVG_RETURN)
    & (approved["oos_validation_pf_ratio"] >= MIN_OOS_TO_VALIDATION_PF)
]

if mc_columns_exist:
    approved = approved[
        (approved["bankruptcy_prob"] < MIN_MC_BANKRUPTCY_PROB)
        & (approved["p90_max_dd"].abs() <= MAX_MC_DD90)
    ]

if approved.empty:
    keep_existing_policy("最終採用条件を満たす戦略なし")

# ★修正(2026-08): 以前はここだけ ["oos_pf","oos_avg_return",...] で
# 並べ替えており、adversarial_strategy_validator.py の profit_objective
# (月次プラス比率40%+複利30%+PF15%+平均リターン10%+期待値5%、勝率は
# 一切使わない設計)と選定基準がズレていた。同じ重み付けに揃える。
if "oos_monthly_positive_ratio" in approved.columns and "oos_compound_return" in approved.columns:
    approved["profit_objective"] = (
        approved["oos_monthly_positive_ratio"] * 0.40
        + approved["oos_compound_return"].clip(-100, 1000) * 0.30
        + approved["oos_pf"].clip(0, 8) * 5.0 * 0.15
        + approved["oos_avg_return"].clip(-5, 5) * 10.0 * 0.10
        + (
            approved["oos_expected_value"].clip(-5, 5) * 10.0 * 0.05
            if "oos_expected_value" in approved.columns
            else 0.0
        )
    )
    sort_cols = ["profit_objective", "oos_monthly_positive_ratio", "oos_compound_return", "oos_pf", "oos_avg_return"]
else:
    # 月次/複利列が無い場合のみ、従来の基準にフォールバックする。
    sort_cols = ["oos_pf", "oos_avg_return", "validation_pf", "validation_signals"]

approved = approved.sort_values(
    sort_cols, ascending=[False] * len(sort_cols)
).reset_index(drop=True)
best = approved.iloc[0]
old_policy = load_existing_policy()

new_policy = {
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
    "oos_signals": safe_int(best["oos_signals"]),
    "oos_win_rate": safe_float(best["oos_win_rate"]),
    "oos_avg_return": safe_float(best["oos_avg_return"]),
    "oos_pf": safe_float(best["oos_pf"]),
    "oos_dd": safe_float(best["oos_dd"]),
    "oos_validation_pf_ratio": safe_float(best["oos_validation_pf_ratio"]),
    "mc_sizing": safe_float(best["sizing"]) if mc_columns_exist else old_policy.get("mc_sizing", 0.005),
    "mc_10y_probability": safe_float(best["prob_10y"]) if mc_columns_exist else old_policy.get("mc_10y_probability", 0.0),
    "mc_15y_probability": safe_float(best["prob_15y"]) if mc_columns_exist else old_policy.get("mc_15y_probability", 0.0),
    "mc_20y_probability": safe_float(best["prob_20y"]) if mc_columns_exist else old_policy.get("mc_20y_probability", 0.0),
    "mc_bankruptcy_probability": safe_float(best["bankruptcy_prob"]) if mc_columns_exist else old_policy.get("mc_bankruptcy_probability", 0.0),
    "mc_p90_max_dd": safe_float(best["p90_max_dd"]) if mc_columns_exist else old_policy.get("mc_p90_max_dd", 0.0),
    "strategy_name": str(best.get("strategy", "")),
    "source": "adversarial_strategy_validator",
    "approval_signature_version": 1,
}
new_policy["approval_signature"] = policy_signature(new_policy, POLICY_SIGNING_SECRET)

try:
    with open(POLICY_FILE, "w", encoding="utf-8") as f:
        json.dump(new_policy, f, ensure_ascii=False, indent=2)
except Exception as e:
    print("❌ strategy_policy.json保存失敗:", e)
    raise SystemExit(1)

print("")
print("=" * 60)
print("✅ strategy_policy.json 更新")
print("=" * 60)
print("採用戦略:", new_policy["strategy_name"])
print("UP:", new_policy["up_threshold"])
print("MIN SCORE:", new_policy["min_score_for_buy"])
print("日経フィルター:", new_policy["nikkei_filter"])
print("ATR TP:", new_policy["atr_tp_multiplier"])
print("ATR SL:", new_policy["atr_sl_multiplier"])
print("Validation PF:", new_policy["validation_pf"])
print("OOS PF:", new_policy["oos_pf"])
print("OOS/Validation PF:", new_policy["oos_validation_pf_ratio"])
