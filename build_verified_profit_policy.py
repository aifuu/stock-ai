import hashlib
import hmac
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

FINAL_FILE = "adversarial_final_candidates.csv"
POLICY_FILE = "strategy_policy.json"
SECRET_NAME = "AI_POLICY_SIGNING_SECRET"
KEYS = [
    "status", "updated_at", "up_threshold", "min_score_for_buy", "nikkei_filter",
    "atr_tp_multiplier", "atr_sl_multiplier", "hold_days", "validation_signals",
    "validation_win_rate", "validation_avg_return", "validation_pf", "validation_dd",
    "oos_signals", "oos_win_rate", "oos_avg_return", "oos_pf", "oos_dd",
    "oos_validation_pf_ratio", "mc_sizing", "mc_10y_probability", "mc_15y_probability",
    "mc_20y_probability", "mc_bankruptcy_probability", "mc_p90_max_dd", "strategy_name",
    "selection_mode", "regime_expectancy_json", "source",
]


def sign(policy: dict, secret: str) -> str:
    payload = json.dumps(
        {k: policy.get(k) for k in KEYS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def write(policy: dict) -> None:
    Path(POLICY_FILE).write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")


def pending(reason: str) -> None:
    old = {}
    if Path(POLICY_FILE).exists():
        try:
            old = json.loads(Path(POLICY_FILE).read_text(encoding="utf-8"))
        except Exception:
            pass
    old.update({
        "status": "PENDING",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "adversarial_strategy_validator",
        "pending_reason": reason,
    })
    old.pop("approval_signature", None)
    old.pop("approval_signature_version", None)
    write(old)
    print("⏸ policy=PENDING")
    print("理由:", reason)
    print("手動APPROVEDへの変更はしません")


if not Path(FINAL_FILE).exists():
    pending(f"{FINAL_FILE} がありません")
    raise SystemExit(0)

try:
    df = pd.read_csv(FINAL_FILE)
except Exception as exc:
    pending(f"候補CSV読み込み失敗: {exc}")
    raise SystemExit(0)

if df.empty or "final_status" not in df.columns:
    pending("4-fold OOSのPASS候補なし")
    raise SystemExit(0)

passed = df[df["final_status"].astype(str).str.upper().eq("PASS")].copy()
if passed.empty:
    pending("4-fold OOSゲートを通過した戦略なし")
    raise SystemExit(0)

required = [
    "oos_monthly_positive_ratio", "oos_compound_return", "oos_pf",
    "oos_avg_return", "oos_expected_value", "selection_mode", "regime_expectancy_json",
]
missing = [c for c in required if c not in passed.columns]
if missing:
    raise RuntimeError("PASS候補に必要な利益最適化列が不足: " + ", ".join(missing))

secret = os.getenv(SECRET_NAME, "").strip()
if not secret:
    raise RuntimeError(f"{SECRET_NAME} が未設定。署名なしAPPROVEDは作成しません")

# Profit-first. Win-rate is deliberately not part of the ranking.
passed["profit_objective"] = (
    pd.to_numeric(passed["oos_monthly_positive_ratio"], errors="coerce") * 0.40
    + pd.to_numeric(passed["oos_compound_return"], errors="coerce").clip(-100, 1000) * 0.30
    + pd.to_numeric(passed["oos_pf"], errors="coerce").clip(0, 8) * 5.0 * 0.15
    + pd.to_numeric(passed["oos_avg_return"], errors="coerce").clip(-5, 5) * 10.0 * 0.10
    + pd.to_numeric(passed["oos_expected_value"], errors="coerce").clip(-5, 5) * 10.0 * 0.05
)
passed = passed.sort_values(
    ["profit_objective", "oos_monthly_positive_ratio", "oos_compound_return", "oos_pf", "oos_avg_return", "oos_expected_value"],
    ascending=[False] * 6,
).reset_index(drop=True)
b = passed.iloc[0]

policy = {
    "status": "APPROVED",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "up_threshold": int(b.get("up_threshold", 0)),
    "min_score_for_buy": int(b.get("score_threshold", 0)),
    "nikkei_filter": bool(b.get("nikkei_filter", False)),
    "atr_tp_multiplier": float(b.get("tp_multiplier", 3.0)),
    "atr_sl_multiplier": float(b.get("sl_multiplier", 1.5)),
    "hold_days": int(b.get("hold_days", 5)),
    "validation_signals": int(b.get("validation_signals", 0)),
    "validation_win_rate": float(b.get("validation_win_rate", 0.0)),
    "validation_avg_return": float(b.get("validation_avg_return", 0.0)),
    "validation_pf": float(b.get("validation_pf", 0.0)),
    "validation_dd": float(b.get("validation_dd", 0.0)),
    "oos_signals": int(b.get("oos_signals", 0)),
    "oos_win_rate": float(b.get("oos_win_rate", 0.0)),
    "oos_avg_return": float(b.get("oos_avg_return", 0.0)),
    "oos_pf": float(b.get("oos_pf", 0.0)),
    "oos_dd": float(b.get("oos_dd", 0.0)),
    "oos_validation_pf_ratio": float(b.get("oos_validation_pf_ratio", 1.0)),
    "mc_sizing": b.get("sizing", None),
    "mc_10y_probability": b.get("prob_10y", None),
    "mc_15y_probability": b.get("prob_15y", None),
    "mc_20y_probability": b.get("prob_20y", None),
    "mc_bankruptcy_probability": b.get("bankruptcy_prob", None),
    "mc_p90_max_dd": b.get("p90_max_dd", None),
    "strategy_name": str(b.get("strategy", "REGIME_EXPECTED_RETURN_TOP1")),
    "selection_mode": "REGIME_EXPECTED_RETURN_TOP1",
    "regime_expectancy_json": str(b["regime_expectancy_json"]),
    "source": "adversarial_strategy_validator",
    "multi_oos_folds": int(b.get("multi_oos_folds", 4)),
    "positive_oos_folds": int(b.get("positive_oos_folds", 0)),
    "oos_compound_return": float(b["oos_compound_return"]),
    "oos_monthly_positive_ratio": float(b["oos_monthly_positive_ratio"]),
    "oos_expected_value": float(b["oos_expected_value"]),
    "profit_objective": float(b["profit_objective"]),
    "selection_version": "regime_profit_core_v1",
}
policy["approval_signature_version"] = 1
policy["approval_signature"] = sign(policy, secret)
write(policy)

print("=" * 70)
print("✅ APPROVED: 4-fold multi-OOS gate passed")
print("strategy:", policy["strategy_name"])
print("selection:", policy["selection_mode"])
print("positive OOS folds:", policy["positive_oos_folds"], "/", policy["multi_oos_folds"])
print("OOS compound:", f"{policy['oos_compound_return']:+.2f}%")
print("monthly positive:", f"{policy['oos_monthly_positive_ratio']:.1f}%")
print("expected return:", f"{policy['oos_expected_value']:+.4f}%")
print("profit objective:", f"{policy['profit_objective']:.4f}")
print("signature: generated")
