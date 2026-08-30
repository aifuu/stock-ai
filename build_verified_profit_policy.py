import hashlib
import hmac
import json
import os
from datetime import datetime
from pathlib import Path

FINAL_FILE = "adversarial_final_candidates.csv"
POLICY_FILE = "strategy_policy.json"
SECRET_NAME = "AI_POLICY_SIGNING_SECRET"

KEYS = [
    "status", "updated_at", "up_threshold", "min_score_for_buy", "nikkei_filter",
    "atr_tp_multiplier", "atr_sl_multiplier", "hold_days", "validation_signals",
    "validation_win_rate", "validation_avg_return", "validation_pf", "validation_dd",
    "oos_signals", "oos_win_rate", "oos_avg_return", "oos_pf", "oos_dd",
    "oos_validation_pf_ratio", "mc_sizing", "mc_10y_probability", "mc_15y_probability",
    "mc_20y_probability", "mc_bankruptcy_probability", "mc_p90_max_dd", "strategy_name", "source",
]


def sign(policy, secret):
    covered = {k: policy.get(k) for k in KEYS}
    payload = json.dumps(covered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def write(policy):
    Path(POLICY_FILE).write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")


def pending(reason):
    old = {}
    if Path(POLICY_FILE).exists():
        try:
            old = json.loads(Path(POLICY_FILE).read_text(encoding="utf-8"))
        except Exception:
            old = {}
    policy = dict(old)
    policy.update({"status": "PENDING", "updated_at": datetime.now().isoformat(timespec="seconds"), "source": "adversarial_strategy_validator", "pending_reason": reason})
    policy.pop("approval_signature", None)
    policy.pop("approval_signature_version", None)
    write(policy)
    print("⏸ policy=PENDING")
    print("理由:", reason)
    print("手動APPROVEDへの変更はしません")


if not Path(FINAL_FILE).exists():
    pending(f"{FINAL_FILE} がありません")
    raise SystemExit(0)

import pandas as pd

try:
    df = pd.read_csv(FINAL_FILE)
except Exception as exc:
    pending(f"候補CSV読み込み失敗: {exc}")
    raise SystemExit(0)

if df.empty or "final_status" not in df.columns:
    pending("4-fold OOSのPASS候補なし")
    raise SystemExit(0)

passed = df[df.final_status.astype(str).str.upper().eq("PASS")].copy()
if passed.empty:
    pending("4-fold OOSゲートを通過した戦略なし")
    raise SystemExit(0)

secret = os.getenv(SECRET_NAME, "").strip()
if not secret:
    raise RuntimeError(f"{SECRET_NAME} が未設定。署名なしAPPROVEDは作成しません")

best = passed.iloc[0]
policy = {
    "status": "APPROVED",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "up_threshold": int(best.up_threshold),
    "min_score_for_buy": int(best.score_threshold),
    "nikkei_filter": bool(best.nikkei_filter),
    "atr_tp_multiplier": float(best.tp_multiplier),
    "atr_sl_multiplier": float(best.sl_multiplier),
    "hold_days": int(best.hold_days),
    "validation_signals": int(best.validation_signals),
    "validation_win_rate": float(best.validation_win_rate),
    "validation_avg_return": float(best.validation_avg_return),
    "validation_pf": float(best.validation_pf),
    "validation_dd": float(best.validation_dd),
    "oos_signals": int(best.oos_signals),
    "oos_win_rate": float(best.oos_win_rate),
    "oos_avg_return": float(best.oos_avg_return),
    "oos_pf": float(best.oos_pf),
    "oos_dd": float(best.oos_dd),
    "oos_validation_pf_ratio": float(best.oos_validation_pf_ratio),
    "mc_sizing": None,
    "mc_10y_probability": None,
    "mc_15y_probability": None,
    "mc_20y_probability": None,
    "mc_bankruptcy_probability": None,
    "mc_p90_max_dd": None,
    "strategy_name": str(best.strategy),
    "source": "adversarial_strategy_validator",
    "multi_oos_folds": int(best.folds),
    "positive_oos_folds": int(best.positive_oos_folds),
    "oos_compound_return": float(best.oos_compound_return),
    "oos_monthly_positive_ratio": float(best.oos_monthly_positive_ratio_mean),
}
policy["approval_signature_version"] = 1
policy["approval_signature"] = sign(policy, secret)
write(policy)
print("=" * 70)
print("✅ APPROVED: 4-fold multi-OOS gate passed")
print("strategy:", policy["strategy_name"])
print("positive OOS folds:", policy["positive_oos_folds"], "/", policy["multi_oos_folds"])
print("OOS compound:", f"{policy['oos_compound_return']:+.2f}%")
print("monthly positive:", f"{policy['oos_monthly_positive_ratio']:.1f}%")
print("signature: generated")
