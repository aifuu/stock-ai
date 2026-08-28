"""
build_intraday_policy.py

intraday_strategy_backtest.py が出力する intraday_strategy_comparison.csv を読み、
STANDARD / RELAXED / LOOSE のうち採用条件を満たす最良段階を
strategy_policy.json の intraday_* キーへ保存する。

5営業日型の build_strategy_policy.py とは完全独立。
5日型のキー(up_threshold / min_score_for_buy / hold_days等)は変更しない。
"""

import json
import os
from datetime import datetime

import pandas as pd

POLICY_FILE = "strategy_policy.json"
INPUT_FILE = "intraday_strategy_comparison.csv"

MIN_TRADES = int(os.getenv("ID_MIN_TRADES", "10"))
MIN_PF = float(os.getenv("ID_MIN_PF", "1.0"))
MIN_AVG_RETURN = float(os.getenv("ID_MIN_AVG_RETURN", "0.0"))

STRATEGY_CONFIGS = {
    "STANDARD": {"min_score": 65.0, "min_up_prob": 55.0, "min_vol_ratio": 1.0},
    "RELAXED": {"min_score": 60.0, "min_up_prob": 52.0, "min_vol_ratio": 0.9},
    "LOOSE": {"min_score": 55.0, "min_up_prob": 50.0, "min_vol_ratio": 0.8},
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


def load_policy():
    if not os.path.exists(POLICY_FILE):
        return {}
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            policy = json.load(f)
        return policy if isinstance(policy, dict) else {}
    except Exception as exc:
        print(f"⚠ {POLICY_FILE} 読み込み失敗: {exc}")
        return {}


def keep_existing(reason):
    print("")
    print("🟡", reason)
    print("デイトレ側 strategy_policy.json は変更しません")
    raise SystemExit(0)


def main():
    if not os.path.exists(INPUT_FILE) or os.path.getsize(INPUT_FILE) == 0:
        keep_existing(f"{INPUT_FILE} がありません/空です")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as exc:
        print(f"❌ {INPUT_FILE} 読み込み失敗: {exc}")
        raise SystemExit(1)

    required = {"strategy", "trades", "pf", "win_rate", "avg_return_pct"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ {INPUT_FILE} に必要列がありません: {sorted(missing)}")
        raise SystemExit(1)

    for col in ["trades", "pf", "win_rate", "avg_return_pct", "total_return_pct", "max_dd_pct", "final_capital"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    qualified = df[
        (df["trades"] >= MIN_TRADES)
        & (df["pf"] >= MIN_PF)
        & (df["avg_return_pct"] > MIN_AVG_RETURN)
    ].copy()

    if qualified.empty:
        keep_existing(
            f"採用条件未達: trades>={MIN_TRADES}, PF>={MIN_PF}, "
            f"avg_return>{MIN_AVG_RETURN}"
        )

    # PFを第一優先、平均リターン、最終資産、件数の順で比較。
    best = qualified.sort_values(
        ["pf", "avg_return_pct", "final_capital", "trades"],
        ascending=[False, False, False, False],
    ).iloc[0]

    strategy_name = str(best["strategy"])
    config = STRATEGY_CONFIGS.get(strategy_name)
    if config is None:
        print(f"❌ 未知の戦略名です: {strategy_name}")
        raise SystemExit(1)

    policy = load_policy()
    policy.update({
        "intraday_strategy": strategy_name,
        "intraday_min_score": config["min_score"],
        "intraday_min_up_prob": config["min_up_prob"],
        "intraday_min_vol_ratio": config["min_vol_ratio"],
        "intraday_atr_tp": safe_float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"), 2.0),
        "intraday_atr_sl": safe_float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"), 1.0),
        "intraday_trades": safe_int(best["trades"]),
        "intraday_pf": safe_float(best["pf"]),
        "intraday_win_rate": safe_float(best["win_rate"]),
        "intraday_avg_return_pct": safe_float(best["avg_return_pct"]),
        "intraday_total_return_pct": safe_float(best.get("total_return_pct", 0.0)),
        "intraday_max_dd_pct": safe_float(best.get("max_dd_pct", 0.0)),
        "intraday_final_capital": safe_float(best.get("final_capital", 0.0)),
        "intraday_min_trades": MIN_TRADES,
        "intraday_min_pf": MIN_PF,
        "intraday_updated_at": datetime.now().isoformat(timespec="seconds"),
    })

    with open(POLICY_FILE, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

    print("")
    print("=" * 60)
    print("✅ strategy_policy.json デイトレ枠 更新")
    print("=" * 60)
    print("採用段階     :", strategy_name)
    print("PF           :", policy["intraday_pf"])
    print("勝率         :", policy["intraday_win_rate"])
    print("平均リターン :", policy["intraday_avg_return_pct"])
    print("累積リターン :", policy["intraday_total_return_pct"])
    print("最大DD       :", policy["intraday_max_dd_pct"])
    print("取引数       :", policy["intraday_trades"])


if __name__ == "__main__":
    main()
