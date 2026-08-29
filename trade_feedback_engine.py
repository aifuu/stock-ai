"""ペーパー売買結果を安全なフィードバックポリシーへ変換する。"""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HISTORY_FILE = "directional_paper_history.csv"
POLICY_FILE = "trade_feedback_policy.json"
REPORT_FILE = "trade_feedback_report.csv"
JST = ZoneInfo("Asia/Tokyo")
LOOKBACK_DAYS = int(os.getenv("FEEDBACK_LOOKBACK_DAYS", "180"))
MIN_GROUP_TRADES = int(os.getenv("FEEDBACK_MIN_GROUP_TRADES", "12"))
OOS_DAYS = int(os.getenv("RETRAIN_HOLDOUT_DAYS", "90"))
MAX_WEIGHT = float(os.getenv("FEEDBACK_MAX_WEIGHT", "1.10"))
MIN_WEIGHT = float(os.getenv("FEEDBACK_MIN_WEIGHT", "0.90"))


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception as e:
        print(f"⚠ feedback履歴読み込み失敗: {e}")
        return pd.DataFrame()
    for col in ["entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def usable_history(df):
    if df.empty or "exit_date" not in df.columns:
        return df
    now = pd.Timestamp.now(tz=JST).tz_localize(None)
    recent_cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    oos_cutoff = now - pd.Timedelta(days=OOS_DAYS)
    # 現在のOOS検証と重なる結果をフィードバックから除外してリーク防止。
    return df[(df["exit_date"] >= recent_cutoff) & (df["exit_date"] < oos_cutoff)].copy()


def stats(df, key):
    if df.empty or key not in df.columns or "pnl" not in df.columns:
        return []
    out = []
    for value, g in df.groupby(key, dropna=False):
        g = g.dropna(subset=["pnl"])
        n = len(g)
        if n < MIN_GROUP_TRADES:
            continue
        gp = float(g.loc[g.pnl > 0, "pnl"].sum())
        gl = float(-g.loc[g.pnl < 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        out.append({"group": str(value), "trades": n,
                    "win_rate": float((g.pnl > 0).mean() * 100),
                    "pf": pf, "pnl": float(g.pnl.sum())})
    return out


def pf_weight(pf):
    if not np.isfinite(pf):
        return MAX_WEIGHT
    if pf >= 1.5:
        return MAX_WEIGHT
    if pf <= 0.7:
        return MIN_WEIGHT
    return float(np.clip(1.0 + (pf - 1.0) * 0.20, MIN_WEIGHT, MAX_WEIGHT))


def main():
    today = datetime.now(JST).strftime("%Y-%m-%d")
    df = usable_history(load_history())
    policy = {
        "version": 1,
        "generated_at": today,
        "source": HISTORY_FILE,
        "leakage_guard": f"excluded last {OOS_DAYS} days",
        "lookback_days": LOOKBACK_DAYS,
        "min_group_trades": MIN_GROUP_TRADES,
        "sample_trades": int(len(df)),
        "direction_weights": {"BUY": 1.0, "SHORT": 1.0},
    }
    direction_stats = stats(df, "direction")
    for row in direction_stats:
        if row["group"] in policy["direction_weights"]:
            policy["direction_weights"][row["group"]] = pf_weight(row["pf"])
    policy["direction_stats"] = direction_stats
    policy["exit_reason_stats"] = stats(df, "exit_reason")
    if "score" in df.columns:
        s = df.copy()
        s["score_bucket"] = pd.cut(s["score"], [-np.inf,50,60,70,80,np.inf],
                                    labels=["<50","50-59","60-69","70-79","80+"],
                                    right=False)
        policy["score_bucket_stats"] = stats(s, "score_bucket")
    else:
        policy["score_bucket_stats"] = []
    with open(POLICY_FILE, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2, allow_nan=False)
    rows = []
    for category in ["direction_stats", "exit_reason_stats", "score_bucket_stats"]:
        for row in policy[category]:
            rows.append({"date": today, "category": category.replace("_stats", ""), **row})
    pd.DataFrame(rows).to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    print(f"🧠 feedback分析完了: usable_trades={len(df)}")
    print(f"direction_weights={policy['direction_weights']}")
    print(f"OOSリーク防止: 直近{OOS_DAYS}日を除外")


if __name__ == "__main__":
    main()
