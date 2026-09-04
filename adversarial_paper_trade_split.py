#!/usr/bin/env python3
"""Split paper-trade results by provenance before adversarial evaluation.

This script intentionally does NOT alter strategy parameters or generate a new
policy. It only evaluates the persisted paper ledger and keeps forced/fallback/
proxy-feature trades out of the validated-model score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = "profit_top10_paper_history.csv"
DEFAULT_OUTPUT = "adversarial_paper_trade_split.csv"
VALID_SELECTION = {"normal", "progressive_level"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _stats(df):
    if df.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "avg_return_pct": 0.0, "profit_factor": 0.0,
                "max_drawdown_pct": 0.0, "expected_value_pct": 0.0,
                "final_compound_return_pct": 0.0}
    x = df.copy()
    r = _num(x.get("return_pct", pd.Series(dtype=float))).fillna(0.0)
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    ordered = x.copy()
    ordered["_ret"] = r.to_numpy()
    if "exit_date" in ordered:
        ordered["_date"] = pd.to_datetime(ordered["exit_date"], errors="coerce")
    else:
        ordered["_date"] = pd.NaT
    ordered = ordered.sort_values(["_date", "ticker"], na_position="last")
    daily = ordered.groupby("_date")["_ret"].mean() if ordered["_date"].notna().any() else pd.Series(dtype=float)
    equity = (1.0 + daily / 100.0).cumprod()
    dd = float(((equity / equity.cummax()) - 1.0).min() * 100.0) if len(equity) else 0.0
    compound = float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else 0.0
    return {
        "trades": int(len(x)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / (wins + losses) * 100.0 if wins + losses else 0.0,
        "avg_return_pct": float(r.mean()),
        "profit_factor": pf,
        "max_drawdown_pct": dd,
        "expected_value_pct": float(r.mean()),
        "final_compound_return_pct": compound,
    }


def classify(df):
    x = df.copy()
    for col, default in {
        "selection_mode": "legacy_unknown",
        "selection_level": "",
        "market_regime": "unknown",
        "model_type": "legacy_unknown",
        "feature_source": "legacy_unknown",
        "validation_eligible": False,
    }.items():
        if col not in x.columns:
            x[col] = default
    x["selection_mode"] = x["selection_mode"].fillna("legacy_unknown").astype(str)
    x["model_type"] = x["model_type"].fillna("legacy_unknown").astype(str)
    x["feature_source"] = x["feature_source"].fillna("legacy_unknown").astype(str)
    x["validation_eligible"] = x["validation_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
    x["bucket"] = "other"
    x.loc[x["selection_mode"].isin(VALID_SELECTION), "bucket"] = "validated_selection"
    x.loc[x["selection_mode"] == "forced_min_trade", "bucket"] = "forced_min_trade"
    x.loc[x["model_type"] == "fallback", "bucket"] = "fallback_model"
    x.loc[x["feature_source"] == "cash_proxy", "bucket"] = "cash_proxy"
    x.loc[x["selection_mode"] == "legacy_unknown", "bucket"] = "legacy_unknown"
    # A trade is eligible only if every provenance gate is clean.
    x["validation_eligible"] = (
        x["selection_mode"].isin(VALID_SELECTION)
        & (x["model_type"] == "validated_model")
        & (x["feature_source"] == "futures")
    )
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"❌ ledger not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("❌ ledger is empty")
    df = classify(df)

    buckets = ["validated_selection", "forced_min_trade", "fallback_model", "cash_proxy", "legacy_unknown", "other"]
    rows = []
    for bucket in buckets:
        s = _stats(df[df["bucket"] == bucket])
        s["bucket"] = bucket
        rows.append(s)
    eligible = _stats(df[df["validation_eligible"]])
    eligible["bucket"] = "VALIDATION_ELIGIBLE"
    rows.append(eligible)
    report = pd.DataFrame(rows)
    cols = ["bucket", "trades", "wins", "losses", "win_rate_pct", "avg_return_pct", "profit_factor", "max_drawdown_pct", "expected_value_pct", "final_compound_return_pct"]
    report = report[cols]
    report.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("🛡️ ADVERSARIAL PAPER TRADE PROVENANCE SPLIT")
    print("=" * 88)
    print(f"入力: {path}")
    print(f"総取引: {len(df)}")
    print(f"検証対象: {int(df['validation_eligible'].sum())}")
    print("")
    print(report.to_string(index=False))
    print("")
    print("ルール:")
    print("  normal / progressive_level + validated_model + futures のみ本番検証成績へ")
    print("  forced_min_trade / fallback_model / cash_proxy / legacy_unknown は分離")
    print(f"📄 出力: {args.output}")

    summary = {
        "total_trades": int(len(df)),
        "validation_eligible_trades": int(df["validation_eligible"].sum()),
        "excluded_trades": int((~df["validation_eligible"]).sum()),
        "forced_trades": int((df["selection_mode"] == "forced_min_trade").sum()),
        "fallback_trades": int((df["model_type"] == "fallback").sum()),
        "cash_proxy_trades": int((df["feature_source"] == "cash_proxy").sum()),
        "legacy_unknown_trades": int((df["selection_mode"] == "legacy_unknown").sum()),
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
