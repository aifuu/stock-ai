"""Shared regime-conditioned profit-first ranking core.

This module is intentionally used by both OOS validation and live TOP1
paper trading so the selection formula cannot silently diverge between them.
No gate threshold is defined here; gates remain in the existing validators.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

SMOOTH = 12.0
REGIMES = ("RISK_ON", "NEUTRAL", "RISK_OFF")
BUCKETS = ("LOW", "MID", "HIGH", "ELITE")


def strength_bucket(strength: float) -> str:
    if strength < 25:
        return "LOW"
    if strength < 45:
        return "MID"
    if strength < 65:
        return "HIGH"
    return "ELITE"


def market_regime_from_nikkei(kairi25: float, ret5: float) -> str:
    if float(kairi25) > 0 and float(ret5) > 0:
        return "RISK_ON"
    if float(kairi25) < 0 and float(ret5) < 0:
        return "RISK_OFF"
    return "NEUTRAL"


def individual_strength(up_probability_pct: float, down_probability_pct: float,
                        score: float, relative_strength_raw: float) -> float:
    return (
        0.45 * (float(up_probability_pct) - float(down_probability_pct))
        + 0.40 * float(score)
        + 0.15 * float(relative_strength_raw) * 100.0
    )


def build_expectancy_table(df: pd.DataFrame) -> dict[str, Any]:
    """Build hierarchical expected-return table from *past-only* rows."""
    if df is None or df.empty:
        return {"global": 0.0, "regimes": {}, "groups": {}}

    x = df.copy()
    x["return"] = pd.to_numeric(x["return"], errors="coerce")
    x = x.dropna(subset=["return", "market_regime", "strength_bucket"])
    if x.empty:
        return {"global": 0.0, "regimes": {}, "groups": {}}

    global_mean = float(x["return"].mean())
    regimes: dict[str, float] = {}
    groups: dict[str, float] = {}

    for regime, row in x.groupby("market_regime")["return"].agg(["mean", "count"]).iterrows():
        n = float(row["count"])
        regimes[str(regime)] = (n * float(row["mean"]) + SMOOTH * global_mean) / (n + SMOOTH)

    for (regime, bucket), row in x.groupby(["market_regime", "strength_bucket"])["return"].agg(["mean", "count"]).iterrows():
        n = float(row["count"])
        parent = regimes.get(str(regime), global_mean)
        groups[f"{regime}|{bucket}"] = (n * float(row["mean"]) + SMOOTH * parent) / (n + SMOOTH)

    return {"global": global_mean, "regimes": regimes, "groups": groups}


def expected_return_from_table(table: dict[str, Any], regime: str, strength: float) -> float:
    bucket = strength_bucket(float(strength))
    groups = table.get("groups", {}) or {}
    regimes = table.get("regimes", {}) or {}
    return float(groups.get(f"{regime}|{bucket}", regimes.get(regime, table.get("global", 0.0))))


def load_expectancy_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {"global": 0.0, "regimes": {}, "groups": {}}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"global": 0.0, "regimes": {}, "groups": {}}


def rank_candidates(candidates: list[dict[str, Any]], expectancy_table: dict[str, Any], top_n: int = 1) -> list[dict[str, Any]]:
    """Identical profit-first ordering used by OOS and live runtime."""
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        regime = str(item.get("market_regime", "NEUTRAL"))
        strength = float(item.get("individual_strength", 0.0))
        item["strength_bucket"] = strength_bucket(strength)
        item["expected_return"] = expected_return_from_table(expectancy_table, regime, strength)
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            float(x.get("expected_return", 0.0)),
            float(x.get("individual_strength", 0.0)),
            float(x.get("score", 0.0)),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(top_n))]


def enrich_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add live/OOS-visible regime and strength fields without using future data."""
    x = df.copy()
    if "relative_strength" not in x.columns:
        x["relative_strength"] = 0.0
    if "market_regime" not in x.columns:
        if "nikkei_kairi25" in x.columns and "nikkei_return_5d" in x.columns:
            x["market_regime"] = [
                market_regime_from_nikkei(a, b)
                for a, b in zip(x["nikkei_kairi25"], x["nikkei_return_5d"])
            ]
        else:
            trend = x.get("nikkei_uptrend", False)
            trend = pd.Series(trend, index=x.index).astype(str).str.lower().isin(["true", "1", "yes"])
            x["market_regime"] = np.where(trend, "RISK_ON", "RISK_OFF")
    x["market_regime"] = x["market_regime"].astype(str).where(
        x["market_regime"].astype(str).isin(REGIMES), "NEUTRAL"
    )
    x["relative_strength"] = pd.to_numeric(x["relative_strength"], errors="coerce").fillna(0.0)
    x["individual_strength"] = [
        individual_strength(u, d, s, rs)
        for u, d, s, rs in zip(x["up_prob"], x["down_prob"], x["score"], x["relative_strength"])
    ]
    x["strength_bucket"] = x["individual_strength"].map(strength_bucket)
    return x
