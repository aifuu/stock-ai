#!/usr/bin/env python3
"""Read-only consistency check for strategy_policy*.json.

Parses each policy file's `strategy_name` (expected shape:
UP{up_threshold}_SCORE{min_score_for_buy}_NIKKEI{ON|OFF}_TP{atr_tp_multiplier}
_SL{atr_sl_multiplier}_H{hold_days}) and compares the values encoded in the
name against the file's own actual field values, reporting any mismatch.

This script is advisory only:
  - It never writes to any file (policy files are opened read-only).
  - It never reads, computes, or touches approval_signature or any signing
    secret.
  - It always exits 0 - a mismatch is reported as a warning, never a
    failure, so it can never block or slow down trading execution.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

TARGET_FILE_NAMES = (
    "strategy_policy.json",
    "strategy_policy_up.json",
    "strategy_policy_down.json",
)

NAME_PATTERN = re.compile(
    r"^UP(?P<up_threshold>-?\d+(?:\.\d+)?)"
    r"_SCORE(?P<min_score_for_buy>-?\d+(?:\.\d+)?)"
    r"_NIKKEI(?P<nikkei_filter>ON|OFF)"
    r"_TP(?P<atr_tp_multiplier>-?\d+(?:\.\d+)?)"
    r"_SL(?P<atr_sl_multiplier>-?\d+(?:\.\d+)?)"
    r"_H(?P<hold_days>-?\d+(?:\.\d+)?)$"
)

NUMERIC_FIELDS = (
    "up_threshold",
    "min_score_for_buy",
    "atr_tp_multiplier",
    "atr_sl_multiplier",
    "hold_days",
)


def numbers_equal(name_value: str, actual_value) -> bool:
    """Compare a numeric-string from strategy_name against the actual field
    value as numbers, so formatting differences (e.g. "4.0" vs 4) are not
    reported as mismatches."""
    try:
        return math.isclose(float(name_value), float(actual_value), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def check_file(path: Path):
    """Inspect one policy file. Returns (status, warnings: list[str]).

    Never raises: any problem (missing file, broken JSON, unexpected shape)
    is captured as a warning/status instead of an exception, per the
    "must never crash and must never affect trading" requirement.
    """
    if not path.exists():
        return "missing", []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return "read_error", [f"{path.name}: ファイルを読み込めませんでした ({exc})"]

    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        return "invalid_json", [f"{path.name}: JSONとして壊れています ({exc})"]

    if not isinstance(policy, dict):
        return "invalid_json", [f"{path.name}: トップレベルがJSONオブジェクトではありません"]

    strategy_name = policy.get("strategy_name")
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        return "no_name", [f"{path.name}: strategy_name フィールドがありません"]

    m = NAME_PATTERN.match(strategy_name.strip())
    if not m:
        return "unknown_format", [
            f"{path.name}: strategy_name '{strategy_name}' が既知パターン "
            "(UP{up_threshold}_SCORE{min_score_for_buy}_NIKKEI{ON|OFF}_TP{atr_tp_multiplier}"
            "_SL{atr_sl_multiplier}_H{hold_days}) に一致しません [形式不明]"
        ]

    parsed = m.groupdict()
    mismatches = []

    for field in NUMERIC_FIELDS:
        name_value = parsed[field]
        actual_value = policy.get(field)
        if actual_value is None:
            mismatches.append((field, name_value, "(フィールドなし)"))
        elif not numbers_equal(name_value, actual_value):
            mismatches.append((field, name_value, actual_value))

    name_nikkei_on = parsed["nikkei_filter"] == "ON"
    actual_nikkei = policy.get("nikkei_filter")
    if not isinstance(actual_nikkei, bool):
        mismatches.append(("nikkei_filter", parsed["nikkei_filter"], repr(actual_nikkei)))
    elif name_nikkei_on != actual_nikkei:
        mismatches.append(("nikkei_filter", parsed["nikkei_filter"], actual_nikkei))

    if not mismatches:
        return "ok", []

    warnings = [
        f"{path.name}: strategy_name '{strategy_name}' 上の {field}={name_value!s} が "
        f"実際の値 {field}={actual_value!s} と食い違っています"
        for field, name_value, actual_value in mismatches
    ]
    return "mismatch", warnings


def emit_github_warning(message: str) -> None:
    safe = message.replace("\r", "").replace("\n", " ").replace("%", "%25")
    print(f"::warning::{safe}")


def write_step_summary(rows) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Policy metadata consistency check",
        "",
        "strategy_name とファイル内の実際の値を突き合わせるだけの読み取り専用チェックです。"
        "取引実行ロジックには一切組み込まれていません。",
        "",
        "| file | status | detail |",
        "| --- | --- | --- |",
    ]
    for file_name, status, detail in rows:
        detail_cell = (detail or "").replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {file_name} | {status} | {detail_cell} |")
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    rows = []
    any_warning = False

    for name in TARGET_FILE_NAMES:
        path = repo_root / name
        status, warnings = check_file(path)
        if status == "missing":
            rows.append((name, "skip", "ファイルなし"))
            continue
        if warnings:
            any_warning = True
            for w in warnings:
                emit_github_warning(w)
            rows.append((name, status, " / ".join(warnings)))
        else:
            rows.append((name, "ok", "strategy_name と実際の値は一致"))

    write_step_summary(rows)

    if any_warning:
        print("WARNING: strategy_name と実際の値に食い違いがあります(警告のみ、取引には影響しません)")
    else:
        print("OK: 全policyファイルで strategy_name と実際の値は一致しています")

    return 0


if __name__ == "__main__":
    sys.exit(main())
