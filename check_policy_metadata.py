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

MANUAL_OVERRIDES_FILE_NAME = "policy_manual_overrides.json"

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


def _analyze(path: Path) -> dict:
    """Inspect one policy file and return structured details.

    Never raises: any problem (missing file, broken JSON, unexpected shape)
    is captured as a status/warning instead of an exception, per the
    "must never crash and must never affect trading" requirement.

    Returns a dict with keys: status, strategy_name, mismatches (list of
    raw (field, name_value, actual_value) tuples, only populated when
    status == "mismatch"), warnings (list[str], formatted messages matching
    the pre-existing check_file() behaviour).
    """
    if not path.exists():
        return {"status": "missing", "strategy_name": None, "mismatches": [], "warnings": []}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "status": "read_error",
            "strategy_name": None,
            "mismatches": [],
            "warnings": [f"{path.name}: ファイルを読み込めませんでした ({exc})"],
        }

    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_json",
            "strategy_name": None,
            "mismatches": [],
            "warnings": [f"{path.name}: JSONとして壊れています ({exc})"],
        }

    if not isinstance(policy, dict):
        return {
            "status": "invalid_json",
            "strategy_name": None,
            "mismatches": [],
            "warnings": [f"{path.name}: トップレベルがJSONオブジェクトではありません"],
        }

    strategy_name = policy.get("strategy_name")
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        return {
            "status": "no_name",
            "strategy_name": None,
            "mismatches": [],
            "warnings": [f"{path.name}: strategy_name フィールドがありません"],
        }

    m = NAME_PATTERN.match(strategy_name.strip())
    if not m:
        return {
            "status": "unknown_format",
            "strategy_name": strategy_name,
            "mismatches": [],
            "warnings": [
                f"{path.name}: strategy_name '{strategy_name}' が既知パターン "
                "(UP{up_threshold}_SCORE{min_score_for_buy}_NIKKEI{ON|OFF}_TP{atr_tp_multiplier}"
                "_SL{atr_sl_multiplier}_H{hold_days}) に一致しません [形式不明]"
            ],
        }

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
        return {"status": "ok", "strategy_name": strategy_name, "mismatches": [], "warnings": []}

    warnings = [
        f"{path.name}: strategy_name '{strategy_name}' 上の {field}={name_value!s} が "
        f"実際の値 {field}={actual_value!s} と食い違っています"
        for field, name_value, actual_value in mismatches
    ]
    return {
        "status": "mismatch",
        "strategy_name": strategy_name,
        "mismatches": mismatches,
        "warnings": warnings,
    }


def check_file(path: Path):
    """Inspect one policy file. Returns (status, warnings: list[str]).

    Never raises: any problem (missing file, broken JSON, unexpected shape)
    is captured as a warning/status instead of an exception, per the
    "must never crash and must never affect trading" requirement.
    """
    info = _analyze(path)
    return info["status"], info["warnings"]


def load_manual_overrides(repo_root: Path):
    """Read policy_manual_overrides.json (read-only, advisory).

    Never raises: a missing file, broken JSON, or unexpected shape is
    treated as "no recorded overrides" rather than an error. This file is
    never written to, and it never touches approval_signature or any
    policy file's contents.
    """
    path = repo_root / MANUAL_OVERRIDES_FILE_NAME
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        entries = data.get("overrides") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not (
                isinstance(entry.get("policy_file"), str)
                and isinstance(entry.get("field"), str)
                and "validated_value" in entry
                and "live_value" in entry
            ):
                continue
            result.append(entry)
        return result
    except Exception:
        return []


def _recorded_value_matches(recorded_value, observed_value) -> bool:
    """Compare a recorded override value against a name-derived or actual
    field value, tolerating numeric-formatting and ON/OFF-vs-bool style
    differences the same way numbers_equal() does for the main check."""
    if numbers_equal(str(recorded_value), observed_value):
        return True
    return str(recorded_value).strip().upper() == str(observed_value).strip().upper()


def find_recorded_override(overrides, policy_file_name: str, field: str):
    for entry in overrides:
        if entry.get("policy_file") == policy_file_name and entry.get("field") == field:
            return entry
    return None


def classify_mismatches(policy_file_name: str, mismatches, overrides):
    """Split a file's mismatches into (warnings, notices) formatted
    messages. A mismatch becomes a "recorded manual override" notice only
    when a matching (policy_file, field) entry exists AND both the
    name-side value equals validated_value AND the actual value equals
    live_value; otherwise it is reported as a warning, same as before."""
    warning_messages = []
    notice_messages = []
    for field, name_value, actual_value in mismatches:
        override = find_recorded_override(overrides, policy_file_name, field)
        if (
            override is not None
            and _recorded_value_matches(override.get("validated_value"), name_value)
            and _recorded_value_matches(override.get("live_value"), actual_value)
        ):
            reason = override.get("reason", "")
            notice_messages.append(
                f"{policy_file_name}: 記録済みの手動上書き - {field} は strategy_name 上 "
                f"{name_value!s}(検証済み)、実際の値は {actual_value!s}(未検証)。{reason}"
            )
        else:
            warning_messages.append(
                f"{policy_file_name}: strategy_name 上の {field}={name_value!s} が "
                f"実際の値 {field}={actual_value!s} と食い違っています"
            )
    return warning_messages, notice_messages


def emit_github_warning(message: str) -> None:
    safe = message.replace("\r", "").replace("\n", " ").replace("%", "%25")
    print(f"::warning::{safe}")


def emit_github_notice(message: str) -> None:
    safe = message.replace("\r", "").replace("\n", " ").replace("%", "%25")
    print(f"::notice::{safe}")


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
    overrides = load_manual_overrides(repo_root)
    rows = []
    any_warning = False

    for name in TARGET_FILE_NAMES:
        path = repo_root / name
        info = _analyze(path)
        status = info["status"]

        if status == "missing":
            rows.append((name, "skip", "ファイルなし"))
            continue

        if status == "mismatch":
            warning_messages, notice_messages = classify_mismatches(
                name, info["mismatches"], overrides
            )
            for w in warning_messages:
                emit_github_warning(w)
            for n in notice_messages:
                emit_github_notice(n)

            detail = " / ".join(warning_messages + notice_messages)
            if warning_messages:
                any_warning = True
                rows.append((name, "mismatch", detail))
            else:
                rows.append((name, "manual_override", detail))
            continue

        if info["warnings"]:
            any_warning = True
            for w in info["warnings"]:
                emit_github_warning(w)
            rows.append((name, status, " / ".join(info["warnings"])))
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
