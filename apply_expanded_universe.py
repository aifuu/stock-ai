#!/usr/bin/env python3
"""Apply the shared expanded universe to scanner/validation scripts at runtime.

This is deliberately a build-time patch: the source files keep their core
universe, while Actions uses one identical expanded universe for all relevant
pipelines. No strategy thresholds are changed here.
"""
import ast
from pathlib import Path
from expanded_universe import TICKERS

TARGETS = [
    Path("daily_directional_top1.py"),
    Path("walk_forward.py"),
    Path("stock_scan.py"),
]


def replace_tickers(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    node = None
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id == "TICKERS":
                    node = n
                    break
        if node is not None:
            break
    if node is None:
        raise RuntimeError(f"{path}: TICKERS が見つかりません")
    lines = source.splitlines(keepends=True)
    start = node.value.lineno - 1
    end = node.value.end_lineno
    replacement = "TICKERS = [\n" + "".join(f'    "{t}",\n' for t in TICKERS) + "]"
    new_source = "".join(lines[:start]) + replacement + "\n" + "".join(lines[end:])
    ast.parse(new_source, filename=str(path))
    path.write_text(new_source, encoding="utf-8")


if __name__ == "__main__":
    if len(TICKERS) < 250 or len(TICKERS) != len(set(TICKERS)):
        raise SystemExit(f"❌ expanded universe 不正: {len(TICKERS)}銘柄")
    for path in TARGETS:
        replace_tickers(path)
    print(f"✅ Expanded universe applied: {len(TICKERS)}銘柄 / 重複なし")
    print("対象: daily_directional_top1.py / walk_forward.py / stock_scan.py")
