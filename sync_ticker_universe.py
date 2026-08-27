#!/usr/bin/env python3
"""Synchronize stock universe between stock_scan.py and walk_forward.py."""

import argparse
import ast
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_FILE = SCRIPT_DIR / "stock_scan.py"
DEFAULT_WF_FILE = SCRIPT_DIR / "walk_forward.py"

# 既存96銘柄に未登録の4銘柄を追加し、100銘柄にする。
DEFAULT_EXTRA_TICKERS = [
    "6762.T",  # TDK
    "7735.T",  # SCREENホールディングス
    "6981.T",  # 村田製作所
    "4543.T",  # テルモ
]

# 285A.T のような英字入りコードも許可。
TICKER_PATTERN = re.compile(r"^[0-9A-Z]{4}\.T$")


def extract_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path}: ファイルが存在しません")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TICKERS":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                        raise RuntimeError(f"{path}: TICKERSが文字列リストではありません")
                    return value

    raise RuntimeError(f"{path}: TICKERSが見つかりません")


def validate_tickers(tickers: list[str]) -> None:
    invalid = [t for t in tickers if not TICKER_PATTERN.fullmatch(t)]
    if invalid:
        raise RuntimeError(
            "銘柄コードのフォーマットが不正です "
            "（4文字の英数字 + .T を許可。例: 9432.T / 285A.T）: "
            f"{invalid}"
        )

    seen = set()
    duplicates = set()
    for ticker in tickers:
        if ticker in seen:
            duplicates.add(ticker)
        seen.add(ticker)
    if duplicates:
        raise RuntimeError(f"重複した銘柄コードがあります: {sorted(duplicates)}")


def replace_tickers(path: Path, tickers: list[str], *, backup: bool = True) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    target_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TICKERS":
                    target_node = node
                    break
        if target_node is not None:
            break

    if target_node is None:
        raise RuntimeError(f"{path}: TICKERSが見つかりません")

    start = target_node.value.lineno - 1
    end = target_node.value.end_lineno
    replacement = "TICKERS = [\n" + "".join(f'    "{t}",\n' for t in tickers) + "]"

    lines = source.splitlines(keepends=True)
    new_source = "".join(lines[:start]) + replacement + "\n" + "".join(lines[end:])

    # 書き換え後もPythonとして解釈できることを確認。
    ast.parse(new_source, filename=str(path))

    if backup:
        path.with_suffix(path.suffix + ".bak").write_text(source, encoding="utf-8")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(new_source, encoding="utf-8")
    tmp_path.replace(path)


def sync(base_file: Path, wf_file: Path, target_count: int = 100, *, backup: bool = True) -> None:
    stock_tickers = list(dict.fromkeys(extract_tickers(base_file)))
    before_count = len(stock_tickers)

    for ticker in DEFAULT_EXTRA_TICKERS:
        if ticker not in stock_tickers:
            stock_tickers.append(ticker)

    validate_tickers(stock_tickers)

    if len(stock_tickers) != target_count:
        raise RuntimeError(
            f"最終銘柄数が目標{target_count}と一致しません: "
            f"開始{before_count}銘柄 → {len(stock_tickers)}銘柄。"
            "追加候補を見直してください。"
        )

    replace_tickers(base_file, stock_tickers, backup=backup)
    replace_tickers(wf_file, stock_tickers, backup=backup)

    final_stock = extract_tickers(base_file)
    final_wf = extract_tickers(wf_file)

    validate_tickers(final_stock)
    validate_tickers(final_wf)

    if len(final_stock) != target_count or len(final_wf) != target_count:
        raise RuntimeError("同期後の銘柄数が100ではありません")

    if final_stock != final_wf:
        raise RuntimeError("stock_scan.py と walk_forward.py のTICKERSが一致していません")

    print(f"✅ 銘柄ユニバース同期完了: {before_count} → {len(final_stock)}銘柄")
    print(f"  stock_scan.py   = {len(final_stock)}")
    print(f"  walk_forward.py = {len(final_wf)}")
    print("✅ 100銘柄・重複なし・2ファイル完全一致")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="銘柄ユニバース同期スクリプト")
    parser.add_argument("--base-file", type=Path, default=DEFAULT_BASE_FILE)
    parser.add_argument("--wf-file", type=Path, default=DEFAULT_WF_FILE)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sync(
            base_file=args.base_file,
            wf_file=args.wf_file,
            target_count=args.target_count,
            backup=not args.no_backup,
        )
    except Exception as e:
        print(f"❌ 失敗: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
