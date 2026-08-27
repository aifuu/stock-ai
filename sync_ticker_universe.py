#!/usr/bin/env python3
"""
=========================================================
銘柄ユニバース同期スクリプト（改善版）

stock_scan.py の TICKERS を正（source of truth）として、
walk_forward.py の TICKERS を完全一致させる。

改善点:
  1. ハードコードされた銘柄数前提を撤廃し、
     現在の銘柄数と追加後の銘柄数を動的に検証・表示する
  2. ticker のフォーマット検証を追加
     ・通常コード: 4桁数字 + .T
     ・新しい形式のコード: 英数字4文字 + .T
       例: 285A.T
  3. 書き込み前に AST として再パース可能か検証してから
     アトミックに書き込む（壊れたファイルで上書きしない）
  4. 上書き前に .bak バックアップを自動作成
  5. スクリプトの実行位置に依存しないよう、パスを
     このファイル自身の場所基準に解決
  6. 重複銘柄・フォーマット不正銘柄を明示的に報告
  7. コマンドライン引数でファイルパス・追加銘柄・目標数を
     指定可能に
=========================================================
"""

import argparse
import ast
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_FILE = SCRIPT_DIR / "stock_scan.py"
DEFAULT_WF_FILE = SCRIPT_DIR / "walk_forward.py"

DEFAULT_EXTRA_TICKERS = [
    "9432.T",  # NTT
    "5401.T",  # 日本製鉄
    "2914.T",  # JT
    "3382.T",  # セブン&アイ
    "4568.T",  # 第一三共
    "6098.T",  # リクルートHD
]

# 東証系ティッカー:
#   通常の4桁数字だけでなく、285A.Tのような
#   英字を含む新形式も許可する。
TICKER_PATTERN = re.compile(r"^[0-9A-Z]{4}\.T$")


def extract_tickers(path: Path) -> list[str]:
    """指定ファイルから TICKERS = [...] の値を取得する。"""
    if not path.exists():
        raise FileNotFoundError(f"{path}: ファイルが存在しません")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TICKERS":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not all(
                        isinstance(x, str) for x in value
                    ):
                        raise RuntimeError(
                            f"{path}: TICKERS が文字列リストではありません"
                        )
                    return value

    raise RuntimeError(f"{path}: TICKERS が見つかりません")


def validate_tickers(tickers: list[str]) -> None:
    """フォーマット不正・重複を検出してエラーにする。"""
    invalid = [t for t in tickers if not TICKER_PATTERN.match(t)]
    if invalid:
        raise RuntimeError(
            "銘柄コードのフォーマットが不正です "
            "（4文字の英数字 + .T を許可。例: 9432.T / 285A.T）: "
            f"{invalid}"
        )

    seen = set()
    duplicates = set()
    for t in tickers:
        if t in seen:
            duplicates.add(t)
        seen.add(t)
    if duplicates:
        raise RuntimeError(f"重複した銘柄コードがあります: {sorted(duplicates)}")


def replace_tickers(path: Path, tickers: list[str], *, backup: bool = True) -> None:
    """TICKERS = [...] を新しいリストに書き換える。"""
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
        raise RuntimeError(f"{path}: TICKERS が見つかりません")

    start = target_node.value.lineno - 1
    end = target_node.value.end_lineno

    replacement = "TICKERS = [\n" + "".join(
        f'    "{t}",\n' for t in tickers
    ) + "]"

    lines = source.splitlines(keepends=True)
    new_source = (
        "".join(lines[:start])
        + replacement
        + "\n"
        + "".join(lines[end:])
    )

    # 書き込み前に構文が壊れていないか検証
    try:
        ast.parse(new_source, filename=str(path))
    except SyntaxError as e:
        raise RuntimeError(
            f"{path}: 書き換え後のコードが構文エラーになるため中止しました: {e}"
        ) from e

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(source, encoding="utf-8")

    # アトミック書き込み
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(new_source, encoding="utf-8")
    tmp_path.replace(path)


def sync(
    base_file: Path,
    wf_file: Path,
    extra_tickers: list[str],
    target_count: int | None,
    *,
    backup: bool = True,
) -> None:
    base_tickers = extract_tickers(base_file)
    before_count = len(base_tickers)

    # 重複を除去しつつ現在の順番を維持
    merged = list(dict.fromkeys(base_tickers))

    for ticker in extra_tickers:
        if ticker not in merged:
            merged.append(ticker)

    validate_tickers(merged)

    if target_count is not None and len(merged) != target_count:
        raise RuntimeError(
            f"最終銘柄数が目標({target_count})と一致しません: "
            f"現在{before_count}銘柄 + 追加候補{len(extra_tickers)}銘柄 "
            f"→ 実際{len(merged)}銘柄（重複や既存分を除いた結果）"
        )

    replace_tickers(
        base_file,
        merged,
        backup=backup,
    )

    replace_tickers(
        wf_file,
        merged,
        backup=backup,
    )

    final_base = extract_tickers(base_file)
    final_wf = extract_tickers(wf_file)

    print(
        f"✅ 銘柄ユニバース同期完了: "
        f"{before_count} → {len(final_base)}銘柄"
    )

    print(
        "  stock_scan.py   =",
        len(final_base),
    )

    print(
        "  walk_forward.py =",
        len(final_wf),
    )

    if final_base != final_wf:
        raise RuntimeError(
            "❌ stock_scan.py と walk_forward.py の"
            "銘柄順・内容が一致していません"
        )

    print(
        "✅ 2ファイルのTICKERS完全一致"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="銘柄ユニバース同期スクリプト"
    )

    parser.add_argument(
        "--base-file",
        type=Path,
        default=DEFAULT_BASE_FILE,
    )

    parser.add_argument(
        "--wf-file",
        type=Path,
        default=DEFAULT_WF_FILE,
    )

    parser.add_argument(
        "--extra-ticker",
        action="append",
        dest="extra_tickers",
        default=None,
        help=(
            "追加する銘柄コード（複数指定可）。"
            "未指定時はデフォルト6銘柄を使用"
        ),
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="最終的な銘柄数の期待値（省略時はチェックしない）",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="書き換え前の .bak バックアップ作成を無効化",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extra_tickers = (
        args.extra_tickers
        if args.extra_tickers is not None
        else DEFAULT_EXTRA_TICKERS
    )

    try:
        sync(
            base_file=args.base_file,
            wf_file=args.wf_file,
            extra_tickers=extra_tickers,
            target_count=args.target_count,
            backup=not args.no_backup,
        )

    except Exception as e:
        print(
            f"❌ 失敗: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
