import ast
from pathlib import Path

# =========================================================
# 銘柄ユニバース同期
#
# stock_scan.py / walk_forward.py の銘柄数を必ず一致させる。
# 現在のstock_scan.pyは94銘柄なので、ここで6銘柄を追加して
# 合計100銘柄にする。
# =========================================================

BASE_FILE = Path("stock_scan.py")
WF_FILE = Path("walk_forward.py")

EXTRA_TICKERS = [
    "9432.T",  # NTT
    "5401.T",  # 日本製鉄
    "2914.T",  # JT
    "3382.T",  # セブン&アイ
    "4568.T",  # 第一三共
    "6098.T",  # リクルートHD
]


def extract_tickers(path: Path):
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


def replace_tickers(path: Path, tickers):
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
    path.write_text(new_source, encoding="utf-8")


def main():
    stock_tickers = extract_tickers(BASE_FILE)

    # 重複を除去しつつ現在の順番を維持。
    stock_tickers = list(dict.fromkeys(stock_tickers))

    for ticker in EXTRA_TICKERS:
        if ticker not in stock_tickers:
            stock_tickers.append(ticker)

    if len(stock_tickers) != 100:
        raise RuntimeError(f"最終銘柄数が100ではありません: {len(stock_tickers)}")

    replace_tickers(BASE_FILE, stock_tickers)
    replace_tickers(WF_FILE, stock_tickers)

    print(f"✅ 銘柄ユニバース同期完了: {len(stock_tickers)}銘柄")
    print("  stock_scan.py   =", len(extract_tickers(BASE_FILE)))
    print("  walk_forward.py =", len(extract_tickers(WF_FILE)))

    if extract_tickers(BASE_FILE) != extract_tickers(WF_FILE):
        raise RuntimeError("❌ stock_scan.py と walk_forward.py の銘柄順・内容が一致していません")

    print("✅ 2ファイルのTICKERS完全一致")


if __name__ == "__main__":
    main()
