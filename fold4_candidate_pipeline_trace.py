import ast
import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CANDIDATE_FILE = Path("walk_forward_all_candidates.csv")
WALK_FORWARD_FILE = Path("walk_forward.py")
OUT_FILE = Path("fold4_candidate_pipeline_trace.csv")

TARGET_DATES = [
    pd.Timestamp("2026-08-18"),
    pd.Timestamp("2026-08-19"),
    pd.Timestamp("2026-08-20"),
]
UP = 45.0
SCORE = 50.0
TOP_N = int(os.getenv("WF_TOP_N", "10"))


def load_tickers():
    if not WALK_FORWARD_FILE.exists():
        raise SystemExit("❌ walk_forward.py がありません")
    tree = ast.parse(WALK_FORWARD_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TICKERS":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, list) and value:
                        return [str(x) for x in value]
    raise SystemExit("❌ walk_forward.py から TICKERS を取得できません")


def normalize_columns(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def download_one(ticker):
    try:
        df = yf.download(
            ticker,
            start="2026-08-01",
            end="2026-08-22",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_columns(df)
    except Exception as e:
        print(f"⚠ {ticker}: {e}")
        return None


def main():
    tickers = load_tickers()
    c = pd.read_csv(CANDIDATE_FILE)
    required = ["date", "ticker", "score", "up_prob", "flat_prob", "down_prob"]
    missing = [x for x in required if x not in c.columns]
    if missing:
        raise SystemExit("❌ 候補CSVの不足列: " + ", ".join(missing))

    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    for col in ["score", "up_prob", "flat_prob", "down_prob"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c["ticker"] = c["ticker"].astype(str)
    c = c.dropna(subset=required)

    print("=" * 100)
    print("🔬 FOLD 4 CANDIDATE PIPELINE TRACE")
    print("目的: 8/18〜8/20に100銘柄級の対象がどの工程で消えたかを分離")
    print("重要: 閾値・policy・ゲートは変更しない。診断のみ。")
    print("=" * 100)
    print(f"TICKERS in walk_forward.py: {len(tickers)}")
    print(f"TARGET_DATES: {[d.date() for d in TARGET_DATES]}")

    # Candidate-side filter trace.
    rows = []
    for date in TARGET_DATES:
        d = c[c["date"] == date].copy()
        n0 = len(d)
        n1 = len(d[d["up_prob"] >= UP])
        n2 = len(d[(d["up_prob"] >= UP) & (d["up_prob"] > d["down_prob"])])
        n3 = len(d[(d["up_prob"] >= UP) & (d["up_prob"] > d["down_prob"]) & (d["flat_prob"] < 50)])
        n4 = len(d[(d["up_prob"] >= UP) & (d["up_prob"] > d["down_prob"]) & (d["flat_prob"] < 50) & (d["score"] >= SCORE)])
        rows.append({
            "date": date.date(),
            "stage": "candidate_csv_rows",
            "count": n0,
            "excluded_from_previous": 0,
        })
        rows.extend([
            {"date": date.date(), "stage": f"UP>={UP:g}", "count": n1, "excluded_from_previous": n0 - n1},
            {"date": date.date(), "stage": "UP>DOWN", "count": n2, "excluded_from_previous": n1 - n2},
            {"date": date.date(), "stage": "FLAT<50", "count": n3, "excluded_from_previous": n2 - n3},
            {"date": date.date(), "stage": f"SCORE>={SCORE:g}", "count": n4, "excluded_from_previous": n3 - n4},
        ])

    # Market-data coverage trace. This is deliberately independent of candidate filters.
    coverage = []
    cache = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"data {i:>3}/{len(tickers)} {ticker}")
        df = download_one(ticker)
        cache[ticker] = df
        time.sleep(0.03)

    candidate_keys = set(zip(c["date"], c["ticker"]))
    for date in TARGET_DATES:
        market_rows = 0
        valid_ohlcv = 0
        sufficient_history = 0
        present_candidate = 0
        missing_data = []
        invalid_ohlcv = []
        no_history = []
        not_candidate = []

        for ticker in tickers:
            df = cache.get(ticker)
            if df is None or date not in df.index:
                missing_data.append(ticker)
                continue
            market_rows += 1
            row = df.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            fields = []
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    fields.append(col)
                else:
                    try:
                        if pd.isna(float(row[col])):
                            fields.append(col)
                    except Exception:
                        fields.append(col)
            if fields:
                invalid_ohlcv.append(f"{ticker}({','.join(fields)})")
                continue
            valid_ohlcv += 1
            hist = df[df.index < date]
            if len(hist) >= 252:
                sufficient_history += 1
            else:
                no_history.append(ticker)
            if (date, ticker) in candidate_keys:
                present_candidate += 1
            else:
                not_candidate.append(ticker)

        rows.extend([
            {"date": date.date(), "stage": "expected_TICKERS", "count": len(tickers), "excluded_from_previous": 0},
            {"date": date.date(), "stage": "market_data_row", "count": market_rows, "excluded_from_previous": len(tickers) - market_rows},
            {"date": date.date(), "stage": "valid_OHLCV", "count": valid_ohlcv, "excluded_from_previous": market_rows - valid_ohlcv},
            {"date": date.date(), "stage": "history>=252d", "count": sufficient_history, "excluded_from_previous": valid_ohlcv - sufficient_history},
            {"date": date.date(), "stage": "candidate_csv_presence", "count": present_candidate, "excluded_from_previous": sufficient_history - present_candidate},
        ])

        print(f"\n📅 {date.date()}")
        print(f"  expected TICKERS : {len(tickers)}")
        print(f"  market data row  : {market_rows}")
        print(f"  valid OHLCV      : {valid_ohlcv}")
        print(f"  history >=252d   : {sufficient_history}")
        print(f"  candidate CSV    : {present_candidate}")
        print(f"  missing data     : {len(missing_data)}")
        if missing_data:
            print("    " + ", ".join(missing_data[:30]))
        print(f"  invalid OHLCV    : {len(invalid_ohlcv)}")
        if invalid_ohlcv:
            print("    " + ", ".join(invalid_ohlcv[:20]))
        print(f"  no history       : {len(no_history)}")
        if no_history:
            print("    " + ", ".join(no_history[:20]))
        print(f"  data-valid but not candidate: {len(not_candidate)}")
        if not_candidate:
            print("    " + ", ".join(not_candidate[:30]))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("判定の読み方")
    print("1) market_data_row が大幅減少 → yfinance/データ取得側が主因")
    print("2) market_data_row は多いが valid_OHLCV が減少 → OHLCV欠損が主因")
    print("3) history>=252d が減少 → ウォームアップ/履歴不足が主因")
    print("4) data-valid が多いのに candidate_csv_presence が少ない → 特徴量/モデル/候補フィルター側を重点調査")
    print(f"診断CSV: {OUT_FILE}")


if __name__ == "__main__":
    main()
