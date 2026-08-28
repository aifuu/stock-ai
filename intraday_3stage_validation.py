import os
import shutil
import subprocess
import sys
from pathlib import Path

STRATEGIES = {
    "STANDARD": {"score": "65", "prob": "55", "vol": "1.0"},
    "RELAXED": {"score": "60", "prob": "52", "vol": "0.9"},
    "LOOSE": {"score": "55", "prob": "50", "vol": "0.8"},
}

BASE_OUTPUT = Path("intraday_top3_backtest_results.csv")
BASE_EQUITY = Path("intraday_top3_equity.csv")


def run_strategy(name, cfg):
    env = os.environ.copy()
    env.update(
        {
            "IT_MIN_SCORE": cfg["score"],
            "IT_MIN_UP_PROB": cfg["prob"],
            "IT_MIN_VOL_RATIO": cfg["vol"],
        }
    )

    print("\n" + "=" * 72)
    print(f"3段階検証: {name}")
    print(
        f"AIスコア>={cfg['score']} / 上昇確率>={cfg['prob']}% / "
        f"出来高倍率>={cfg['vol']}"
    )
    print("=" * 72)

    for path in (BASE_OUTPUT, BASE_EQUITY):
        if path.exists():
            path.unlink()

    completed = subprocess.run(
        [sys.executable, "intraday_top3_backtest.py"],
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        print(f"⚠ {name}: バックテスト終了コード={completed.returncode}")
        return False

    if BASE_OUTPUT.exists():
        shutil.move(BASE_OUTPUT, f"intraday_top3_backtest_{name}.csv")
    else:
        print(f"⚠ {name}: 結果CSVなし")

    if BASE_EQUITY.exists():
        shutil.move(BASE_EQUITY, f"intraday_top3_equity_{name}.csv")

    return True


def summarize():
    import numpy as np
    import pandas as pd

    rows = []
    for name in STRATEGIES:
        f = Path(f"intraday_top3_backtest_{name}.csv")
        if not f.exists():
            rows.append({"strategy": name, "trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "pf": 0.0})
            continue

        df = pd.read_csv(f)
        trades = df[df["status"] == "TRADE"].copy()
        if trades.empty:
            rows.append({"strategy": name, "trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "pf": 0.0})
            continue

        returns = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
        wins = int((returns > 0).sum())
        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(-returns[returns < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
        rows.append(
            {
                "strategy": name,
                "trades": len(trades),
                "win_rate_pct": wins / len(trades) * 100.0,
                "avg_return_pct": float(returns.mean()),
                "pf": pf,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("intraday_3stage_summary.csv", index=False, encoding="utf-8-sig")
    print("\n📊 3段階比較")
    print(out.to_string(index=False))

    valid = out[out["trades"] > 0].copy()
    if not valid.empty:
        best = valid.sort_values(["pf", "avg_return_pct", "trades"], ascending=[False, False, False]).iloc[0]
        print(
            f"\n✅ 暫定ベスト: {best['strategy']} "
            f"件数={int(best['trades'])} "
            f"勝率={best['win_rate_pct']:.1f}% "
            f"平均={best['avg_return_pct']:+.3f}% "
            f"PF={best['pf']:.3f}"
        )


if __name__ == "__main__":
    ok = True
    for name, cfg in STRATEGIES.items():
        ok = run_strategy(name, cfg) and ok
    summarize()
    raise SystemExit(0 if ok else 1)
