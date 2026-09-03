import numpy as np

import profit_top10_paper as app

_original_scan_candidates = app.scan_candidates


def scan_candidates_profit_first(policy):
    result = _original_scan_candidates(policy)
    # profit_top10_paper.main() expects (candidates, scanned), while the
    # progressive runtime may return an optional third selection-level value.
    # Normalize both shapes here so the wrapper cannot reintroduce the
    # "too many values to unpack" failure.
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("scan_candidates returned an invalid result")
    candidates, scanned = result[0], result[1]
    for z in candidates:
        up = float(z.get("up_probability", 0.0)) / 100.0
        down = float(z.get("down_probability", 0.0)) / 100.0
        price = float(z.get("price", 0.0))
        if price <= 0:
            z["expected_value"] = -999.0
            z["profit_score"] = -999.0
            continue
        tp_ret = (float(z.get("tp", price)) - price) / price * 100.0
        sl_ret = (price - float(z.get("sl", price))) / price * 100.0
        z["expected_value"] = up * tp_ret - down * sl_ret
        z["profit_score"] = float(
            np.clip(
                0.70 * float(z.get("score", 0.0))
                + 0.30 * np.clip(z["expected_value"], -5.0, 5.0) * 10.0,
                0.0,
                100.0,
            )
        )
    candidates.sort(
        key=lambda z: (
            float(z.get("profit_score", -999.0)),
            float(z.get("expected_value", -999.0)),
            float(z.get("score", -999.0)),
            float(z.get("up_probability", -999.0)),
        ),
        reverse=True,
    )
    print("✅ 本番TOP10: 確率加重期待利益を含むProfit Score順")
    return candidates, scanned


app.scan_candidates = scan_candidates_profit_first

if __name__ == "__main__":
    app.main()
