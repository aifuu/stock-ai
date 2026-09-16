"""
日経MACD逆張り(単一指標) ペーパートレード ― 実験的セカンドトラック
====================================================================

★このスクリプトは本番トレードロジック(daily_directional_top1.py の main()、
profit_top10_paper.py の scan()/open_positions()/mark_and_close() など)を
一切呼び出さず、本番の状態ファイル(profit_top10_paper_state.json,
strategy_policy.json 等)も一切読み書きしない、完全に独立した実験用の
セカンドトラック(研究用のsingle_indicator_backtest.pyと同じ位置付けだが、
こちらは毎営業日1回自動実行してポジションを実際に持ち越すペーパートレード)。

背景:
  single_indicator_backtest.py の拡張検証(run #2)で、
  nikkei_macd_bottom10pct(日経225指数自体のMACDが直近1年(252営業日)の
  分布で下位10パーセンタイル以下 = 市場全体が深く売られすぎている局面で
  個別銘柄をBUY)が単体で最も強いエッジ(平均リターン+1.62%、取引1,685件、
  勝率65.22%)だった。このシグナルだけを使い、本番と同じAIモデルで
  225銘柄中スコア最高の1銘柄(TOP1)に集中投資するペーパートレードを、
  本番とは別枠の仮想資金で回す。

3パターン並列トラック(nikkei_macd_dip_backtest_compare.py のTP_MULTグリッド
検証結果を受け、TP_MULT=[0.22, 0.5, 0.88, 1.25, 1.75, 2.5, 3.5]の7段階のうち
一番上(top)・中間(mid)・一番下(bottom)の3パターンを、それぞれ独立した
仮想資金100万円(合計300万円)で同時に走らせる):
  - top    : TP_MULT=0.22 / SL_MULT=0.1257(現行と同じ、旧来からの継続ポジションを引き継ぐ)
  - mid    : TP_MULT=1.25 / SL_MULT=1.25/1.75(RR比1.75:1)
  - bottom : TP_MULT=3.5  / SL_MULT=2.0(旧設定)

設計:
  - シグナル判定(日経225指数のMACD分位点)とAIモデルによるTOP1銘柄選定は
    1日1回だけ共通で行う(3パターンともAIスコアは同一のため、同じ日に
    エントリーする場合の銘柄自体は3パターンとも同じになるのが自然)。
    エントリー価格・TP価格・SL価格はTP_MULT/SL_MULTがパターンごとに異なる
    ため、パターンごとに計算する。
  - ポジションの保有有無・決済判定・資金(capital/peak/max_dd)は
    パターンごとに完全に独立して管理する(state/historyファイルもパターン
    ごとに分離)。保有期間やTP/SL到達タイミングはパターンごとに異なり得る
    ため、決済判定(check_exit)もパターンごとに個別に行う。
  - シグナル判定は single_indicator_backtest.py の continuous_entry_signals()
    と完全に同一の計算式(直近252営業日ローリングウィンドウの10%分位点
    以下かどうか)を、日経225指数自身のmacd列(daily_directional_top1.py の
    make_nikkei()が返す)に対してそのまま適用する。
  - TOP1選定は daily_directional_top1.py の load_model()/features()/
    directional_score() を流用し、二重実装しない。BUY方向のスコア
    (long_score)が最高の銘柄を選ぶ(up_threshold/min_scoreの下限は設けない。
    バックテストのエッジは日経条件のみで発生していたため)。
  - 決済シミュレーションは single_indicator_backtest.py の
    simulate_trades() と同じロジック(TP/SLのATR倍率、最大保有3営業日、
    日次High/LowでTP/SL判定)を、実際に日をまたいでポジションを保有し
    続ける形で再実装している(状態ファイルに position を保持し、実行の
    たびにエントリー日からの日次バーを遡ってTP/SL/保有期限到達を判定する)。
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_directional_top1 import (  # noqa: E402
    FEATURES,
    NAMES,
    TICKERS,
    atr,
    directional_score,
    download,
    features,
    load_model,
    make_nikkei,
)

TZ = ZoneInfo("Asia/Tokyo")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

# 本番(AI_INITIAL_CAPITAL)とは別枠の新規資金。あえて別のenv変数名にして
# 本番の資金設定と混ざらないようにする。3パターンとも同額(デフォルト100万円)。
BASE_INITIAL_CAPITAL = float(os.getenv("NIKKEI_MACD_DIP_INITIAL_CAPITAL", "1000000"))
LOT_SIZE = 100
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))

# single_indicator_backtest.py の nikkei_macd_bottom10pct と完全に同一の定義
LOOKBACK_WINDOW = 252
PERCENTILE = 0.10

# 3パターンとも共通(検証時と同じ最大保有日数)
HOLD_DAYS = 3

BASE_LABEL = "🔬 日経MACD逆張(別トラック)"


def _fmt(x):
    return f"{round(float(x), 4):g}"


CONFIGS = [
    {
        "name": "top",
        "tp_mult": 0.22,
        "sl_mult": 0.1257,
        "state_file": "nikkei_macd_dip_paper_state_top.json",
        "history_file": "nikkei_macd_dip_paper_history_top.csv",
        "initial_capital": BASE_INITIAL_CAPITAL,
    },
    {
        "name": "mid",
        "tp_mult": 1.25,
        "sl_mult": 1.25 / 1.75,
        "state_file": "nikkei_macd_dip_paper_state_mid.json",
        "history_file": "nikkei_macd_dip_paper_history_mid.csv",
        "initial_capital": BASE_INITIAL_CAPITAL,
    },
    {
        "name": "bottom",
        "tp_mult": 3.5,
        "sl_mult": 2.0,
        "state_file": "nikkei_macd_dip_paper_state_bottom.json",
        "history_file": "nikkei_macd_dip_paper_history_bottom.csv",
        "initial_capital": BASE_INITIAL_CAPITAL,
    },
]
for _cfg in CONFIGS:
    _cfg["label"] = f"{BASE_LABEL}[{_cfg['name']} TP{_fmt(_cfg['tp_mult'])}/SL{_fmt(_cfg['sl_mult'])}]"


def send(msg):
    text = str(msg)
    print(text)
    if not WEBHOOK_URL:
        return False
    text = text if len(text) <= 1900 else text[:1897] + "..."
    try:
        r = requests.post(WEBHOOK_URL, json={"content": text}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"⚠️ Discord通知失敗: {exc}")
        return False


def default_state(cfg):
    return {
        "capital": cfg["initial_capital"],
        "peak": cfg["initial_capital"],
        "max_dd": 0.0,
        "position": None,
        "trades_total": 0,
    }


def load_state(cfg):
    s = default_state(cfg)
    if os.path.exists(cfg["state_file"]):
        try:
            with open(cfg["state_file"], encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception:
            pass
    return s


def save_state(cfg, s):
    tmp = cfg["state_file"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cfg["state_file"])


def append_history(cfg, row):
    df = pd.DataFrame([row])
    if os.path.exists(cfg["history_file"]):
        try:
            df = pd.concat([pd.read_csv(cfg["history_file"]), df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(cfg["history_file"], index=False, encoding="utf-8-sig")


def nikkei_macd_signal(nikkei):
    """single_indicator_backtest.py の continuous_entry_signals(x, 'nikkei_macd', 0.10)
    の low_signal(下位10%)と完全に同一の計算式を、日経指数自身のmacd列に適用する。"""
    macd = nikkei["macd"]
    if len(macd) < LOOKBACK_WINDOW:
        return None
    roll = macd.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW)
    lo_thr_series = roll.quantile(PERCENTILE)
    today_macd = float(macd.iloc[-1])
    today_thr = lo_thr_series.iloc[-1]
    today_thr = float(today_thr) if pd.notna(today_thr) else None
    window = macd.iloc[-LOOKBACK_WINDOW:]
    pct_rank = float((window <= today_macd).mean() * 100)
    signal = today_thr is not None and today_macd <= today_thr
    return {
        "today_macd": today_macd,
        "today_thr": today_thr,
        "pct_rank": pct_rank,
        "signal": signal,
        "nikkei_date": str(nikkei.index[-1].date()),
    }


def check_exit(position, today):
    """simulate_trades()のTP/SL判定ロジック(SL優先の同日判定、最大保有日数到達で
    HOLD_LIMIT決済)を、実際にエントリー日から今日までの日次バーを遡って適用する。
    TP/SL価格はエントリー時にconfigごとのTP_MULT/SL_MULTで計算済みのものを
    positionに保持しているため、ここではconfig非依存。"""
    df = download(position["ticker"], period="6mo")
    if df is None or df.empty:
        return None
    entry_date = pd.Timestamp(position["entry_date"])
    bdays = pd.bdate_range(entry_date + pd.Timedelta(days=1), pd.Timestamp(today))
    if len(bdays) == 0:
        return None
    bars = df[df.index.normalize().isin(bdays)].sort_index()
    if bars.empty:
        return None
    tp, sl = float(position["tp"]), float(position["sl"])
    exit_price = reason = exit_dt = None
    for day_idx, (ts, bar) in enumerate(bars.iterrows(), start=1):
        hi, lo = float(bar["High"]), float(bar["Low"])
        if lo <= sl and hi >= tp:
            exit_price, reason = sl, "SL"
        elif hi >= tp:
            exit_price, reason = tp, "TP"
        elif lo <= sl:
            exit_price, reason = sl, "SL"
        if reason:
            exit_dt = ts
            break
        if day_idx >= HOLD_DAYS:
            exit_price, reason, exit_dt = float(bar["Close"]), "HOLD_LIMIT", ts
            break
    if reason is None:
        return None
    return exit_price, reason, exit_dt


def close_position(cfg, state):
    p = state["position"]
    result = check_exit(p, datetime.now(TZ).strftime("%Y-%m-%d"))
    if result is None:
        return None
    exit_price, reason, exit_dt = result
    entry_price = float(p["entry_price"])
    shares = int(p["shares"])
    gross = (exit_price - entry_price) * shares
    fee = (entry_price + exit_price) * shares * FEE_RATE
    pnl = gross - fee
    state["capital"] = float(state["capital"]) + pnl
    state["peak"] = max(float(state.get("peak", state["capital"])), float(state["capital"]))
    state["max_dd"] = max(
        float(state.get("max_dd", 0.0)),
        (state["peak"] - state["capital"]) / state["peak"] * 100 if state["peak"] else 0.0,
    )
    exit_date_str = str(pd.Timestamp(exit_dt).date())
    hold_days_actual = len(pd.bdate_range(pd.Timestamp(p["entry_date"]), pd.Timestamp(exit_dt)))
    append_history(cfg, {
        "entry_date": p["entry_date"],
        "exit_date": exit_date_str,
        "ticker": p["ticker"],
        "company": p["company"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "invested_amount": p["invested_amount"],
        "tp": p["tp"],
        "sl": p["sl"],
        "score": p.get("score"),
        "up_probability": p.get("up_probability"),
        "down_probability": p.get("down_probability"),
        "nikkei_macd_at_entry": p.get("nikkei_macd_at_entry"),
        "nikkei_macd_percentile_at_entry": p.get("nikkei_macd_percentile_at_entry"),
        "return_pct": round(pnl / p["invested_amount"] * 100, 3) if p["invested_amount"] else 0,
        "pnl": round(pnl, 2),
        "result": reason,
        "hold_days": hold_days_actual,
        "capital_after": round(state["capital"], 2),
    })
    state["position"] = None
    result_label = {"TP": "利確(TP)", "SL": "損切(SL)", "HOLD_LIMIT": "期限到達"}.get(reason, reason)
    emoji = "✅" if pnl >= 0 else "❌"
    return (
        f"{emoji} {cfg['label']}｜決済\n"
        f"📅 {exit_date_str}\n"
        f"{p['ticker']} {p['company']}\n"
        f"エントリー {entry_price:,.1f} → 決済 {exit_price:,.1f}\n"
        f"結果: {result_label}\n"
        f"損益: {pnl:+,.0f}円\n"
        f"保有: {hold_days_actual}営業日\n\n"
        f"💰 仮想資産(別枠): {state['capital']:,.0f}円"
    )


def find_best_candidate(nikkei):
    """AIモデルによるTOP1銘柄選定。3パターンともAIスコアは共通のため1日1回だけ
    実行し、結果を各configのエントリー計算に使い回す。"""
    model = load_model()
    if model is None:
        return None, 0, "model_load_failed"
    cols = list(getattr(model, "feature_names_in_", [])) or list(FEATURES)
    best = None
    scanned = 0
    for ticker in TICKERS:
        df = download(ticker, period="3y")
        if df is None or len(df) < 150:
            continue
        x = features(df, nikkei)
        try:
            x = x.dropna(subset=cols)
        except KeyError:
            continue
        if x.empty:
            continue
        scanned += 1
        last = x.iloc[-1]
        try:
            probs = model.predict_proba(x[cols].iloc[-1:])[0]
            classes = list(model.classes_)
            down = float(probs[classes.index(0)])
            up = float(probs[classes.index(2)])
            long_s, _short_s = directional_score(last, up, down)
            price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            if not np.isfinite(a) or a <= 0 or price <= 0:
                continue
            if best is None or long_s > best["score"]:
                best = {
                    "ticker": ticker,
                    "company": NAMES.get(ticker, ticker),
                    "score": float(long_s),
                    "up_probability": up * 100,
                    "down_probability": down * 100,
                    "price": price,
                    "atr": a,
                }
        except Exception as exc:
            print(ticker, "score error", exc)
    if best is None:
        return None, scanned, "no_candidates"
    return best, scanned, None


def try_entry(cfg, state, today, sig, best, scanned, fail_reason):
    if best is None:
        if fail_reason == "model_load_failed":
            send(f"❌ {cfg['label']}｜シグナル成立もAIモデル読込失敗のためエントリー見送り")
        else:
            send(f"⚠️ {cfg['label']}｜シグナル成立({sig['pct_rank']:.1f}%タイル)もスコアリング可能な銘柄なし(対象{scanned}銘柄)")
        return None

    price, a = best["price"], best["atr"]
    tp = price + a * cfg["tp_mult"]
    sl = max(0.01, price - a * cfg["sl_mult"])
    capital = float(state["capital"])
    shares = (int(capital // price) // LOT_SIZE) * LOT_SIZE
    if shares < LOT_SIZE:
        if capital >= price * LOT_SIZE:
            shares = LOT_SIZE
        else:
            send(f"⚠️ {cfg['label']}｜シグナル成立もエントリー資金不足(必要 {price*LOT_SIZE:,.0f}円 > 残高 {capital:,.0f}円)")
            return None
    invested = shares * price
    state["position"] = {
        "ticker": best["ticker"],
        "company": best["company"],
        "entry_date": today,
        "entry_price": price,
        "shares": shares,
        "invested_amount": invested,
        "tp": tp,
        "sl": sl,
        "atr": a,
        "score": best["score"],
        "up_probability": best["up_probability"],
        "down_probability": best["down_probability"],
        "nikkei_macd_at_entry": sig["today_macd"],
        "nikkei_macd_percentile_at_entry": sig["pct_rank"],
    }
    state["trades_total"] = int(state.get("trades_total", 0)) + 1
    return (
        f"🆕 {cfg['label']}｜新規エントリー\n"
        f"📅 {today}\n"
        f"日経MACD: {sig['today_macd']:.3f}(直近1年{sig['pct_rank']:.1f}%タイル、下位{int(PERCENTILE*100)}%到達)\n"
        f"BUY｜{best['ticker']} {best['company']}\n"
        f"スコア: {best['score']:.1f}｜上昇確率 {best['up_probability']:.1f}%\n"
        f"エントリー: {price:,.1f}円｜{shares:,}株｜投資額 {invested:,.0f}円\n"
        f"TP: {tp:,.1f}｜SL: {sl:,.1f}｜最大保有{HOLD_DAYS}営業日\n"
        f"対象{scanned}銘柄からTOP1選定"
    )


def _run():
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    nikkei = make_nikkei()
    if nikkei is None:
        send(f"❌ {BASE_LABEL}｜日経225データ取得失敗(ネットワーク不通の可能性)")
        return

    sig = nikkei_macd_signal(nikkei)
    if sig is None:
        send(f"⚠️ {BASE_LABEL}｜日経データ不足({len(nikkei)}件<{LOOKBACK_WINDOW})、シグナル判定不可")
        return

    thr_log = f"{sig['today_thr']:.4f}" if sig["today_thr"] is not None else "N/A"
    print(f"日経MACD: {sig['today_macd']:.4f}｜下位{int(PERCENTILE*100)}%しきい値: {thr_log}")
    print(f"直近1年パーセンタイル: {sig['pct_rank']:.2f}%｜シグナル成立: {sig['signal']}")

    states = {cfg["name"]: load_state(cfg) for cfg in CONFIGS}

    closed_msgs = {}
    for cfg in CONFIGS:
        state = states[cfg["name"]]
        if state.get("position"):
            closed_msgs[cfg["name"]] = close_position(cfg, state)

    need_entry = [cfg for cfg in CONFIGS if not states[cfg["name"]].get("position") and sig["signal"]]

    best = scanned = fail_reason = None
    if need_entry:
        best, scanned, fail_reason = find_best_candidate(nikkei)

    entry_msgs = {}
    for cfg in need_entry:
        entry_msgs[cfg["name"]] = try_entry(cfg, states[cfg["name"]], today, sig, best, scanned, fail_reason)

    for cfg in CONFIGS:
        save_state(cfg, states[cfg["name"]])

    for cfg in CONFIGS:
        name = cfg["name"]
        if closed_msgs.get(name):
            send(closed_msgs[name])
        if entry_msgs.get(name):
            send(entry_msgs[name])

    thr_str = f"{sig['today_thr']:.3f}" if sig["today_thr"] is not None else "N/A"
    lines = [
        f"{BASE_LABEL}｜日次判定(3パターン並列)\n"
        f"📅 {today}(日経データ日付: {sig['nikkei_date']})\n"
        f"日経MACD: {sig['today_macd']:.3f}｜直近1年{sig['pct_rank']:.1f}%タイル｜"
        f"下位{int(PERCENTILE*100)}%しきい値: {thr_str}\n"
        f"シグナル: {'成立(下位' + str(int(PERCENTILE*100)) + '%)' if sig['signal'] else '不成立'}"
    ]
    total_capital = 0.0
    total_initial = 0.0
    for cfg in CONFIGS:
        state = states[cfg["name"]]
        position = state.get("position")
        if position:
            pos_desc = (
                f"保有中: {position['ticker']} {position['company']}｜"
                f"エントリー {position['entry_price']:,.1f}(={position['entry_date']})｜"
                f"TP {position['tp']:,.1f}｜SL {position['sl']:,.1f}"
            )
        else:
            pos_desc = "保有ポジションなし"
        total_capital += float(state["capital"])
        total_initial += float(cfg["initial_capital"])
        lines.append(
            f"\n[{cfg['name']} TP{_fmt(cfg['tp_mult'])}/SL{_fmt(cfg['sl_mult'])}] {pos_desc}\n"
            f"💰 {state['capital']:,.0f}円(開始{cfg['initial_capital']:,.0f}円から "
            f"{state['capital']-cfg['initial_capital']:+,.0f}円)｜"
            f"最大DD {state.get('max_dd',0):.2f}%｜累計取引 {state.get('trades_total',0)}件"
        )
    lines.append(
        f"\n💰 3パターン合計: {total_capital:,.0f}円(開始{total_initial:,.0f}円から "
        f"{total_capital-total_initial:+,.0f}円)\n"
        f"⚠️ 実注文なし・本番システム(profit_top10_paper.py等)とは完全に独立した実験トラック"
    )
    send("\n".join(lines))


def main():
    try:
        _run()
    except Exception as exc:
        try:
            send(f"❌ {BASE_LABEL}｜実行エラー: {exc}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
