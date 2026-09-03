import os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

import profit_top10_paper as pt
from daily_directional_top1 import TICKERS, NAMES, make_nikkei, load_model, features, atr, directional_score

TZ = ZoneInfo('Asia/Tokyo')
AUDIT_FILE = 'selection_audit.csv'
TOP50 = 50
TOP10 = 10


def finite(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def component_scores(row, up, down):
    r = finite(row.get('rsi', np.nan), 50)
    macd = finite(row.get('macd', np.nan))
    sig = finite(row.get('signal', np.nan))
    ma25 = finite(row.get('ma25', np.nan))
    ma75 = finite(row.get('ma75', np.nan))
    vol = finite(row.get('vol_ratio', np.nan))
    low = finite(row.get('from_low', np.nan))
    high = finite(row.get('from_high', np.nan))
    momentum = finite(row.get('momentum_score', np.nan))

    rsi_pts = 25 if r < 35 else 0
    macd_pts = 25 if macd > sig else 0
    ma_pts = 20 if ma25 > ma75 else 0
    vol_pts = 20 if vol > 1.5 else 0
    range_pts = 15 if high > -10 else (8 if high > -20 else 0)
    tech_raw = rsi_pts + macd_pts + ma_pts + vol_pts + range_pts
    tech_component = tech_raw / 105 * 100 * 0.525
    prob_component = up * 0.225
    momentum_component = momentum * 0.25
    total = tech_component + prob_component + momentum_component
    return {
        'rsi_points': rsi_pts,
        'macd_points': macd_pts,
        'ma_points': ma_pts,
        'volume_points': vol_pts,
        'range_points': range_pts,
        'technical_component': tech_component,
        'probability_component': prob_component,
        'momentum_component': momentum_component,
        'total_score_recalc': total,
    }


def run():
    now = datetime.now(TZ)
    today = now.strftime('%Y-%m-%d')
    policy = pt.load_policy()
    nikkei = make_nikkei()
    model = load_model()
    if nikkei is None or model is None:
        raise RuntimeError('日経またはAIモデル取得失敗')
    cols = list(getattr(model, 'feature_names_in_', []))
    rows = []
    scanned = 0

    # 430銘柄を同じ runtime universe から取得し、まず出来高・トレンド等の一次候補を作る。
    for ticker in TICKERS:
        d = pt.download(ticker)
        if d is None or len(d) < 150:
            continue
        scanned += 1
        x = features(d, nikkei).dropna(subset=cols)
        if x.empty:
            continue
        try:
            last = x.iloc[-1]
            pr = model.predict_proba(x.iloc[-1:])[0]
            classes = list(model.classes_)
            if not all(c in classes for c in (0, 1, 2)):
                continue
            down = finite(pr[classes.index(0)]) * 100
            flat = finite(pr[classes.index(1)]) * 100
            up = finite(pr[classes.index(2)]) * 100
            price = finite(d['Close'].iloc[-1])
            volume_ratio = finite(last.get('vol_ratio', np.nan))
            trend_alignment = finite(last.get('trend_alignment', np.nan))
            momentum = finite(last.get('momentum_score', np.nan))
            preliminary = (
                min(volume_ratio / 2.0, 1.0) * 35
                + trend_alignment / 3.0 * 35
                + momentum / 100.0 * 30
            )
            long_score, short_score = directional_score(last, up / 100, down / 100)
            comps = component_scores(last, up, down)
            rows.append({
                'date': today,
                'ticker': ticker,
                'company': NAMES.get(ticker, ticker),
                'price': price,
                'volume_ratio': volume_ratio,
                'trend_alignment': trend_alignment,
                'momentum_score': momentum,
                'preliminary_score': preliminary,
                'up_probability': up,
                'down_probability': down,
                'flat_probability': flat,
                'directional_score': finite(long_score),
                **comps,
            })
        except Exception as exc:
            print(f'{ticker}: {exc}')

    if not rows:
        raise RuntimeError('選定候補が0件')

    df = pd.DataFrame(rows)
    # 第1段階: 出来高・トレンド・モメンタムでTOP50
    df = df.sort_values(['preliminary_score', 'volume_ratio', 'trend_alignment'], ascending=False).reset_index(drop=True)
    df['stage1_rank'] = np.arange(1, len(df) + 1)
    top50 = df.head(TOP50).copy()

    # 第2段階: AI確率・テクニカル・モメンタムを含む総合スコアでTOP10
    top50['final_score'] = top50['total_score_recalc']
    top50 = top50.sort_values(['final_score', 'up_probability', 'directional_score'], ascending=False).reset_index(drop=True)
    top50['final_rank'] = np.arange(1, len(top50) + 1)
    top10 = top50.head(TOP10).copy()
    top1 = top10.iloc[0]

    # 1位との差を全項目について保存。これで「なぜ4812.Tが1位か」を毎日再現できる。
    for i, r in top10.iterrows():
        out = r.to_dict()
        out['rank'] = int(i + 1)
        out['gap_vs_top1_score'] = finite(top1['final_score']) - finite(r['final_score'])
        for col in ['up_probability','down_probability','flat_probability','preliminary_score','directional_score','momentum_score','volume_ratio','technical_component','probability_component','momentum_component']:
            out[f'gap_vs_top1_{col}'] = finite(top1[col]) - finite(r[col])
        rows_out = out
        # append below
        if i == 0:
            result = []
        result.append(rows_out)

    audit = pd.DataFrame(result)
    audit.insert(0, 'generated_at', now.isoformat())
    audit.to_csv(AUDIT_FILE, index=False, encoding='utf-8-sig')

    msg = [
        '🔎 TOP1選定監査',
        f'📅 {today} {now:%H:%M} JST',
        f'対象ユニバース: {len(TICKERS)}銘柄｜取得成功: {scanned}｜一次候補: {len(df)}',
        '1次: 出来高・トレンド・モメンタム → TOP50',
        '2次: AI上昇確率・テクニカル・モメンタム → TOP10',
        f'🥇 最終TOP1: {top1["ticker"]} {top1["company"]}｜総合 {top1["final_score"]:.2f}',
        '',
        '順位 | 銘柄 | AI上昇 | AI下落 | 出来高 | トレンド | テクニカル | 確率寄与 | モメンタム | 総合 | TOP1差',
    ]
    for _, r in top10.iterrows():
        msg.append(
            f'{int(r["final_rank"])} | {r["ticker"]} {r["company"]} | '
            f'{r["up_probability"]:.1f}% | {r["down_probability"]:.1f}% | '
            f'{r["volume_ratio"]:.2f}倍 | {r["trend_alignment"]:.0f}/3 | '
            f'{r["technical_component"]:.2f} | {r["probability_component"]:.2f} | '
            f'{r["momentum_component"]:.2f} | {r["final_score"]:.2f} | '
            f'{finite(top1["final_score"])-finite(r["final_score"]):+.2f}'
        )
    msg += [
        '',
        '📌 TOP1の根拠:',
        f'AI上昇確率 {top1["up_probability"]:.1f}%｜出来高 {top1["volume_ratio"]:.2f}倍｜トレンド {top1["trend_alignment"]:.0f}/3｜モメンタム {top1["momentum_score"]:.1f}',
        f'テクニカル寄与 {top1["technical_component"]:.2f}｜確率寄与 {top1["probability_component"]:.2f}｜モメンタム寄与 {top1["momentum_component"]:.2f}',
        '📄 詳細: selection_audit.csv',
    ]
    text = '\n'.join(msg)
    if os.getenv('DISCORD_WEBHOOK'):
        pt.discord_send(text)
    print(text)


if __name__ == '__main__':
    run()
