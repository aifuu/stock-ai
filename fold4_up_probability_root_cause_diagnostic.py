import pandas as pd
import numpy as np
from pathlib import Path

P = Path('walk_forward_all_candidates.csv')
OUT = Path('fold4_up_probability_root_cause.csv')
DETAIL = Path('fold4_up_probability_root_cause_detail.csv')

if not P.exists():
    raise SystemExit('❌ walk_forward_all_candidates.csv がありません')

df = pd.read_csv(P)
if 'date' not in df.columns:
    raise SystemExit('❌ date列がありません')
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
df = df.dropna(subset=['date']).copy()

dates = pd.DatetimeIndex(sorted(df['date'].drop_duplicates()))
if len(dates) < 239:
    raise SystemExit(f'❌ unique dates不足: {len(dates)} < 239')
oos_dates = dates[-239:]
oos = df[df['date'].isin(oos_dates)].copy()

prob_col = next((c for c in ['prob_up','up_probability','probability_up','up_prob','p_up','up_prob_pct','prob'] if c in oos.columns), None)
if prob_col is None:
    raise SystemExit('❌ UP確率列がありません')

oos['up_prob_pct'] = pd.to_numeric(oos[prob_col], errors='coerce')
if oos['up_prob_pct'].dropna().max() <= 1:
    oos['up_prob_pct'] *= 100

for c in ['score','testa_score','rsi','vol','return','result','nikkei_uptrend']:
    if c in oos.columns:
        if c != 'nikkei_uptrend':
            oos[c] = pd.to_numeric(oos[c], errors='coerce')

print('='*80)
print('🔬 FOLD 4 OOS UP-PROBABILITY ROOT-CAUSE DIAGNOSTIC')
print('目的: 8/19・8/20の候補激減がモデル確率低下なのか、入力特徴量・スコア・データ欠落なのかを分離')
print('重要: ゲート条件・探索閾値・policyは変更しない。診断のみ。')
print('='*80)
print(f'OOS: {oos_dates[0].date()} -> {oos_dates[-1].date()} / rows={len(oos):,} / tickers={oos.ticker.nunique() if "ticker" in oos else "NA"}')

# 日別のモデル確率と利用可能な説明変数の急変を測定
agg_cols = {'up_prob_pct':['count','mean','median','min','max']}
for c in ['score','testa_score','rsi','vol']:
    if c in oos.columns:
        agg_cols[c] = ['mean','median','min','max']
daily = oos.groupby('date').agg(agg_cols)
daily.columns = ['_'.join(x).strip('_') for x in daily.columns]
daily = daily.reset_index()

# 日別変化率/変化量。特に8/18->8/19->8/20を可視化
for c in ['up_prob_pct','score','testa_score','rsi','vol']:
    mean_col = f'{c}_mean'
    if mean_col in daily:
        daily[f'{c}_delta'] = daily[mean_col].diff()

print('\n--- 日別急落検出 TOP 15 ---')
if 'up_prob_pct_delta' in daily:
    print(daily.nsmallest(15,'up_prob_pct_delta').to_string(index=False))

for d in [pd.Timestamp('2026-08-18'),pd.Timestamp('2026-08-19'),pd.Timestamp('2026-08-20')]:
    z=oos[oos['date']==d].copy()
    print(f'\n--- {d.date()} ROOT-CAUSE SNAPSHOT rows={len(z)} ---')
    if len(z):
        cols=[c for c in ['date','ticker','company','up_prob_pct','score','testa_score','rsi','vol','nikkei_uptrend','price'] if c in z.columns]
        print(z.sort_values('up_prob_pct',ascending=False)[cols].head(20).to_string(index=False))
        for c in ['score','testa_score','rsi','vol']:
            if c in z.columns:
                s=z[c].dropna()
                if len(s): print(f'{c}: min={s.min():.4f} median={s.median():.4f} max={s.max():.4f}')

# 各特徴量とup_probの横断相関。因果ではなく診断上の関連度。
print('\n--- CROSS-SECTIONAL CORRELATION WITH UP_PROB ---')
for c in ['score','testa_score','rsi','vol']:
    if c in oos.columns:
        z=oos[['up_prob_pct',c]].dropna()
        print(f'{c:12s}: corr={z.up_prob_pct.corr(z[c]): .4f} n={len(z):,}')

# 8/19,8/20で全銘柄が存在しない問題も明示
print('\n--- COVERAGE / DATA DENSITY ---')
for d in [pd.Timestamp('2026-08-14'),pd.Timestamp('2026-08-15'),pd.Timestamp('2026-08-18'),pd.Timestamp('2026-08-19'),pd.Timestamp('2026-08-20')]:
    z=oos[oos.date==d]
    print(f'{d.date()}: rows={len(z)} unique_tickers={z.ticker.nunique() if "ticker" in z else "NA"}')

# 重要: 239日全体で確率帯別の実績を集計。モデル確率が低いだけなのかも検証。
if 'return' in oos.columns:
    oos['prob_band'] = pd.cut(oos['up_prob_pct'], bins=[-np.inf,20,25,30,35,40,42.5,45,np.inf], right=False)
    perf=oos.groupby('prob_band',observed=False).agg(rows=('up_prob_pct','size'), avg_return=('return','mean'), positive=('return',lambda s:int((s>0).sum())))
    perf['positive_rate_pct']=100*perf['positive']/perf['rows'].replace(0,np.nan)
    print('\n--- PROBABILITY BAND REALIZED RETURN ---')
    print(perf.to_string())
else:
    perf=pd.DataFrame()

# 詳細CSV: 原データ + 各行の原因候補フラグ
if 'ticker' in oos.columns:
    oos['ticker_for_diag']=oos['ticker'].astype(str)
else:
    oos['ticker_for_diag']='ROW'
for t in [20,25,30,35,40,42.5,45,50,55,60]:
    oos[f'up_ge_{str(t).replace(".","_")}'] = oos.up_prob_pct >= t

detail_cols=[c for c in ['date','ticker_for_diag','company','up_prob_pct','score','testa_score','rsi','vol','nikkei_uptrend','price','return','result'] if c in oos.columns]
oos[detail_cols].sort_values(['date','up_prob_pct'],ascending=[True,False]).to_csv(DETAIL,index=False,encoding='utf-8-sig')
daily.to_csv(OUT,index=False,encoding='utf-8-sig')
print(f'\n診断CSV: {OUT} / {len(daily)} rows')
print(f'詳細CSV: {DETAIL} / {len(oos)} rows')
print('\n⚠ 判定: この診断は原因候補を特定するものであり、閾値・policy・ゲートを変更しません。')
print('⚠ MACD/MA等がcandidate CSVに無い場合は「保存されていないため直接検証不能」と明示します。')
