import pandas as pd
import numpy as np
from pathlib import Path

P = Path('walk_forward_all_candidates.csv')
OUT = Path('fold4_up_probability_diagnostic.csv')
DETAIL = Path('fold4_oos_prob_up_detail.csv')

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

prob_candidates = ['prob_up','up_probability','probability_up','up_prob','p_up','up_prob_pct','prob']
prob_col = next((c for c in prob_candidates if c in oos.columns), None)
print('============================================================')
print('🔎 FOLD 4 OOS COMPLETE UP-PROBABILITY DIAGNOSTIC')
print('ゲート条件・探索閾値・policyは変更しない。診断のみ。')
print('============================================================')
print(f'OOS期間: {oos_dates[0].date()} -> {oos_dates[-1].date()} ({len(oos_dates)}営業日)')
print(f'RAW_OOS: {len(oos):,}')
print('利用可能列:', ', '.join(oos.columns.tolist()))
if prob_col is None:
    print('❌ UP確率列を候補CSVから特定できません。推論元スクリプト側でprob_upを保存する必要があります。')
    raise SystemExit(2)

x = pd.to_numeric(oos[prob_col], errors='coerce')
scale = '0-1 -> %' if x.notna().any() and x.max() <= 1.0 else '%'
oos['prob_up_raw'] = x
oos['prob_up_pct'] = x * 100.0 if scale == '0-1 -> %' else x
p = oos['prob_up_pct']

# 実際のUP45判定を明示的に保存し、確率値との整合性を検査。
oos['up_ge_45'] = p >= 45
oos['up_ge_50'] = p >= 50
for t in [20,25,30,35,40,42.5,45,50,55,60]:
    oos[f'up_ge_{str(t).replace(".", "_")}'] = p >= t

# 銘柄列を可能な範囲で特定。なければ行番号を一意キーにする。
ticker_col = next((c for c in ['ticker','symbol','code','Code','Ticker'] if c in oos.columns), None)
oos['ticker_for_diag'] = oos[ticker_col].astype(str) if ticker_col else 'ROW'
oos['diag_date'] = oos['date'].dt.strftime('%Y-%m-%d')
keep = ['diag_date','ticker_for_diag',prob_col,'prob_up_raw','prob_up_pct','up_ge_20','up_ge_25','up_ge_30','up_ge_35','up_ge_40','up_ge_42_5','up_ge_45','up_ge_50','up_ge_55','up_ge_60']
# 存在する列だけ出力
keep = [c for c in keep if c in oos.columns]
detail = oos[keep].sort_values(['diag_date','ticker_for_diag'])
detail.to_csv(DETAIL,index=False,encoding='utf-8-sig')

valid = p.dropna()
print(f'probability column: {prob_col} ({scale})')
print(f'valid probability rows: {len(valid):,} / {len(oos):,}')
if len(valid):
    print('--- UP probability distribution (%) ---')
    print(f'min={valid.min():.4f} max={valid.max():.4f} mean={valid.mean():.4f} median={valid.median():.4f}')
    for q,v in valid.quantile([.01,.05,.10,.25,.50,.75,.90,.95,.99]).items():
        print(f'p{int(q*100):02d}={v:.4f}')
    print('--- exact threshold counts ---')
    for t in [20,25,30,35,40,42.5,45,50,55,60]:
        n=int((valid>=t).sum())
        print(f'UP>={t:>4}: {n:>6} ({100*n/len(valid):6.2f}%)')

# 日別: 特に2026-08-19/20を明示。
daily=oos.groupby('date').agg(rows=('prob_up_pct','size'),valid_prob=('prob_up_pct',lambda s:int(s.notna().sum())),up20=('prob_up_pct',lambda s:int((s>=20).sum())),up25=('prob_up_pct',lambda s:int((s>=25).sum())),up30=('prob_up_pct',lambda s:int((s>=30).sum())),up35=('prob_up_pct',lambda s:int((s>=35).sum())),up40=('prob_up_pct',lambda s:int((s>=40).sum())),up42_5=('prob_up_pct',lambda s:int((s>=42.5).sum())),up45=('prob_up_pct',lambda s:int((s>=45).sum())),up50=('prob_up_pct',lambda s:int((s>=50).sum())),up55=('prob_up_pct',lambda s:int((s>=55).sum())),up60=('prob_up_pct',lambda s:int((s>=60).sum())),mean_up=('prob_up_pct','mean'),median_up=('prob_up_pct','median'),max_up=('prob_up_pct','max')).reset_index()
print('--- last 20 OOS dates ---')
print(daily.tail(20).to_string(index=False))
for d in [pd.Timestamp('2026-08-19'),pd.Timestamp('2026-08-20')]:
    z=oos[oos['date']==d]
    print(f'--- DETAIL {d.date()} ({len(z)} rows) ---')
    cols=[c for c in ['date',ticker_col,prob_col,'prob_up_pct','up_ge_45'] if c]
    print(z[cols].sort_values(prob_col,ascending=False).to_string(index=False) if len(z) else 'NO ROWS')

daily.to_csv(OUT,index=False,encoding='utf-8-sig')
print(f'診断CSV: {OUT} / {len(daily)} rows')
print(f'銘柄・日別詳細CSV: {DETAIL} / {len(detail)} rows')
