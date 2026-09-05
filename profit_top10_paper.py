import hashlib, hmac, json, os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from daily_directional_top1 import TICKERS, NAMES, download, make_nikkei, load_model, features, atr, directional_score
import paper_risk_policy

TZ=ZoneInfo('Asia/Tokyo'); POLICY_FILE='strategy_policy.json'; STATE_FILE='profit_top10_paper_state.json'; HISTORY_FILE='profit_top10_paper_history.csv'; MONTHLY_FILE='profit_top10_monthly_performance.csv'
INITIAL_CAPITAL=float(os.getenv('AI_INITIAL_CAPITAL','1000000')); TOP_N=10; MAX_TRADES_PER_TICKER_PER_DAY=10; MAX_TOTAL_TRADES_PER_DAY=30; FEE_RATE=float(os.getenv('INTRADAY_FEE_RATE','0.00055')); FORCED_EXIT=dtime(15,25); SHORT_ENABLED=os.getenv('ENABLE_SHORT_PAPER','1').lower() in ('1','true','yes','on')

def discord_send(message,required=False):
    webhook=os.getenv('DISCORD_WEBHOOK','').strip()
    if not webhook:
        print('❌ DISCORD_WEBHOOK 未設定'); return False
    try:
        import requests; r=requests.post(webhook,json={'content':message[:1950]},timeout=30); r.raise_for_status(); return True
    except Exception as e:
        print(f'❌ Discord通知失敗: {e}')
        if required: raise
        return False

def _canonical(p):
    keys=('status','updated_at','up_threshold','min_score_for_buy','nikkei_filter','atr_tp_multiplier','atr_sl_multiplier','hold_days','validation_signals','validation_avg_month_return','validation_avg_return','validation_pf','validation_dd','oos_signals','oos_avg_month_return','oos_monthly_plus5_ratio','oos_compound_return','oos_avg_return','oos_pf','oos_dd','oos_validation_pf_ratio','mc_sizing','mc_10y_probability','mc_15y_probability','mc_20y_probability','mc_bankruptcy_probability','mc_p90_max_dd','strategy_name','source')
    return json.dumps({k:p.get(k) for k in keys},ensure_ascii=False,sort_keys=True,separators=(',',':'))

def load_policy():
    with open(POLICY_FILE,encoding='utf-8') as f:p=json.load(f)
    req=['status','up_threshold','min_score_for_buy','nikkei_filter','atr_tp_multiplier','atr_sl_multiplier','hold_days']; miss=[k for k in req if k not in p]
    if miss: raise RuntimeError('policy不足: '+','.join(miss))
    status=str(p['status']).upper()
    if status=='APPROVED':
        secret=os.getenv('AI_POLICY_SIGNING_SECRET','').strip()
        if not secret or p.get('source')!='adversarial_strategy_validator' or int(p.get('approval_signature_version',0))!=1: raise RuntimeError('承認済みpolicy検証失敗')
        exp=hmac.new(secret.encode(),_canonical(p).encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(p.get('approval_signature','')),exp): raise RuntimeError('policy署名不一致')
    elif not(status=='PENDING' and os.getenv('PAPER_TRADE_MODE','1')=='1'): raise RuntimeError(f'policy={status} 新規取引停止')
    p['up_threshold']=float(p['up_threshold']); p['min_score_for_buy']=float(p['min_score_for_buy']); p['nikkei_filter']=str(p['nikkei_filter']).lower() in ('true','1','yes','on'); p['atr_tp_multiplier']=float(p['atr_tp_multiplier']); p['atr_sl_multiplier']=float(p['atr_sl_multiplier']); p['hold_days']=int(p['hold_days']); return p

def default_state(): return {'capital':INITIAL_CAPITAL,'peak':INITIAL_CAPITAL,'max_dd':0.0,'positions':[],'trade_count_date':None,'trades_today':0,'trades_by_ticker_today':{},'daily_start_capital':INITIAL_CAPITAL}
def load_state():
    s=default_state()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,encoding='utf-8') as f:s.update(json.load(f))
        except Exception: pass
    s.setdefault('positions',[]); s.setdefault('trades_by_ticker_today',{}); return s
def save_state(s):
    tmp=STATE_FILE+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f: json.dump(s,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,STATE_FILE)
def reset_daily(s,today):
    if s.get('trade_count_date')!=today:s.update({'trade_count_date':today,'trades_today':0,'trades_by_ticker_today':{},'daily_start_capital':float(s.get('capital',INITIAL_CAPITAL))})

def append_history(row):
    df=pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        try: df=pd.concat([pd.read_csv(HISTORY_FILE),df],ignore_index=True)
        except Exception: pass
    df.to_csv(HISTORY_FILE,index=False,encoding='utf-8-sig')

def download_5m(t):
    try:
        d=yf.download(t,period='5d',interval='5m',auto_adjust=False,progress=False,threads=False)
        if d is None or d.empty:return None
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        idx=pd.to_datetime(d.index); idx=idx.tz_convert(TZ).tz_localize(None) if getattr(idx,'tz',None) is not None else idx.tz_localize('UTC').tz_convert(TZ).tz_localize(None); d.index=idx; return d.sort_index()
    except Exception:return None

def trade_reasons(last,up,down,nikkei):
    p=float(last.get('Close',np.nan)); ma25=float(last.get('MA25',last.get('ma25',np.nan))); ma75=float(last.get('MA75',last.get('ma75',np.nan))); rsi=float(last.get('RSI',last.get('rsi',np.nan))); macd=float(last.get('MACD',last.get('macd',np.nan))); vr=float(last.get('vol_ratio',np.nan))
    trend='上昇トレンド' if np.isfinite(p) and np.isfinite(ma25) and np.isfinite(ma75) and p>ma25>ma75 else ('下落トレンド' if np.isfinite(p) and np.isfinite(ma25) and np.isfinite(ma75) and p<ma25<ma75 else 'トレンド中立')
    vals=[]
    if np.isfinite(vr): vals.append(f'出来高{vr:.1f}倍')
    vals += [trend, f'25日線{"上" if np.isfinite(ma25) and p>ma25 else "下"}', f'75日線{"上" if np.isfinite(ma75) and p>ma75 else "下"}', f'MACD{("買い" if macd>0 else "弱い") if np.isfinite(macd) else "判定不可"}', f'RSI{rsi:.1f}' if np.isfinite(rsi) else 'RSI判定不可', f'AI上昇{up:.1f}%/下落{down:.1f}%']
    if nikkei is not None:
        try: vals.append(f'日経{("上昇" if float(nikkei.get("ret5",0))>0 else "下落")}')
        except Exception: pass
    return '｜'.join(vals)

def scan(policy):
    nik,model=make_nikkei(),load_model(); cand=[]; fallback=[]; scanned=0
    if model is None or nik is None: raise RuntimeError('日経またはAIモデル取得失敗')
    cols=list(getattr(model,'feature_names_in_',[]))
    for t in TICKERS:
        d=download(t)
        if d is None or len(d)<150: continue
        scanned+=1; x=features(d,nik).dropna(subset=cols)
        if x.empty: continue
        try:
            last=x.iloc[-1]; pr=model.predict_proba(x.iloc[-1:])[0]; cl=list(model.classes_)
            if not all(c in cl for c in (0,1,2)): continue
            down,up,flat=float(pr[cl.index(0)])*100,float(pr[cl.index(2)])*100,float(pr[cl.index(1)])*100
            ls,ss=directional_score(last,up/100,down/100); a=float(atr(d).iloc[-1])
            if not np.isfinite(a) or a<=0:continue
            intr=download_5m(t); price=float(d['Close'].iloc[-1])
            if intr is not None and not intr.empty: price=float(intr['Close'].iloc[-1])
            nlast=nik.reindex(x.index).ffill().iloc[-1]
            reason=trade_reasons(last,up,down,nlast)
            base={'ticker':t,'company':NAMES.get(t,t),'price':price,'up_probability':up,'down_probability':down,'flat_probability':flat,'data_date':str(x.index[-1].date())}
            for direction,score in [('BUY',float(ls)),('SHORT',float(ss))]:
                item=dict(base); item.update(direction=direction,score=score,tp=price+(a*policy['atr_tp_multiplier'] if direction=='BUY' else -a*policy['atr_tp_multiplier']),sl=max(.01,price-a*policy['atr_sl_multiplier']) if direction=='BUY' else price+a*policy['atr_sl_multiplier'],buy_reason=reason)
                if direction=='BUY':
                    reward_pct=(item['tp']-price)/price*100 if price>0 else 0.0; risk_pct=(price-item['sl'])/price*100 if price>0 else 0.0; win_prob=up/100; loss_prob=down/100
                else:
                    reward_pct=(price-item['tp'])/price*100 if price>0 else 0.0; risk_pct=(item['sl']-price)/price*100 if price>0 else 0.0; win_prob=down/100; loss_prob=up/100
                item['expected_value_pct']=win_prob*reward_pct-loss_prob*max(risk_pct,0.0)
                fallback.append(item)
                ok=(up>=policy['up_threshold'] and up>down and flat<50 and score>=policy['min_score_for_buy']) if direction=='BUY' else (SHORT_ENABLED and down>=policy['up_threshold'] and down>up and flat<50 and score>=policy['min_score_for_buy'])
                if ok:cand.append(item)
        except Exception as e: print(t,e)
    cand.sort(key=lambda z:(z['expected_value_pct'],z['score']),reverse=True); fallback.sort(key=lambda z:(z['expected_value_pct'],z['score']),reverse=True)
    return (cand or fallback)[:TOP_N],scanned

def open_positions(s,policy,cands,today):
    active={p['ticker'] for p in s['positions']}; out=[]
    for c in cands:
        if c['ticker'] in active:continue
        allowed,reason=paper_risk_policy.position_allowed(s,c['ticker'])
        if not allowed:
            if reason.startswith('日次損失上限') or reason.startswith('最大DD'):
                print(f'⛔ リスクポリシーにより新規エントリー全停止: {reason}'); break
            continue
        cnt=int(s.get('trades_by_ticker_today',{}).get(c['ticker'],0))
        if cnt>=MAX_TRADES_PER_TICKER_PER_DAY or int(s.get('trades_today',0))>=MAX_TOTAL_TRADES_PER_DAY: continue
        budget=float(s['capital'])/TOP_N; price=float(c['price']); shares=int(budget//price) if price>0 else 0
        if shares<=0:continue
        invested=shares*price
        s['positions'].append({**c,'entry_date':today,'entry_time':datetime.now(TZ).strftime('%H:%M'),'entry_price':price,'shares':shares,'invested_amount':invested,'allocation':1/TOP_N,'policy_updated_at':policy.get('updated_at'),'current_price':price,'unrealized_pnl':0.0})
        s['trades_today']=int(s.get('trades_today',0))+1; s.setdefault('trades_by_ticker_today',{})[c['ticker']]=cnt+1; active.add(c['ticker']); out.append(c)
    return out

def mark_and_close(s,now):
    remaining=[]; msgs=[]
    for p in s['positions']:
        d=download_5m(p['ticker'])
        if d is None or d.empty:remaining.append(p);continue
        bars=d[d.index.date==now.date()]; last=float(bars['Close'].iloc[-1]) if not bars.empty else float(p['entry_price']); p['current_price']=last
        ep=float(p['entry_price']); sh=int(p.get('shares',0)); direction=p.get('direction','BUY'); unreal=((last-ep)*sh if direction=='BUY' else (ep-last)*sh)-ep*sh*FEE_RATE-last*sh*FEE_RATE; p['unrealized_pnl']=unreal
        exit_price=reason=None
        for ts,b in bars.iterrows():
            if ts.time()<dtime(9,0):continue
            hi,lo=float(b['High']),float(b['Low'])
            if direction=='BUY':
                if lo<=p['sl']:exit_price,reason=float(p['sl']),'SL'
                elif hi>=p['tp']:exit_price,reason=float(p['tp']),'TP'
            else:
                if hi>=p['sl']:exit_price,reason=float(p['sl']),'SL'
                elif lo<=p['tp']:exit_price,reason=float(p['tp']),'TP'
            if reason or ts.time()>=FORCED_EXIT:
                if not reason:exit_price,reason=float(b['Close']),'EOD'
                break
        if exit_price is None:remaining.append(p);continue
        gross=(exit_price-ep)*sh if direction=='BUY' else (ep-exit_price)*sh; pnl=gross-(ep+exit_price)*sh*FEE_RATE; s['capital']=float(s['capital'])+pnl; exit_value=exit_price*sh; total=s['capital']
        append_history({'entry_date':p['entry_date'],'entry_time':p['entry_time'],'exit_date':str(now.date()),'exit_time':now.strftime('%H:%M'),'ticker':p['ticker'],'company':p['company'],'direction':direction,'entry_price':ep,'exit_price':exit_price,'shares':sh,'invested_amount':p['invested_amount'],'exit_value':exit_value,'tp':p['tp'],'sl':p['sl'],'score':p['score'],'up_probability':p['up_probability'],'down_probability':p['down_probability'],'expected_value_pct':p.get('expected_value_pct',0),'return_pct':pnl/p['invested_amount']*100 if p['invested_amount'] else 0,'pnl':pnl,'result':reason,'total_assets':total,'buy_reason':p.get('buy_reason','')})
        msgs.append(f"{'🟢' if pnl>=0 else '🔴'} 決済｜{p['company']}（{p['ticker']}）｜{direction}\n決済価格 {exit_price:,.1f}円｜{sh:,}株｜投資額 {p['invested_amount']:,.0f}円\n確定損益 {pnl:+,.0f}円｜💰総資産 {total:,.0f}円｜開始100万円から {total-INITIAL_CAPITAL:+,.0f}円")
    s['positions']=remaining; s['peak']=max(float(s.get('peak',s['capital'])),float(s['capital'])); return msgs

def main():
    now=datetime.now(TZ); today=now.strftime('%Y-%m-%d'); policy=load_policy(); s=load_state(); reset_daily(s,today)
    if not(now.weekday()<5 and dtime(9,0)<=now.time()<=dtime(15,30)):
        discord_send(f'🤖 PROFIT LOOP｜待機\n{today} {now:%H:%M} JST\n市場時間外｜実注文なし'); return
    closed=mark_and_close(s,now); cands,scanned=scan(policy); opened=open_positions(s,policy,cands,today); save_state(s)
    equity=float(s['capital'])+sum(float(p.get('unrealized_pnl',0)) for p in s['positions']); daily=equity-float(s.get('daily_start_capital',INITIAL_CAPITAL)); cum=(equity/INITIAL_CAPITAL-1)*100
    rows=[]
    for i,p in enumerate(s['positions'],1):rows.append(f"{i}. {'買い' if p['direction']=='BUY' else '空売り'} {p['company']}（{p['ticker']}）\n   {p['shares']:,}株｜投資額 {p['invested_amount']:,.0f}円｜取得 {p['entry_price']:,.1f}円｜現在値 {p['current_price']:,.1f}円｜含み損益 {p['unrealized_pnl']:+,.0f}円\n   利確 {p['tp']:,.1f}｜損切 {p['sl']:,.1f}｜期待値 {p.get('expected_value_pct',0):+.2f}%\n   🧠 買った基準: {p.get('buy_reason','')}")
    msg=('🤖 利益優先ループ｜TOP10 ペーパートレード\n━━━━━━━━━━━━━━━━━━\n'
         f'📅 {today} {now:%H:%M} JST｜⚠️ 実注文なし\n対象430銘柄｜取得成功 {scanned}｜候補 {len(cands)}｜新規 {len(opened)}件\n'
         f'条件: 確率≥{policy["up_threshold"]:.0f}%｜AIスコア≥{policy["min_score_for_buy"]:.0f}｜TP×{policy["atr_tp_multiplier"]:.1f}｜SL×{policy["atr_sl_multiplier"]:.1f}\n'
         f'💰総資産 {equity:,.0f}円｜本日 {daily:+,.0f}円｜累計 {cum:+.2f}%\n'
         f'📦 保有 {len(s["positions"])}件\n' + ('\n'.join(rows) if rows else 'なし'))
    for m in closed: discord_send(m)
    discord_send(msg)

if __name__=='__main__': main()
