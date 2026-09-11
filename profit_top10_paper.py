import hashlib, hmac, json, os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from daily_directional_top1 import TICKERS, NAMES, download, make_nikkei, load_model, features, atr, directional_score
import paper_risk_policy
import futures_trend
import discord_progress

TZ=ZoneInfo('Asia/Tokyo'); POLICY_FILE='strategy_policy.json'; POLICY_FILE_UP='strategy_policy_up.json'; POLICY_FILE_DOWN='strategy_policy_down.json'; STATE_FILE='profit_top10_paper_state.json'; HISTORY_FILE='profit_top10_paper_history.csv'; MONTHLY_FILE='profit_top10_monthly_performance.csv'

def select_policy_file():
    """先物トレンド(案3)に応じて、その日使うpolicyファイルを選ぶ。

    上昇用/下落用の個別policyファイルが両方揃っていない間は、
    既存の単一strategy_policy.jsonにフォールバックする(安全側の既定動作)。
    判定結果は本番選定には使わず、futures_trend_history.csvへ記録するだけの
    週次専用戦略検討用データとしても蓄積される。
    """
    result = futures_trend.detect_futures_trend()
    try:
        futures_trend.log_daily_trend(result)
    except Exception as e:
        print(f'⚠️ futures_trend記録失敗: {e}')
    candidate = POLICY_FILE_DOWN if result.get('trend') == futures_trend.DOWN else POLICY_FILE_UP
    if os.path.exists(candidate):
        print(f'📈 先物トレンド判定: {result.get("trend")}｜{result.get("reason")}｜使用policy: {candidate}')
        return candidate, result
    print(f'📈 先物トレンド判定: {result.get("trend")}｜{result.get("reason")}｜{candidate}未整備のため{POLICY_FILE}を使用')
    return POLICY_FILE, result
INITIAL_CAPITAL=float(os.getenv('AI_INITIAL_CAPITAL','1000000')); TOP_N=10; MAX_TRADES_PER_TICKER_PER_DAY=10; MAX_TOTAL_TRADES_PER_DAY=30; FEE_RATE=float(os.getenv('INTRADAY_FEE_RATE','0.00055')); FORCED_EXIT=dtime(15,25); SHORT_ENABLED=os.getenv('ENABLE_SHORT_PAPER','1').lower() in ('1','true','yes','on'); LOT_SIZE=100

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

def load_policy(policy_file=None):
    path = policy_file or POLICY_FILE
    with open(path,encoding='utf-8') as f:p=json.load(f)
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

def _tp_sl(price,direction,a,policy):
    if direction=='BUY':
        return price+a*policy['atr_tp_multiplier'],max(.01,price-a*policy['atr_sl_multiplier'])
    return price-a*policy['atr_tp_multiplier'],price+a*policy['atr_sl_multiplier']

def _reward_risk(price,tp,sl,direction):
    if direction=='BUY':
        return ((tp-price)/price*100 if price>0 else 0.0),((price-sl)/price*100 if price>0 else 0.0)
    return ((price-tp)/price*100 if price>0 else 0.0),((sl-price)/price*100 if price>0 else 0.0)

def scan(policy):
    nik,model=make_nikkei(),load_model(); cand=[]; fallback=[]; scanned=0
    if model is None or nik is None: raise RuntimeError('日経またはAIモデル取得失敗')
    cols=list(getattr(model,'feature_names_in_',[])); nikkei_filter_on=bool(policy.get('nikkei_filter'))
    for t in TICKERS:
        d=download(t)
        if d is None or len(d)<150: continue
        scanned+=1; x=features(d,nik).dropna(subset=cols)
        if x.empty: continue
        try:
            last=x.iloc[-1]; pr=model.predict_proba(x[cols].iloc[-1:])[0]; cl=list(model.classes_)
            if not all(c in cl for c in (0,1,2)): continue
            down,up,flat=float(pr[cl.index(0)])*100,float(pr[cl.index(2)])*100,float(pr[cl.index(1)])*100
            ls,ss=directional_score(last,up/100,down/100); a=float(atr(d).iloc[-1])
            if not np.isfinite(a) or a<=0:continue
            # 5分足による価格精緻化は、フィルタ通過済みの候補にだけ後段でまとめて行う
            # (プレフィルター廃止で全225銘柄スキャンになったため、個別ネットワーク
            # 呼び出しが増えすぎないようにする)。
            price=float(d['Close'].iloc[-1])
            nlast=nik.reindex(x.index).ffill().iloc[-1]; nikkei_up=bool(nlast.get('nikkei_uptrend',True))
            reason=trade_reasons(last,up,down,nlast)
            base={'ticker':t,'company':NAMES.get(t,t),'price':price,'up_probability':up,'down_probability':down,'flat_probability':flat,'data_date':str(x.index[-1].date())}
            pending=[]
            for direction,score in [('BUY',float(ls)),('SHORT',float(ss))]:
                tp,sl=_tp_sl(price,direction,a,policy)
                item=dict(base); item.update(direction=direction,score=score,tp=tp,sl=sl,buy_reason=reason)
                reward_pct,risk_pct=_reward_risk(price,tp,sl,direction)
                win_prob,loss_prob=(up/100,down/100) if direction=='BUY' else (down/100,up/100)
                item['expected_value_pct']=win_prob*reward_pct-loss_prob*max(risk_pct,0.0)
                fallback.append(item)
                ok=(up>=policy['up_threshold'] and up>down and flat<50 and score>=policy['min_score_for_buy']) if direction=='BUY' else (SHORT_ENABLED and down>=policy['up_threshold'] and down>up and flat<50 and score>=policy['min_score_for_buy'])
                if ok and nikkei_filter_on and direction=='BUY' and not nikkei_up: ok=False
                if ok and nikkei_filter_on and direction=='SHORT' and nikkei_up: ok=False
                if ok: pending.append((item,win_prob,loss_prob))
            if pending:
                intr=download_5m(t); refined=price
                if intr is not None and not intr.empty: refined=float(intr['Close'].iloc[-1])
                for item,win_prob,loss_prob in pending:
                    if refined>0 and refined!=price:
                        direction=item['direction']
                        tp,sl=_tp_sl(refined,direction,a,policy)
                        reward_pct,risk_pct=_reward_risk(refined,tp,sl,direction)
                        item.update(price=refined,tp=tp,sl=sl,expected_value_pct=win_prob*reward_pct-loss_prob*max(risk_pct,0.0))
                    cand.append(item)
        except Exception as e: print(t,e)
    cand.sort(key=lambda z:(z['expected_value_pct'],z['score']),reverse=True); fallback.sort(key=lambda z:(z['expected_value_pct'],z['score']),reverse=True)
    return (cand or fallback)[:TOP_N],scanned

def open_positions(s,policy,cands,today):
    active={p['ticker'] for p in s['positions']}; out=[]
    capital=float(s['capital'])
    remaining=capital-sum(float(p.get('invested_amount',0)) for p in s['positions'])
    for c in cands:
        if c['ticker'] in active:continue
        allowed,reason=paper_risk_policy.position_allowed(s,c['ticker'])
        if not allowed:
            if reason.startswith('日次損失上限') or reason.startswith('最大DD'):
                print(f'⛔ リスクポリシーにより新規エントリー全停止: {reason}'); break
            continue
        cnt=int(s.get('trades_by_ticker_today',{}).get(c['ticker'],0))
        if cnt>=MAX_TRADES_PER_TICKER_PER_DAY or int(s.get('trades_today',0))>=MAX_TOTAL_TRADES_PER_DAY: continue
        price=float(c['price'])
        if price<=0:continue
        # ★変更(2026-09): 資産をTOP_Nで分割せず、毎回その時点の総資産を丸ごと
        # 予算として使う(ユーザー承認済みのハイリスク運用)。これにより実質的に
        # 1銘柄集中投資となる(最初に処理される最有力候補が残り現金のほぼ全額を
        # 使うため、同一スキャン内で複数銘柄に新規エントリーすることは通常ない)。
        # 1単元(100株)すら買えない銘柄は、残り資金があれば1単元だけ買う。
        slot_budget=capital
        shares=(int(slot_budget//price)//LOT_SIZE)*LOT_SIZE
        if shares<LOT_SIZE:
            shares=LOT_SIZE if remaining>=price*LOT_SIZE else 0
        if shares<=0:continue
        invested=shares*price
        if invested>remaining:continue
        remaining-=invested
        s['positions'].append({**c,'entry_date':today,'entry_time':datetime.now(TZ).strftime('%H:%M'),'entry_price':price,'shares':shares,'invested_amount':invested,'allocation':(invested/capital) if capital else 0,'policy_updated_at':policy.get('updated_at'),'current_price':price,'unrealized_pnl':0.0})
        s['trades_today']=int(s.get('trades_today',0))+1; s.setdefault('trades_by_ticker_today',{})[c['ticker']]=cnt+1; active.add(c['ticker']); out.append(s['positions'][-1])
        try: discord_progress.notify_trade(c['ticker'],c['direction'],price)
        except Exception as e: print(f'⚠️ discord_progress notify_trade失敗: {e}')
    return out

def mark_and_close(s,now,policy):
    # policy['hold_days']が示す営業日数だけ保有を許す(以前は毎日15:25に全ポジション
    # 強制決済していたが、検証済み戦略は複数営業日保有を前提としているため一致させる)。
    # まずエントリー翌営業日〜前営業日を日足で遡り、当日のイントラデイ監視を挟めずに
    # TP/SLを踏んでいないか確認(セッション未実行日などの取りこぼし対策)。
    remaining=[]; msgs=[]; hold_limit=max(1,int(policy.get('hold_days') or 1))
    for p in s['positions']:
        ep=float(p['entry_price']); sh=int(p.get('shares',0)); direction=p.get('direction','BUY'); entry_date=pd.Timestamp(p['entry_date'])
        exit_price=reason=exit_dt=None
        prior_days=pd.bdate_range(entry_date+pd.Timedelta(days=1),pd.Timestamp(now.date())-pd.Timedelta(days=1))
        if len(prior_days)>0:
            daily=download(p['ticker'],period='3mo')
            if daily is not None and not daily.empty:
                for ts,b in daily[daily.index.normalize().isin(prior_days)].iterrows():
                    hi,lo=float(b['High']),float(b['Low'])
                    if direction=='BUY':
                        if lo<=p['sl'] and hi>=p['tp']:exit_price,reason=float(p['sl']),'SL'
                        elif hi>=p['tp']:exit_price,reason=float(p['tp']),'TP'
                        elif lo<=p['sl']:exit_price,reason=float(p['sl']),'SL'
                    else:
                        if hi>=p['sl'] and lo<=p['tp']:exit_price,reason=float(p['sl']),'SL'
                        elif lo<=p['tp']:exit_price,reason=float(p['tp']),'TP'
                        elif hi>=p['sl']:exit_price,reason=float(p['sl']),'SL'
                    if reason:exit_dt=ts;break
        held=len(pd.bdate_range(entry_date+pd.Timedelta(days=1),pd.Timestamp(now.date())))
        if exit_price is None:
            d=download_5m(p['ticker'])
            if d is not None and not d.empty:
                bars=d[d.index.date==now.date()]; last=float(bars['Close'].iloc[-1]) if not bars.empty else float(p.get('current_price',ep)); p['current_price']=last
                for ts,b in bars.iterrows():
                    if ts.time()<dtime(9,0):continue
                    hi,lo=float(b['High']),float(b['Low'])
                    if direction=='BUY':
                        if lo<=p['sl']:exit_price,reason=float(p['sl']),'SL'
                        elif hi>=p['tp']:exit_price,reason=float(p['tp']),'TP'
                    else:
                        if hi>=p['sl']:exit_price,reason=float(p['sl']),'SL'
                        elif lo<=p['tp']:exit_price,reason=float(p['tp']),'TP'
                    if reason:exit_dt=ts;break
                    if held>=hold_limit and ts.time()>=FORCED_EXIT:exit_price,reason=float(b['Close']),'HOLD_LIMIT';exit_dt=ts;break
        cur=float(p.get('current_price',ep)); unreal=((cur-ep)*sh if direction=='BUY' else (ep-cur)*sh)-ep*sh*FEE_RATE-cur*sh*FEE_RATE; p['unrealized_pnl']=unreal
        if exit_price is None:remaining.append(p);continue
        gross=(exit_price-ep)*sh if direction=='BUY' else (ep-exit_price)*sh; pnl=gross-(ep+exit_price)*sh*FEE_RATE; s['capital']=float(s['capital'])+pnl; exit_value=exit_price*sh; total=s['capital']
        exit_date_str=str(pd.Timestamp(exit_dt).date()) if exit_dt is not None else str(now.date())
        append_history({'entry_date':p['entry_date'],'entry_time':p['entry_time'],'exit_date':exit_date_str,'exit_time':now.strftime('%H:%M'),'ticker':p['ticker'],'company':p['company'],'direction':direction,'entry_price':ep,'exit_price':exit_price,'shares':sh,'invested_amount':p['invested_amount'],'exit_value':exit_value,'tp':p['tp'],'sl':p['sl'],'score':p['score'],'up_probability':p['up_probability'],'down_probability':p['down_probability'],'expected_value_pct':p.get('expected_value_pct',0),'return_pct':pnl/p['invested_amount']*100 if p['invested_amount'] else 0,'pnl':pnl,'result':reason,'total_assets':total,'buy_reason':p.get('buy_reason','')})
        msgs.append(f"{'🟢' if pnl>=0 else '🔴'} 決済｜{p['company']}（{p['ticker']}）｜{direction}\n決済価格 {exit_price:,.1f}円｜{sh:,}株｜投資額 {p['invested_amount']:,.0f}円\n確定損益 {pnl:+,.0f}円｜💰総資産 {total:,.0f}円｜開始100万円から {total-INITIAL_CAPITAL:+,.0f}円")
        try: discord_progress.notify_exit(p['ticker'],exit_price,pnl)
        except Exception as e: print(f'⚠️ discord_progress notify_exit失敗: {e}')
    s['positions']=remaining; s['peak']=max(float(s.get('peak',s['capital'])),float(s['capital'])); return msgs

def _run():
    now=datetime.now(TZ); today=now.strftime('%Y-%m-%d'); policy_file,trend_result=select_policy_file(); policy=load_policy(policy_file); s=load_state(); reset_daily(s,today)
    if not(now.weekday()<5 and dtime(9,0)<=now.time()<=dtime(15,30)):
        discord_send(f'🤖 PROFIT LOOP｜待機\n{today} {now:%H:%M} JST\n市場時間外｜実注文なし'); return
    closed=mark_and_close(s,now,policy); cands,scanned=scan(policy)
    if cands:
        top1=cands[0]
        try:
            if discord_progress.top1_changed(top1.get('ticker',''),top1.get('direction','BUY')):
                discord_progress.notify_selection_start(); discord_progress.notify_top10(cands); discord_progress.notify_top1(top1.get('ticker',''),top1.get('direction','BUY'),float(top1.get('score',0) or 0))
        except Exception as e: print(f'⚠️ discord_progress TOP1通知失敗: {e}')
    opened=open_positions(s,policy,cands,today); save_state(s)
    entry_msgs=[
        f"🆕 エントリー｜{p['company']}（{p['ticker']}）｜{'買い' if p['direction']=='BUY' else '空売り'}\n"
        f"取得価格 {p['entry_price']:,.1f}円｜{p['shares']:,}株｜投資額 {p['invested_amount']:,.0f}円\n"
        f"利確 {p['tp']:,.1f}｜損切 {p['sl']:,.1f}｜期待値 {p.get('expected_value_pct',0):+.2f}%\n"
        f"🧠 買った基準: {p.get('buy_reason','')}"
        for p in opened
    ]
    equity=float(s['capital'])+sum(float(p.get('unrealized_pnl',0)) for p in s['positions']); daily=equity-float(s.get('daily_start_capital',INITIAL_CAPITAL)); cum=(equity/INITIAL_CAPITAL-1)*100
    rows=[]
    for i,p in enumerate(s['positions'],1):rows.append(f"{i}. {'買い' if p['direction']=='BUY' else '空売り'} {p['company']}（{p['ticker']}）\n   {p['shares']:,}株｜投資額 {p['invested_amount']:,.0f}円｜取得 {p['entry_price']:,.1f}円｜現在値 {p['current_price']:,.1f}円｜含み損益 {p['unrealized_pnl']:+,.0f}円\n   利確 {p['tp']:,.1f}｜損切 {p['sl']:,.1f}｜期待値 {p.get('expected_value_pct',0):+.2f}%\n   🧠 買った基準: {p.get('buy_reason','')}")
    msg=('🤖 利益優先ループ｜TOP10 ペーパートレード\n━━━━━━━━━━━━━━━━━━\n'
         f'📅 {today} {now:%H:%M} JST｜⚠️ 実注文なし\n対象225銘柄(日経225)｜取得成功 {scanned}｜候補 {len(cands)}｜新規 {len(opened)}件\n'
         f'条件: 確率≥{policy["up_threshold"]:.0f}%｜AIスコア≥{policy["min_score_for_buy"]:.0f}｜TP×{policy["atr_tp_multiplier"]:.1f}｜SL×{policy["atr_sl_multiplier"]:.1f}｜日経フィルター{"ON" if policy.get("nikkei_filter") else "OFF"}｜最大保有{policy["hold_days"]}営業日\n'
         f'📈 先物トレンド: {trend_result.get("trend")}｜{trend_result.get("reason")}｜使用policy: {policy_file}\n'
         f'💰総資産 {equity:,.0f}円｜本日 {daily:+,.0f}円｜累計 {cum:+.2f}%\n'
         f'📦 保有 {len(s["positions"])}件\n' + ('\n'.join(rows) if rows else 'なし'))
    for m in closed: discord_send(m)
    for m in entry_msgs: discord_send(m)
    discord_send(msg)
    try: discord_progress.notify_progress(msg)
    except Exception as e: print(f'⚠️ discord_progress notify_progress失敗: {e}')

def main():
    try:
        _run()
    except Exception as e:
        try: discord_progress.notify_error('main',str(e))
        except Exception: pass
        raise

if __name__=='__main__': main()
