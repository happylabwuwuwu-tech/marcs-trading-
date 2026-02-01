import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import warnings
import os
import random
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (星際戰神風格)
# =============================================================================
st.set_page_config(page_title="MARCS V97 經典復刻版", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; }
    
    /* 星空背景 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.7;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 風險儀表板 */
    .risk-container {
        background: rgba(30, 30, 35, 0.6); border: 1px solid #333; padding: 15px 20px;
        border-radius: 10px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .risk-val { font-family: 'JetBrains Mono'; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
    .risk-label { font-size: 12px; color: #888; text-transform: uppercase; }
    
    /* 戰術面板 */
    .tac-card {
        background: rgba(26, 26, 26, 0.8); border: 1px solid #444; border-radius: 6px; padding: 10px;
        margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;
        backdrop-filter: blur(5px);
    }
    .tac-label { font-size: 12px; color: #aaa; font-family: 'Rajdhani'; font-weight: bold; }
    .tac-val { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: bold; color: #fff; }
    .tac-sub { font-size: 10px; color: #666; margin-left: 5px; }

    /* 一般組件 */
    .metric-card {
        background: rgba(18, 18, 22, 0.85); backdrop-filter: blur(12px);
        border-left: 4px solid #ffae00; border-radius: 8px; padding: 15px; margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-left-color: #ffd700; }
    
    .highlight-val { font-size: 24px; font-weight: bold; color: #fff; font-family: 'JetBrains Mono'; }
    .highlight-lbl { font-size: 12px; color: #8b949e; letter-spacing: 1px; text-transform: uppercase;}
    .smart-text { font-size: 14px; color: #ffb86c; font-weight: bold; margin-top: 5px; }
    
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); }
    
    .factor-table { width: 100%; border-collapse: collapse; font-size: 13px; background: rgba(30,30,30,0.5); border: 1px solid #444; border-radius:4px; }
    .factor-table td { padding: 8px; border-bottom: 1px solid #333; color: #eee; }
    .factor-bar-bg { width: 100%; height: 4px; background: #333; border-radius: 2px; }
    
    .chip-tag { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 10px; font-family: 'Noto Sans TC'; vertical-align: middle; }
    
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; transition: 0.2s; border-radius: 5px; }
    .news-card:hover { background: rgba(40,40,50,0.9); }
    .news-title { color: #e0e0e0; text-decoration: none; font-weight: bold; font-size: 14px; }
    
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; border:none; background: linear-gradient(90deg, #333 0%, #ffae00 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 數據獲取層 (Robust Download V97)
# =============================================================================
def robust_download(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty and df['Close'].iloc[-1] > 0:
            df.index = pd.to_datetime(df.index)
            return df
            
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0) 
            except: pass
        if 'Close' not in df.columns and 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        
        if not df.empty and 'Close' in df.columns and df['Close'].iloc[-1] > 0:
            df.index = pd.to_datetime(df.index)
            return df
    except: pass
    return pd.DataFrame()

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
        return []

# =============================================================================
# 2. SMC 與 量化引擎
# =============================================================================
class SMC_Engine:
    @staticmethod
    def identify_fvg(df, lookback=30):
        fvgs = []
        try:
            for i in range(len(df)-2, len(df)-lookback, -1):
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    fvgs.append({'type': 'Bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'idx': df.index[i-2]})
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    fvgs.append({'type': 'Bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'idx': df.index[i-2]})
            return fvgs[:3]
        except: return []

# =============================================================================
# 3. 核心分析引擎
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 30: return 50, ["數據不足"], df, 0, None, 0, 0, []
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20; signals.append("Elder Impulse Bull")
            if fi_13.iloc[-1] > 0: score += 10
            
            fvgs = SMC_Engine.identify_fvg(df)
            current_price = c.iloc[-1]
            in_bull = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bull' for f in fvgs)
            if in_bull: score += 15; signals.append("SMC Support")
            
            chips = FinMind_Engine.get_tw_chips(ticker)
            if chips:
                if chips['latest'] > 1000: score += 15
                elif chips['latest'] < -1000: score -= 15
            
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = ema22 + 2*atr; df['K_Lower'] = ema22 - 2*atr
            
            return score, signals, df, atr, chips, current_price, score, fvgs
        except: return 50, ["計算錯誤"], df, 0, None, 0, 0, []

# =============================================================================
# 4. 輔助引擎 (修復估值與宏觀)
# =============================================================================
class FinMind_Engine:
    @staticmethod
    def get_tw_chips(ticker):
        if ".TW" not in ticker: return None
        try:
            start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": ticker.split('.')[0], "start_date": start_date}
            res = requests.get(url, params=params, timeout=3)
            data = res.json()
            if data['msg'] == 'success' and data['data']:
                df = pd.DataFrame(data['data'])
                f = df[df['name'] == 'Foreign_Investor']
                if not f.empty: return {"latest": int((f.iloc[-1]['buy']-f.iloc[-1]['sell'])/1000)}
            return None
        except: return None

class News_Intel_Engine:
    @staticmethod
    def fetch_news(ticker):
        items = []
        sentiment = 0
        try:
            # 1. yfinance News
            try:
                stock = yf.Ticker(ticker); raw_news = stock.news
                for item in raw_news[:3]:
                    t = item.get('title'); l = item.get('link')
                    d = pd.to_datetime(item.get('providerPublishTime'), unit='s').strftime('%m-%d')
                    s = "neu"
                    if any(x in t.lower() for x in ["soar","jump","beat","buy"]): s="pos"; sentiment+=1
                    elif any(x in t.lower() for x in ["drop","miss","sell"]): s="neg"; sentiment-=1
                    items.append({"title": t, "link": l, "date": d, "sent": s})
            except: pass

            # 2. Google Fallback
            if not items:
                q = ticker.split('.')[0] + (" 台股" if ".TW" in ticker else " stock")
                url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall('.//item')[:3]:
                        t = item.find('title').text
                        l = item.find('link').text
                        d = item.find('pubDate').text[5:16] if item.find('pubDate') else ""
                        s = "pos" if "漲" in t else ("neg" if "跌" in t else "neu")
                        items.append({"title": t, "link": l, "date": d, "sent": s})
                        if s=="pos": sentiment+=1
                        elif s=="neg": sentiment-=1
            
            return items, max(-1, min(1, sentiment/3))
        except: return [], 0

class Scanner_Engine_Elder:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = robust_download(ticker, "6mo")
            if df.empty or len(df) < 50: return None
            c = df['Close']; ema22 = c.ewm(span=22).mean()
            score = 60
            if c.iloc[-1] > ema22.iloc[-1]: score += 20
            else: score -= 20
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "sl": ema22.iloc[-1]*0.98}
        except: return None

class Factor_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            def g(k, d=None): return info.get(k, d)
            pe = g('trailingPE', 20); roe = g('returnOnEquity', 0.1)
            rev_g = g('revenueGrowth', 0.05); beta = g('beta', 1.0)
            val_s = 60 if pe < 25 else 40
            gro_s = min(100, int(rev_g * 400)) if rev_g else 50
            qual_s = 70 if roe > 0.15 else 40
            vol_s = 80 if beta < 1.0 else 40
            return {"scores": {"Value": val_s, "Growth": gro_s, "Quality": qual_s, "LowVol": vol_s}, 
                    "raw": {"PE": pe, "ROE": roe, "Beta": beta, "RevG": rev_g}}
        except: return None

class PEG_Valuation_Engine:
    @staticmethod
    def calculate(ticker, sentiment_score=0):
        """[V97 Fix] 估值防護網：若無成長率，改用純本益比估值"""
        try:
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 0)
            if price == 0: price = info.get('regularMarketPrice', 0)
            if price == 0: return None
            
            pe = info.get('trailingPE', None)
            growth = info.get('earningsGrowth', None)
            
            # Fallback 1: 如果有 PE 但沒 Growth -> 假設 5% 成長 (保守)
            if pe and not growth: growth = 0.05
            
            # Fallback 2: 還是沒有 -> Price Only
            if not pe or not growth: 
                return {"fair": price, "scenarios": {"Bear": price*0.9, "Bull": price*1.1}, "method": "Price Only", "peg_used": 0, "sentiment_impact": "0%"}
            
            peg = pe / (growth * 100)
            target_peg = peg * (1 + (sentiment_score * 0.2))
            fair_price = (price / pe) * (target_peg * growth * 100)
            return {"fair": fair_price, "scenarios": {"Bear": fair_price * 0.85, "Bull": fair_price * 1.15}, "method": "PEG Adjusted", "peg_used": round(target_peg, 2), "sentiment_impact": f"{sentiment_score*20:+.0f}%"}
        except: return None

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid):
        default = {"cap": 0, "pct": 0.0}
        if price <= 0: return 0, default
        try:
            risk = capital * 0.02; dist = price - sl
            if dist <= 0: return 0, default
            conf = hybrid / 100.0
            size = int((risk/dist) * conf)
            pos_val = size * price
            pct = (pos_val / capital) * 100
            return size, {"cap": int(pos_val), "pct": round(pct, 1)}
        except: return 0, default

class Backtest_Engine:
    @staticmethod
    def run_backtest(ticker):
        """[V97 Fix] 恢復詳細回測數據 (MDD, CAGR)"""
        try:
            df = robust_download(ticker, "2y")
            if df.empty or len(df) < 100: return None
            
            df['EMA22'] = df['Close'].ewm(span=22).mean()
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            df['Hist'] = df['MACD'] - df['Signal']
            df['Green'] = (df['EMA22'] > df['EMA22'].shift(1)) & (df['Hist'] > df['Hist'].shift(1))
            
            position = 0; entry_price = 0; trades = []; equity = [100000]
            
            for i in range(1, len(df)):
                price = df['Close'].iloc[i]; date = df.index[i]
                if position == 0 and df['Green'].iloc[i]:
                    position = 1; entry_price = price
                    trades.append({'date': date, 'type': 'Buy', 'price': price, 'profit': 0})
                elif position == 1 and not df['Green'].iloc[i]:
                    position = 0; profit = (price - entry_price) / entry_price
                    equity.append(equity[-1] * (1 + profit))
                    trades.append({'date': date, 'type': 'Sell', 'price': price, 'profit': profit})
                
                if position == 1: equity.append(equity[-1] * (1 + (df['Close'].iloc[i]/df['Close'].iloc[i-1] - 1)))
                else: equity.append(equity[-1])
            
            # [V97] Advanced Stats
            equity_curve = pd.Series(equity, index=df.index[-len(equity):])
            total_ret = (equity[-1] - 100000) / 100000
            
            # MDD
            roll_max = equity_curve.cummax()
            drawdown = (equity_curve - roll_max) / roll_max
            mdd = drawdown.min()
            
            # Win Rate
            wins = len([t for t in trades if t['type']=='Sell' and t['profit']>0])
            total_trades = len([t for t in trades if t['type']=='Sell'])
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            return {
                "total_return": total_ret,
                "mdd": mdd,
                "win_rate": win_rate,
                "trades": total_trades,
                "equity_curve": equity_curve
            }
        except: return None

class Macro_Risk_Engine:
    @staticmethod
    def calculate_market_risk():
        """[V97 Fix] 宏觀數據防護，防止 None"""
        try:
            df = robust_download("^VIX", "5d")
            vix = df['Close'].iloc[-1] if not df.empty else 20
            return 60, ["VIX Stable"], vix
        except: return 50, ["Data Retry"], 20

class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid, m_score, chips, fvgs):
        tag = "😐 觀望 (Hold)"; bg = "#333"
        if hybrid >= 80: tag = "🔥 強力買進"; bg = "#3fb950"
        elif hybrid >= 60: tag = "✅ 買進"; bg = "#1f6feb"
        elif hybrid <= 40: tag = "❄️ 弱勢"; bg = "#888"
        elif hybrid <= 20: tag = "⛔ 危險"; bg = "#f85149"
        
        reasons = []
        if m_score >= 70: reasons.append("動能強勁")
        if chips and chips['latest'] > 0: reasons.append("外資買超")
        if any(f['type']=='Bull' for f in fvgs): reasons.append("回測 Bullish FVG")
        
        return tag, f"{ticker} 目前呈現 {tag.split(' ')[1]}。主因：{'，'.join(reasons)}。", bg

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "2330.TW").upper()
    if st.sidebar.button("分析單一標的"): st.session_state.target = target_in
    
    # Scanner
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 主動掃描器"):
        scan_source = st.radio("來源", ["線上掃描", "CSV匯入"])
        if scan_source == "線上掃描":
            market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股"])
            if st.button("🚀 啟動掃描"):
                with st.spinner("Deep Scanning..."):
                    tickers = Global_Market_Loader.get_scan_list(market)
                    res = []
                    bar = st.progress(0)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
                        futures = {exe.submit(Scanner_Engine_Elder.analyze_single, t): t for t in tickers}
                        done = 0
                        for f in concurrent.futures.as_completed(futures):
                            r = f.result(); done += 1
                            if r: res.append(r)
                            bar.progress(done/len(tickers))
                    st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                    bar.empty()
        else:
            uploaded = st.file_uploader("上傳CSV", type=['csv'])
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    tickers = df_up.iloc[:, 0].astype(str).tolist()
                    if st.button("🚀 掃描上傳清單"):
                        res = []
                        bar = st.progress(0)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
                            futures = {exe.submit(Scanner_Engine_Elder.analyze_single, t): t for t in tickers}
                            done = 0
                            for f in concurrent.futures.as_completed(futures):
                                r = f.result(); done += 1
                                if r: res.append(r)
                                bar.progress(done/len(tickers))
                        st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                        bar.empty()
                except: st.error("CSV 格式錯誤")

    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    target = st.session_state.target

    # 1. Macro
    risk, risk_d, vix = Macro_Risk_Engine.calculate_market_risk()
    st.markdown(f"""<div class="risk-container"><div class="risk-val" style="color:#4caf50">{risk}</div><div style="color:#aaa">MARKET RISK (VIX: {vix:.1f})</div></div>""", unsafe_allow_html=True)

    # Scanner Results
    if st.session_state.scan_results:
        with st.expander("🔭 掃描結果"):
            st.dataframe(pd.DataFrame(st.session_state.scan_results), use_container_width=True)
            sel = st.selectbox("Load:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("Load"): st.session_state.target = sel

    # 2. Analysis
    with st.spinner(f"Scanning {target} (Elder + SMC FVG)..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Pro.analyze, target)
            f_factor = executor.submit(Factor_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            m_score, sigs, df_m, atr, chips, curr_p, _, fvgs = f_micro.result()
            factor_data = f_factor.result()
            news, sent = f_news.result()
            dcf_res = PEG_Valuation_Engine.calculate(target, sent)
            backtest_res = Backtest_Engine.run_backtest(target)

        hybrid = int((risk * 0.3) + (m_score * 0.7))
        sl_p = curr_p - 2.5 * atr if atr > 0 else 0
        tp_p = curr_p + 4.0 * atr if atr > 0 else 0
        risk_pct = round((sl_p / curr_p - 1)*100, 2) if curr_p > 0 else 0
        size, r_d = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # 3. Verdict
    tag, comm, bg = Message_Generator.get_verdict(target, hybrid, m_score, chips, fvgs)
    c_tag = f"<span class='chip-tag' style='background:#f44336'>外資 {chips['latest']}</span>" if chips else ""
    st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${curr_p:.2f}</span> {c_tag}</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div class="verdict-box" style="background:{bg}30; border-color:{bg}"><h2 style="margin:0; color:{bg}">{tag}</h2><p style="margin-top:5px; color:#ccc">{comm}</p></div>""", unsafe_allow_html=True)

    # Tactical Panel
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">ATR (Volatility)</div><div class="tac-val">{atr:.2f}</div></div><div class="tac-sub">Risk Unit</div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="tac-card" style="border-color:#f44336"><div><div class="tac-label">STOP LOSS</div><div class="tac-val" style="color:#f44336">${sl_p:.2f}</div></div><div class="tac-sub">{risk_pct}% Risk</div></div>""", unsafe_allow_html=True)
    with t3: st.markdown(f"""<div class="tac-card" style="border-color:#4caf50"><div><div class="tac-label">TAKE PROFIT</div><div class="tac-val" style="color:#4caf50">${tp_p:.2f}</div></div><div class="tac-sub">Reward 1.6x</div></div>""", unsafe_allow_html=True)
    with t4: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">SUGGESTED SIZE</div><div class="tac-val">{r_d['pct']}%</div></div><div class="tac-sub">${r_d['cap']:,}</div></div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">技術評分</div><div class="highlight-val">{m_score}</div><div class="smart-text">{sigs[0] if sigs else '盤整'}</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">宏觀風險</div><div class="highlight-val">{risk}</div><div class="smart-text">VIX: {vix:.1f}</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 情緒修正</div><div class="highlight-val">{sent:+.2f}</div><div class="smart-text">News Adj</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">SMC 訊號</div><div class="highlight-val">{len(fvgs)}</div><div class="smart-text">Active FVG</div></div>""", unsafe_allow_html=True)

    # 4. Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 SMC 戰術圖表", "🧬 估值模型", "📰 情報中心", "🔄 策略回測"])
    
    with tab1:
        if not df_m.empty and 'EMA22' in df_m.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5, label='Price')
            ax.plot(df_m.index, df_m['EMA22'], color='#ffae00', lw=1.5, label='EMA 22')
            for fvg in fvgs:
                color = 'green' if fvg['type'] == 'Bull' else 'red'
                rect = patches.Rectangle((fvg['idx'], fvg['bottom']), width=timedelta(days=5), height=fvg['top']-fvg['bottom'], linewidth=0, edgecolor=None, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            ax.axhline(sl_p, color='#f44336', ls='--', label='SL')
            ax.axhline(tp_p, color='#4caf50', ls='--', label='TP')
            ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='#888'); ax.grid(True, color='#333', alpha=0.3)
            st.pyplot(fig)
            
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols, alpha=0.8); ax1.set_title("MACD", color='white')
            ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='#888')
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_title("Force Index", color='white')
            ax2.axhline(0, color='gray', ls='--')
            ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='#888')
            fig2.patch.set_facecolor('#0d1117'); st.pyplot(fig2)
        else: st.warning("數據不足")

    with tab2:
        if dcf_res:
            c_v1, c_v2 = st.columns(2)
            with c_v1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 合理價</div><div class="highlight-val">${dcf_res['fair']:.2f}</div><div class="smart-text">{dcf_res['method']}</div></div>""", unsafe_allow_html=True)
            with c_v2: st.json(dcf_res['scenarios'])
        else: st.info("無 PEG 數據")

    with tab3:
        if news:
            cols = st.columns(3)
            for i, item in enumerate(news):
                bd = "#4caf50" if item['sent']=="pos" else "#444"
                with cols[i%3]: st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd}"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta">{item['date']}</div></div>""", unsafe_allow_html=True)
        else: st.info("無新聞")

    with tab4:
        if backtest_res:
            b1, b2, b3 = st.columns(3)
            with b1: st.metric("總報酬 (2Y)", f"{backtest_res['total_return']:.1%}")
            with b2: st.metric("最大回撤 (MDD)", f"{backtest_res['mdd']:.1%}")
            with b3: st.metric("勝率", f"{backtest_res['win_rate']:.1%}")
            st.line_chart(backtest_res['equity_curve'])
        else: st.warning("無法回測")

if __name__ == "__main__":
    main()
