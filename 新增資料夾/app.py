import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import warnings
import os
import random
import concurrent.futures
from scipy.stats import wasserstein_distance

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺與訊息設計 (V75 人性化 + V74 戰神美學)
# =============================================================================
st.set_page_config(page_title="MARCS V76 完全體", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; }
    
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.7;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* V75 借鑑 Daily Dip 的卡片設計 */
    .metric-card {
        background: rgba(18, 18, 22, 0.85); 
        backdrop-filter: blur(12px);
        border-left: 4px solid #ffae00; /* Elder Orange */
        border-radius: 8px; padding: 15px; text-align: left;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-left-color: #ffd700; }

    .highlight-val { font-size: 28px; font-weight: bold; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.1); }
    .highlight-lbl { font-size: 12px; color: #8b949e; letter-spacing: 1px; text-transform: uppercase;}
    .smart-text { font-size: 14px; color: #ffb86c; font-weight: bold; margin-top: 5px; }
    
    /* V75 智能點評區塊 */
    .verdict-box {
        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; border:none; background: linear-gradient(90deg, #333 0%, #ffae00 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 訊息轉譯模組 (V75 靈魂)
# =============================================================================
class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid_score, m_score, impact):
        # 1. 狀態標籤
        tag = "😐 觀望 (Hold)"
        bg_color = "#30363d"
        
        if hybrid_score >= 80: 
            tag = "🔥 強力買進 (Strong Buy)"; bg_color = "#3fb950" # Green
        elif hybrid_score >= 60: 
            tag = "✅ 買進 (Buy)"; bg_color = "#1f6feb" # Blue
        elif hybrid_score <= 40: 
            tag = "❄️ 弱勢 (Weak)"; bg_color = "#888" # Grey
        elif hybrid_score <= 20: 
            tag = "⛔ 危險 (Danger)"; bg_color = "#f85149" # Red
        
        # 2. 智能短評
        reasons = []
        if m_score >= 70: reasons.append("艾爾德動能強勁")
        elif m_score <= 40: reasons.append("艾爾德動能疲弱")
        
        if impact > 10: reasons.append("宏觀順風 (+Macro)")
        elif impact < -10: reasons.append("宏觀逆風 (-Macro)")
        
        comment = f"{ticker} 目前呈現 {tag.split(' ')[1]} 狀態。"
        if reasons: comment += "主因：" + "，且".join(reasons) + "。"
        else: comment += "多空力道膠著，建議耐心等待明確訊號。"
            
        return tag, comment, bg_color

# =============================================================================
# 2. 資料庫與宏觀 (V72 利差邏輯)
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {"^VIX": {"name": "VIX 恐慌", "type": "Sentiment"}, "^TNX": {"name": "US10Y 殖利率", "type": "Yield"}, "JPY=X": {"name": "USD/JPY 匯率", "type": "Currency"}, "^SOX": {"name": "SOX 費半", "type": "Equity"}, "DX-Y.NYB": {"name": "DXY 美元", "type": "Currency"}}

    @staticmethod
    def get_correlation_impact(ticker, macro_data):
        impact_score = 0
        us10y = macro_data.get('^TNX', {}).get('trend', 'Neutral')
        dxy = macro_data.get('DX-Y.NYB', {}).get('trend', 'Neutral')
        sox = macro_data.get('^SOX', {}).get('trend', 'Neutral')

        if any(x in ticker for x in [".TW", ".TWO"]): 
            if "Bull" in us10y: impact_score -= 15
            if "Bull" in dxy: impact_score -= 10
            if "Bull" in sox: impact_score += 20
        elif "=F" in ticker:
            if "Bull" in us10y: impact_score -= 25
            if "Bull" in dxy: impact_score -= 15
        elif "-USD" in ticker:
            if "Bull" in us10y: impact_score -= 20
        return int(impact_score)

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_full_market():
        try:
            tickers = []
            headers = {'User-Agent': 'Mozilla/5.0'}
            for m, s in [(2, '.TW'), (4, '.TWO')]:
                res = requests.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}", headers=headers, timeout=5)
                if res.status_code == 200:
                    df = pd.read_html(res.text)[0]
                    for item in df.iloc[:, 0].astype(str):
                        parts = item.split()
                        if len(parts)>=1 and len(parts[0])==4 and parts[0].isdigit(): tickers.append(f"{parts[0]}{s}")
            if len(tickers)<50: raise Exception("Blocked")
            random.shuffle(tickers)
            return tickers
        except: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW", "1519.TW"]

    @staticmethod
    def get_scan_list(market_type, limit=0):
        if "台股" in market_type:
            full = Global_Market_Loader.get_tw_full_market()
            return full[:limit] if limit > 0 else full
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "AVGO", "TSM", "SOXL", "TQQQ"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "PEPE-USD"]
        elif "貴金屬" in market_type: return ["GC=F", "SI=F", "HG=F", "CL=F"]
        return []

# =============================================================================
# 3. 分析引擎 (V74 Elder 核心)
# =============================================================================
class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; ma20 = c.rolling(20).mean()
            trend = "Bullish/High" if c.iloc[-1] > ma20.iloc[-1] else "Bearish/Low"
            return {"name": name, "price": c.iloc[-1], "trend": trend}
        except: return None

class Scanner_Engine_Elder:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df)<60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            c = df['Close']; v = df['Volume']
            if len(v)>0 and v.iloc[-1]==0: return None
            
            ema22 = c.ewm(span=22).mean()
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26; signal = macd.ewm(span=9).mean(); hist = macd - signal
            
            score = 40
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            if hist.iloc[-1] > hist.iloc[-2]: score += 20
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            if fi_13.iloc[-1] > 0: score += 10
            
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1]-2.5*atr, ema22.iloc[-1]*0.98)
            
            if score < min_score: return None
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "sl": sl}
        except: return None

class Micro_Engine_Elder:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df, 0
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            score = 50; signals = []
            
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10; signals.append("EMA趨勢向上")
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26; signal = macd.ewm(span=9).mean(); hist = macd - signal
            
            fi = c.diff() * v
            fi_13 = fi.ewm(span=13).mean(); fi_2 = fi.ewm(span=2).mean()
            
            is_green = (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2])
            if is_green: score += 20; signals.append("🔥 Elder 綠燈")
            if fi_13.iloc[-1] > 0: score += 10; signals.append("主力買盤")
            if (fi_13.iloc[-1] > 0) and (fi_2.iloc[-1] < 0) and is_green: score += 15; signals.append("✨ 完美回檔買點")

            atr = (h-l).rolling(14).mean()
            k_upper = ema22 + 2.0 * atr.rolling(10).mean()
            k_lower = ema22 - 2.0 * atr.rolling(10).mean()
            
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force_Index'] = fi_13
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return score, signals, df, atr.iloc[-1]
        except: return 50, [], pd.DataFrame(), 0

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid_score):
        if any(x in ticker for x in ["-USD", "BTC", "ETH"]): vol_cap = 1.0; atype = "Crypto"
        elif "=F" in ticker: vol_cap = 0.4; atype = "Metal"
        elif any(x in ticker for x in [".TW", ".TWO"]): vol_cap = 0.5; atype = "TW Stock"
        else: vol_cap = 0.6; atype = "US Stock"
        
        risk = capital * 0.02; dist = price - sl
        if dist <= 0: return 0, {}
        
        conf = hybrid_score / 100.0
        size = int((risk/dist) * (0.5 if vol_cap>0.8 else 1.0) * conf)
        if vol_cap > 0.8: size = round((risk/dist)*0.5*conf, 4)
        
        position_value = size * price
        pct_capital = (position_value / capital) * 100
        if pct_capital > 50: 
            pct_capital = 50; size = (capital * 0.5) / price
            if vol_cap <= 0.8: size = int(size)
            position_value = size * price

        return size, {"risk": int(risk), "type": atype, "cap": int(position_value), "pct": round(pct_capital, 1)}

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("總本金", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 被動輸入 (Quick Check)")
    manual_input = st.sidebar.text_input("輸入代碼 (e.g. 2330.TW)", value="").upper()
    run_manual = st.sidebar.button("分析單一標的")

    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 主動掃描 (Scanner)"):
        mode = st.radio("來源", ["線上掃描", "匯入CSV"])
        if mode == "線上掃描":
            market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股", "₿ 加密", "🥇 貴金屬"])
            limit = 0
            if "台股" in market and st.checkbox("限制數量", value=True): limit = st.slider("上限", 100, 2000, 300)
            if st.button("啟動掃描"):
                with st.spinner("Elder Scanner Running..."):
                    tickers = Global_Market_Loader.get_scan_list(market, limit)
                    res = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
                        futures = {exe.submit(Scanner_Engine_Elder.analyze_single, t, 60): t for t in tickers}
                        for f in concurrent.futures.as_completed(futures):
                            r = f.result()
                            if r: res.append(r)
                    st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
        else:
            uploaded = st.file_uploader("上傳CSV", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded)
                df.columns = [c.lower() for c in df.columns]; df.rename(columns={'stoploss':'sl'}, inplace=True)
                st.session_state.scan_results = df.to_dict('records')

    st.sidebar.markdown("---")
    video_file = "demo.mp4"
    if os.path.exists(video_file): st.sidebar.video(video_file)

    st.markdown("<h1 style='text-align:center; color:#ffae00; text-shadow: 0 0 10px rgba(255,174,0,0.5);'>🛡️ MARCS V76 完全體</h1>", unsafe_allow_html=True)

    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "macro_data" not in st.session_state: st.session_state.macro_data = {}
    if "target" not in st.session_state: st.session_state.target = "BTC-USD"

    if run_manual and manual_input: st.session_state.target = manual_input

    # ZONE 1: Macro
    st.markdown("### 📡 1. 全球宏觀 (Macro)")
    if st.button("🔄 同步全球數據 (Yield Update)"):
        with st.spinner("Analyzing Yield Spreads..."):
            macro_res = {}
            cols = st.columns(5)
            idx = 0
            for t, info in Global_Market_Loader.get_indices().items():
                r = Macro_Engine.analyze(t, info['name'])
                if r:
                    macro_res[t] = r
                    is_bad = "Bull" in r['trend'] and ("VIX" in t or "TNX" in t or "DXY" in t)
                    clr = "#f85149" if is_bad else "#3fb950"
                    with cols[idx]:
                        st.markdown(f"""<div class="metric-card" style="border-top:2px solid {clr}">
                            <div class="highlight-lbl">{r['name']}</div>
                            <div class="highlight-val" style="font-size:20px">{r['price']:.2f}</div>
                            <div class="metric-sub" style="color:{clr}">{r['trend']}</div>
                        </div>""", unsafe_allow_html=True)
                    idx += 1
            st.session_state.macro_data = macro_res

    # ZONE 2: Spotlight (V75 Feature)
    if st.session_state.scan_results:
        st.markdown("### 🔥 今日焦點 (Top Picks)")
        top_3 = st.session_state.scan_results[:3]
        cols = st.columns(len(top_3))
        for i, stock in enumerate(top_3):
            with cols[i]:
                if st.button(f"{stock['ticker']} (Score: {stock['score']})", key=f"btn_{i}"):
                    st.session_state.target = stock['ticker']
        
        with st.expander("🔭 完整掃描列表"):
            df = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True)

    # ZONE 3: Report
    target = st.session_state.target
    if target:
        st.markdown("---")
        st.markdown(f"### 🎯 深度戰略分析: {target}")
        
        with st.spinner(f"Applying Triple Screen Analysis for {target}..."):
            m_score, sigs, df_m, atr = Micro_Engine_Elder.analyze(target)
            impact = 0
            if st.session_state.macro_data:
                impact = Global_Market_Loader.get_correlation_impact(target, st.session_state.macro_data)
            hybrid = m_score + impact
            
            info = next((r for r in st.session_state.scan_results if r['ticker'] == target), None)
            if not df_m.empty:
                curr_p = df_m['Close'].iloc[-1]
                sl_p = curr_p - 2.5 * atr
            elif info: curr_p = info['price']; sl_p = info['sl']
            else: curr_p = 0
            
            if curr_p > 0:
                size, dets = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)
                
                # --- V75 智能點評區塊 ---
                tag, comment, bg_color = Message_Generator.get_verdict(target, hybrid, m_score, impact)
                st.markdown(f"""
                <div class="verdict-box" style="background:{bg_color}30; border-color:{bg_color}">
                    <h2 style="margin:0; color:{bg_color}; text-shadow:0 0 10px {bg_color}80;">{tag}</h2>
                    <p style="margin-top:10px; font-size:18px; color:#e6edf3;">{comment}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- 卡片顯示 (V75 Style) ---
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">艾爾德評分</div><div class="highlight-val">{m_score}</div><div class="smart-text">{', '.join(sigs) if sigs else '盤整'}</div></div>""", unsafe_allow_html=True)
                with c2: 
                    clr = "#3fb950" if impact>0 else "#f85149"
                    st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">宏觀修正</div><div class="highlight-val" style="color:{clr}">{impact}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">綜合總分</div><div class="highlight-val" style="color:#ffae00">{hybrid}</div></div>""", unsafe_allow_html=True)
                with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">建議倉位 %</div><div class="highlight-val">{dets['pct']}%</div><div class="smart-text">${dets['cap']:,}</div></div>""", unsafe_allow_html=True)
                
                # --- 圖表區 (V74 Elder Style) ---
                st.markdown("#### 📊 戰術圖表 (Tactical Chart)")
                tab1, tab2 = st.tabs(["🕯️ Keltner 主圖", "🌊 MACD & Force Index (籌碼)"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sub = df_m.tail(120)
                    ax.plot(sub.index, sub['Close'], color='#e6edf3', lw=1.5, label='Price')
                    ax.plot(sub.index, sub['EMA22'], color='#ffae00', lw=1.5, label='EMA 22 (Trend)')
                    ax.plot(sub.index, sub['K_Upper'], color='#00f2ff', ls='--', alpha=0.3)
                    ax.plot(sub.index, sub['K_Lower'], color='#00f2ff', ls='--', alpha=0.3)
                    ax.fill_between(sub.index, sub['K_Upper'], sub['K_Lower'], color='#00f2ff', alpha=0.05)
                    ax.axhline(sl_p, color='#f85149', ls='-', label=f'SL: {sl_p:.2f}')
                    ax.legend()
                    ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
                    ax.tick_params(colors='#8b949e'); ax.grid(True, color='#30363d', alpha=0.3)
                    st.pyplot(fig)
                
                with tab2:
                    if not df_m.empty:
                        fig2, (ax_macd, ax_fi) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 1])
                        sub_ind = df_m.tail(120)
                        
                        hist = sub_ind['MACD_Hist']
                        colors = ['#3fb950' if hist.iloc[i] > hist.iloc[i-1] else '#f85149' for i in range(len(hist))]
                        ax_macd.bar(sub_ind.index, hist, color=colors, alpha=0.9)
                        ax_macd.set_title('MACD Histogram (Momentum)', color='white', fontsize=10)
                        ax_macd.set_facecolor('#0d1117'); ax_macd.tick_params(colors='#8b949e')
                        
                        ax_fi.plot(sub_ind.index, sub_ind['Force_Index'], color='#00f2ff', lw=1.5)
                        ax_fi.axhline(0, color='gray', ls='--', alpha=0.5)
                        ax_fi.set_title('Force Index (13-Day)', color='white', fontsize=10)
                        ax_fi.set_facecolor('#0d1117'); ax_fi.tick_params(colors='#8b949e')
                        
                        fig2.patch.set_facecolor('#0d1117')
                        st.pyplot(fig2)
            else:
                st.error("無法獲取數據，請確認代碼是否正確。")

if __name__ == "__main__":
    main()
