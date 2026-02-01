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
# 0. 視覺核心 (V66 星際美學 + V57 霓虹特效)
# =============================================================================
st.set_page_config(page_title="MARCS V71 奇點戰情室", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* 深邃宇宙背景 */
    .stApp { background-color: #020202; font-family: 'Rajdhani', sans-serif; }
    
    /* 粒子星空特效 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        background-position: 0 0, 40px 60px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.7;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 懸浮毛玻璃卡片 */
    .metric-card {
        background: rgba(18, 18, 22, 0.75); 
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 255, 0.15);
        border-radius: 12px; padding: 15px; text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .metric-card:hover { 
        transform: translateY(-5px); 
        border-color: rgba(0, 242, 255, 0.6); 
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }

    /* 科技感文字 */
    .metric-value { color: #fff; font-size: 24px; font-weight: 700; text-shadow: 0 0 10px rgba(255,255,255,0.1); }
    .metric-label { color: #8b949e; font-size: 12px; letter-spacing: 1px; font-family: 'Roboto Mono'; text-transform: uppercase; }
    
    /* 按鈕與組件優化 */
    .stButton>button { 
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%); 
        border: 1px solid #30363d; color: #58a6ff; 
        transition: 0.3s;
    }
    .stButton>button:hover {
        border-color: #00f2ff; color: #00f2ff; box-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
    }
    .stProgress > div > div > div > div { background-color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 資料與邏輯引擎 (V70 深度核心)
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        # V70: 擴充宏觀因子以支援連動分析
        return {
            "^VIX": {"name": "VIX 恐慌", "type": "Sentiment"},
            "DX-Y.NYB": {"name": "DXY 美元", "type": "Currency"},
            "TLT": {"name": "TLT 美債", "type": "Rates"},
            "^SOX": {"name": "SOX 費半", "type": "Equity"},
            "^NDX": {"name": "NDX 那指", "type": "Equity"}
        }

    @staticmethod
    def get_correlation_impact(ticker, macro_data):
        """[V70] 計算資產連動影響分"""
        impact_score = 0
        # 簡單定義關聯 (正相關/負相關)
        if any(x in ticker for x in [".TW", ".TWO"]): weights = {'^SOX': 0.5, '^NDX': 0.3, 'DX-Y.NYB': -0.3}
        elif "-USD" in ticker: weights = {'^NDX': 0.6, 'DX-Y.NYB': -0.4}
        elif "=F" in ticker: weights = {'DX-Y.NYB': -0.6, 'TLT': 0.4}
        else: weights = {'^NDX': 0.5, '^VIX': -0.3}

        for key, w in weights.items():
            if key in macro_data:
                trend = macro_data[key]['trend']
                if w > 0: # 正相關
                    if "Bull" in trend: impact_score += 10 * abs(w)
                    elif "Bear" in trend: impact_score -= 10 * abs(w)
                else: # 負相關
                    if "Bull" in trend: impact_score -= 10 * abs(w)
                    elif "Bear" in trend: impact_score += 10 * abs(w)
        return int(impact_score)

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_full_market():
        tickers = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            for m, s in [(2, '.TW'), (4, '.TWO')]:
                res = requests.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}", headers=headers, timeout=10)
                if res.status_code == 200:
                    df = pd.read_html(res.text)[0]
                    for item in df.iloc[:, 0].astype(str):
                        parts = item.split()
                        if len(parts)>=1 and len(parts[0])==4 and parts[0].isdigit(): tickers.append(f"{parts[0]}{s}")
            if len(tickers)<50: raise Exception("Blocked")
            random.shuffle(tickers)
            return tickers
        except:
            # V67: 靜態備援清單
            return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "3231.TW", "2376.TW", "2356.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW", "2882.TW", "1519.TW"]

    @staticmethod
    def get_scan_list(market_type, limit=0):
        if "台股" in market_type:
            full = Global_Market_Loader.get_tw_full_market()
            return full[:limit] if limit > 0 else full
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "AVGO", "TSM", "SOXL", "TQQQ"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "PEPE-USD"]
        elif "貴金屬" in market_type: return ["GC=F", "SI=F", "HG=F", "CL=F"]
        return []

class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']
            rs = c.diff().pipe(lambda x: x.where(x>0,0).rolling(14).mean() / (-x.where(x<0,0).rolling(14).mean()))
            rsi = 100 - (100/(1+rs)).iloc[-1]
            try: w2 = wasserstein_distance(np.log(c).diff().dropna().tail(20), np.log(c).diff().dropna().iloc[-40:-20])
            except: w2 = 0.5
            chaos = w2 / (np.log(c).diff().dropna().rolling(40).std().mean()*0.1 + 1e-9)
            
            trend = "Neutral"
            if rsi > 60: trend = "Bullish/High"
            elif rsi < 40: trend = "Bearish/Low"
            return {"name": name, "price": c.iloc[-1], "trend": trend, "chaos": chaos}
        except: return None

class Scanner_Engine:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df)<60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            c = df['Close']; v = df['Volume']
            if len(v)>0 and v.iloc[-1]==0: return None
            ma20 = c.rolling(20).mean().iloc[-1]; ma60 = c.rolling(60).mean().iloc[-1]
            if not (c.iloc[-1] > ma20 > ma60): return None
            rs = c.diff().pipe(lambda x: x.where(x>0,0).rolling(14).mean() / (-x.where(x<0,0).rolling(14).mean()))
            rsi = 100 - (100/(1+rs)).iloc[-1]
            score = 40 + (20 if 55<=rsi<=75 else (10 if rsi>75 else 0))
            if v.iloc[-1] > v.rolling(5).mean().iloc[-1]*1.3: score += 15
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1]-2.5*atr, ma20*0.98)
            if score < min_score: return None
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "rsi": rsi, "sl": sl}
        except: return None

class Micro_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df, 0
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; score = 50; signals = []
            ema20 = c.ewm(span=20).mean()
            atr = (df['High']-df['Low']).rolling(14).mean()
            k_upper = ema20 + 2.0 * atr.rolling(10).mean()
            k_lower = ema20 - 2.0 * atr.rolling(10).mean()
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner Breakout")
            obv = (np.sign(c.diff()) * df['Volume']).fillna(0).cumsum()
            if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV Bullish")
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
        
        # [V70] 混合分數信心加權
        confidence = hybrid_score / 100.0
        size = int((risk/dist) * (0.5 if vol_cap>0.8 else 1.0) * confidence)
        if vol_cap>0.8: size = round((risk/dist)*0.5*confidence, 4)
        return size, {"risk": int(risk), "type": atype, "cap": int(size*price), "conf": round(confidence, 2)}

# =============================================================================
# MAIN UI (V71)
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 系統核心 (System)")
    mode = st.sidebar.radio("模式 (Mode)", ["☁️ 線上掃描 (Live)", "📂 匯入報告 (Import)"])
    capital = st.sidebar.number_input("本金 (Capital)", value=1000000, step=100000)
    
    # [V71] 影片回歸
    st.sidebar.markdown("---")
    video_file = "demo.mp4"
    if os.path.exists(video_file): 
        with st.sidebar.expander("🎥 系統演示 (Demo)", expanded=True):
            st.video(video_file)
    else:
        st.sidebar.info("⚠️ 影片檔 model_arch.mp4.mp4 未上傳")

    # --- Header ---
    st.markdown("<h1 style='text-align:center; color:#00f2ff; text-shadow: 0 0 15px rgba(0,242,255,0.6);'>👁️ MARCS V71 奇點戰情室</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b949e; letter-spacing:3px;'>THE SINGULARITY EDITION</p>", unsafe_allow_html=True)

    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "macro_data" not in st.session_state: st.session_state.macro_data = {}
    if "target" not in st.session_state: st.session_state.target = "BTC-USD"

    # =================================================
    # ZONE 1: 宏觀矩陣 (V70 Logic + V66 Visuals)
    # =================================================
    st.markdown("### 📡 1. 宏觀矩陣 (Macro Matrix)")
    if st.button("🔄 初始化全域數據 (INITIALIZE)"):
        with st.spinner("Establishing Uplink..."):
            macro_res = {}
            cols = st.columns(5) # 5 columns for expanded indices
            idx = 0
            for t, info in Global_Market_Loader.get_indices().items():
                r = Macro_Engine.analyze(t, info['name'])
                if r:
                    macro_res[t] = r
                    clr = "#f85149" if "High" in r['trend'] else ("#3fb950" if "Low" in r['trend'] else "#888")
                    with cols[idx]:
                        st.markdown(f"""<div class="metric-card" style="border-top:2px solid {clr}">
                            <div class="metric-label">{r['name']}</div>
                            <div class="metric-value" style="font-size:20px">{r['price']:.2f}</div>
                            <div class="metric-label" style="color:{clr}">{r['trend']}</div>
                        </div>""", unsafe_allow_html=True)
                    idx += 1
            st.session_state.macro_data = macro_res
    
    if not st.session_state.macro_data:
        st.warning("⚠️ 請先點擊上方按鈕初始化宏觀數據，以啟用 Impact Analysis。")

    # =================================================
    # ZONE 2: 雙模掃描 (V69 Architecture)
    # =================================================
    st.markdown("---")
    st.markdown("### 🔭 2. 獵殺雷達 (Hunter Radar)")
    
    if mode == "☁️ 線上掃描 (Live)":
        c1, c2 = st.columns([1, 2])
        with c1:
            market = st.selectbox("戰場選擇", ["🇹🇼 台股", "🇺🇸 美股", "₿ 加密", "🥇 貴金屬"])
            limit = 0
            if "台股" in market and st.checkbox("限制數量 (加速)", value=True): limit = st.slider("上限", 100, 2000, 300)
            
            if st.button("🚀 啟動掃描"):
                tickers = Global_Market_Loader.get_scan_list(market, limit)
                res = []
                bar = st.progress(0); txt = st.empty()
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
                    futures = {exe.submit(Scanner_Engine.analyze_single, t, 60): t for t in tickers}
                    done = 0
                    for f in concurrent.futures.as_completed(futures):
                        r = f.result(); done += 1
                        if r: res.append(r)
                        bar.progress(done/len(tickers))
                        txt.text(f"Scanning: {done}/{len(tickers)} | Hits: {len(res)}")
                st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                bar.empty(); txt.empty()
        with c2:
            if st.session_state.scan_results:
                df = pd.DataFrame(st.session_state.scan_results)
                st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True, height=200)
                if st.button("分析 Top 1"): st.session_state.target = df.iloc[0]['ticker']
    
    else: # Import Mode
        uploaded = st.file_uploader("上傳 CSV (marcs_scan_report.csv)", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            # 兼容處理：確保欄位名正確
            df.columns = [c.lower() for c in df.columns] 
            df.rename(columns={'stoploss':'sl'}, inplace=True) 
            st.session_state.scan_results = df.to_dict('records')
            st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True, height=200)

    # =================================================
    # ZONE 3: 全知分析 (V70 Logic + V66 Visuals)
    # =================================================
    st.markdown("---")
    col_in, col_go = st.columns([3, 1])
    with col_in: manual = st.text_input("戰術目標代碼 (Manual Input):", value=st.session_state.target).upper()
    with col_go: 
        st.write(""); st.write("")
        if st.button("執行精密打擊"): st.session_state.target = manual

    target = st.session_state.target
    if target:
        st.markdown(f"### 🎯 3. 全知分析報告: {target}")
        with st.spinner(f"Connecting to Matrix for {target}..."):
            # A. 微觀
            m_score, sigs, df_m, atr = Micro_Engine.analyze(target)
            
            # B. 宏觀影響 (V70 Feature)
            impact = 0
            if st.session_state.macro_data:
                impact = Global_Market_Loader.get_correlation_impact(target, st.session_state.macro_data)
            
            # C. 混合分數
            hybrid = m_score + impact
            
            # D. 價格數據 (優先用即時)
            info = next((r for r in st.session_state.scan_results if r['ticker'] == target), None)
            if not df_m.empty:
                curr_p = df_m['Close'].iloc[-1]
                sl_p = curr_p - 2.5 * atr
            elif info:
                curr_p = info['price']; sl_p = info['sl']
            else: curr_p = 0
            
            if curr_p > 0:
                size, dets = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)
                
                # 視覺化 (Glassmorphism Cards)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"""<div class="metric-card"><div class="metric-label">微觀技術分</div><div class="metric-value">{m_score}</div><div class="metric-label">{', '.join(sigs)}</div></div>""", unsafe_allow_html=True)
                with c2: 
                    clr = "#3fb950" if impact>0 else "#f85149"
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">宏觀影響修正</div><div class="metric-value" style="color:{clr}">{impact}</div></div>""", unsafe_allow_html=True)
                with c3: 
                    st.markdown(f"""<div class="metric-card" style="border-color:#00f2ff"><div class="metric-label">MARCS 混合總分</div><div class="metric-value" style="color:#00f2ff">{hybrid}</div></div>""", unsafe_allow_html=True)
                with c4: st.markdown(f"""<div class="metric-card"><div class="metric-label">建議倉位 ({dets['type']})</div><div class="metric-value">{size}</div><div class="metric-label">Conf: {int(dets['conf']*100)}%</div></div>""", unsafe_allow_html=True)
                
                # Chart
                fig, ax = plt.subplots(figsize=(12, 5))
                sub = df_m.tail(120)
                ax.plot(sub.index, sub['Close'], color='#e6edf3', lw=1.5)
                ax.plot(sub.index, sub['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                ax.plot(sub.index, sub['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                ax.fill_between(sub.index, sub['K_Upper'], sub['K_Lower'], color='#00f2ff', alpha=0.1)
                ax.axhline(sl_p, color='#f85149', ls='-', label='SL')
                ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
                ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2)
                st.pyplot(fig)
            else: st.error("Data Unavailable")

if __name__ == "__main__":
    main()
