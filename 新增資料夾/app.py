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
# 0. 視覺核心 (V57 星際美學回歸)
# =============================================================================
st.set_page_config(page_title="MARCS V66 星際旗艦版", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* 背景與字體 */
    .stApp {
        background-color: #050505;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* 動態星空特效 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 3px);
        background-size: 550px 550px, 350px 350px, 250px 250px;
        background-position: 0 0, 40px 60px, 130px 270px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.8;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 毛玻璃卡片 (Glassmorphism) */
    .metric-card {
        background: rgba(20, 20, 25, 0.7);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: rgba(88, 166, 255, 0.6); }

    /* 文字與數據 */
    .metric-label { color: #8b949e; font-family: 'Roboto Mono'; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #ffffff; font-size: 26px; font-weight: 700; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
    .metric-sub { font-size: 12px; margin-top: 5px; font-family: 'Roboto Mono'; }

    /* 按鈕與組件 */
    .stButton>button { background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%); color: #000; font-weight: bold; border: none; }
    .stProgress > div > div > div > div { background-color: #00f2ff; }
    
    /* 修正表格樣式 */
    [data-testid="stDataFrame"] { background: rgba(20, 20, 25, 0.8); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 後端引擎 (V65 強力核心)
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {
            "^VIX": {"name": "VIX 恐慌", "type": "Sentiment"},
            "DX-Y.NYB": {"name": "DXY 美元", "type": "Currency"},
            "TLT": {"name": "TLT 美債", "type": "Rates"},
            "JPY=X": {"name": "JPY 日圓", "type": "Currency"}
        }

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_full_market():
        tickers = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            for mode, suffix in [(2, '.TW'), (4, '.TWO')]:
                url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
                res = requests.get(url, headers=headers, timeout=10)
                df = pd.read_html(res.text)[0]
                raw_col = df.iloc[:, 0].astype(str)
                for item in raw_col:
                    parts = item.split()
                    if len(parts) >= 1 and len(parts[0]) == 4 and parts[0].isdigit():
                        tickers.append(f"{parts[0]}{suffix}")
            if len(tickers) < 100: raise Exception("Crawl failed")
            random.shuffle(tickers)
            return tickers
        except:
            return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "3231.TW", "2376.TW", "2356.TW", "3035.TWO", "8069.TWO", "3293.TWO", "3017.TW"]

    @staticmethod
    def get_scan_list(market_type, limit=0):
        if "台股" in market_type:
            full = Global_Market_Loader.get_tw_full_market()
            return full[:limit] if limit > 0 else full
        elif "美股" in market_type:
            return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "AVGO", "TSM", "SOXL", "TQQQ"]
        elif "加密貨幣" in market_type:
            return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "PEPE-USD"]
        elif "貴金屬" in market_type:
            return ["GC=F", "SI=F", "HG=F", "CL=F"]
        return []

class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']
            
            # RSI
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Chaos
            returns = np.log(c).diff().dropna()
            try: w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            except: w2 = 0.5
            chaos = w2 / (returns.rolling(40).std().mean() * 0.1 + 1e-9)
            
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {"ticker": ticker, "name": name, "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend}
        except: return None

    @staticmethod
    def calculate_mmi(results):
        score = 50.0
        d = {r['ticker']: r for r in results if r}
        if d.get('^VIX'): score += 15 if d['^VIX']['trend']=='Overbought' else (-15 if d['^VIX']['trend']=='Oversold' else 0)
        if d.get('DX-Y.NYB'): score -= 12 if d['DX-Y.NYB']['trend']=='Overbought' else (12 if d['DX-Y.NYB']['trend']=='Oversold' else 0)
        return min(100, max(0, score))

class Scanner_Engine_V38:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
            c = df['Close']; v = df['Volume']
            if len(v)>0 and v.iloc[-1] == 0: return None
            
            ma20 = c.rolling(20).mean().iloc[-1]
            ma60 = c.rolling(60).mean().iloc[-1]
            if not (c.iloc[-1] > ma20 > ma60): return None
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            score = 40
            if 55 <= rsi <= 75: score += 20
            elif rsi > 75: score += 10
            
            if v.iloc[-1] > v.rolling(5).mean().iloc[-1] * 1.3: score += 15
            
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1] - 2.5 * atr, ma20 * 0.98)
            
            if score < min_score: return None
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "rsi": rsi, "sl": sl}
        except: return None

class Micro_Structure_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            score = 50; signals = []
            
            ema20 = c.ewm(span=20).mean()
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr10 = tr.rolling(10).mean()
            k_upper = ema20 + 2.0 * atr10
            k_lower = ema20 - 2.0 * atr10
            
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner Breakout")
            if c.iloc[-1] > c.iloc[-2] * 1.015: score += 5; signals.append("Power Candle")
            
            obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
            if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV Bullish")
            
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return min(100, max(0, score)), signals, df
        except: return 50, [], pd.DataFrame()

class Antifragile_Position_Sizing:
    @staticmethod
    def calculate(capital, price, sl, ticker):
        if any(x in ticker for x in ["-USD", "BTC", "ETH"]): vol_cap = 1.0; asset_type = "Crypto"
        elif "=F" in ticker: vol_cap = 0.4; asset_type = "Metal"
        elif any(x in ticker for x in [".TW", ".TWO"]): vol_cap = 0.5; asset_type = "TW Stock"
        else: vol_cap = 0.6; asset_type = "US Stock"

        risk = capital * 0.02
        dist = price - sl
        if dist <= 0: return 0, {}
        
        size = int((risk / dist) * (0.5 if vol_cap > 0.8 else 1.0)) # 簡化版 Taleb 係數
        if vol_cap > 0.8: size = round((risk / dist) * 0.5, 4) # Crypto 小數點
        
        return size, {"risk": int(risk), "asset_type": asset_type, "capital": int(size * price)}

# =============================================================================
# MAIN UI (V57 架構回歸)
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 系統核心 (System Core)")
    capital = st.sidebar.number_input("總本金 (Capital)", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    market_select = st.sidebar.selectbox("雷達掃描範圍", ["🇹🇼 台股 (全市場)", "🇺🇸 美股", "₿ 加密貨幣", "🥇 貴金屬"])
    scan_limit = 0
    if "台股" in market_select:
        use_limit = st.sidebar.checkbox("限制掃描數量 (加速)", value=True)
        if use_limit: scan_limit = st.sidebar.slider("上限", 100, 2000, 300)
    
    # 影片模組
    st.sidebar.markdown("---")
    video_file = "demo.mp4"
    if os.path.exists(video_file): 
        with st.sidebar.expander("🎥 系統架構 (Architecture)"):
            st.video(video_file)

    # --- Header (V57 Style) ---
    st.markdown("<h1 style='text-align:center; color:#00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.5);'>🛡️ MARCS V66 星際旗艦版</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b949e; letter-spacing:2px;'>QUANTUM MACRO INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)

    # Session State
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "target" not in st.session_state: st.session_state.target = "BTC-USD" # 預設

    # =================================================
    # ZONE 1: 宏觀天候 (The Weather) - 始終顯示
    # =================================================
    st.markdown("### 📡 MACRO WEATHER")
    if st.button("🔄 REFRESH MACRO DATA", type="secondary"):
        with st.spinner("Syncing Global Exchanges..."):
            res_list = []
            cols = st.columns(4)
            for idx, (t, info) in enumerate(Global_Market_Loader.get_indices().items()):
                r = Macro_Engine.analyze(t, info['name'])
                res_list.append(r)
                if r:
                    clr = "#f85149" if r['trend']=='Overbought' else ("#3fb950" if r['trend']=='Oversold' else "#8b949e")
                    with cols[idx%4]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top:2px solid {clr}">
                            <div class="metric-label">{r['name']}</div>
                            <div class="metric-value">{r['price']:.2f}</div>
                            <div class="metric-sub" style="color:{clr}">{r['trend']}</div>
                        </div>""", unsafe_allow_html=True)
            mmi = Macro_Engine.calculate_mmi(res_list)
            st.markdown(f"<div style='margin-top:10px; text-align:center; color:#8b949e'>MMI RISK INDEX: <b style='color:#00f2ff'>{mmi:.1f}</b></div>", unsafe_allow_html=True)

    # =================================================
    # ZONE 2: 掃描雷達 (The Scanner) - 折疊式設計
    # =================================================
    st.markdown("---")
    with st.expander("🔭 啟動掃描雷達 (ACTIVATE SCANNER)", expanded=False):
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button(f"🚀 掃描 {market_select}"):
                with st.spinner("Deploying Hunter Drones..."):
                    tickers = Global_Market_Loader.get_scan_list(market_select, scan_limit)
                    results = []
                    bar = st.progress(0); txt = st.empty()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                        future_to_ticker = {executor.submit(Scanner_Engine_V38.analyze_single, t, 60): t for t in tickers}
                        completed = 0
                        for future in concurrent.futures.as_completed(future_to_ticker):
                            r = future.result()
                            if r: results.append(r)
                            completed += 1
                            bar.progress(completed/len(tickers))
                            txt.text(f"Scanning: {completed}/{len(tickers)} | Hits: {len(results)}")
                    
                    st.session_state.scan_results = sorted(results, key=lambda x: x['score'], reverse=True)
                    bar.empty(); txt.empty()
        
        with c2:
            if st.session_state.scan_results:
                df = pd.DataFrame(st.session_state.scan_results)
                st.dataframe(df[['ticker', 'score', 'price', 'rsi', 'sl']], use_container_width=True, height=200)
                # 掃描後的連動選擇
                scan_target = st.selectbox("👉 從掃描結果中選擇標的 (Select Target):", [r['ticker'] for r in st.session_state.scan_results])
                if st.button("分析選定標的 (Analyze Selection)"):
                    st.session_state.target = scan_target

    # =================================================
    # ZONE 3: 戰術面板 (Tactical Panel) - 核心展示區
    # =================================================
    st.markdown("---")
    
    # 混合輸入區：可以手動打，也可以接收掃描結果
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        manual_input = st.text_input("戰術目標代碼 (MANUAL TARGET INPUT):", value=st.session_state.target).upper()
    with col_btn:
        st.write(""); st.write("")
        if st.button("執行精密打擊 (EXECUTE)"):
            st.session_state.target = manual_input

    # 執行深度分析
    target = st.session_state.target
    if target:
        with st.spinner(f"Decoding Market Structure for {target}..."):
            m_score, sigs, df_m = Micro_Structure_Engine.analyze(target)
            
            # 獲取價格與停損 (優先從掃描結果拿，沒有則重算)
            scan_info = next((r for r in st.session_state.scan_results if r['ticker'] == target), None)
            if scan_info:
                curr_p = scan_info['price']; sl_p = scan_info['sl']
            elif not df_m.empty:
                curr_p = df_m['Close'].iloc[-1]
                atr = (df_m['High']-df_m['Low']).rolling(14).mean().iloc[-1]
                sl_p = curr_p - 2.5 * atr
            else:
                curr_p = 0
            
            if curr_p > 0:
                size, dets = Antifragile_Position_Sizing.calculate(capital, curr_p, sl_p, target)
                
                # 視覺化呈現 (V57 Style)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">MICRO SCORE</div>
                        <div class="metric-value" style="color:{'#3fb950' if m_score>60 else '#f85149'}">{m_score}</div>
                        <div class="metric-sub">{', '.join(sigs) if sigs else 'NEUTRAL'}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">POSITION SIZE ({dets['asset_type']})</div>
                        <div class="metric-value">{size} <span style="font-size:14px">UNIT</span></div>
                        <div class="metric-sub" style="color:#00f2ff">Est. Capital: ${dets['capital']:,}</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">INTELLIGENT SL</div>
                        <div class="metric-value" style="color:#f85149">{sl_p:.2f}</div>
                        <div class="metric-sub">Risk Amount: -${dets['risk']}</div>
                    </div>""", unsafe_allow_html=True)
                
                # Chart
                st.markdown("#### 📊 TACTICAL CHART (Keltner Channel)")
                fig, ax = plt.subplots(figsize=(12, 5))
                sub = df_m.tail(120)
                ax.plot(sub.index, sub['Close'], color='#e6edf3', lw=1.5, label='Price')
                ax.plot(sub.index, sub['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                ax.plot(sub.index, sub['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                ax.fill_between(sub.index, sub['K_Upper'], sub['K_Lower'], color='#00f2ff', alpha=0.1)
                ax.axhline(sl_p, color='#f85149', ls='-', lw=1.5, label='Stop Loss')
                
                ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
                ax.tick_params(colors='#8b949e'); ax.grid(True, color='#30363d', alpha=0.3)
                ax.legend(frameon=False, labelcolor='white')
                st.pyplot(fig)
            else:
                st.error("DATA UNAVILABLE. CHECK TICKER SYMBOL.")

if __name__ == "__main__":
    main()
