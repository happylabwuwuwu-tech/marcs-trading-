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
# 0. 視覺核心 (星際美學)
# =============================================================================
st.set_page_config(page_title="MARCS V69 雙模旗艦版", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    
    /* 動態星空 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.8;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 元件樣式 */
    .metric-card {
        background: rgba(20, 20, 25, 0.8); border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px; padding: 15px; text-align: center; backdrop-filter: blur(10px);
    }
    .metric-value { color: white; font-size: 24px; font-weight: bold; }
    .metric-label { color: #8b949e; font-size: 12px; letter-spacing: 1px; }
    .stButton>button { background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%); color: black; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 共用引擎 (Macro, Micro, Risk)
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {"^VIX": {"name": "VIX", "type": "Sentiment"}, "DX-Y.NYB": {"name": "DXY", "type": "Currency"}, "TLT": {"name": "TLT", "type": "Rates"}, "JPY=X": {"name": "JPY", "type": "Currency"}}

    @staticmethod
    def get_static_backup_list():
        # [安全網] 內建 100+ 檔台股熱門清單
        return [
            "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2603.TW", "2609.TW", "2615.TW",
            "2382.TW", "3231.TW", "2376.TW", "2356.TW", "6669.TW", "3035.TWO",
            "3037.TW", "2368.TW", "3017.TW", "3044.TW", "2498.TW", "3008.TW",
            "2881.TW", "2882.TW", "2891.TW", "5871.TW", "1519.TW", "1513.TW",
            "3529.TWO", "6274.TWO", "8069.TWO", "6147.TWO", "3293.TWO",
            "2412.TW", "1301.TW", "1303.TW", "2002.TW", "2344.TW", "2408.TW"
        ]

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
            # 自動切換備援清單
            return Global_Market_Loader.get_static_backup_list()

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
            trend = "Overbought" if rsi>70 else ("Oversold" if rsi<30 else "Neutral")
            return {"name": name, "price": c.iloc[-1], "trend": trend, "chaos": chaos}
        except: return None

    @staticmethod
    def calculate_mmi(results):
        score = 50.0; d = {r['name']: r for r in results if r}
        if d.get('VIX'): score += 15 if d['VIX']['trend']=='Overbought' else (-15 if d['VIX']['trend']=='Oversold' else 0)
        if d.get('DXY'): score -= 12 if d['DXY']['trend']=='Overbought' else (12 if d['DXY']['trend']=='Oversold' else 0)
        return min(100, max(0, score))

class Scanner_Engine_V38:
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
            if df.empty: return 50, [], df
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; score = 50; signals = []
            
            ema20 = c.ewm(span=20).mean()
            atr10 = (df['High']-df['Low']).rolling(10).mean()
            k_upper = ema20 + 2.0 * atr10; k_lower = ema20 - 2.0 * atr10
            
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner Breakout")
            obv = (np.sign(c.diff()) * df['Volume']).fillna(0).cumsum()
            if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV Bullish")
            
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return score, signals, df
        except: return 50, [], pd.DataFrame()

class Antifragile_Sizing:
    @staticmethod
    def calculate(capital, price, sl, ticker):
        if any(x in ticker for x in ["-USD", "BTC", "ETH"]): vol_cap = 1.0; atype = "Crypto"
        elif "=F" in ticker: vol_cap = 0.4; atype = "Metal"
        elif any(x in ticker for x in [".TW", ".TWO"]): vol_cap = 0.5; atype = "TW Stock"
        else: vol_cap = 0.6; atype = "US Stock"
        
        risk = capital * 0.02; dist = price - sl
        if dist <= 0: return 0, {}
        
        size = int((risk/dist) * (0.5 if vol_cap>0.8 else 1.0))
        if vol_cap>0.8: size = round((risk/dist)*0.5, 4)
        return size, {"risk": int(risk), "type": atype, "cap": int(size*price)}

# =============================================================================
# MAIN APP (雙模架構)
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 系統核心 (Core)")
    mode = st.sidebar.radio("數據來源模式 (Mode)", ["☁️ 線上即時掃描 (Live)", "📂 匯入 Colab 報告 (Import)"])
    capital = st.sidebar.number_input("本金 (Capital)", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    video_file = "model_arch.mp4.mp4"
    if os.path.exists(video_file): 
        with st.sidebar.expander("🎥 系統演示"): st.video(video_file)

    st.markdown("<h1 style='text-align:center; color:#00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.5);'>🛡️ MARCS V69 雙模旗艦版</h1>", unsafe_allow_html=True)

    # Session State
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "target" not in st.session_state: st.session_state.target = "BTC-USD"

    # =================================================
    # ZONE 1: 宏觀 (共用)
    # =================================================
    st.markdown("### 📡 MACRO WEATHER")
    if st.button("🔄 REFRESH MACRO"):
        with st.spinner("Syncing..."):
            res_list = []
            cols = st.columns(4)
            for idx, (t, info) in enumerate(Global_Market_Loader.get_indices().items()):
                r = Macro_Engine.analyze(t, info['name'])
                res_list.append(r)
                if r:
                    clr = "#f85149" if r['trend']=='Overbought' else ("#3fb950" if r['trend']=='Oversold' else "#8b949e")
                    with cols[idx%4]: st.markdown(f"""<div class="metric-card" style="border-top:2px solid {clr}"><div class="metric-label">{r['name']}</div><div class="metric-value">{r['price']:.2f}</div><div class="metric-label" style="color:{clr}">{r['trend']}</div></div>""", unsafe_allow_html=True)
            mmi = Macro_Engine.calculate_mmi(res_list)
            st.markdown(f"<div style='text-align:center; margin-top:10px; color:#8b949e'>MMI RISK INDEX: <b style='color:#00f2ff'>{mmi:.1f}</b></div>", unsafe_allow_html=True)

    # =================================================
    # ZONE 2: 雙模數據源 (Hybrid Input)
    # =================================================
    st.markdown("---")
    
    if mode == "☁️ 線上即時掃描 (Live)":
        with st.expander("🔭 啟動掃描雷達 (ACTIVATE SCANNER)", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                market = st.selectbox("選擇戰場", ["🇹🇼 台股 (全市場)", "🇺🇸 美股", "₿ 加密貨幣", "🥇 貴金屬"])
                limit = 0
                if "台股" in market:
                    if st.checkbox("限制數量 (加速)", value=True): limit = st.slider("上限", 100, 2000, 300)
                
                if st.button("🚀 啟動掃描"):
                    tickers = Global_Market_Loader.get_scan_list(market, limit)
                    res = []
                    bar = st.progress(0); txt = st.empty()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
                        futures = {exe.submit(Scanner_Engine_V38.analyze_single, t, 60): t for t in tickers}
                        done = 0
                        for f in concurrent.futures.as_completed(futures):
                            r = f.result()
                            if r: res.append(r)
                            done += 1
                            bar.progress(done/len(tickers))
                            txt.text(f"Scanning: {done}/{len(tickers)} | Hits: {len(res)}")
                    st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                    bar.empty(); txt.empty()
            
            with c2:
                if st.session_state.scan_results:
                    df = pd.DataFrame(st.session_state.scan_results)
                    st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True, height=200)
                    sel = st.selectbox("👉 選擇分析標的:", [r['ticker'] for r in st.session_state.scan_results])
                    if st.button("分析"): st.session_state.target = sel

    else: # 離線模式
        st.markdown("### 📂 匯入 Colab 掃描報告")
        uploaded = st.file_uploader("上傳 CSV 檔", type=['csv'])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.rename(columns={'Ticker':'ticker', 'Price':'price', 'Score':'score', 'StopLoss':'sl'}, inplace=True)
                st.session_state.scan_results = df.to_dict('records')
                st.success(f"已匯入 {len(df)} 檔資料")
                
                st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True, height=200)
                sel = st.selectbox("👉 選擇分析標的:", df['ticker'].tolist())
                if st.button("分析"): st.session_state.target = sel
            except Exception as e: st.error(f"格式錯誤: {e}")

    # =================================================
    # ZONE 3: 戰術面板 (共用)
    # =================================================
    st.markdown("---")
    col_in, col_go = st.columns([3, 1])
    with col_in: manual = st.text_input("戰術目標代碼 (MANUAL INPUT):", value=st.session_state.target).upper()
    with col_go: 
        st.write(""); st.write("")
        if st.button("執行精密打擊"): st.session_state.target = manual

    target = st.session_state.target
    if target:
        with st.spinner(f"Decoding {target}..."):
            m_score, sigs, df_m = Micro_Engine.analyze(target)
            
            info = next((r for r in st.session_state.scan_results if r['ticker'] == target), None)
            
            if info: curr_p = info['price']; sl_p = info['sl']
            elif not df_m.empty: 
                curr_p = df_m['Close'].iloc[-1]
                atr = (df_m['High']-df_m['Low']).rolling(14).mean().iloc[-1]
                sl_p = curr_p - 2.5 * atr
            else: curr_p = 0
            
            if curr_p > 0:
                size, dets = Antifragile_Sizing.calculate(capital, curr_p, sl_p, target)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"""<div class="metric-card"><div class="metric-label">MICRO SCORE</div><div class="metric-value" style="color:{'#3fb950' if m_score>60 else '#f85149'}">{m_score}</div><div class="metric-label">{', '.join(sigs)}</div></div>""", unsafe_allow_html=True)
                with c2: st.markdown(f"""<div class="metric-card"><div class="metric-label">SIZE ({dets['type']})</div><div class="metric-value">{size}</div><div class="metric-label" style="color:#00f2ff">${dets['cap']:,}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="metric-card"><div class="metric-label">STOP LOSS</div><div class="metric-value" style="color:#f85149">{sl_p:.2f}</div><div class="metric-label">Risk: -${dets['risk']}</div></div>""", unsafe_allow_html=True)
                
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
            else: st.error("DATA UNAVILABLE")

if __name__ == "__main__":
    main()
