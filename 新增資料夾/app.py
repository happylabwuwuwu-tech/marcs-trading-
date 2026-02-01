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
st.set_page_config(page_title="MARCS V73 戰略修復版", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.7;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    .metric-card {
        background: rgba(18, 18, 22, 0.75); 
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 255, 0.15);
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .metric-card:hover { 
        transform: translateY(-5px); 
        border-color: rgba(0, 242, 255, 0.6); 
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }

    .metric-value { color: #fff; font-size: 28px; font-weight: 700; text-shadow: 0 0 10px rgba(255,255,255,0.1); }
    .metric-label { color: #8b949e; font-size: 12px; letter-spacing: 1px; font-family: 'Roboto Mono'; text-transform: uppercase; }
    .metric-sub { font-size: 12px; color: #58a6ff; margin-top: 5px; font-family: 'Roboto Mono'; }
    
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%); color:black; border:none;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 宏觀與資料庫
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {
            "^VIX": {"name": "VIX 恐慌", "type": "Sentiment"},
            "^TNX": {"name": "US10Y 殖利率", "type": "Yield"},
            "JPY=X": {"name": "USD/JPY 匯率", "type": "Currency"},
            "^SOX": {"name": "SOX 費半", "type": "Equity"},
            "DX-Y.NYB": {"name": "DXY 美元", "type": "Currency"}
        }

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
        except:
            return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW", "1519.TW"]

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
# 2. 分析引擎
# =============================================================================
class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']
            ma20 = c.rolling(20).mean()
            trend = "Neutral"
            if c.iloc[-1] > ma20.iloc[-1]: trend = "Bullish/High"
            else: trend = "Bearish/Low"
            return {"name": name, "price": c.iloc[-1], "trend": trend}
        except: return None

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
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            score = 40
            if 55 <= rsi <= 75: score += 20
            elif rsi > 75: score += 10
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
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            score = 50; signals = []
            
            ema20 = c.ewm(span=20).mean()
            atr = (h-l).rolling(14).mean()
            k_upper = ema20 + 2.0 * atr.rolling(10).mean()
            k_lower = ema20 - 2.0 * atr.rolling(10).mean()
            
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner 突破")
            obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
            if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV 強勢")
            
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
        
        risk = capital * 0.02
        dist = price - sl
        if dist <= 0: return 0, {}
        
        conf = hybrid_score / 100.0
        
        # 計算倉位股數/單位
        size = int((risk/dist) * (0.5 if vol_cap>0.8 else 1.0) * conf)
        if vol_cap > 0.8: size = round((risk/dist)*0.5*conf, 4)
        
        # [V73] 計算百分比
        position_value = size * price
        pct_capital = (position_value / capital) * 100
        
        # 限制最大倉位 50% 防止爆倉
        if pct_capital > 50: 
            pct_capital = 50
            size = (capital * 0.5) / price
            if vol_cap <= 0.8: size = int(size)
            position_value = size * price

        return size, {"risk": int(risk), "type": atype, "cap": int(position_value), "pct": round(pct_capital, 1)}

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    # --- Sidebar ---
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
                with st.spinner("Scanning..."):
                    tickers = Global_Market_Loader.get_scan_list(market, limit)
                    res = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
                        futures = {exe.submit(Scanner_Engine_V38.analyze_single, t, 60): t for t in tickers}
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

    # --- Main ---
    st.markdown("<h1 style='text-align:center; color:#00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.5);'>🛡️ MARCS V73 戰略修復版</h1>", unsafe_allow_html=True)

    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "macro_data" not in st.session_state: st.session_state.macro_data = {}
    if "target" not in st.session_state: st.session_state.target = "BTC-USD"

    if run_manual and manual_input: st.session_state.target = manual_input

    # ZONE 1: Macro
    st.markdown("### 📡 1. 全球宏觀 (Macro)")
    if st.button("🔄 同步全球數據 (Update)"):
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
                            <div class="metric-label">{r['name']}</div>
                            <div class="metric-value" style="font-size:20px">{r['price']:.2f}</div>
                            <div class="metric-sub" style="color:{clr}">{r['trend']}</div>
                        </div>""", unsafe_allow_html=True)
                    idx += 1
            st.session_state.macro_data = macro_res

    # ZONE 2: Scanner Result
    if st.session_state.scan_results:
        with st.expander("🔭 掃描結果列表"):
            df = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True)
            sel = st.selectbox("選擇分析:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("分析選定標的"): st.session_state.target = sel

    # ZONE 3: Report
    target = st.session_state.target
    if target:
        st.markdown("---")
        st.markdown(f"### 🎯 深度戰略分析: {target}")
        
        with st.spinner(f"Processing {target}..."):
            m_score, sigs, df_m, atr = Micro_Engine.analyze(target)
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
                
                # --- V73 更新：倉位改成百分比顯示 ---
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"""<div class="metric-card"><div class="metric-label">微觀評分</div><div class="metric-value">{m_score}</div><div class="metric-sub">{', '.join(sigs)}</div></div>""", unsafe_allow_html=True)
                with c2: 
                    clr = "#3fb950" if impact>0 else "#f85149"
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">宏觀修正</div><div class="metric-value" style="color:{clr}">{impact}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="metric-card" style="border-color:#00f2ff"><div class="metric-label">總體評分</div><div class="metric-value" style="color:#00f2ff">{hybrid}</div></div>""", unsafe_allow_html=True)
                
                # 這裡改為顯示 %
                with c4: st.markdown(f"""<div class="metric-card"><div class="metric-label">建議倉位 %</div><div class="metric-value">{dets['pct']}%</div><div class="metric-sub">{size} 股 (${dets['cap']:,})</div></div>""", unsafe_allow_html=True)
                
                # Chart
                st.markdown("#### 📊 戰術圖表 (Tactical Chart)")
                tab1, tab2 = st.tabs(["🕯️ Keltner 主圖", "🌊 趨勢細節 (RSI/OBV)"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sub = df_m.tail(120)
                    ax.plot(sub.index, sub['Close'], color='#e6edf3', lw=1.5, label='Price')
                    ax.plot(sub.index, sub['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                    ax.plot(sub.index, sub['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                    ax.fill_between(sub.index, sub['K_Upper'], sub['K_Lower'], color='#00f2ff', alpha=0.1)
                    ax.axhline(sl_p, color='#f85149', ls='-', label=f'SL: {sl_p:.2f}')
                    ax.legend()
                    ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
                    ax.tick_params(colors='#8b949e'); ax.grid(True, color='#30363d', alpha=0.3)
                    st.pyplot(fig)
                
                # --- V73 更新：補上趨勢細節圖表 ---
                with tab2:
                    st.caption("展示 RSI 動能指標與成交量變化")
                    if not df_m.empty:
                        # 計算指標
                        delta = df_m['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        fig2, (ax_rsi, ax_vol) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 1])
                        sub_ind = df_m.tail(120)
                        rsi_sub = rsi.tail(120)
                        
                        # RSI Plot
                        ax_rsi.plot(rsi_sub.index, rsi_sub, color='#d2a8ff', lw=1.5)
                        ax_rsi.axhline(70, color='#f85149', ls='--', alpha=0.5)
                        ax_rsi.axhline(30, color='#3fb950', ls='--', alpha=0.5)
                        ax_rsi.set_title('Relative Strength Index (RSI)', color='white', fontsize=10)
                        ax_rsi.set_facecolor('#0d1117')
                        ax_rsi.tick_params(colors='#8b949e')
                        
                        # Volume Plot
                        colors = ['#f85149' if o > c else '#3fb950' for o, c in zip(sub_ind['Open'], sub_ind['Close'])]
                        ax_vol.bar(sub_ind.index, sub_ind['Volume'], color=colors, alpha=0.8)
                        ax_vol.set_title('Volume Profile', color='white', fontsize=10)
                        ax_vol.set_facecolor('#0d1117')
                        ax_vol.tick_params(colors='#8b949e')
                        
                        fig2.patch.set_facecolor('#0d1117')
                        st.pyplot(fig2)
            else:
                st.error("無法獲取數據，請確認代碼是否正確。")

if __name__ == "__main__":
    main()
