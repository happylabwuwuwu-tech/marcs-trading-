import sys
import os
import types
import warnings
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# =============================================================================
# 0. 系統補丁 & Imports
# =============================================================================
try:
    import distutils.version
except ImportError:
    if 'distutils' not in sys.modules:
        sys.modules['distutils'] = types.ModuleType('distutils')
    if 'distutils.version' not in sys.modules:
        sys.modules['distutils.version'] = types.ModuleType('distutils.version')
    try:
        from packaging.version import Version as LooseVersion
    except ImportError:
        class LooseVersion:
            def __init__(self, vstring): self.vstring = vstring
            def __ge__(self, other): return str(self.vstring) >= str(other.vstring)
            def __lt__(self, other): return str(self.vstring) < str(other.vstring)
    sys.modules['distutils.version'].LooseVersion = LooseVersion

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from scipy.stats import norm
from FinMind.data import DataLoader

# 兼容性處理
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        return np.mean(np.abs(np.sort(u_values) - np.sort(v_values)))

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 視覺核心
# =============================================================================
st.set_page_config(page_title="MARCS OMEGA TRINITY", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 全局 */
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; color: #c9d1d9; }
    
    /* 星空背景 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.6;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 元件樣式 */
    .signal-box { background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95)); border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; backdrop-filter: blur(10px); }
    .big-signal { font-size: 42px; font-weight: 800; margin: 10px 0; font-family: 'JetBrains Mono'; }
    .metric-card { background: rgba(18, 18, 22, 0.85); border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    .highlight-val { font-size: 24px; font-weight: bold; color: #e6edf3; font-family: 'JetBrains Mono'; }
    .dna-bar-bg { width: 100%; background: #21262d; height: 6px; border-radius: 3px; margin-top: 5px; }
    .dna-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    
    /* 側邊欄與標籤 */
    .stats-sidebar { background-color: rgba(13, 17, 23, 0.8); border-left: 1px solid #30363d; padding: 15px; height: 100%; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0 0; color: #8b949e; font-weight: bold; }
    .stTabs [aria-selected="true"] { background-color: rgba(30, 30, 35, 0.5); color: #d2a8ff; border-bottom: 2px solid #d2a8ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. 數據與物理層 (核心引擎)
# =============================================================================
@st.cache_data(ttl=3600)
def robust_download(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False) # 單線程保平安
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

class Causal_Physics_Engine:
    @staticmethod
    def rolling_hilbert(series, window=64):
        values = series.values
        n = len(values)
        analytic_signal = np.zeros(n, dtype=complex)
        for i in range(window, n):
            segment = values[i-window : i] * np.hanning(window)
            analytic_signal[i] = hilbert(segment)[-1]
        return analytic_signal

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def calc_metrics_cached(df):
        if df.empty or len(df) < 100: return df
        df = df.copy()
        c = df['Close']; v = df['Volume']
        
        # 1. Causal Sync
        ema = c.ewm(span=20).mean()
        detrend_c = (c - ema).fillna(0)
        detrend_v = (v - v.rolling(20).mean()).fillna(0)
        
        analytic_c = Causal_Physics_Engine.rolling_hilbert(detrend_c, window=64)
        analytic_v = Causal_Physics_Engine.rolling_hilbert(detrend_v, window=64)
        phase_c = np.angle(analytic_c)
        phase_v = np.angle(analytic_v)
        
        sync_raw = np.cos(phase_c - phase_v)
        sync_raw[:64] = 0 
        df['Sync_Smooth'] = pd.Series(sync_raw).rolling(5).mean().fillna(0)
        
        # 2. VPIN
        delta_p = c.diff()
        sigma = delta_p.rolling(20).std() + 1e-9
        cdf = norm.cdf(delta_p / sigma)
        oi = (v * cdf - v * (1 - cdf)).abs()
        total_vol = v.rolling(20).sum() + 1e-9
        df['VPIN'] = (oi.rolling(20).sum() / total_vol).fillna(0)
        
        # 3. Technicals
        df['EMA20'] = c.ewm(span=20).mean()
        df['EMA50'] = c.ewm(span=50).mean()
        df['ATR'] = (df['High']-df['Low']).rolling(14).mean()
        
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

# =============================================================================
# 3. 外部數據與戰術層
# =============================================================================
class FinMind_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_data(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        # 請填入 Token
        USER_TOKEN = "" 
        try:
            stock_id = ticker.split('.')[0]
            api = DataLoader()
            if USER_TOKEN: api.login_by_token(api_token=USER_TOKEN)
            
            data = {"chips": 0, "pe": None, "growth": None}
            # 簡單模擬，實戰請解開註解
            # df = api.taiwan_stock_institutional_investors(...)
            return data
        except: return None

class News_Intel_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_news(ticker):
        items = []; sentiment_score = 0
        try:
            query = ticker.split('.')[0]
            lang = "hl=zh-TW&gl=TW" if ".TW" in ticker else "hl=en-US&gl=US"
            url = f"https://news.google.com/rss/search?q={query}&{lang}"
            resp = requests.get(url, timeout=2) # Fast timeout
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title').text
                    date = item.find('pubDate').text[:16]
                    items.append({"title": title, "date": date, "link": item.find('link').text})
                    if any(x in title for x in ["漲","High","Bull"]): sentiment_score += 1
            return items, max(-1, min(1, sentiment_score/3))
        except: return [], 0

class SMC_Engine:
    @staticmethod
    def identify_fvg(df):
        fvgs = []
        if len(df) < 5: return []
        for i in range(len(df)-2, len(df)-30, -1):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({'type': 'Bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'date': df.index[i-2]})
            elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({'type': 'Bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'date': df.index[i-2]})
        return fvgs[:3]

# =============================================================================
# 4. 分析整合 (Universal Analyst)
# =============================================================================
class Universal_Analyst:
    @staticmethod
    def analyze(ticker, fast_mode=False):
        # 1. 數據
        period = "6mo" if fast_mode else "2y"
        df = robust_download(ticker, period)
        if df.empty or len(df) < 60: return None
        
        # 2. 物理
        df = Causal_Physics_Engine.calc_metrics_cached(df)
        df = df.fillna(method='bfill')
        last = df.iloc[-1]
        
        # 3. DNA Score
        dna = {}
        
        # Trend
        trend_s = 50
        if last['Close'] > last['EMA20']: trend_s += 20
        if last['EMA20'] > last['EMA50']: trend_s += 20
        dna['Trend'] = min(trend_s, 100)
        
        # Momentum (Z-Score)
        rsi_z = (last['RSI'] - df['RSI'].tail(60).mean()) / (df['RSI'].tail(60).std() + 1e-9)
        dna['Momentum'] = min(max(50 + rsi_z*20, 0), 100)
        
        # Physics
        dna['Physics'] = min(max(50 + last['Sync_Smooth']*40, 0), 100)
        
        # Flow (Simplified for speed)
        dna['Flow'] = min(max(100 - last['VPIN']*100, 0), 100)
        
        # Value & Stability
        dna['Value'] = min(max(100 - last['RSI'], 0), 100) # Proxy
        atr_z = (last['ATR'] - df['ATR'].tail(60).mean()) / (df['ATR'].tail(60).std() + 1e-9)
        dna['Stability'] = min(max(50 - atr_z*20, 0), 100)
        
        avg_score = np.mean(list(dna.values()))
        
        # 4. 如果是詳細模式，抓取更多資料
        fvgs = []
        news = []
        sent = 0
        fm_data = None
        
        if not fast_mode:
            fvgs = SMC_Engine.identify_fvg(df)
            news, sent = News_Intel_Engine.fetch_news(ticker)
            fm_data = FinMind_Engine.get_tw_data(ticker)
            # 修正 DNA (如有額外數據)
            dna['Momentum'] += sent * 10
            dna['Momentum'] = min(max(dna['Momentum'], 0), 100)
            
        return {
            "df": df, "last": last, "dna": dna, "score": avg_score,
            "fvgs": fvgs, "news": news, "sent": sent, "fm_data": fm_data
        }

# =============================================================================
# 5. UI 模組 (Trinity Layout)
# =============================================================================
def render_macro_oracle():
    st.markdown("### 🌍 Macro Oracle (宏觀預言機)")
    col1, col2, col3, col4 = st.columns(4)
    
    # 模擬宏觀數據 (因為 yfinance 多次請求會卡)
    # 實戰中請使用 robust_download 抓取 ^VIX, DX-Y.NYB, TLT
    vix = 21.5; dxy = 104.2; tlt = 92.5; crypto_corr = 0.85
    
    regime = "NEUTRAL"
    c_reg = "#888"
    if vix > 25: regime = "FEAR (避險)"; c_reg = "#f85149"
    elif vix < 15 and dxy < 103: regime = "GOLDILOCKS (看多)"; c_reg = "#3fb950"
    
    col1.markdown(f"<div class='metric-card'><div class='highlight-lbl'>REGIME</div><div class='highlight-val' style='color:{c_reg}'>{regime}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='highlight-lbl'>VIX (Fear)</div><div class='highlight-val'>{vix}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='highlight-lbl'>DXY (USD)</div><div class='highlight-val'>{dxy}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='highlight-lbl'>Risk Corr</div><div class='highlight-val'>{crypto_corr}</div></div>", unsafe_allow_html=True)
    
    st.info("💡 宏觀建議: 當 VIX > 25 時，減少部位；當 DXY 突破 105 時，現金為王。")

def render_quantum_scanner():
    st.markdown("### 🔭 Quantum Scanner (量子掃描器)")
    
    market = st.selectbox("選擇市場", ["🇹🇼 台股 (TW)", "🇺🇸 美股 (US)", "🪙 加密 (Crypto)"])
    if "TW" in market: tickers = ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW"]
    elif "US" in market: tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN"]
    else: tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
    
    if st.button("🚀 啟動掃描 (Fast Mode)"):
        results = []
        bar = st.progress(0)
        
        # 單線程安全掃描
        for i, t in enumerate(tickers):
            try:
                res = Universal_Analyst.analyze(t, fast_mode=True)
                if res:
                    results.append({
                        "Ticker": t,
                        "Price": res['last']['Close'],
                        "Score": res['score'],
                        "Sync": res['last']['Sync_Smooth'],
                        "RSI": res['last']['RSI']
                    })
            except: pass
            bar.progress((i+1)/len(tickers))
            
        if results:
            df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(
                df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.warning("No Data Found.")

def render_sovereign_lab():
    st.markdown("### 🛡️ Sovereign Lab (深度分析)")
    ticker = st.text_input("輸入代碼", "2330.TW")
    
    if st.button("Deep Analyze"):
        with st.spinner("Processing Physics & DNA..."):
            res = Universal_Analyst.analyze(ticker, fast_mode=False)
            
        if res is None:
            st.error("Data Insufficient.")
            return
            
        # Layout
        c1, c2 = st.columns([3, 1])
        
        with c1:
            # 戰術板
            score = res['score']
            sig = "WAIT"; color="#888"
            if score >= 70: sig="BUY"; color="#3fb950"
            elif score <= 30: sig="SELL"; color="#f85149"
            
            st.markdown(f"""
            <div class="signal-box" style="border-top: 4px solid {color}">
                <div style="color:#aaa; font-size:14px">TACTICAL SIGNAL</div>
                <div class="big-signal" style="color:{color}">{sig}</div>
                <div>Score: {score:.0f} | Price: ${res['last']['Close']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 圖表
            df = res['df']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#ffae00', width=1), name='EMA20'), row=1, col=1)
            
            for f in res['fvgs']:
                c = "rgba(0,255,0,0.2)" if f['type']=='Bull' else "rgba(255,0,0,0.2)"
                fig.add_shape(type="rect", x0=f['date'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor=c, line_width=0, row=1, col=1)
                
            fig.add_trace(go.Scatter(x=df.index, y=df['Sync_Smooth'], line=dict(color='#d2a8ff', width=2), name='Phase Sync'), row=2, col=1)
            fig.add_hrect(y0=0.5, y1=1.0, row=2, col=1, fillcolor="#d2a8ff", opacity=0.1, line_width=0)
            fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="#555")
            
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### 🧬 DNA")
            for k, v in res['dna'].items():
                c = "#3fb950" if v>60 else "#f85149" if v<40 else "#8b949e"
                st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:12px'><span style='color:#aaa'>{k}</span><span style='color:{c}; font-weight:bold'>{v:.0f}</span></div><div class='dna-bar-bg'><div class='dna-bar-fill' style='width:{v}%; background:{c}'></div></div>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown("#### 📰 News")
            for item in res['news']:
                st.markdown(f"<div style='background:#222; padding:8px; border-radius:4px; margin-bottom:5px; font-size:12px'><a href='{item['link']}' style='color:#e0e0e0; text-decoration:none'>{item['title']}</a></div>", unsafe_allow_html=True)

# =============================================================================
# 6. 主程序 (Tab Layout)
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ MARCS TRINITY")
    st.sidebar.caption("Macro | Scanner | Lab")
    
    # 使用 Radio 來切換三大模式，這是最穩定的導航方式
    mode = st.sidebar.radio("MODE SELECT", ["🌍 Macro Oracle", "🔭 Quantum Scanner", "🛡️ Sovereign Lab"])
    
    if mode == "🌍 Macro Oracle":
        render_macro_oracle()
    elif mode == "🔭 Quantum Scanner":
        render_quantum_scanner()
    elif mode == "🛡️ Sovereign Lab":
        render_sovereign_lab()

if __name__ == "__main__":
    main()
