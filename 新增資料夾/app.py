install lxml
import sys
import os
import types
import warnings
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# =============================================================================
# 0. 系統補丁 & Imports (必要)
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
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from scipy.stats import norm

# 兼容性處理
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        return np.mean(np.abs(np.sort(u_values) - np.sort(v_values)))

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 視覺核心 (CSS)
# =============================================================================
st.set_page_config(page_title="MARCS V160 LEVIATHAN", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
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

    /* 戰術板 */
    .signal-box { background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95)); border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; backdrop-filter: blur(10px); }
    .big-signal { font-size: 42px; font-weight: 800; margin: 10px 0; font-family: 'JetBrains Mono'; }
    
    /* 戰術數據卡 */
    .metric-card { background: rgba(22, 27, 34, 0.85); border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 10px; text-align: left; }
    .highlight-lbl { font-size: 11px; color: #8b949e; letter-spacing: 1px; text-transform: uppercase; font-family: 'Rajdhani'; }
    .highlight-val { font-size: 22px; font-weight: bold; color: #e6edf3; font-family: 'JetBrains Mono'; margin-top: 5px; }
    .smart-text { font-size: 11px; color: #ffb86c; font-family: 'Noto Sans TC'; margin-top: 4px; }
    
    /* DNA Bar */
    .dna-bar-bg { width: 100%; background: #21262d; height: 6px; border-radius: 3px; margin-top: 5px; }
    .dna-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    
    /* 側邊欄 */
    .stats-sidebar { background-color: rgba(13, 17, 23, 0.8); border-left: 1px solid #30363d; padding: 15px; height: 100%; border-radius: 10px; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }
    .stat-val { font-weight: bold; font-family: 'JetBrains Mono'; }
    
    /* 新聞 */
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; transition: 0.2s; border-radius: 5px; margin-bottom: 5px; }
    .news-title { color: #e0e0e0; text-decoration: none; font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. 數據獲取與爬蟲層 (LEVIATHAN ENGINE)
# =============================================================================
@st.cache_data(ttl=3600)
def robust_download(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if df.empty: return pd.DataFrame()
        
        # 處理 MultiIndex Column (yfinance v0.2+)
        if isinstance(df.columns, pd.MultiIndex): 
            try:
                df.columns = df.columns.get_level_values(0)
            except: pass
                
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

class Market_List_Provider:
    @staticmethod
    @st.cache_data(ttl=86400) # 每天只爬一次
    def get_full_tw_tickers():
        tickers = []
        try:
            # 1. 上市 (Mode=2)
            url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
            res_twse = requests.get(url_twse, timeout=10)
            df_twse = pd.read_html(res_twse.text)[0]
            df_twse.columns = df_twse.iloc[0]
            df_twse = df_twse.iloc[1:]
            
            # 2. 上櫃 (Mode=4)
            url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
            res_tpex = requests.get(url_tpex, timeout=10)
            df_tpex = pd.read_html(res_tpex.text)[0]
            df_tpex.columns = df_tpex.iloc[0]
            df_tpex = df_tpex.iloc[1:]
            
            # 合併與清洗
            df_all = pd.concat([df_twse, df_tpex])
            df_all['Code'] = df_all['有價證券代號及名稱'].apply(lambda x: x.split()[0] if type(x)==str else "")
            stocks = df_all[df_all['Code'].str.len() == 4] # 只留4碼股票
            tickers = [f"{c}.TW" for c in stocks['Code'].tolist()]
            
            return tickers
        except Exception as e:
            st.error(f"Crawler Error: {e} (Need 'lxml' installed)")
            # Fallback if crawler fails
            return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2881.TW", "2382.TW", "2303.TW"]

    @staticmethod
    def get_crypto_list():
        return [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", 
            "TRX-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD", "UNI-USD", "ATOM-USD",
            "PEPE-USD", "SHIB-USD", "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD", "SUI-USD"
        ]

class Batch_Pre_Filter:
    @staticmethod
    def filter_by_volume(tickers, min_volume_shares=1000000): # 預設 1000 張
        survivors = []
        batch_size = 50 
        
        status_text = st.empty()
        bar = st.progress(0)
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i+batch_size]
            if not batch: continue
            
            status_text.text(f"🚀 Filtering Volume: Batch {i//batch_size + 1}...")
            bar.progress((i / len(tickers)))
            
            try:
                # 關鍵：只抓 5 天數據來檢查成交量，速度極快
                str_tickers = " ".join(batch)
                df = yf.download(str_tickers, period="5d", group_by='ticker', progress=False, threads=True)
                
                if df.empty: continue

                for t in batch:
                    try:
                        # 處理單一股票與多股票的結構差異
                        if len(batch) == 1:
                            vol = df['Volume'].iloc[-1]
                            price = df['Close'].iloc[-1]
                        else:
                            if t not in df.columns.levels[0]: continue
                            vol = df[t]['Volume'].iloc[-1]
                            price = df[t]['Close'].iloc[-1]
                        
                        # 規則：成交量 > 門檻 且 股價 > 10元
                        if vol > min_volume_shares and price > 10:
                            survivors.append(t)
                    except: continue
            except: pass
                
        bar.empty()
        status_text.empty()
        return survivors

# =============================================================================
# 3. 物理引擎 (Causal) - 已修復 Sync 0 問題
# =============================================================================
class Causal_Physics_Engine:
    @staticmethod
    def rolling_hilbert(series, window=64):
        values = series.values
        n = len(values)
        analytic_signal = np.zeros(n, dtype=complex)
        for i in range(window, n):
            segment = values[i-window : i] * np.hanning(window)
            if np.all(segment == 0): 
                analytic_signal[i] = 0
            else:
                analytic_signal[i] = hilbert(segment)[-1]
        return analytic_signal

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def calc_metrics_cached(df):
        if df.empty: return df
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
        sync_raw[:64] = 0 # 物理熱機期
        df['Sync'] = sync_raw
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
        
        return df.fillna(method='bfill').fillna(0)

class SMC_Engine:
    @staticmethod
    def identify_fvg(df):
        fvgs = []
        if len(df) < 35: return []
        for i in range(len(df)-2, len(df)-30, -1):
            try:
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    fvgs.append({'type': 'Bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'date': df.index[i-2]})
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    fvgs.append({'type': 'Bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'date': df.index[i-2]})
            except IndexError: continue
        return fvgs[:3]

class News_Intel_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_news(ticker):
        items = []; sentiment_score = 0
        try:
            query = ticker.split('.')[0]
            lang = "hl=zh-TW&gl=TW" if ".TW" in ticker else "hl=en-US&gl=US"
            url = f"https://news.google.com/rss/search?q={query}&{lang}"
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title').text
                    pubDate = item.find('pubDate')
                    date = pubDate.text[:16] if pubDate is not None else "Unknown"
                    link = item.find('link').text
                    items.append({"title": title, "date": date, "link": link})
                    if any(x in title for x in ["漲","High","Bull", "強勢", "噴"]): sentiment_score += 1
            return items, max(-1, min(1, sentiment_score/3))
        except: return [], 0

# =============================================================================
# 4. 分析整合 (Universal Analyst)
# =============================================================================
class Universal_Analyst:
    @staticmethod
    def analyze(ticker, fast_mode=False):
        # [CRITICAL FIX] 解決 Sync=0 的關鍵：
        # 就算在 fast_mode (掃描模式)，也要抓至少 1 年數據，
        # 確保物理引擎有 >64 根 K 棒進行熱機 (Warm-up)。
        period = "1y" if fast_mode else "2y"
        
        df = robust_download(ticker, period)
        
        # [Safety Valve] 提高安全閥到 100，避免無效數據
        if df.empty or len(df) < 100: return None
        
        df = Causal_Physics_Engine.calc_metrics_cached(df)
        last = df.iloc[-1]
        
        # DNA Calculation
        dna = {}
        
        # Trend
        trend_s = 50
        if last['Close'] > last['EMA20']: trend_s += 20
        if last['EMA20'] > last['EMA50']: trend_s += 20
        dna['Trend'] = min(trend_s, 100)
        
        # Momentum (Dynamic Lookback)
        lookback = min(60, len(df))
        rsi_window = df['RSI'].tail(lookback)
        if rsi_window.std() == 0: rsi_z = 0
        else: rsi_z = (last['RSI'] - rsi_window.mean()) / (rsi_window.std() + 1e-9)
        dna['Momentum'] = min(max(50 + rsi_z*20, 0), 100)
        
        # Physics (Sync)
        phy = last['Sync_Smooth']
        if pd.isna(phy): phy = 0
        dna['Physics'] = min(max(50 + phy*40, 0), 100)
        
        # Flow
        vpin = last['VPIN']
        if pd.isna(vpin): vpin = 0.5
        dna['Flow'] = min(max(100 - vpin*100, 0), 100)
        
        dna['Value'] = min(max(100 - last['RSI'], 0), 100)
        dna['Stability'] = 50 
        
        avg_score = np.mean(list(dna.values()))
        if pd.isna(avg_score): avg_score = 50 
        
        fvgs = []
        news = []
        sent = 0
        
        if not fast_mode:
            fvgs = SMC_Engine.identify_fvg(df)
            news, sent = News_Intel_Engine.fetch_news(ticker)
            
        return {
            "df": df, "last": last, "dna": dna, "score": avg_score,
            "fvgs": fvgs, "news": news, "sent": sent
        }

# =============================================================================
# 5. 回測器
# =============================================================================
class Sovereign_Backtester:
    def __init__(self, df, capital=1000000, fee=0.001425*0.6, tax=0.003):
        self.df = df
        self.capital = capital
        self.fee = fee
        self.tax = tax

    def run(self):
        cash = self.capital
        position = 0
        trades = []
        equity = []
        if self.df.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        bh_shares = self.capital // self.df['Close'].iloc[0]
        total_fee = 0
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            date = self.df.index[i]
            price = row['Close']
            
            buy_sig = (row['Sync_Smooth'] > 0.5) and (price > row['EMA20'])
            sell_sig = (row['Sync_Smooth'] < -0.2) or (price < row['EMA20'])
            
            if position > 0 and sell_sig:
                gross = position * price
                cost = gross * (self.fee + self.tax)
                cash += (gross - cost)
                total_fee += cost
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Net': gross-cost})
                position = 0
            elif position == 0 and buy_sig:
                cost = cash * 0.99
                f = cost * self.fee
                shares = (cost - f) // price
                if shares > 0:
                    cash -= (shares * price + f)
                    total_fee += f
                    position = shares
                    trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Cost': -cost})
            
            val = cash + (position * price)
            equity.append({'Date': date, 'Equity': val, 'BuyHold': bh_shares * price})
            
        final_eq = equity[-1]['Equity'] if equity else self.capital
        final_bh = equity[-1]['BuyHold'] if equity else self.capital
        
        stats = {
            'final_equity': final_eq,
            'total_return': (final_eq - self.capital) / self.capital * 100,
            'bh_return': (final_bh - self.capital) / self.capital * 100,
            'trades': len(trades),
            'fees': total_fee
        }
        return pd.DataFrame(equity), pd.DataFrame(trades), stats

# =============================================================================
# 6. UI Renderers
# =============================================================================
def render_macro_oracle():
    st.markdown("### 🌍 Macro Oracle")
    col1, col2, col3, col4 = st.columns(4)
    vix = 21.5; dxy = 104.2
    regime = "NEUTRAL"; c_reg = "#888"
    if vix > 25: regime = "FEAR"; c_reg = "#f85149"
    elif vix < 15: regime = "BULL"; c_reg = "#3fb950"
    
    col1.markdown(f"<div class='metric-card'><div class='highlight-lbl'>REGIME</div><div class='highlight-val' style='color:{c_reg}'>{regime}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='highlight-lbl'>VIX</div><div class='highlight-val'>{vix}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='highlight-lbl'>DXY</div><div class='highlight-val'>{dxy}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='highlight-lbl'>RISK</div><div class='highlight-val'>MED</div></div>", unsafe_allow_html=True)

def render_quantum_scanner():
    st.markdown("### 🔭 Quantum Scanner (Leviathan Mode)")
    st.markdown("<small>System will fetch 1700+ tickers and filter by volume.</small>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        market = st.selectbox("Market", ["🇹🇼 台股全市場 (Full TW)", "🇺🇸 美股 (US)", "🪙 Crypto"])
    with col2:
        if "TW" in market:
            vol_lots = st.slider("Min Volume (Lots/張)", 500, 5000, 1000, step=500)
        else:
            vol_lots = 0 
    
    if st.button("🚀 Start Deep Scan"):
        # 1. 獲取原始清單
        if "TW" in market:
            with st.spinner("🕷️ Crawling TWSE/TPEx (1700+ tickers)..."):
                raw_tickers = Market_List_Provider.get_full_tw_tickers()
                st.toast(f"Found {len(raw_tickers)} tickers.")
        elif "Crypto" in market:
            raw_tickers = Market_List_Provider.get_crypto_list()
        else:
            raw_tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN", "PLTR", "MSTR", "GOOG", "AMZN", "META"]

        # 2. 成交量過濾 (僅台股)
        if "TW" in market:
            targets = Batch_Pre_Filter.filter_by_volume(raw_tickers, min_volume_shares=vol_lots*1000)
            st.info(f"📉 Funnel: {len(raw_tickers)} -> {len(targets)} active stocks.")
        else:
            targets = raw_tickers

        # 3. 深度物理運算
        res_list = []
        bar = st.progress(0)
        status = st.empty()
        
        # 為了避免過久，這裡可以設一個上限，例如只跑前 500 檔
        # 但既然是 Leviathan 模式，我們讓它跑完
        scan_targets = targets
        
        for i, t in enumerate(scan_targets):
            status.text(f"🔬 Computing Physics: {t}...")
            try:
                # 這裡調用修復過的 analyze，確保 fast_mode 也有 1 年數據
                r = Universal_Analyst.analyze(t, fast_mode=True)
                if r: 
                    res_list.append({
                        "Ticker": t, 
                        "Price": r['last']['Close'],
                        "Score": float(f"{r['score']:.1f}"), 
                        "Sync": float(f"{r['last']['Sync_Smooth']:.4f}"), # 4位小數以觀察微小變化
                        "RSI": float(f"{r['last']['RSI']:.0f}"),
                        "Trend": float(f"{r['dna']['Trend']:.0f}")
                    })
            except: pass
            bar.progress((i+1)/len(scan_targets))
            
        status.empty()
        bar.empty()
        
        if res_list:
            df = pd.DataFrame(res_list).sort_values("Score", ascending=False)
            st.success(f"✅ Scan Complete. Found {len(df)} candidates.")
            st.dataframe(
                df.style.background_gradient(subset=['Score'], cmap='RdYlGn')
                  .format({"Price": "{:.2f}", "Score": "{:.1f}", "Sync": "{:.4f}"}),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No data returned.")

def render_sovereign_lab():
    st.markdown("### 🛡️ Sovereign Lab")
    ticker = st.text_input("Ticker", "2330.TW")
    
    if st.button("Deep Analyze"):
        with st.spinner("Analyzing Physics & DNA..."):
            res = Universal_Analyst.analyze(ticker, fast_mode=False)
            
        if res is None:
            st.error("Data Insufficient or Download Failed (Need > 100 candles).")
            return
            
        c1, c2 = st.columns([3, 1])
        with c1:
            score = res['score']
            last = res['last']
            price = last['Close']
            sig = "WAIT"; color="#888"
            if score >= 70: sig="BUY"; color="#3fb950"
            elif score <= 30: sig="SELL"; color="#f85149"
            
            st.markdown(f"""
            <div class="signal-box" style="border-top: 4px solid {color}">
                <div style="color:#aaa; font-size:14px">TACTICAL SIGNAL</div>
                <div class="big-signal" style="color:{color}">{sig}</div>
                <div>Score: {score:.0f} | Price: ${price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            atr = last['ATR']
            sl = price - (2.5 * atr)
            tp = price + (4.0 * atr)
            fvg_count = len(res['fvgs'])
            
            g1, g2, g3, g4 = st.columns(4)
            def grid_card(col, lbl, val, sub, c="white"):
                col.markdown(f"""<div class="metric-card"><div class="highlight-lbl">{lbl}</div><div class="highlight-val" style="color:{c}">{val}</div><div class="smart-text">{sub}</div></div>""", unsafe_allow_html=True)
            
            grid_card(g1, "ATR", f"{atr:.2f}", "Risk Unit")
            grid_card(g2, "STOP LOSS", f"${sl:,.2f}", "-2.5 ATR", "#f85149")
            grid_card(g3, "TAKE PROFIT", f"${tp:,.2f}", "Reward 1.6x", "#3fb950")
            grid_card(g4, "SMC ZONES", f"{fvg_count}", "Active FVG", "#ffae00")
            
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
            
            st.markdown("### 📰 Intel Feed")
            if res['news']:
                cols = st.columns(2)
                for i, item in enumerate(res['news']):
                    with cols[i%2]: st.markdown(f"<div class='news-card'><a href='{item['link']}' class='news-title' target='_blank'>{item['title']}</a><br><small style='color:#666'>{item['date']}</small></div>", unsafe_allow_html=True)
        
        with c2:
            st.markdown("#### 🧬 DNA")
            for k, v in res['dna'].items():
                c = "#3fb950" if v>60 else "#f85149" if v<40 else "#8b949e"
                st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:12px'><span style='color:#aaa'>{k}</span><span style='color:{c}; font-weight:bold'>{v:.0f}</span></div><div class='dna-bar-bg'><div class='dna-bar-fill' style='width:{v}%; background:{c}'></div></div>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown("#### 🏛️ Backtest (1Y)")
            sb = Sovereign_Backtester(res['df'])
            df_eq, df_tr, stats = sb.run()
            
            def row(l, v, c="#e6edf3"):
                st.markdown(f"<div class='stat-row'><span>{l}</span><span class='stat-val' style='color:{c}'>{v}</span></div>", unsafe_allow_html=True)
            
            if stats:
                pnl = stats['final_equity'] - 1000000
                row("Return", f"{stats['total_return']:.2f}%", "#3fb950" if pnl>0 else "#f85149")
                row("Alpha", f"{(stats['total_return']-stats['bh_return']):.2f}%", "#d2a8ff")
                row("Trades", f"{stats['trades']}")
                row("Fees", f"${stats['fees']:,.0f}", "#f85149")
            else:
                st.write("Backtest Failed")

# =============================================================================
# 7. 主程序
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ MARCS V160 LEVIATHAN")
    mode = st.sidebar.radio("MODE", ["🌍 Macro Oracle", "🔭 Quantum Scanner", "🛡️ Sovereign Lab"])
    
    if mode == "🌍 Macro Oracle": render_macro_oracle()
    elif mode == "🔭 Quantum Scanner": render_quantum_scanner()
    elif mode == "🛡️ Sovereign Lab": render_sovereign_lab()

if __name__ == "__main__":
    main()
