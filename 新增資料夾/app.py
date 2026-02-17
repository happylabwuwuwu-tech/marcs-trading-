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
import statsmodels.api as sm
import pandas_datareader.data as web
import plotly.graph_objects as go
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
# 1. 視覺核心 (星空 + 戰術板)
# =============================================================================
st.set_page_config(page_title="MARCS V120 OMEGA", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 全局設定 */
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; color: #c9d1d9; }
    
    /* 動態星空背景 (來自 V98) */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.6;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 戰術指令板 (來自 V110) */
    .signal-box {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95));
        border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center;
        margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); backdrop-filter: blur(10px);
    }
    .signal-buy { border-top: 4px solid #3fb950; }
    .signal-sell { border-top: 4px solid #f85149; }
    .signal-wait { border-top: 4px solid #8b949e; }
    
    .big-signal { font-size: 42px; font-weight: 800; letter-spacing: 2px; margin: 10px 0; font-family: 'JetBrains Mono'; }
    .signal-reason { font-family: 'Noto Sans TC'; font-size: 14px; color: #8b949e; }
    
    /* 數據卡片 (融合風格) */
    .metric-card {
        background: rgba(18, 18, 22, 0.85); backdrop-filter: blur(12px);
        border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 10px;
    }
    .highlight-lbl { font-size: 11px; color: #8b949e; letter-spacing: 1px; text-transform: uppercase; font-family: 'Rajdhani'; }
    .highlight-val { font-size: 24px; font-weight: bold; color: #e6edf3; font-family: 'JetBrains Mono'; }
    .smart-text { font-size: 12px; color: #ffb86c; font-family: 'Noto Sans TC'; margin-top: 4px; }
    
    /* DNA Bar */
    .dna-bar-bg { width: 100%; background: #21262d; height: 6px; border-radius: 3px; margin-top: 5px; }
    .dna-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    
    /* 側邊欄統計 */
    .stats-sidebar { background-color: rgba(13, 17, 23, 0.8); border-left: 1px solid #30363d; padding: 15px; height: 100%; border-radius: 10px; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }
    .stat-val { font-weight: bold; font-family: 'JetBrains Mono'; }
    
    /* 新聞卡片 */
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; transition: 0.2s; border-radius: 5px; margin-bottom: 5px; }
    .news-card:hover { background: rgba(40,40,50,0.9); }
    .news-title { color: #e0e0e0; text-decoration: none; font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. 數據獲取與工具層
# =============================================================================
@st.cache_data(ttl=3600)
def robust_download(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        # 修正時區問題
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "PLTR"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]
        return []

# =============================================================================
# 3. 戰術與因子引擎 (V98 移植)
# =============================================================================
class SMC_Engine:
    @staticmethod
    def identify_fvg(df, lookback=60):
        fvgs = []
        try:
            start_idx = max(len(df) - lookback, 2)
            for i in range(len(df)-2, start_idx, -1):
                # Bull FVG
                if df['Low'].iloc[i] > df['High'].iloc[i-2]: 
                    top, bottom = df['Low'].iloc[i], df['High'].iloc[i-2]
                    # Check mitigation
                    is_mitigated = any(df['Low'].iloc[j] < bottom for j in range(i+1, len(df)))
                    if not is_mitigated: fvgs.append({'type': 'Bull', 'top': top, 'bottom': bottom, 'idx': df.index[i-2], 'date': df.index[i-2]})
                # Bear FVG
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    top, bottom = df['Low'].iloc[i-2], df['High'].iloc[i]
                    is_mitigated = any(df['High'].iloc[j] > top for j in range(i+1, len(df)))
                    if not is_mitigated: fvgs.append({'type': 'Bear', 'top': top, 'bottom': bottom, 'idx': df.index[i-2], 'date': df.index[i-2]})
            return fvgs[:5] # 只回傳最近的5個
        except: return []

class FinMind_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_data(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        # 👇 請在這裡貼上您的 Token (如果沒有就留空字串 "")
        USER_TOKEN = ""  
        stock_id = ticker.split('.')[0]
        api = DataLoader()
        if USER_TOKEN: api.login_by_token(api_token=USER_TOKEN)
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        data = {"chips": 0, "pe": None, "growth": None}
        try:
            # 外資籌碼
            df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            if not df.empty:
                f = df[df['name'] == 'Foreign_Investor']
                if not f.empty: data['chips'] = int((f.iloc[-1]['buy'] - f.iloc[-1]['sell']) / 1000)
            # 月營收
            rev_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df_rev = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=rev_start)
            if not df_rev.empty: data['growth'] = df_rev.iloc[-1]['revenue_year_growth'] / 100.0
            return data
        except: return None

class News_Intel_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_news(ticker):
        items = []; sentiment_score = 0
        try:
            query = ticker.split('.')[0]
            if ".TW" in ticker: query += " (營收 OR 法說 OR 外資) when:7d"; lang = "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            else: query += " stock finance when:7d"; lang = "hl=en-US&gl=US&ceid=US:en"
            
            url = f"https://news.google.com/rss/search?q={query}&{lang}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:4]:
                    title = item.find('title').text
                    if any(x in title for x in ["影片","直播"]): continue
                    link = item.find('link').text
                    pubDate = item.find('pubDate')
                    date = pubDate.text[5:16] if pubDate is not None else "Recent"
                    
                    s_val = 0
                    if any(x in title for x in ["漲","高","Bull","優於","新高","Surge"]): s_val=1
                    elif any(x in title for x in ["跌","低","Bear","不如","重挫","Drop"]): s_val=-1
                    
                    items.append({"title": title, "link": link, "date": date, "sent": s_val})
                    sentiment_score += s_val
            
            final_sent = max(-1, min(1, sentiment_score / 3))
            return items, final_sent
        except: return [], 0

# =============================================================================
# 4. 物理與全譜分析引擎 (V110 + V98 Fusion)
# =============================================================================
class Universal_Analyst:
    @staticmethod
    def analyze(ticker):
        # 1. 基礎數據
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 50: return None
        
        c = df['Close'].values; v = df['Volume'].values
        
        # 2. 物理層 (Physics) - V110
        # Hilbert Sync
        ema = pd.Series(c).ewm(span=20).mean()
        detrend_c = (c - ema).fillna(0).values
        detrend_v = (v - pd.Series(v).rolling(20).mean()).fillna(0).values
        phase_c = np.angle(hilbert(detrend_c))
        phase_v = np.angle(hilbert(detrend_v))
        df['Sync'] = np.cos(phase_c - phase_v)
        df['Sync_Smooth'] = df['Sync'].rolling(5).mean()
        
        # VPIN (Order Flow Toxicity Proxy)
        delta_p = np.diff(c, prepend=c[0])
        sigma = np.std(delta_p) + 1e-9
        cdf = norm.cdf(delta_p / sigma)
        oi = np.abs(v*cdf - v*(1-cdf))
        df['VPIN'] = pd.Series(oi).rolling(20).sum() / (pd.Series(v).rolling(20).sum() + 1e-9)
        
        # Chaos (Wasserstein)
        log_ret = np.log(df['Close']).diff().fillna(0)
        chaos_list = [0]*40
        for i in range(40, len(df)):
            w2 = wasserstein_distance(log_ret[i-20:i], log_ret[i-40:i-20])
            chaos_list.append(w2 * 1000)
        df['Chaos'] = pd.Series(chaos_list, index=df.index).fillna(0)
        
        # 3. 技術層 (Technicals)
        df['EMA20'] = pd.Series(c).ewm(span=20).mean()
        df['EMA50'] = pd.Series(c).ewm(span=50).mean()
        df['ATR'] = (df['High']-df['Low']).rolling(14).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df = df.fillna(method='bfill')
        last = df.iloc[-1]
        
        # 4. 戰術層 (SMC) - V98
        fvgs = SMC_Engine.identify_fvg(df)
        
        # 5. 基本面層 (Data Fetching) - V98
        fm_data = FinMind_Engine.get_tw_data(ticker)
        news, sent = News_Intel_Engine.fetch_news(ticker)
        
        # 6. DNA Score Calculation (Mapping Factors)
        dna = {}
        
        # Trend: EMA Alignment
        trend_s = 50
        if last['Close'] > last['EMA20']: trend_s += 20
        if last['EMA20'] > last['EMA50']: trend_s += 20
        dna['Trend'] = min(trend_s, 100)
        
        # Momentum: RSI + News Sentiment
        mom_s = 50 + (sent * 20)
        if last['RSI'] > 50: mom_s += 10
        dna['Momentum'] = min(max(mom_s, 0), 100)
        
        # Physics: Hilbert Sync
        phy_s = 50 + (last['Sync_Smooth'] * 40)
        dna['Physics'] = min(max(phy_s, 0), 100)
        
        # Flow: VPIN + Foreign Chips
        flow_s = 100 - (last['VPIN'] * 100)
        if fm_data and fm_data['chips'] > 0: flow_s += 10
        elif fm_data and fm_data['chips'] < 0: flow_s -= 10
        dna['Flow'] = min(max(flow_s, 0), 100)
        
        # Value: PEG (Simplified)
        val_s = 50
        if fm_data and fm_data['growth'] and fm_data['growth'] > 0:
            pe = fm_data['pe'] if fm_data['pe'] else 20
            peg = pe / (fm_data['growth'] * 100)
            if peg < 1: val_s = 90
            elif peg < 1.5: val_s = 70
            else: val_s = 30
        else:
            # Fallback to RSI inverse for Value proxy if no data
            val_s = 100 - last['RSI']
        dna['Value'] = min(max(val_s, 0), 100)
        
        # Volatility: ATR relative to price (Lower is better score usually, but stable trend needs some vol)
        vol_pct = (last['ATR'] / last['Close']) * 100
        vol_s = 100 - (vol_pct * 10)
        dna['Stability'] = min(max(vol_s, 0), 100)
        
        avg_score = np.mean(list(dna.values()))
        
        return {
            "df": df, "last": last, "dna": dna, "score": avg_score,
            "fvgs": fvgs, "fm_data": fm_data, "news": news, "sent": sent
        }

class Scanner_Engine:
    @staticmethod
    def scan_single(ticker):
        try:
            df = robust_download(ticker, "6mo")
            if df.empty or len(df) < 30: return None
            c = df['Close']; ema20 = c.ewm(span=20).mean()
            # 簡易評分
            score = 50
            if c.iloc[-1] > ema20.iloc[-1]: score += 20
            # RSI
            delta = c.diff()
            rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean() + 1e-9)
            rsi = 100 - (100/(1+rs)).iloc[-1]
            if rsi > 50: score += 10
            
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "rsi": rsi}
        except: return None

# =============================================================================
# 5. 至尊回測器 (Sovereign Backtester V110)
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
        
        bh_shares = self.capital // self.df['Close'].iloc[0]
        
        total_fee = 0
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            date = self.df.index[i]
            price = row['Close']
            
            # 策略: 物理共振 + 趨勢
            buy_sig = (row['Sync_Smooth'] > 0.5) and (price > row['EMA20'])
            sell_sig = (row['Sync_Smooth'] < -0.2) or (price < row['EMA20'])
            
            if position > 0 and sell_sig:
                gross = position * price
                f = gross * self.fee
                t = gross * self.tax
                net = gross - f - t
                cash += net
                total_fee += (f+t)
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Net': net})
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
            
        stats = {
            'final_equity': equity[-1]['Equity'],
            'total_return': (equity[-1]['Equity'] - self.capital) / self.capital * 100,
            'bh_return': (equity[-1]['BuyHold'] - self.capital) / self.capital * 100,
            'trades': len(trades),
            'fees': total_fee
        }
        return pd.DataFrame(equity), pd.DataFrame(trades), stats

# =============================================================================
# 6. 渲染組件
# =============================================================================
def render_tactical_board(res, ticker):
    score = res['score']
    last = res['last']
    
    # 決定訊號
    signal = "WAIT"
    sig_class = "signal-wait"
    reason = "Market is choppy."
    
    if score >= 70:
        signal = "BUY"
        sig_class = "signal-buy"
        reason = "Physics Sync + Strong DNA"
    elif score <= 40:
        signal = "SELL"
        sig_class = "signal-sell"
        reason = "Weak Structure / Divergence"
        
    atr = last['ATR']
    price = last['Close']
    sl = price - (2.5 * atr)
    tp = price + (4.0 * atr)
    
    # 戰術板上方
    st.markdown(f"""
    <div class="signal-box {sig_class}">
        <div style="font-size:14px; color:#8b949e; margin-bottom:5px;">TACTICAL SIGNAL</div>
        <div class="big-signal" style="color:{'#3fb950' if signal=='BUY' else ('#f85149' if signal=='SELL' else '#8b949e')}">{signal}</div>
        <div class="signal-reason">{reason}</div>
        <hr style="border-color:#30363d; margin:15px 0;">
        <div style="display:flex; justify-content:center; gap:40px;">
             <div><span style="color:#8b949e; font-size:12px;">PRICE</span><br><span style="font-size:24px; font-weight:bold;">${price:,.2f}</span></div>
             <div><span style="color:#8b949e; font-size:12px;">SCORE</span><br><span style="font-size:24px; font-weight:bold; color:#d2a8ff;">{score:.0f}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 數據網格
    c1, c2, c3, c4 = st.columns(4)
    def kpi(col, lbl, val, sub, color="white"):
        col.markdown(f"""<div class="metric-card"><div class="highlight-lbl">{lbl}</div><div class="highlight-val" style="color:{color}">{val}</div><div class="smart-text">{sub}</div></div>""", unsafe_allow_html=True)
    
    kpi(c1, "ATR (Risk Unit)", f"{atr:.2f}", f"{(atr/price)*100:.1f}% Vol")
    kpi(c2, "STOP LOSS", f"${sl:,.2f}", "-2.5 ATR", "#f85149")
    kpi(c3, "TAKE PROFIT", f"${tp:,.2f}", "Reward 1.6x", "#3fb950")
    kpi(c4, "SMC FVG", f"{len(res['fvgs'])}", "Active Zones", "#ffae00")

    # DNA 條狀圖
    st.markdown("### 🧬 Factor DNA")
    cols = st.columns(6)
    for i, (k, v) in enumerate(res['dna'].items()):
        color = "#3fb950" if v > 60 else ("#f85149" if v < 40 else "#8b949e")
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:11px; color:#8b949e;">{k}</div>
                <div style="font-size:16px; font-weight:bold; color:{color};">{v:.0f}</div>
                <div class="dna-bar-bg">
                    <div class="dna-bar-fill" style="width:{v}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 7. 主程序
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ MARCS OMEGA")
    capital = st.sidebar.number_input("Capital", 1000000, step=10000)
    
    # 掃描器與輸入
    with st.sidebar.expander("🔭 Scanner"):
        market = st.selectbox("Market", ["🇹🇼 台股", "🇺🇸 美股", "🪙 Crypto"])
        if st.button("Run Scan"):
            with st.spinner("Scanning..."):
                tickers = Global_Market_Loader.get_scan_list(market)
                res = []
                bar = st.progress(0)
                with concurrent.futures.ThreadPoolExecutor() as exe:
                    futures = {exe.submit(Scanner_Engine.scan_single, t): t for t in tickers}
                    for i, f in enumerate(concurrent.futures.as_completed(futures)):
                        r = f.result()
                        if r: res.append(r)
                        bar.progress((i+1)/len(tickers))
                st.session_state.scan_res = sorted(res, key=lambda x: x['score'], reverse=True)
                bar.empty()
    
    if "scan_res" in st.session_state and st.session_state.scan_res:
        st.sidebar.dataframe(pd.DataFrame(st.session_state.scan_res)[['ticker','score']], use_container_width=True)

    ticker = st.sidebar.text_input("Ticker", "2330.TW")
    if st.sidebar.button("Analyze Target"):
        st.session_state.target = ticker

    if "target" in st.session_state:
        target = st.session_state.target
        
        # 執行全譜分析
        with st.spinner(f"Decoding {target} Physics & DNA..."):
            res = Universal_Analyst.analyze(target)
        
        if not res:
            st.error("No Data.")
            return

        # 布局
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            # 1. 戰術板
            render_tactical_board(res, target)
            
            # 2. 圖表 (Plotly Ultimate)
            st.markdown("### 🔭 Quantum Chart (FVG + Physics)")
            df = res['df']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # K線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            # EMA
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#ffae00', width=1), name='EMA20'), row=1, col=1)
            
            # FVG 視覺化 (矩形)
            for fvg in res['fvgs']:
                color = "rgba(0, 255, 0, 0.2)" if fvg['type'] == 'Bull' else "rgba(255, 0, 0, 0.2)"
                fig.add_shape(type="rect", x0=fvg['date'], y0=fvg['bottom'], x1=df.index[-1], y1=fvg['top'], fillcolor=color, line_width=0, row=1, col=1)

            # Physics Sync
            fig.add_trace(go.Scatter(x=df.index, y=df['Sync_Smooth'], line=dict(color='#d2a8ff', width=2), name='Phase Sync'), row=2, col=1)
            fig.add_hrect(y0=0.5, y1=1.0, row=2, col=1, fillcolor="#d2a8ff", opacity=0.1, line_width=0)
            
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, paper_bgcolor="#050505", plot_bgcolor="#050505", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 新聞情報
            st.markdown("### 📰 Intel Feed")
            if res['news']:
                cols = st.columns(2)
                for i, item in enumerate(res['news']):
                    with cols[i%2]:
                        st.markdown(f"<div class='news-card'><a href='{item['link']}' class='news-title' target='_blank'>{item['title']}</a><br><small style='color:#666'>{item['date']}</small></div>", unsafe_allow_html=True)

        with col_side:
            # 右側統計欄 (Sovereign Style)
            st.markdown('<div class="stats-sidebar">', unsafe_allow_html=True)
            st.markdown("#### 🏛️ Account")
            
            # 即時回測
            sb = Sovereign_Backtester(res['df'], capital)
            df_eq, df_tr, stats = sb.run()
            
            def row(l, v, c="#e6edf3"):
                st.markdown(f"<div class='stat-row'><span>{l}</span><span class='stat-val' style='color:{c}'>{v}</span></div>", unsafe_allow_html=True)
            
            pnl = stats['final_equity'] - capital
            c_pnl = "#3fb950" if pnl > 0 else "#f85149"
            
            row("Equity", f"${stats['final_equity']:,.0f}")
            row("Return", f"{stats['total_return']:.2f}%", c_pnl)
            row("Alpha", f"{(stats['total_return']-stats['bh_return']):.2f}%", "#d2a8ff")
            st.divider()
            row("Trades", f"{stats['trades']}")
            row("Fees", f"${stats['fees']:,.0f}", "#f85149")
            
            st.divider()
            st.markdown("#### 🏗️ Fundamental")
            fm = res['fm_data']
            if fm:
                row("Foreign Chips", f"{fm['chips']}", "#3fb950" if fm['chips']>0 else "#f85149")
                row("Rev Growth", f"{fm['growth']}%" if fm['growth'] else "N/A")
            else:
                st.caption("No Fundamental Data")
                
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
