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
st.set_page_config(page_title="MARCS OMEGA V130", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 全局設定 */
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; color: #c9d1d9; }
    
    /* 動態星空背景 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px;
        animation: stars 120s linear infinite; z-index: -1; opacity: 0.6;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }

    /* 戰術指令板 */
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
    
    /* 數據卡片 */
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
# 2. 數據獲取層 (Robust Download)
# =============================================================================
@st.cache_data(ttl=3600)
def robust_download(ticker, period="2y"): # 恢復 2年數據以支持 Fama-French
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

# =============================================================================
# 3. 物理引擎 (重啟 Causal Hilbert)
# =============================================================================
class Causal_Physics_Engine:
    @staticmethod
    def rolling_hilbert(series, window=64):
        """因果滾動希爾伯特變換 (慢但正確)"""
        values = series.values
        n = len(values)
        analytic_signal = np.zeros(n, dtype=complex)
        
        for i in range(window, n):
            segment = values[i-window : i]
            segment = segment * np.hanning(window)
            h_segment = hilbert(segment)
            analytic_signal[i] = h_segment[-1]
        return analytic_signal

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False) # 快取！這是防止黑屏的關鍵
    def calc_metrics_cached(df):
        if df.empty or len(df) < 100: return df
        df = df.copy()
        
        c = df['Close']
        v = df['Volume']
        
        # 1. Causal Hilbert Sync
        ema = c.ewm(span=20).mean()
        detrend_c = (c - ema).fillna(0)
        v_ma = v.rolling(20).mean()
        detrend_v = (v - v_ma).fillna(0)
        
        analytic_c = Causal_Physics_Engine.rolling_hilbert(detrend_c, window=64)
        analytic_v = Causal_Physics_Engine.rolling_hilbert(detrend_v, window=64)
        
        phase_c = np.angle(analytic_c)
        phase_v = np.angle(analytic_v)
        
        sync_raw = np.cos(phase_c - phase_v)
        sync_raw[:64] = 0 
        
        df['Sync'] = sync_raw
        df['Sync_Smooth'] = pd.Series(sync_raw).rolling(5).mean().fillna(0)
        
        # 2. VPIN (Toxicity Proxy)
        delta_p = c.diff()
        sigma = delta_p.rolling(20).std() + 1e-9
        cdf = norm.cdf(delta_p / sigma)
        buy_vol = v * cdf
        sell_vol = v * (1 - cdf)
        oi = (buy_vol - sell_vol).abs()
        total_vol = v.rolling(20).sum() + 1e-9
        df['VPIN'] = (oi.rolling(20).sum() / total_vol).fillna(0) # 補0修復
        
        # 3. Chaos (Wasserstein)
        log_ret = np.log(c).diff().fillna(0)
        chaos_list = [0]*60
        for i in range(60, len(df)):
            w2 = wasserstein_distance(log_ret.iloc[i-20:i], log_ret.iloc[i-40:i-20])
            chaos_list.append(w2 * 1000)
        df['Chaos'] = pd.Series(chaos_list, index=df.index).fillna(0)
        
        # 4. Technicals
        df['EMA20'] = c.ewm(span=20).mean()
        df['EMA50'] = c.ewm(span=50).mean()
        df['ATR'] = (df['High']-df['Low']).rolling(14).mean()
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

# =============================================================================
# 4. 六因子引擎 (Fama-French Restored)
# =============================================================================
class SixFactor_Engine:
    @staticmethod
    @st.cache_data(ttl=86400) # 每天只抓一次
    def get_ff_factors(start_date):
        try:
            # 這裡使用 requests 直接抓取 CSV 以繞過 pandas_datareader 的不穩定性
            # 但為了演示，我們使用 yfinance 的替代因子 (ETF Proxy) 來模擬
            # 這是更穩定的工程解法：使用 MTUM(動能), VLUE(價值), USMV(低波) 代替 FF5
            # 如果堅持要 FF5 原始數據，請使用 Kenneth French Data Library
            # 這裡我們先回傳 None，以防卡死，待用戶提供本地 CSV
            return None 
        except: return None

    @staticmethod
    def analyze_exposure(ticker_df):
        # 暫時回傳 None，因為 FF 數據源不穩定
        # 這裡保留接口，未來可接入本地 CSV
        return None

# =============================================================================
# 5. 基本面與新聞引擎 (Restored)
# =============================================================================
class FinMind_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_data(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        USER_TOKEN = ""  # 填入 Token
        stock_id = ticker.split('.')[0]
        api = DataLoader()
        if USER_TOKEN: api.login_by_token(api_token=USER_TOKEN)
        
        data = {"chips": 0, "pe": None, "growth": None}
        try:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            if not df.empty:
                f = df[df['name'] == 'Foreign_Investor']
                if not f.empty: data['chips'] = int((f.iloc[-1]['buy'] - f.iloc[-1]['sell']) / 1000)
            
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
            resp = requests.get(url, timeout=3) # 強制 Timeout
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
# 6. SMC 戰術引擎
# =============================================================================
class SMC_Engine:
    @staticmethod
    def identify_fvg(df, lookback=60):
        fvgs = []
        try:
            start_idx = max(len(df) - lookback, 2)
            for i in range(len(df)-2, start_idx, -1):
                if df['Low'].iloc[i] > df['High'].iloc[i-2]: 
                    top, bottom = df['Low'].iloc[i], df['High'].iloc[i-2]
                    is_mitigated = any(df['Low'].iloc[j] < bottom for j in range(i+1, len(df)))
                    if not is_mitigated: fvgs.append({'type': 'Bull', 'top': top, 'bottom': bottom, 'idx': df.index[i-2], 'date': df.index[i-2]})
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    top, bottom = df['Low'].iloc[i-2], df['High'].iloc[i]
                    is_mitigated = any(df['High'].iloc[j] > top for j in range(i+1, len(df)))
                    if not is_mitigated: fvgs.append({'type': 'Bear', 'top': top, 'bottom': bottom, 'idx': df.index[i-2], 'date': df.index[i-2]})
            return fvgs[:5]
        except: return []

# =============================================================================
# 7. 全譜分析整合 (The Fusion)
# =============================================================================
class Universal_Analyst:
    @staticmethod
    def analyze(ticker):
        # 1. 基礎數據
        df = robust_download(ticker, "2y")
        if df.empty or len(df) < 100: return None
        
        # 2. 物理計算 (有 Cache 保護)
        df = Causal_Physics_Engine.calc_metrics_cached(df)
        df = df.fillna(method='bfill')
        last = df.iloc[-1]
        
        # 3. 外部數據 (並行處理)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_fm = executor.submit(FinMind_Engine.get_tw_data, ticker)
            f_news = executor.submit(News_Intel_Engine.fetch_news, ticker)
            # f_ff = executor.submit(SixFactor_Engine.analyze_exposure, df) # 暫時關閉以保穩定
            
            fm_data = f_fm.result()
            news, sent = f_news.result()
            # ff_data = f_ff.result()
        
        # 4. 戰術層
        fvgs = SMC_Engine.identify_fvg(df)
        
        # 5. DNA Score (數據熔斷機制)
        dna = {}
        missing_data_count = 0
        
        # Trend
        trend_s = 50
        if last['Close'] > last['EMA20']: trend_s += 20
        if last['EMA20'] > last['EMA50']: trend_s += 20
        dna['Trend'] = min(trend_s, 100)
        
        # Momentum (Dynamic Z-Score)
        recent_rsi = df['RSI'].tail(60)
        rsi_z = (last['RSI'] - recent_rsi.mean()) / (recent_rsi.std() + 1e-9)
        mom_s = 50 + (rsi_z * 20) + (sent * 10)
        dna['Momentum'] = min(max(mom_s, 0), 100)
        
        # Physics (Causal Sync)
        phy_s = 50 + (last['Sync_Smooth'] * 40)
        dna['Physics'] = min(max(phy_s, 0), 100)
        
        # Flow (VPIN + Chips)
        if pd.isna(last['VPIN']) or last['VPIN'] == 0: 
            flow_s = 50; missing_data_count += 0.5
        else:
            flow_s = 100 - (last['VPIN'] * 100)
            
        if fm_data:
            if fm_data['chips'] > 0: flow_s += 10
            elif fm_data['chips'] < 0: flow_s -= 10
        else: missing_data_count += 0.5
        dna['Flow'] = min(max(flow_s, 0), 100)
        
        # Value (PEG)
        if fm_data and fm_data['growth']:
            pe = fm_data['pe'] if fm_data['pe'] else 20
            peg = pe / (fm_data['growth'] * 100)
            val_s = 90 if peg < 1 else (70 if peg < 1.5 else 30)
        else:
            val_s = 100 - last['RSI']
            missing_data_count += 1
        dna['Value'] = min(max(val_s, 0), 100)
        
        # Stability (ATR Z-Score)
        recent_atr = df['ATR'].tail(60)
        atr_z = (last['ATR'] - recent_atr.mean()) / (recent_atr.std() + 1e-9)
        stab_s = 50 - (atr_z * 20)
        dna['Stability'] = min(max(stab_s, 0), 100)
        
        if missing_data_count >= 2.5: return None
        
        avg_score = np.mean(list(dna.values()))
        
        return {
            "df": df, "last": last, "dna": dna, "score": avg_score,
            "fvgs": fvgs, "fm_data": fm_data, "news": news, "sent": sent
        }

# =============================================================================
# 8. 至尊回測器 (Sovereign Backtester)
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
                f = gross * self.fee; t = gross * self.tax
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
# 9. 主程序
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ MARCS OMEGA v130")
    st.sidebar.caption("Resurrection | Physics + Fund + News")
    
    ticker = st.sidebar.text_input("Ticker", "2330.TW")
    capital = st.sidebar.number_input("Capital", 1000000)
    
    if st.sidebar.button("Deep Analyze"):
        with st.spinner("Initializing Causal Engines..."):
            res = Universal_Analyst.analyze(ticker)
            
        if res is None:
            st.error("❌ Data Insufficient or Connection Error.")
            return
            
        # --- Layout ---
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            # 1. 戰術板
            score = res['score']
            price = res['last']['Close']
            sig = "WAIT"; sig_class = "signal-wait"
            if score >= 70: sig="BUY"; sig_class="signal-buy"
            elif score <= 30: sig="SELL"; sig_class="signal-sell"
            
            st.markdown(f"""
            <div class="signal-box {sig_class}">
                <div class="big-signal" style="color:{'#3fb950' if sig=='BUY' else '#f85149' if sig=='SELL' else '#8b949e'}">{sig}</div>
                <div style="font-size:14px; color:#aaa">Score: {score:.1f} | Price: ${price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. 圖表 (Physics + FVG)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=res['df'].index, open=res['df']['Open'], high=res['df']['High'], low=res['df']['Low'], close=res['df']['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['EMA20'], line=dict(color='#ffae00', width=1), name='EMA20'), row=1, col=1)
            
            for fvg in res['fvgs']:
                c = "rgba(0,255,0,0.2)" if fvg['type']=='Bull' else "rgba(255,0,0,0.2)"
                fig.add_shape(type="rect", x0=fvg['date'], x1=res['df'].index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=c, line_width=0, row=1, col=1)
            
            fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Sync_Smooth'], line=dict(color='#d2a8ff', width=2), name='Causal Sync'), row=2, col=1)
            fig.add_hrect(y0=0.5, y1=1.0, row=2, col=1, fillcolor="#d2a8ff", opacity=0.1, line_width=0)
            fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="#333")
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 新聞
            st.markdown("### 📰 Intel Feed")
            if res['news']:
                cols = st.columns(2)
                for i, item in enumerate(res['news']):
                    with cols[i%2]: st.markdown(f"<div class='news-card'><a href='{item['link']}' class='news-title' target='_blank'>{item['title']}</a><br><small style='color:#666'>{item['date']}</small></div>", unsafe_allow_html=True)
        
        with col_side:
            # 右側 DNA 與 統計
            st.markdown("### 🧬 DNA")
            for k, v in res['dna'].items():
                c = "#3fb950" if v > 60 else "#f85149" if v < 40 else "#8b949e"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:#aaa">{k}</span>
                        <span style="color:{c}; font-weight:bold">{v:.0f}</span>
                    </div>
                    <div class="dna-bar-bg"><div class="dna-bar-fill" style="width:{v}%; background:{c}"></div></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 🏛️ Backtest (1Y)")
            sb = Sovereign_Backtester(res['df'], capital)
            df_eq, df_tr, stats = sb.run()
            
            def row(l, v, c="#e6edf3"):
                st.markdown(f"<div class='stat-row'><span>{l}</span><span class='stat-val' style='color:{c}'>{v}</span></div>", unsafe_allow_html=True)
            
            pnl = stats['final_equity'] - capital
            row("Return", f"{stats['total_return']:.2f}%", "#3fb950" if pnl>0 else "#f85149")
            row("Alpha", f"{(stats['total_return']-stats['bh_return']):.2f}%", "#d2a8ff")
            row("Trades", f"{stats['trades']}")
            row("Fees", f"${stats['fees']:,.0f}", "#f85149")

if __name__ == "__main__":
    main()
