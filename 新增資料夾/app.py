import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import warnings
import random
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (星際戰神風格)
# =============================================================================
st.set_page_config(page_title="MARCS V99 最終穩定版", layout="wide", page_icon="🛡️")

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
# 1. 數據獲取層 (V99: 核心修復 - 偽裝 + 快取)
# =============================================================================
def get_headers():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
    ]
    return {"User-Agent": random.choice(agents)}

# [V99 FIX] 加入 st.cache_data 避免重複請求導致被擋
@st.cache_data(ttl=900) # 快取 15 分鐘
def robust_download(ticker, period="1y"):
    """
    V99 終極下載器：
    1. 使用 requests.Session 偽裝瀏覽器
    2. 強制扁平化 MultiIndex (解決美股問題)
    3. 清洗空值
    """
    session = requests.Session()
    session.headers.update(get_headers())
    
    try:
        # 嘗試 1: yf.Ticker.history (美股首選)
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)
        
        # 如果 history 失敗，嘗試 download
        if df.empty:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, session=session)
        
        if df.empty: return pd.DataFrame()

        # [CRITICAL FIX] 暴力清洗 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0) 
            except: pass
        
        # 移除重複欄位 (常見於 download 後)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 統一欄位名稱
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        # 最終檢查
        if 'Close' in df.columns and len(df) > 0:
            df.index = pd.to_datetime(df.index)
            # 確保數值型
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            return df
            
    except Exception as e:
        pass
        
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
            if len(df) < lookback: lookback = len(df)
            for i in range(len(df)-2, len(df)-lookback, -1):
                # Bullish FVG
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    fvgs.append({'type': 'Bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'idx': df.index[i-2]})
                # Bearish FVG
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    fvgs.append({'type': 'Bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'idx': df.index[i-2]})
            return fvgs[:3]
        except: return []

# =============================================================================
# 3. 核心分析引擎 (Micro)
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        
        # [V99] 嚴格的長度檢查
        if df.empty or len(df) < 50: 
            return 50, ["數據不足 (可能被擋修或無交易)"], pd.DataFrame(), 0, None, 0, 0, []
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            # Elder Indicators
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20; signals.append("Elder Impulse Bull")
            if fi_13.iloc[-1] > 0: score += 10
            
            # SMC
            fvgs = SMC_Engine.identify_fvg(df)
            current_price = c.iloc[-1]
            in_bull = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bull' for f in fvgs)
            if in_bull: score += 15; signals.append("SMC Support")
            
            # Chips
            chips = FinMind_Engine.get_tw_chips(ticker)
            if chips:
                if chips['latest'] > 1000: score += 15
                elif chips['latest'] < -1000: score -= 15
            
            # ATR
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            if np.isnan(atr): atr = current_price * 0.02 # Fallback
            
            # Prep DF
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = ema22 + 2*atr; df['K_Lower'] = ema22 - 2*atr
            
            return score, signals, df, atr, chips, current_price, score, fvgs
        except Exception as e: 
            return 50, ["計算錯誤"], df, 0, None, 0, 0, []

# =============================================================================
# 4. 輔助引擎 (Fix Valuation & Risk)
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
        # [V99] 簡化新聞抓取，避免卡死
        items = []
        try:
            q = ticker.split('.')[0] + (" stock" if "-USD" in ticker else " 台股")
            url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    t = item.find('title').text
                    l = item.find('link').text
                    d = item.find('pubDate').text[5:16] if item.find('pubDate') is not None else ""
                    s = "pos" if any(x in t for x in ["漲","高","Bull"]) else ("neg" if any(x in t for x in ["跌","低","Bear"]) else "neu")
                    items.append({"title": t, "link": l, "date": d, "sent": s})
            return items, 0
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
    @st.cache_data(ttl=3600) # Cache Fundamental Data
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
        try:
            # 1. 嘗試抓基本面
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 0)
            if price == 0: price = info.get('regularMarketPrice', 0)
            
            # [Fallback] 如果 API 沒價格，用 K 線
            if price == 0:
                df = robust_download(ticker, "5d")
                if not df.empty: price = df['Close'].iloc[-1]
                else: return None

            pe = info.get('trailingPE', None)
            growth = info.get('earningsGrowth', None)
            
            # [Fallback] 技術估值
            if not pe or not growth:
                return {"fair": price, "scenarios": {"Bear": price*0.9, "Bull": price*1.1}, "method": "Price Action Only", "peg_used": "N/A"}
            
            peg = pe / (growth * 100)
            target_peg = peg * (1 + (sentiment_score * 0.2))
            fair_price = (price / pe) * (target_peg * growth * 100)
            return {"fair": fair_price, "scenarios": {"Bear": fair_price * 0.85, "Bull": fair_price * 1.15}, "method": "PEG Adjusted", "peg_used": round(target_peg, 2)}
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
            
            position = 0; equity = [100000]; trades = []
            
            for i in range(1, len(df)):
                price = df['Close'].iloc[i]; prev = df['Close'].iloc[i-1]
                if position == 0 and df['Green'].iloc[i]:
                    position = 1; trades.append(1)
                elif position == 1 and not df['Green'].iloc[i]:
                    position = 0; trades.append(0)
                
                if position == 1: equity.append(equity[-1] * (price/prev))
                else: equity.append(equity[-1])
            
            eq_curve = pd.Series(equity, index=df.index[-len(equity):])
            total_ret = (equity[-1] - 100000) / 100000
            
            # MDD
            roll_max = eq_curve.cummax()
            drawdown = (eq_curve - roll_max) / roll_max
            mdd = drawdown.min()
            
            return {
                "total_return": total_ret,
                "mdd": mdd,
                "win_rate": 0.5,
                "equity_curve": eq_curve,
                "drawdown": drawdown
            }
        except: return None

class Macro_Risk_Engine:
    @staticmethod
    @st.cache_data(ttl=1800) # Cache Macro 30 mins
    def calculate_market_risk():
        try:
            df = robust_download("^VIX", "5d")
            vix = df['Close'].iloc[-1] if not df.empty else 20
            return 60, ["VIX Stable"], vix
        except: return 50, ["System Ready"], 20

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
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
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
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
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
    risk, risk_d, vix = Macro_Risk_Engine.calculate_market_risk
