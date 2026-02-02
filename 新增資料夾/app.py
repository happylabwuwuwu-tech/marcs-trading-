import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import warnings
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (星際戰神風格 + SMC 戰術面板)
# =============================================================================
st.set_page_config(page_title="MARCS V96 SMC戰術版", layout="wide", page_icon="🛡️")

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
    
    /* [V96] 戰術數據卡 (Tactical Card) */
    .tac-card {
        background: rgba(26, 26, 26, 0.8); border: 1px solid #444; border-radius: 6px; padding: 10px;
        margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;
        backdrop-filter: blur(5px);
    }
    .tac-label { font-size: 12px; color: #aaa; font-family: 'Rajdhani'; font-weight: bold; }
    .tac-val { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: bold; color: #fff; }
    .tac-sub { font-size: 10px; color: #666; margin-left: 5px; }

    /* 卡片與表格 */
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
    
    .chip-tag { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 10px; font-family: 'Noto Sans TC'; vertical-align: middle; }
    
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; transition: 0.2s; border-radius: 5px; }
    .news-card:hover { background: rgba(40,40,50,0.9); }
    .news-title { color: #e0e0e0; text-decoration: none; font-weight: bold; font-size: 14px; }
    
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; border:none; background: linear-gradient(90deg, #333 0%, #ffae00 100%); color: white; }
    
    /* 調整 Matplotlib 背景 */
    div[data-testid="stImage"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 數據獲取層 (Robust Download + Caching)
# =============================================================================
@st.cache_data(ttl=3600)  # 緩存數據 1 小時
def robust_download(ticker, period="1y"):
    try:
        # 嘗試直接獲取 history
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        # 如果數據為空或格式不對，嘗試 yf.download
        if df.empty or len(df) == 0:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        # 處理 MultiIndex columns (yfinance 新版常見問題)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass
        
        # 確保有 Close 欄位
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        if not df.empty and 'Close' in df.columns and len(df) > 0:
            df.index = pd.to_datetime(df.index)
            # 簡單過濾無效數據
            if df['Close'].iloc[-1] > 0:
                return df
    except Exception as e:
        print(f"Download Error for {ticker}: {e}")
    return pd.DataFrame()

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW", "2609.TW", "2615.TW"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "PLTR"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
        return []

# =============================================================================
# 2. [V96 New] SMC 引擎 (Smart Money Concepts)
# =============================================================================
class SMC_Engine:
    @staticmethod
    def identify_fvg(df, lookback=60):
        """
        識別公允價值缺口 (Fair Value Gap)
        只回傳最近且尚未被完全回補的 FVG
        """
        fvgs = []
        try:
            # 遍歷最近 N 根 K 線 (倒序)
            # 確保索引不越界
            start_idx = max(len(df) - lookback, 2)
            
            for i in range(len(df)-2, start_idx, -1):
                # Bullish FVG: Low[i] > High[i-2] (中間那根是大陽線)
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    top = df['Low'].iloc[i]
                    bottom = df['High'].iloc[i-2]
                    # 檢查是否已被回補 (之後的 K 線低點是否跌破 bottom)
                    is_mitigated = False
                    for j in range(i+1, len(df)):
                        if df['Low'].iloc[j] < bottom:
                            is_mitigated = True; break
                    
                    if not is_mitigated:
                        fvgs.append({'type': 'Bull', 'top': top, 'bottom': bottom, 'idx': df.index[i-2]})

                # Bearish FVG: High[i] < Low[i-2] (中間那根是大陰線)
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    top = df['Low'].iloc[i-2]
                    bottom = df['High'].iloc[i]
                    is_mitigated = False
                    for j in range(i+1, len(df)):
                        if df['High'].iloc[j] > top:
                            is_mitigated = True; break
                    
                    if not is_mitigated:
                        fvgs.append({'type': 'Bear', 'top': top, 'bottom': bottom, 'idx': df.index[i-2]})
            
            return fvgs[:3] # 只取最近的 3 個
        except Exception as e:
            return []

# =============================================================================
# 3. 核心分析引擎 (Micro + SMC 整合)
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 30: 
            return 50, ["數據不足"], df, 0, None, 0, 0, []
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            # 1. Elder Logic (EMA + MACD + Force Index)
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            
            # Force Index
            fi = c.diff() * v
            fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): 
                score += 20; signals.append("Elder Impulse Bull")
            elif (ema22.iloc[-1] < ema22.iloc[-2]) and (hist.iloc[-1] < hist.iloc[-2]):
                score -= 20; signals.append("Elder Impulse Bear")
                
            if fi_13.iloc[-1] > 0: score += 10
            
            # 2. SMC Logic (FVG)
            fvgs = SMC_Engine.identify_fvg(df)
            current_price = c.iloc[-1]
            
            # 檢查是否處於 FVG 區間內
            in_bull_fvg = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bull' for f in fvgs)
            in_bear_fvg = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bear' for f in fvgs)
            
            if in_bull_fvg: score += 15; signals.append("Testing Bullish FVG (Support)")
            if in_bear_fvg: score -= 15; signals.append("Testing Bearish FVG (Resist)")
            
            # 3. Chips & ATR
            chips = FinMind_Engine.get_tw_chips(ticker)
            if chips:
                if chips['latest'] > 1000: score += 15
                elif chips['latest'] < -1000: score -= 15
            
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            
            # Prep DataFrame for plotting
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            
            return score, signals, df, atr, chips, current_price, score, fvgs
        except Exception as e: 
            print(f"Analyze Error: {e}")
            return 50, ["計算錯誤"], df, 0, None, 0, 0, []

# =============================================================================
# 4. 輔助引擎
# =============================================================================
class FinMind_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_chips(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        try:
            # 簡單模擬或從 FinMind 獲取 (需考慮 API 限制，這裡做容錯)
            start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
            url = "https://api.finmindtrade.com/api/v4/data"
            stock_id = ticker.split('.')[0]
            params = {
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell", 
                "data_id": stock_id, 
                "start_date": start_date
            }
            res = requests.get(url, params=params, timeout=3)
            data = res.json()
            if data['msg'] == 'success' and data['data']:
                df = pd.DataFrame(data['data'])
                f = df[df['name'] == 'Foreign_Investor']
                if not f.empty: 
                    latest_buy = f.iloc[-1]['buy'] - f.iloc[-1]['sell']
                    return {"latest": int(latest_buy/1000)} # 張數
            return None
        except: return None

class News_Intel_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_news(ticker):
        items = []
        sentiment_score = 0
        try:
            # 1. YFinance News
            if "-USD" in ticker or ".TW" not in ticker:
                try:
                    stock = yf.Ticker(ticker)
                    raw_news = stock.news
                    for item in raw_news[:5]:
                        title = item.get('title'); link = item.get('link')
                        pub_time = item.get('providerPublishTime')
                        date = pd.to_datetime(pub_time, unit='s').strftime('%m-%d')
                        sent = "neu"; s_val = 0
                        tl = title.lower()
                        if any(x in tl for x in ["soar","jump","beat","upgrade","buy","surge"]): sent="pos"; s_val=1
                        elif any(x in tl for x in ["plunge","drop","miss","downgrade","sell","crash"]): sent="neg"; s_val=-1
                        items.append({"title": title, "link": link, "date": date, "sent": sent})
                        sentiment_score += s_val
                except: pass

            # 2. Google RSS Fallback
            if not items:
                query = ticker.split('.')[0]
                if ".TW" in ticker: 
                    query += " (營收 OR 法說 OR 外資) when:7d"
                    lang = "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                else: 
                    query += " stock finance when:7d"
                    lang = "hl=en-US&gl=US&ceid=US:en"
                
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
                        sent = "neu"; s_val = 0
                        if any(x in title for x in ["漲","高","Bull","優於","新高"]): sent="pos"; s_val=1
                        elif any(x in title for x in ["跌","低","Bear","不如","重挫"]): sent="neg"; s_val=-1
                        items.append({"title": title, "link": link, "date": date, "sent": sent})
                        sentiment_score += s_val
            
            # Normalize sentiment
            final_sent = max(-1, min(1, sentiment_score / 3))
            return items, final_sent
        except: return [], 0

class Scanner_Engine_Elder:
    @staticmethod
    def analyze_single(ticker):
        try:
            df = robust_download(ticker, "6mo")
            if df.empty or len(df) < 50: return None
            c = df['Close']; ema22 = c.ewm(span=22).mean()
            score = 60
            if c.iloc[-1] > ema22.iloc[-1]: score += 20
            else: score -= 20
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "sl": ema22.iloc[-1]*0.98}
        except: return None

class PEG_Valuation_Engine:
    @staticmethod
    def calculate(ticker, sentiment_score=0):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice', 0)
            if price == 0: price = info.get('regularMarketPrice', 0)
            if price == 0: return None
            
            pe = info.get('trailingPE', None)
            growth = info.get('earningsGrowth', None) # 這是季度成長，有時需用其他欄位
            
            # 如果沒有 earningsGrowth，嘗試用 revenueGrowth 替代估算
            if not growth: growth = info.get('revenueGrowth', None)

            if not pe or not growth: 
                return {"fair": price, "method": "Price Only", "peg_used": 0, "sentiment_impact": "0%"}
            
            # PEG 模型
            peg = pe / (growth * 100)
            # 根據情緒調整 PEG 目標 (情緒好給予更高溢價)
            target_peg = peg * (1 + (sentiment_score * 0.2))
            
            # 簡單反推合理價: Fair P/E = Growth * Target_PEG
            fair_pe = (growth * 100) * 1.0 # 假設合理 PEG 為 1
            fair_price = (price / pe) * fair_pe
            
            # 根據情緒微調
            fair_price = fair_price * (1 + sentiment_score * 0.1)

            return {
                "fair": fair_price, 
                "scenarios": {"Bear": fair_price * 0.85, "Bull": fair_price * 1.15}, 
                "method": "PEG Adjusted", 
                "peg_used": round(target_peg, 2), 
                "sentiment_impact": f"{sentiment_score*20:+.0f}%"
            }
        except: return None

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid):
        default = {"cap": 0, "pct": 0.0}
        if price <= 0: return 0, default
        try:
            # 基礎風險：本金的 2%
            risk_amount = capital * 0.02
            dist = price - sl
            if dist <= 0: return 0, default # SL 高於現價，邏輯錯誤或已止損
            
            # 信心係數調整 (Hybrid Score 越高，倉位越大)
            conf = max(0.2, hybrid / 100.0) 
            
            # 倉位大小 (股數) = 風險金額 / 每股虧損距離 * 信心係數
            size = int((risk_amount / dist) * conf)
            pos_val = size * price
            
            # 限制單一倉位不超過本金 30% (安全閥)
            if pos_val > capital * 0.3:
                size = int((capital * 0.3) / price)
                pos_val = size * price

            pct = (pos_val / capital) * 100
            return size, {"cap": int(pos_val), "pct": round(pct, 1)}
        except: return 0, default

class Backtest_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def run_backtest(ticker):
        try:
            df = robust_download(ticker, "2y")
            if df.empty or len(df) < 100: return None
            
            # 簡單策略：EMA22 向上 + MACD 黃金交叉
            df['EMA22'] = df['Close'].ewm(span=22).mean()
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            df['Hist'] = df['MACD'] - df['Signal']
            
            # 進場條件
            df['Green'] = (df['EMA22'] > df['EMA22'].shift(1)) & (df['Hist'] > 0) & (df['Hist'] > df['Hist'].shift(1))
            
            position = 0; entry_price = 0; equity = [100000]
            
            for i in range(1, len(df)):
                price = df['Close'].iloc[i]
                
                # Buy
                if position == 0 and df['Green'].iloc[i]:
                    position = 1; entry_price = price
                
                # Sell (MACD 死叉 或 跌破 EMA22)
                elif position == 1 and (df['Hist'].iloc[i] < 0 or price < df['EMA22'].iloc[i]):
                    position = 0
                    profit_pct = (price - entry_price) / entry_price
                    equity.append(equity[-1] * (1 + profit_pct))
                
                # Hold logic for equity curve
                if position == 1:
                    change = (df['Close'].iloc[i] / df['Close'].iloc[i-1]) - 1
                    equity.append(equity[-1] * (1 + change))
                else:
                    equity.append(equity[-1])
            
            total_ret = (equity[-1] - 100000) / 100000
            return {"total_return": total_ret, "equity_curve": pd.DataFrame({'Equity': equity[-len(df):]}, index=df.index)}
        except: return None

class Macro_Risk_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_market_risk():
        try:
            vix_df = robust_download("^VIX", "5d")
            if not vix_df.empty:
                vix = vix_df['Close'].iloc[-1]
                score = max(0, 100 - (vix * 2)) # VIX 越高 分數越低
                return int(score), ["VIX Monitor"], vix
            return 50, ["Neutral"], 20
        except: return 50, ["Loading"], 20

class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid, m_score, chips, fvgs):
        tag = "😐 觀望 (Hold)"; bg = "#333"
        if hybrid >= 80: tag = "🔥 強力買進 (Strong Buy)"; bg = "#3fb950"
        elif hybrid >= 60: tag = "✅ 買進 (Buy)"; bg = "#1f6feb"
        elif hybrid <= 40: tag = "❄️ 弱勢 (Weak)"; bg = "#888"
        elif hybrid <= 20: tag = "⛔ 危險 (Sell)"; bg = "#f85149"
        
        reasons = []
        if m_score >= 70: reasons.append("動能強勁")
        if chips and chips['latest'] > 0: reasons.append("外資買超")
        if any(f['type']=='Bull' for f in fvgs): reasons.append("回測 Bullish FVG (支撐有效)")
        if any(f['type']=='Bear' for f in fvgs): reasons.append("遭遇 Bearish FVG (壓力)")
        
        reason_str = "，".join(reasons) if reasons else "多空不明"
        return tag, f"{ticker} 目前呈現 {tag.split(' ')[1]}。主因：{reason_str}。", bg

# =============================================================================
# MAIN UI
# =============================================================================
# =============================================================================
# MAIN UI (修復完整版)
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金 (Capital)", value=1000000, step=10000)
    target_in = st.sidebar.text_input("代碼 (Ticker)", "2330.TW").upper()
    
    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    
    if st.sidebar.button("分析單一標的"): 
        st.session_state.target = target_in
    
    # Scanner
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 主動掃描器 (Scanner)"):
        scan_source = st.radio("來源", ["線上掃描", "CSV匯入"])
        if scan_source == "線上掃描":
            market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股", "🪙 加密貨幣"])
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
                    # 假設第一欄是代碼
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

    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    target = st.session_state.target

    # 1. Macro Risk
    risk, risk_d, vix = Macro_Risk_Engine.calculate_market_risk()
    st.markdown(f"""<div class="risk-container"><div class="risk-val" style="color:#4caf50">{risk}</div><div style="color:#aaa">MARKET RISK (VIX: {vix:.1f})</div></div>""", unsafe_allow_html=True)

    # Scanner Results Display
    if st.session_state.scan_results:
        with st.expander("🔭 掃描結果 (Scan Results)"):
            st.dataframe(pd.DataFrame(st.session_state.scan_results), use_container_width=True)
            # 讓使用者選擇掃描結果
            scan_tickers = [r['ticker'] for r in st.session_state.scan_results]
            sel = st.selectbox("Load Result:", scan_tickers)
            if st.button("Load Selected"): 
                st.session_state.target = sel
                st.rerun()

    # 2. Analysis Execution
    with st.spinner(f"Scanning {target} (Elder + SMC FVG)..."):
        # 使用 ThreadPool 平行處理多個分析任務
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Pro.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            
            # 獲取結果
            m_score, sigs, df_m, atr, chips, curr_p, _, fvgs = f_micro.result()
            news, sent = f_news.result()
            
            # 這些計算依賴上面的結果，所以放在外面
            dcf_res = PEG_Valuation_Engine.calculate(target, sent)
            backtest = Backtest_Engine.run_backtest(target)

        # 混合評分 (Macro 30% + Micro 70%)
        hybrid = int((risk * 0.3) + (m_score * 0.7))
        
        # [V96] 計算 SL / TP (戰術面板)
        sl_p = curr_p - 2.5 * atr if atr > 0 else 0
        tp_p = curr_p + 4.0 * atr if atr > 0 else 0
        risk_pct = round((sl_p / curr_p - 1)*100, 2) if curr_p > 0 else 0
        size, r_d = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # 3. Verdict & UI
    tag, comm, bg = Message_Generator.get_verdict(target, hybrid, m_score, chips, fvgs)
    
    # 標題區
    c_tag = f"<span class='chip-tag' style='background:#f44336'>外資 {chips['latest']}</span>" if chips else ""
    st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${curr_p:.2f}</span> {c_tag}</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div class="verdict-box" style="background:{bg}30; border-color:{bg}"><h2 style="margin:0; color:{bg}">{tag}</h2><p style="margin-top:5px; color:#ccc">{comm}</p></div>""", unsafe_allow_html=True)

    # [V96] 戰術面板 (Tactical Panel)
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">ATR (Volatility)</div><div class="tac-val">{atr:.2f}</div></div><div class="tac-sub">Risk Unit</div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="tac-card" style="border-color:#f44336"><div><div class="tac-label">STOP LOSS</div><div class="tac-val" style="color:#f44336">${sl_p:.2f}</div></div><div class="tac-sub">{risk_pct}% Risk</div></div>""", unsafe_allow_html=True)
    with t3: st.markdown(f"""<div class="tac-card" style="border-color:#4caf50"><div><div class="tac-label">TAKE PROFIT</div><div class="tac-val" style="color:#4caf50">${tp_p:.2f}</div></div><div class="tac-sub">Reward 1.6x</div></div>""", unsafe_allow_html=True)
    with t4: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">SUGGESTED SIZE</div><div class="tac-val">{r_d['pct']}%</div></div><div class="tac-sub">${r_d['cap']:,}</div></div>""", unsafe_allow_html=True)

    # 數據卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">技術評分</div><div class="highlight-val">{m_score}</div><div class="smart-text">{sigs[0] if sigs else '盤整'}</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">宏觀風險</div><div class="highlight-val">{risk}</div><div class="smart-text">VIX: {vix:.1f}</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 情緒修正</div><div class="highlight-val">{sent:+.2f}</div><div class="smart-text">News Adj</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">SMC 訊號</div><div class="highlight-val">{len(fvgs)}</div><div class="smart-text">Active FVG</div></div>""", unsafe_allow_html=True)

    # 4. Tabs & Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📊 SMC 戰術圖表", "🧬 估值模型", "📰 情報中心", "🔄 策略回測"])
    
    with tab1:
        if not df_m.empty and 'EMA22' in df_m.columns:
            # Price Chart
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5, label='Price')
            ax.plot(df_m.index, df_m['EMA22'], color='#ffae00', lw=1.5, label='EMA 22')
            
            # [V96] 繪製 FVG 區塊
            for fvg in fvgs:
                color = 'green' if fvg['type'] == 'Bull' else 'red'
                # 繪製矩形
                rect = patches.Rectangle((fvg['idx'], fvg['bottom']), width=timedelta(days=5), height=fvg['top']-fvg['bottom'], linewidth=0, edgecolor=None, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                ax.text(fvg['idx'], fvg['top'], f" {fvg['type']} FVG", color=color, fontsize=8, verticalalignment='bottom')

            ax.axhline(sl_p, color='#f44336', ls='--', label=f'SL: {sl_p:.2f}')
            ax.axhline(tp_p, color='#4caf50', ls='--', label=f'TP: {tp_p:.2f}')
            ax.legend(loc='upper left')
            ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='#888'); ax.grid(True, color='#333', alpha=0.3)
            st.pyplot(fig)
            plt.close(fig) # 釋放記憶體
            
            # Indicators Chart
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            
            # MACD
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols, alpha=0.8); ax1.set_title("MACD Histogram", color='white', fontsize=10)
            ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='#888')
            
            # Force Index
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_title("Force Index (13)", color='white', fontsize=10)
            ax2.axhline(0, color='gray', ls='--')
            ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='#888')
            
            fig2.patch.set_facecolor('#0d1117')
            st.pyplot(fig2)
            plt.close(fig2)
        else: st.warning("數據不足，無法繪圖")

    with tab2:
        if dcf_res:
            c_v1, c_v2 = st.columns(2)
            with c_v1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 合理價</div><div class="highlight-val">${dcf_res['fair']:.2f}</div><div class="smart-text">Method: {dcf_res['method']}</div></div>""", unsafe_allow_html=True)
            with c_v2: 
                st.write("#### 估值情境 (Scenarios)")
                st.json(dcf_res['scenarios'])
                st.caption(f"PEG Used: {dcf_res['peg_used']} | Sentiment Impact: {dcf_res['sentiment_impact']}")
        else: st.info("無法計算 PEG (可能缺乏盈利數據)")

    with tab3:
        if news:
            cols = st.columns(3)
            for i, item in enumerate(news):
                bd = "#4caf50" if item['sent']=="pos" else "#f44336" if item['sent']=="neg" else "#444"
                with cols[i%3]: st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd}"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta" style="color:#666; font-size:12px; margin-top:5px;">{item['date']}</div></div>""", unsafe_allow_html=True)
        else: st.info("無近期新聞")

    with tab4:
        if backtest:
            b1, b2 = st.columns([1, 3])
            with b1: 
                ret_color = "green" if backtest['total_return'] > 0 else "red"
                st.markdown(f"### 總報酬 (2Y)\n<span style='color:{ret_color}; font-size:24px; font-weight:bold'>{backtest['total_return']:.1%}</span>", unsafe_allow_html=True)
                st.caption("策略：EMA22 趨勢 + MACD 動能")
            with b2:
                st.line_chart(backtest['equity_curve'], color="#ffae00")
        else: st.warning("數據不足，無法回測")

if __name__ == "__main__":
    main()
