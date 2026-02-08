import sys
import os
import types

# =============================================================================
# 0. 系統補丁 (Fix for Python 3.12+ & pandas_datareader)
# =============================================================================
# 必須放在所有 import 之前，解決 distutils 被移除的問題
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

# =============================================================================
# 正常 Import
# =============================================================================
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
import statsmodels.api as sm
import pandas_datareader.data as web
from FinMind.data import DataLoader

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 視覺核心 (星際戰神風格 + SMC 戰術面板)
# =============================================================================
st.set_page_config(page_title="MARCS V97 6-Factor Elite", layout="wide", page_icon="🛡️")

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
    
    div[data-testid="stImage"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. 數據獲取層
# =============================================================================
@st.cache_data(ttl=3600)  
def robust_download(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty or len(df) == 0:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        if not df.empty and 'Close' in df.columns and len(df) > 0:
            df.index = pd.to_datetime(df.index).tz_localize(None) 
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
# 3. SMC 引擎
# =============================================================================
class SMC_Engine:
    @staticmethod
    def identify_fvg(df, lookback=60):
        fvgs = []
        try:
            start_idx = max(len(df) - lookback, 2)
            for i in range(len(df)-2, start_idx, -1):
                if df['Low'].iloc[i] > df['High'].iloc[i-2]: # Bull
                    top = df['Low'].iloc[i]; bottom = df['High'].iloc[i-2]
                    is_mitigated = False
                    for j in range(i+1, len(df)):
                        if df['Low'].iloc[j] < bottom: is_mitigated = True; break
                    if not is_mitigated: fvgs.append({'type': 'Bull', 'top': top, 'bottom': bottom, 'idx': df.index[i-2]})

                elif df['High'].iloc[i] < df['Low'].iloc[i-2]: # Bear
                    top = df['Low'].iloc[i-2]; bottom = df['High'].iloc[i]
                    is_mitigated = False
                    for j in range(i+1, len(df)):
                        if df['High'].iloc[j] > top: is_mitigated = True; break
                    if not is_mitigated: fvgs.append({'type': 'Bear', 'top': top, 'bottom': bottom, 'idx': df.index[i-2]})
            return fvgs[:3]
        except Exception: return []

# =============================================================================
# 4. 六因子模型引擎
# =============================================================================
class SixFactor_Engine:
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_ff_factors(start_date):
        """下載 Fama-French 5因子 + Momentum"""
        try:
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date)[0]
            mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start=start_date)[0]
            factors = pd.concat([ff5, mom], axis=1).dropna()
            factors = factors / 100.0
            factors.rename(columns={'Mkt-RF': 'Mkt_RF', 'Mom   ': 'MOM'}, inplace=True)
            return factors
        except Exception as e:
            print(f"F-F Data Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def analyze_exposure(ticker_df, ticker_symbol):
        try:
            if ticker_df.empty or len(ticker_df) < 60: return None
            start_date = ticker_df.index[0]
            factors = SixFactor_Engine.get_ff_factors(start_date)
            if factors.empty: return None
            
            combined = pd.concat([ticker_df['Close'].pct_change().dropna(), factors], axis=1, join='inner').dropna()
            if len(combined) < 30: return None
            
            y = combined['Close'] - combined['RF']
            X = combined[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            betas = model.params
            pvalues = model.pvalues
            rsquared = model.rsquared
            
            analysis = {
                'Beta_Mkt': {'val': betas['Mkt_RF'], 'sig': pvalues['Mkt_RF'] < 0.05, 'desc': '市場連動性'},
                'Beta_SMB': {'val': betas['SMB'], 'sig': pvalues['SMB'] < 0.05, 'desc': '規模 (Size)'}, 
                'Beta_HML': {'val': betas['HML'], 'sig': pvalues['HML'] < 0.05, 'desc': '價值 (Value)'}, 
                'Beta_RMW': {'val': betas['RMW'], 'sig': pvalues['RMW'] < 0.05, 'desc': '獲利 (Profit)'}, 
                'Beta_CMA': {'val': betas['CMA'], 'sig': pvalues['CMA'] < 0.05, 'desc': '投資 (Inv)'},    
                'Beta_MOM': {'val': betas['MOM'], 'sig': pvalues['MOM'] < 0.05, 'desc': '動能 (Mom)'},    
                'R2': rsquared
            }
            return analysis
        except Exception as e:
            print(f"6-Factor Analysis Error: {e}")
            return None

# =============================================================================
# 5. FinMind & Valuation (UPGRADED)
# =============================================================================
class FinMind_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_data(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: 
            return None
        
        stock_id = ticker.split('.')[0]
        api = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        data_pack = {
            "chips": 0,
            "pe": None,
            "growth": None,
            "price": 0
        }
        
        try:
            # 1. 外資籌碼
            df_chips = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            if not df_chips.empty:
                f = df_chips[df_chips['name'] == 'Foreign_Investor']
                if not f.empty:
                    latest = f.iloc[-1]
                    data_pack['chips'] = int((latest['buy'] - latest['sell']) / 1000)

            # 2. 本益比 (PER)
            df_per = api.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start_date)
            if not df_per.empty:
                latest_pe = df_per.iloc[-1]['PER']
                if latest_pe > 0: data_pack['pe'] = latest_pe

            # 3. 月營收成長率
            rev_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df_rev = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=rev_start)
            if not df_rev.empty:
                latest_growth = df_rev.iloc[-1]['revenue_year_growth']
                if latest_growth is not None:
                    data_pack['growth'] = latest_growth / 100.0

            return data_pack
        except Exception as e:
            print(f"FinMind Error: {e}")
            return None

class PEG_Valuation_Engine:
    @staticmethod
    def calculate(ticker, sentiment_score=0):
        try:
            # --- 分支 A: 台股模式 (FinMind) ---
            if ".TW" in ticker:
                fm_data = FinMind_Engine.get_tw_data(ticker)
                stock = yf.Ticker(ticker)
                # 嘗試抓現價
                try: price = stock.history(period="1d")['Close'].iloc[-1]
                except: price = 0
                
                if fm_data and price > 0:
                    pe = fm_data['pe']
                    growth = fm_data['growth']
                    if not pe: pe = stock.info.get('trailingPE')
                    
                    if pe and growth:
                        peg = pe / (growth * 100)
                        if peg <= 0: peg = 99 
                        
                        target_peg = peg * (1 + (sentiment_score * 0.2))
                        fair_pe = (growth * 100) * 1.0 
                        fair_price = (price / pe) * fair_pe * (1 + sentiment_score * 0.1)
                        
                        return {
                            "fair": fair_price, 
                            "scenarios": {"Bear": fair_price * 0.85, "Bull": fair_price * 1.15}, 
                            "method": "FinMind PEG (TW)", 
                            "peg_used": round(peg, 2), 
                            "sentiment_impact": f"{sentiment_score*20:+.0f}%"
                        }

            # --- 分支 B: 美股/通用模式 (YF) ---
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
            if price == 0: return None
            
            pe = info.get('trailingPE', None)
            growth = info.get('earningsGrowth', None) or info.get('revenueGrowth', None)

            # [安全網] 避免 KeyError
            if not pe or not growth: 
                return {
                    "fair": price, 
                    "scenarios": {"Bear": price * 0.9, "Bull": price * 1.1},
                    "method": "Price Only (No Data)", 
                    "peg_used": 0, 
                    "sentiment_impact": "0%"
                }
            
            peg = pe / (growth * 100)
            target_peg = peg * (1 + (sentiment_score * 0.2))
            fair_pe = (growth * 100) * 1.0 
            fair_price = (price / pe) * fair_pe * (1 + sentiment_score * 0.1)

            return {
                "fair": fair_price, 
                "scenarios": {"Bear": fair_price * 0.85, "Bull": fair_price * 1.15}, 
                "method": "PEG Adjusted (YF)", 
                "peg_used": round(target_peg, 2), 
                "sentiment_impact": f"{sentiment_score*20:+.0f}%"
            }
        except Exception as e:
            print(f"Valuation Error: {e}")
            return None

# =============================================================================
# 6. 核心分析引擎
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 30: 
            return 50, ["數據不足"], df, 0, None, 0, 0, [], None
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            # 1. Elder Logic
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v
            fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): 
                score += 20; signals.append("Elder Impulse Bull")
            elif (ema22.iloc[-1] < ema22.iloc[-2]) and (hist.iloc[-1] < hist.iloc[-2]):
                score -= 20; signals.append("Elder Impulse Bear")
                
            if fi_13.iloc[-1] > 0: score += 10
            
            # 2. SMC Logic
            fvgs = SMC_Engine.identify_fvg(df)
            current_price = c.iloc[-1]
            in_bull_fvg = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bull' for f in fvgs)
            in_bear_fvg = any(f['bottom'] <= current_price <= f['top'] and f['type']=='Bear' for f in fvgs)
            
            if in_bull_fvg: score += 15; signals.append("Testing Bullish FVG")
            if in_bear_fvg: score -= 15; signals.append("Testing Bearish FVG")
            
            # 3. Chips (Hybrid: FinMind or YF)
            chips = None
            if ".TW" in ticker:
                fm_data = FinMind_Engine.get_tw_data(ticker)
                if fm_data and fm_data['chips'] != 0:
                    chips = {'latest': fm_data['chips']}
                    if chips['latest'] > 1000: score += 15
                    elif chips['latest'] < -1000: score -= 15
            
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            
            # 4. Run 6-Factor Model
            ff_analysis = SixFactor_Engine.analyze_exposure(df, ticker)
            if ff_analysis:
                if ff_analysis['Beta_RMW']['val'] > 0 and ff_analysis['Beta_RMW']['sig']: score += 5
                if ff_analysis['Beta_MOM']['val'] > 0 and ff_analysis['Beta_MOM']['sig']: score += 5
            
            return score, signals, df, atr, chips, current_price, score, fvgs, ff_analysis
        except Exception as e: 
            print(f"Analyze Error: {e}")
            return 50, ["計算錯誤"], df, 0, None, 0, 0, [], None

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

# =============================================================================
# 7. 其他輔助與 UI
# =============================================================================
class News_Intel_Engine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_news(ticker):
        items = []; sentiment_score = 0
        try:
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
                    if any(x in title for x in ["漲","高","Bull","優於","新高","Surge"]): sent="pos"; s_val=1
                    elif any(x in title for x in ["跌","低","Bear","不如","重挫","Drop"]): sent="neg"; s_val=-1
                    items.append({"title": title, "link": link, "date": date, "sent": sent})
                    sentiment_score += s_val
            
            final_sent = max(-1, min(1, sentiment_score / 3))
            return items, final_sent
        except: return [], 0

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid):
        default = {"cap": 0, "pct": 0.0}
        if price <= 0: return 0, default
        try:
            risk_amount = capital * 0.02
            dist = price - sl
            if dist <= 0: return 0, default 
            
            conf = max(0.2, hybrid / 100.0) 
            size = int((risk_amount / dist) * conf)
            pos_val = size * price
            
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
            
            df['EMA22'] = df['Close'].ewm(span=22).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            df['Hist'] = df['MACD'] - df['Signal']
            
            df['Green'] = (df['EMA22'] > df['EMA22'].shift(1)) & (df['Hist'] > 0) & (df['Hist'] > df['Hist'].shift(1))
            
            position = 0; entry_price = 0; equity = [100000]
            
            for i in range(1, len(df)):
                price = df['Close'].iloc[i]
                if position == 0 and df['Green'].iloc[i]:
                    position = 1; entry_price = price
                elif position == 1 and (df['Hist'].iloc[i] < 0 or price < df['EMA22'].iloc[i]):
                    position = 0
                    profit_pct = (price - entry_price) / entry_price
                    equity.append(equity[-1] * (1 + profit_pct))
                
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
                score = max(0, 100 - (vix * 2)) 
                return int(score), ["VIX Monitor"], vix
            return 50, ["Neutral"], 20
        except: return 50, ["Loading"], 20

class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid, m_score, chips, fvgs, ff_data):
        tag = "😐 觀望 (Hold)"; bg = "#333"
        if hybrid >= 80: tag = "🔥 強力買進 (Strong Buy)"; bg = "#3fb950"
        elif hybrid >= 60: tag = "✅ 買進 (Buy)"; bg = "#1f6feb"
        elif hybrid <= 40: tag = "❄️ 弱勢 (Weak)"; bg = "#888"
        elif hybrid <= 20: tag = "⛔ 危險 (Sell)"; bg = "#f85149"
        
        reasons = []
        if m_score >= 70: reasons.append("技術動能強勁")
        if chips and chips['latest'] > 0: reasons.append("外資買超")
        if any(f['type']=='Bull' for f in fvgs): reasons.append("回測 Bullish FVG")
        
        if ff_data:
            if ff_data['Beta_RMW']['val'] > 0 and ff_data['Beta_RMW']['sig']: reasons.append("高獲利因子")
            if ff_data['Beta_MOM']['val'] > 0 and ff_data['Beta_MOM']['sig']: reasons.append("動能因子強勢")
            if ff_data['Beta_HML']['val'] < 0 and ff_data['Beta_HML']['sig']: reasons.append("高成長屬性")

        reason_str = "，".join(reasons) if reasons else "多空不明"
        return tag, f"{ticker} 目前呈現 {tag.split(' ')[1]}。主因：{reason_str}。", bg

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金 (Capital)", value=1000000, step=10000)
    target_in = st.sidebar.text_input("代碼 (Ticker)", "2330.TW").upper()
    
    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    if st.sidebar.button("分析單一標的"): st.session_state.target = target_in
    
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
            if uploaded and st.button("🚀 掃描上傳清單"):
                try:
                    df_up = pd.read_csv(uploaded)
                    tickers = df_up.iloc[:, 0].astype(str).tolist()
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
    
    if st.session_state.scan_results:
        with st.expander("🔭 掃描結果 (Scan Results)"):
            st.dataframe(pd.DataFrame(st.session_state.scan_results), use_container_width=True)
            scan_tickers = [r['ticker'] for r in st.session_state.scan_results]
            sel = st.selectbox("Load Result:", scan_tickers)
            if st.button("Load Selected"): 
                st.session_state.target = sel
                st.rerun()

    target = st.session_state.target

    # 1. Macro Risk
    risk, risk_d, vix = Macro_Risk_Engine.calculate_market_risk()
    st.markdown(f"""<div class="risk-container"><div class="risk-val" style="color:#4caf50">{risk}</div><div style="color:#aaa">MARKET RISK (VIX: {vix:.1f})</div></div>""", unsafe_allow_html=True)

    # 2. Analysis Execution
    with st.spinner(f"Engaging Fama-French 6-Factor Model on {target}..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Pro.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            
            m_score, sigs, df_m, atr, chips, curr_p, _, fvgs, ff_analysis = f_micro.result()
            news, sent = f_news.result()
            
            dcf_res = PEG_Valuation_Engine.calculate(target, sent)
            backtest = Backtest_Engine.run_backtest(target)

        hybrid = int((risk * 0.3) + (m_score * 0.7))
        
        sl_p = curr_p - 2.5 * atr if atr > 0 else 0
        tp_p = curr_p + 4.0 * atr if atr > 0 else 0
        risk_pct = round((sl_p / curr_p - 1)*100, 2) if curr_p > 0 else 0
        size, r_d = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # 3. Verdict & UI
    tag, comm, bg = Message_Generator.get_verdict(target, hybrid, m_score, chips, fvgs, ff_analysis)
    
    c_tag = f"<span class='chip-tag' style='background:#f44336'>外資 {chips['latest']}</span>" if chips else ""
    st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${curr_p:.2f}</span> {c_tag}</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div class="verdict-box" style="background:{bg}30; border-color:{bg}"><h2 style="margin:0; color:{bg}">{tag}</h2><p style="margin-top:5px; color:#ccc">{comm}</p></div>""", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">ATR (Volatility)</div><div class="tac-val">{atr:.2f}</div></div><div class="tac-sub">Risk Unit</div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="tac-card" style="border-color:#f44336"><div><div class="tac-label">STOP LOSS</div><div class="tac-val" style="color:#f44336">${sl_p:.2f}</div></div><div class="tac-sub">{risk_pct}% Risk</div></div>""", unsafe_allow_html=True)
    with t3: st.markdown(f"""<div class="tac-card" style="border-color:#4caf50"><div><div class="tac-label">TAKE PROFIT</div><div class="tac-val" style="color:#4caf50">${tp_p:.2f}</div></div><div class="tac-sub">Reward 1.6x</div></div>""", unsafe_allow_html=True)
    with t4: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">SUGGESTED SIZE</div><div class="tac-val">{r_d['pct']}%</div></div><div class="tac-sub">${r_d['cap']:,}</div></div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">技術評分</div><div class="highlight-val">{m_score}</div><div class="smart-text">{sigs[0] if sigs else '盤整'}</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">FF5 模型品質 (R²)</div><div class="highlight-val">{ff_analysis['R2']:.2f}</div><div class="smart-text">學術因子解釋力</div></div>""" if ff_analysis else "", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 情緒修正</div><div class="highlight-val">{sent:+.2f}</div><div class="smart-text">News Adj</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">SMC 訊號</div><div class="highlight-val">{len(fvgs)}</div><div class="smart-text">Active FVG</div></div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 SMC 戰術圖表", "🧬 六因子與估值", "📰 情報中心", "🔄 策略回測"])
    
    with tab1:
        if not df_m.empty and 'EMA22' in df_m.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5, label='Price')
            ax.plot(df_m.index, df_m['EMA22'], color='#ffae00', lw=1.5, label='EMA 22')
            
            for fvg in fvgs:
                color = 'green' if fvg['type'] == 'Bull' else 'red'
                rect = patches.Rectangle((fvg['idx'], fvg['bottom']), width=timedelta(days=5), height=fvg['top']-fvg['bottom'], linewidth=0, edgecolor=None, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                ax.text(fvg['idx'], fvg['top'], f" {fvg['type']} FVG", color=color, fontsize=8, verticalalignment='bottom')

            ax.axhline(sl_p, color='#f44336', ls='--', label=f'SL: {sl_p:.2f}')
            ax.axhline(tp_p, color='#4caf50', ls='--', label=f'TP: {tp_p:.2f}')
            ax.legend(loc='upper left')
            ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='#888'); ax.grid(True, color='#333', alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols, alpha=0.8); ax1.set_title("MACD Histogram", color='white', fontsize=10)
            ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='#888')
            
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_title("Force Index (13)", color='white', fontsize=10)
            ax2.axhline(0, color='gray', ls='--')
            ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='#888')
            fig2.patch.set_facecolor('#0d1117')
            st.pyplot(fig2)
            plt.close(fig2)
        else: st.warning("數據不足，無法繪圖")

    with tab2:
        if ff_analysis:
            st.markdown("### 🧬 Fama-French 6-Factor DNA")
            st.caption("基於 FF (2018) 'Choosing factors' 論文模型，解析股票的風險屬性。")
            f_cols = st.columns(6)
            factors_map = [
                ('Mkt', 'Beta_Mkt', '市場 Beta'),
                ('Size', 'Beta_SMB', '規模 (SMB)'),
                ('Value', 'Beta_HML', '價值 (HML)'),
                ('Profit', 'Beta_RMW', '獲利 (RMW)'),
                ('Invest', 'Beta_CMA', '投資 (CMA)'),
                ('Momentum', 'Beta_MOM', '動能 (MOM)')
            ]
            
            for i, (label, key, desc) in enumerate(factors_map):
                val = ff_analysis[key]['val']
                sig = ff_analysis[key]['sig']
                color = "#4caf50" if val > 0 else "#f44336"
                opacity = "1.0" if sig else "0.3" 
                
                with f_cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center; opacity:{opacity}; border:1px solid #444; border-radius:5px; padding:5px; background:#111;">
                        <div style="font-size:12px; color:#888;">{label}</div>
                        <div style="font-size:18px; font-weight:bold; color:{color}; font-family:'JetBrains Mono'">{val:+.2f}</div>
                        <div style="font-size:10px; color:#aaa;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("📚 如何解讀這些因子？"):
                st.markdown("""
                * **SMB (規模)**: 正值代表小型股特徵，負值代表大型股。
                * **HML (價值)**: 正值代表價值股（低 P/B），負值代表成長股。
                * **RMW (獲利)**: 正值代表高獲利品質（Robust），負值代表獲利疲弱（Weak）。
                * **CMA (投資)**: 正值代表保守投資（資產擴張慢），負值代表積極擴張（Aggressive）。
                * **MOM (動能)**: 正值代表順勢動能股，負值代表逆勢或動能反轉。
                """)
            st.divider()

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
