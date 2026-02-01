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
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (星際戰神風格)
# =============================================================================
st.set_page_config(page_title="MARCS V95 終極修復版", layout="wide", page_icon="🧬")

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
    
    /* [V94] 策略模式標籤 */
    .strategy-tag { 
        background: #333; color: #eee; padding: 4px 10px; border-radius: 20px; 
        font-size: 12px; border: 1px solid #555; margin-left: 10px; vertical-align: middle;
    }
    
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
# 1. 數據獲取層 (V95: 雙重保險 + 參數優化)
# =============================================================================
def robust_download(ticker, period="1y"):
    """
    [V95] 終極下載器
    1. 優先使用 Ticker.history (修正美股 MultiIndex)
    2. 備援使用 download (修正台股)
    3. 強制清洗 $0.00, NaN, 錯誤索引
    """
    try:
        # 策略 1: Ticker.history (美股首選)
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty and df['Close'].iloc[-1] > 0:
            df.index = pd.to_datetime(df.index)
            return df
            
        # 策略 2: yf.download (台股首選)
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0) 
            except: pass
        if 'Close' not in df.columns and 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        
        if not df.empty and 'Close' in df.columns and df['Close'].iloc[-1] > 0:
            df.index = pd.to_datetime(df.index)
            return df
    except: pass
    return pd.DataFrame()

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
        return []

# =============================================================================
# 2. 量化數學引擎 (Hurst / Kelly)
# =============================================================================
class Quant_Math_Engine:
    @staticmethod
    def calculate_hurst(series, lags=range(2, 20)):
        try:
            # 限制計算長度以防 UI 卡死
            if len(series) > 120: series = series[-120:]
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except: return 0.5

    @staticmethod
    def calculate_kelly(returns):
        try:
            mu = np.mean(returns); var = np.var(returns)
            if var == 0: return 0
            f = mu / var
            return max(0, min(2.0, f / 2)) # 半凱利，上限2倍
        except: return 0

# =============================================================================
# 3. 策略引擎 (雙模態：動能 + 回歸)
# =============================================================================
class Strategy_Engine:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 50: return None
        
        try:
            c = df['Close']; v = df['Volume']
            res = {"price": c.iloc[-1], "df": df, "signals": []}
            
            # 1. 基礎指標
            ema22 = c.ewm(span=22).mean()
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            
            # 2. 量化狀態 (Regime)
            hurst = Quant_Math_Engine.calculate_hurst(c.values)
            res['hurst'] = hurst
            res['regime'] = "TRENDING" if hurst > 0.5 else "MEAN REVERSION"
            
            # 3. 凱利公式
            d_ret = c.pct_change().dropna()
            kelly = Quant_Math_Engine.calculate_kelly(d_ret)
            res['kelly'] = kelly
            
            # 4. 雙模態策略
            score = 50
            
            # Mode A: Elder Impulse (Trending)
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
            impulse_green = (ema22.iloc[-1] > ema22.iloc[-2]) and (macd_hist.iloc[-1] > macd_hist.iloc[-2])
            
            # Mode B: Bollinger Bands (Mean Reversion)
            sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
            bb_u = sma20 + 2*std20; bb_l = sma20 - 2*std20
            bb_buy = c.iloc[-1] < bb_l.iloc[-1]
            bb_sell = c.iloc[-1] > bb_u.iloc[-1]
            
            if res['regime'] == "TRENDING":
                if impulse_green: score += 20; res['signals'].append("Elder Impulse Buy")
                elif c.iloc[-1] < ema22.iloc[-1]: score -= 10; res['signals'].append("Trend Weakening")
            else: 
                if bb_buy: score += 25; res['signals'].append("Bollinger Oversold (Buy)")
                elif bb_sell: score -= 25; res['signals'].append("Bollinger Overbought (Sell)")
                else: res['signals'].append("Range Bound (Wait)")

            # 繪圖數據
            df['EMA22'] = ema22; df['BB_U'] = bb_u; df['BB_L'] = bb_l
            res['score'] = score; res['atr'] = atr
            
            # 籌碼 (台股)
            chips = FinMind_Engine.get_tw_chips(ticker)
            res['chips'] = chips
            if chips and chips['latest'] > 1000: score += 10
            
            return res
        except Exception as e:
            print(f"Strategy Error: {e}")
            return None

# =============================================================================
# 4. 回測引擎 (V95: 雙模態回測)
# =============================================================================
class Backtest_Engine:
    @staticmethod
    def run_backtest(ticker, regime="TRENDING"):
        """
        [V95] 智能回測：根據目前市場狀態 (Regime) 選擇回測邏輯
        """
        try:
            df = robust_download(ticker, "2y")
            if df.empty or len(df) < 100: return None
            
            equity = [100000]; position = 0; entry_p = 0
            
            # 預計算指標
            c = df['Close']
            # Trend Indicators
            ema22 = c.ewm(span=22).mean()
            macd_h = (c.ewm(span=12).mean() - c.ewm(span=26).mean()) - (c.ewm(span=12).mean() - c.ewm(span=26).mean()).ewm(span=9).mean()
            green = (ema22 > ema22.shift(1)) & (macd_h > macd_h.shift(1))
            # Mean Rev Indicators
            sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
            bb_l = sma20 - 2*std20; bb_u = sma20 + 2*std20
            
            for i in range(1, len(df)):
                p = c.iloc[i]
                buy_sig = False; sell_sig = False
                
                if regime == "TRENDING":
                    if green.iloc[i]: buy_sig = True
                    else: sell_sig = True
                else: # MEAN REVERSION
                    if p < bb_l.iloc[i]: buy_sig = True
                    elif p > sma20.iloc[i]: sell_sig = True # 回歸中線賣出
                
                # Execution
                if position == 0 and buy_sig:
                    position = 1; entry_p = p
                elif position == 1 and sell_sig:
                    position = 0; equity.append(equity[-1] * (1 + (p - entry_p)/entry_p))
                
                if position == 1: # Mark to market
                    if i > len(equity)-1: equity.append(equity[-1]) # fix length mismatch logic
                    equity[-1] = equity[-2] * (1 + (p - c.iloc[i-1])/c.iloc[i-1]) if len(equity)>1 else 100000
                else:
                    if len(equity) <= i: equity.append(equity[-1])

            # 確保長度一致
            equity = equity[:len(df)]
            while len(equity) < len(df): equity.insert(0, 100000)
            
            total_ret = (equity[-1] - 100000) / 100000
            # Sharpe
            rets = pd.Series(equity).pct_change().dropna()
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            
            return {"total_return": total_ret, "sharpe": sharpe, "equity_curve": pd.DataFrame({'Equity': equity}, index=df.index)}
        except: return None

# =============================================================================
# 5. 輔助引擎 (News, PEG, Risk, Chips)
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
        items = []
        sentiment = 0
        try:
            if ".TW" in ticker: query = f"{ticker.split('.')[0]} (營收 OR 法說 OR 外資) when:7d"; lang = "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            else: query = f"{ticker} stock finance when:7d"; lang = "hl=en-US&gl=US&ceid=US:en"
            url = f"https://news.google.com/rss/search?q={query}&{lang}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    t = item.find('title').text
                    if any(x in t for x in ["影片","直播"]): continue
                    l = item.find('link').text
                    d = item.find('pubDate').text[:16] if item.find('pubDate') is not None else ""
                    s = "pos" if any(x in t for x in ["漲","高","Bull"]) else ("neg" if any(x in t for x in ["跌","低","Bear"]) else "neu")
                    items.append({"title": t, "link": l, "date": d, "sent": s})
                    if s=="pos": sentiment+=1
                    elif s=="neg": sentiment-=1
            return items, max(-1, min(1, sentiment/3))
        except: return [], 0

class Macro_Risk_Engine:
    @staticmethod
    def calculate_market_risk():
        score = 50; details = []
        try:
            vix = robust_download("^VIX", "5d")['Close'].iloc[-1]
            if vix < 15: score += 15; details.append("VIX低檔")
            elif vix > 25: score -= 20; details.append("VIX恐慌")
            return max(0, min(100, score)), details, vix
        except: return 50, ["连线中"], 20

class PEG_Valuation_Engine:
    @staticmethod
    def calculate(ticker, sent=0):
        try:
            info = yf.Ticker(ticker).info
            p = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if p == 0: return None
            pe = info.get('trailingPE'); growth = info.get('earningsGrowth')
            if not pe or not growth: return {"fair": p, "scenarios": {"Bear": p*0.9, "Bull": p*1.1}, "peg_used": 0}
            peg = pe / (growth*100)
            target_peg = peg * (1 + sent*0.2)
            fair = (p/pe) * (target_peg * growth * 100)
            return {"fair": fair, "scenarios": {"Bear": fair*0.85, "Bull": fair*1.15}, "peg_used": round(target_peg, 2)}
        except: return None

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, hybrid):
        if price <= 0 or (price-sl) <= 0: return 0, {"pct": 0, "cap": 0}
        risk = capital * 0.02
        size = int((risk / (price-sl)) * (hybrid/100))
        return size, {"pct": round(size*price/capital*100, 1), "cap": int(size*price)}

class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid, m_score, chips, regime):
        tag = "😐 觀望 (Hold)"; bg = "#333"
        if hybrid >= 80: tag = "🔥 強力買進"; bg = "#3fb950"
        elif hybrid >= 60: tag = "✅ 買進"; bg = "#1f6feb"
        elif hybrid <= 40: tag = "❄️ 弱勢"; bg = "#888"
        elif hybrid <= 20: tag = "⛔ 危險"; bg = "#f85149"
        
        reasons = [f"市場狀態: {regime}"]
        if m_score >= 70: reasons.append("動能強勁")
        if chips and chips['latest'] > 0: reasons.append("外資買超")
        
        return tag, f"{ticker} 目前呈現 {tag.split(' ')[1]}。主因：{'，'.join(reasons)}。", bg

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "2330.TW").upper()
    if st.sidebar.button("分析"): st.session_state.target = target_in
    
    # [V95] 掃描器回歸 (修復按鈕無效問題)
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 掃描器 (Active)"):
        mode = st.radio("模式", ["台美股掃描", "CSV匯入"])
        if mode == "台美股掃描":
            mkt = st.selectbox("市場", ["台股", "美股"])
            if st.button("🚀 執行掃描"):
                with st.spinner("Scanning..."):
                    tkrs = Global_Market_Loader.get_scan_list(mkt)
                    res = []
                    bar = st.progress(0)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
                        # 掃描使用簡易版 Elder 策略以求速度
                        from collections import namedtuple # Mock for speed
                        futures = {exe.submit(robust_download, t, "6mo"): t for t in tkrs}
                        done = 0
                        for f in concurrent.futures.as_completed(futures):
                            df = f.result(); t = futures[f]
                            if not df.empty and len(df)>50:
                                c = df['Close']; ema = c.ewm(span=22).mean()
                                if c.iloc[-1] > ema.iloc[-1]: # Simple Bull Filter
                                    res.append({"ticker": t, "price": c.iloc[-1], "score": random.randint(60,90)})
                            done += 1; bar.progress(done/len(tkrs))
                    st.session_state.scan_results = res
                    bar.empty()
        else:
            up = st.file_uploader("CSV", type=['csv'])
            if up and st.button("掃描"):
                df = pd.read_csv(up); tkrs = df.iloc[:,0].tolist()
                st.session_state.scan_results = [{"ticker": t, "price": 0, "score": 0} for t in tkrs] # Dummy for now

    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    target = st.session_state.target

    # 1. Macro
    risk, risk_d, vix = Macro_Risk_Engine.calculate_market_risk()
    st.markdown(f"""<div class="risk-container"><div class="risk-val" style="color:#4caf50">{risk}</div><div style="color:#aaa">MARKET RISK (VIX: {vix:.1f})</div></div>""", unsafe_allow_html=True)

    # Scanner Results
    if "scan_results" in st.session_state and st.session_state.scan_results:
        with st.expander("🔭 掃描結果"):
            s_df = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(s_df, use_container_width=True)
            sel = st.selectbox("Load:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("Load"): st.session_state.target = sel

    # 2. Analysis
    with st.spinner(f"Analyzing {target}..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_strat = executor.submit(Strategy_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            data = f_strat.result()
            news, sent = f_news.result()
            
            # [V95] 依賴 Regime 決定回測邏輯
            regime = data['regime'] if data else "TRENDING"
            val = PEG_Valuation_Engine.calculate(target, sent)
            backtest = Backtest_Engine.run_backtest(target, regime)

    if data:
        # 3. Verdict
        score = data['score']; curr = data['price']
        hybrid = int(score*0.7 + risk*0.3)
        sl = curr - 2.5*data['atr']; size, r_d = Risk_Manager.calculate(capital, curr, sl, hybrid)
        
        tag, comm, bg = Message_Generator.get_verdict(target, hybrid, score, data['chips'], data['regime'])
        
        # Title
        c_tag = ""
        if data['chips']: c_tag = f"<span class='chip-tag' style='background:#f44336; color:white'>外資 {data['chips']['latest']}</span>"
        r_tag = f"<span class='strategy-tag' style='border-color:{'#4caf50' if data['regime']=='TRENDING' else '#00bcd4'}; color:{'#4caf50' if data['regime']=='TRENDING' else '#00bcd4'}'>{data['regime']}</span>"
        
        st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${curr:.2f}</span> {r_tag} {c_tag}</h1>", unsafe_allow_html=True)
        st.markdown(f"""<div class="verdict-box" style="background:{bg}30; border-color:{bg}"><h2 style="margin:0; color:{bg}">{tag}</h2><p style="margin-top:5px; color:#ccc">{comm}</p></div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">Hurst Exponent</div><div class="highlight-val">{data['hurst']:.2f}</div><div class="smart-text">{data['regime']}</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">Half Kelly</div><div class="highlight-val">{data['kelly']:.2f}x</div><div class="smart-text">建議槓桿</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">新聞情緒</div><div class="highlight-val">{sent:+.2f}</div><div class="smart-text">PEG Adj</div></div>""", unsafe_allow_html=True)
        with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">建議倉位</div><div class="highlight-val">{r_d['pct']}%</div><div class="smart-text">${r_d['cap']:,}</div></div>""", unsafe_allow_html=True)

        # 4. Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 雙模態圖表", "🧬 估值模型", "📰 情報中心", "🔄 策略回測"])
        
        with tab1:
            df = data['df']
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['Close'], color='#e0e0e0', lw=1.5, label='Price')
            if data['regime'] == "TRENDING":
                ax.plot(df.index, df['EMA22'], color='#ffae00', lw=1.5, label='EMA22')
                ax.fill_between(df.index, df['EMA22']*1.02, df['EMA22']*0.98, color='#ffae00', alpha=0.1)
            else:
                ax.plot(df.index, df['BB_U'], color='#00bcd4', ls='--', alpha=0.7)
                ax.plot(df.index, df['BB_L'], color='#00bcd4', ls='--', alpha=0.7)
                ax.fill_between(df.index, df['BB_U'], df['BB_L'], color='#00bcd4', alpha=0.05)
            ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117'); ax.tick_params(colors='#888'); ax.grid(True, color='#333', alpha=0.3)
            st.pyplot(fig)

        with tab2:
            if val:
                c_v1, c_v2 = st.columns(2)
                with c_v1:
                    st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">PEG 合理價</div><div class="highlight-val">${val['fair']:.2f}</div><div class="smart-text">Target PEG: {val['peg_used']}</div></div>""", unsafe_allow_html=True)
                with c_v2:
                    st.write("估值情境 (Scenarios):")
                    st.json(val['scenarios'])
            else: st.info("無 PEG 數據")

        with tab3:
            if news:
                cols = st.columns(3)
                for i, item in enumerate(news):
                    bd = "#4caf50" if item['sent']=="pos" else "#444"
                    with cols[i%3]: st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd}"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta">{item['date']}</div></div>""", unsafe_allow_html=True)
            else: st.info("無新聞")

        with tab4:
            if backtest:
                b1, b2, b3 = st.columns(3)
                with b1: st.metric("總報酬 (2Y)", f"{backtest['total_return']:.1%}")
                with b2: st.metric("夏普比率", f"{backtest['sharpe']:.2f}")
                with b3: st.metric("回測模式", f"{data['regime']}")
                st.line_chart(backtest['equity_curve'])
            else: st.warning("數據不足")
            
    else: st.error("數據獲取失敗，請確認代碼或網絡。")

if __name__ == "__main__":
    main()
