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
# 0. 視覺核心 (維持 V88 星際風格)
# =============================================================================
st.set_page_config(page_title="MARCS V89 PEG動態戰情室", layout="wide", page_icon="🧬")

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
    
    /* 卡片與表格 */
    .metric-card {
        background: rgba(18, 18, 22, 0.85); backdrop-filter: blur(12px);
        border-left: 4px solid #ffae00; border-radius: 8px; padding: 15px; margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-left-color: #ffd700; }
    
    .highlight-val { font-size: 28px; font-weight: bold; color: #fff; font-family: 'JetBrains Mono'; }
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
# 1. 數據獲取層 (Robust Data)
# =============================================================================
def robust_download(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        # [V87 Fix] 處理 yfinance 多重索引問題
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0) 
            except: pass
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df
    except: return pd.DataFrame()

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
        return []

# =============================================================================
# 2. 核心引擎 (V86 邏輯 + V89 PEG升級)
# =============================================================================
class Macro_Risk_Engine:
    @staticmethod
    def calculate_market_risk():
        score = 50; details = []
        try:
            vix_df = robust_download("^VIX", "5d")
            vix = vix_df['Close'].iloc[-1] if not vix_df.empty else 20
            
            tnx_df = robust_download("^TNX", "5d")
            tnx = tnx_df['Close'].iloc[-1] if not tnx_df.empty else 4.0
            
            sox_df = robust_download("^SOX", "20d")
            sox = sox_df['Close'] if not sox_df.empty else pd.Series([100])
            
            if vix < 15: score += 15; details.append("VIX低檔")
            elif vix > 25: score -= 20; details.append("VIX恐慌")
            if tnx > 4.5: score -= 10; details.append("美債高利")
            if sox.iloc[-1] > sox.mean(): score += 15
            else: score -= 15; details.append("費半弱勢")
        except: return 50, ["數據連線中..."], 0
        return max(0, min(100, score)), details, vix

class FinMind_Engine:
    @staticmethod
    def get_tw_chips(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        stock_id = ticker.split('.')[0]
        try:
            start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start_date}
            res = requests.get(url, params=params, timeout=3)
            data = res.json()
            if data['msg'] == 'success' and data['data']:
                df = pd.DataFrame(data['data'])
                foreign = df[df['name'] == 'Foreign_Investor']
                if not foreign.empty:
                    latest = foreign.iloc[-1]['buy'] - foreign.iloc[-1]['sell']
                    cum_5d = (foreign.tail(5)['buy'] - foreign.tail(5)['sell']).sum()
                    return {"latest": int(latest/1000), "5d": int(cum_5d/1000), "date": foreign.iloc[-1]['date']}
            return None
        except: return None

class News_Intel_Engine:
    @staticmethod
    def fetch_news(ticker):
        """
        [V89 Upgrade] 高可信度新聞引擎
        1. 美股：使用 yfinance 官方新聞源 (Reuters/Bloomberg)
        2. 台股：使用 Google RSS 但嚴格過濾關鍵字
        3. 回傳：新聞列表 + 綜合情緒分數 (-1.0 ~ 1.0)
        """
        items = []
        sentiment_score = 0 # 累計情緒分
        
        try:
            # --- 策略 A: 美股 (yf.News) ---
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
                        if any(x in tl for x in ["soar","jump","beat","upgrade","record","buy"]): sent="pos"; s_val=1
                        elif any(x in tl for x in ["plunge","drop","miss","downgrade","warn","sell"]): sent="neg"; s_val=-1
                        
                        items.append({"title": title, "link": link, "date": date, "sent": sent})
                        sentiment_score += s_val
                except: pass

            # --- 策略 B: 台股/備援 (Google RSS + Strict Filter) ---
            if not items:
                query = ticker.split('.')[0]
                if ".TW" in ticker: 
                    # [V89] 嚴格關鍵字：只看營收/法說/外資
                    query += " (營收 OR 法說 OR 外資 OR EPS OR 財報) when:7d"
                    lang = "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                else: 
                    query += " stock finance when:7d"
                    lang = "hl=en-US&gl=US&ceid=US:en"
                
                url = f"https://news.google.com/rss/search?q={query}&{lang}"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    count = 0
                    for item in root.findall('.//item'):
                        if count >= 4: break
                        title = item.find('title').text
                        # [V89] 垃圾過濾
                        if any(x in title for x in ["影片","直播","開箱","閒聊","PTT"]): continue
                        
                        link = item.find('link').text
                        pub_date = item.find('pubDate')
                        date = pub_date.text[5:16] if pub_date is not None else "Recent"
                        
                        sent = "neu"; s_val = 0
                        if any(x in title for x in ["漲","高","Bull","Beat","大增","看好"]): sent="pos"; s_val=1
                        elif any(x in title for x in ["跌","低","Bear","Miss","大減","看淡"]): sent="neg"; s_val=-1
                        
                        items.append({"title": title, "link": link, "date": date, "sent": sent})
                        sentiment_score += s_val
                        count += 1
            
            # 正規化情緒分數 (-1 ~ 1)
            final_sentiment = max(-1, min(1, sentiment_score / 3)) if items else 0
            return items, final_sentiment
            
        except: return [], 0

class Scanner_Engine_Elder:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = robust_download(ticker, "6mo")
            if df.empty or len(df) < 50: return None
            
            c = df['Close']; ema22 = c.ewm(span=22).mean()
            score = 60 # 基礎分
            
            if c.iloc[-1] > ema22.iloc[-1]: score += 20
            else: score -= 20
            
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "sl": ema22.iloc[-1]*0.98}
        except: return None

# =============================================================================
# 3. 分析引擎 (Micro / Factor / PEG Valuation / Risk)
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        df = robust_download(ticker, "1y")
        if df.empty or len(df) < 30: return 50, ["數據不足"], df, 0, None
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            # Elder Indicators
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            # MACD & Force
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20; signals.append("Impulse Green")
            if fi_13.iloc[-1] > 0: score += 10
            
            # Chips Integration
            chips = FinMind_Engine.get_tw_chips(ticker)
            if chips:
                if chips['latest'] > 1000: score += 15; signals.append(f"外資買超{chips['latest']}")
                elif chips['latest'] < -1000: score -= 15; signals.append(f"外資賣超{abs(chips['latest'])}")
            
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = ema22 + 2*atr; df['K_Lower'] = ema22 - 2*atr
            
            return score, signals, df, atr, chips
        except: return 50, ["計算錯誤"], df, 0, None

class Factor_Engine:
    @staticmethod
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
        """
        [V89 New] PEG 動態估值模型
        邏輯：新聞情緒影響市場對 PEG 的容忍度，進而修正合理價。
        Fair Price = EPS * (Base PEG * Sentiment_Adj) * Growth
        """
        try:
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 0)
            if price == 0: price = info.get('regularMarketPrice', 0)
            if price == 0: return None
            
            # 1. 獲取核心參數
            pe = info.get('trailingPE', None)
            peg = info.get('pegRatio', None)
            growth = info.get('earningsGrowth', None) # 預估成長率
            
            # 防呆：如果數據缺失，給予保守預設或直接回傳 None
            if not pe or not growth: 
                # 備案：簡單動能估值
                return {"fair": price, "method": "Price Only (No Data)", "peg_used": 0}

            # 如果 PEG 沒抓到，自己算
            if not peg: peg = pe / (growth * 100)
            
            # 2. 新聞情緒修正 (Sentiment Adjustment)
            # 情緒 +1 (極好) -> PEG 可上調 20%
            # 情緒 -1 (極差) -> PEG 下調 20%
            adj_factor = 1 + (sentiment_score * 0.2) 
            target_peg = peg * adj_factor
            
            # 3. 計算合理價
            # Implied Fair PE = Target PEG * (Growth * 100)
            # Fair Price = (Price / PE) * Fair PE  => Price * (Fair PE / PE)
            fair_pe = target_peg * (growth * 100)
            fair_price = (price / pe) * fair_pe
            
            return {
                "fair": fair_price,
                "scenarios": {
                    "Bear": fair_price * 0.85, 
                    "Bull": fair_price * 1.15
                },
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
            risk = capital * 0.02; dist = price - sl
            if dist <= 0: return 0, default
            conf = hybrid / 100.0
            size = int((risk/dist) * conf)
            pos_val = size * price
            pct = (pos_val / capital) * 100
            return size, {"cap": int(pos_val), "pct": round(pct, 1)}
        except: return 0, default

class Message_Generator:
    @staticmethod
    def get_verdict(ticker, hybrid_score, m_score, chips):
        tag = "😐 觀望 (Hold)"; bg_color = "#30363d"
        if hybrid_score >= 80: tag = "🔥 強力買進 (Strong Buy)"; bg_color = "#3fb950"
        elif hybrid_score >= 60: tag = "✅ 買進 (Buy)"; bg_color = "#1f6feb"
        elif hybrid_score <= 40: tag = "❄️ 弱勢 (Weak)"; bg_color = "#888"
        elif hybrid_score <= 20: tag = "⛔ 危險 (Danger)"; bg_color = "#f85149"
        
        reasons = []
        if m_score >= 70: reasons.append("技術面動能強勁")
        if chips and chips['5d'] > 0: reasons.append("外資籌碼進駐")
        
        comment = f"{ticker} 目前呈現 {tag.split(' ')[1]} 狀態。"
        if reasons: comment += "主因：" + "，且".join(reasons) + "。"
        else: comment += "建議耐心等待明確訊號。"
        return tag, comment, bg_color

# =============================================================================
# MAIN UI (V89 整合)
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "2330.TW").upper()
    if st.sidebar.button("分析單一標的"): st.session_state.target = target_in
    
    # 掃描器
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 主動掃描器"):
        market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股"])
        if st.button("🚀 啟動掃描"):
            with st.spinner("Deep Scanning..."):
                tickers = Global_Market_Loader.get_scan_list(market)
                res = []
                bar = st.progress(0)
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
                    futures = {exe.submit(Scanner_Engine_Elder.analyze_single, t): t for t in tickers}
                    done = 0
                    for f in concurrent.futures.as_completed(futures):
                        r = f.result(); done += 1
                        if r: res.append(r)
                        bar.progress(done/len(tickers))
                st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                bar.empty()

    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    target = st.session_state.target

    # --- 1. Risk Gauge ---
    risk_score, risk_dtls, vix = Macro_Risk_Engine.calculate_market_risk()
    r_color = "#4caf50" if risk_score >= 60 else ("#ff9800" if risk_score >= 40 else "#f44336")
    r_text = "BULLISH" if risk_score >= 60 else ("NEUTRAL" if risk_score >= 40 else "BEARISH")
    
    st.markdown(f"""
    <div class="risk-container">
        <div style="display:flex; align-items:center;">
            <div style="border-right:1px solid #555; padding-right:20px; margin-right:20px; text-align:center;">
                <div class="risk-val" style="color:{r_color}">{risk_score}</div>
                <div style="font-size:10px; color:#888;">MARKET RISK</div>
            </div>
            <div>
                <div style="font-size:20px; font-weight:bold; color:#fff;">{r_text}</div>
                <div style="color:#aaa; font-size:12px;">VIX: {vix:.1f} | {' '.join(risk_dtls)}</div>
            </div>
        </div>
        <div style="font-family:'Rajdhani'; color:#ffae00; font-size:24px; font-weight:bold;">MARCS <span style="font-size:14px; color:#888;">V89 PEG</span></div>
    </div>
    """, unsafe_allow_html=True)

    # --- Scanner Results ---
    if st.session_state.scan_results:
        with st.expander(f"🔭 掃描結果 ({len(st.session_state.scan_results)})"):
            df_scan = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df_scan, use_container_width=True)
            sel = st.selectbox("Load:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("Load Ticker"): st.session_state.target = sel

    # --- 2. Main Analysis ---
    with st.spinner(f"Analyzing {target} (PEG Logic + News)..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Pro.analyze, target)
            f_factor = executor.submit(Factor_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target) # 回傳 items, sentiment
            
            m_score, sigs, df_m, atr, chips = f_micro.result()
            factor_data = f_factor.result()
            news_items, sentiment_score = f_news.result()
            
            # [V89] PEG 估值需要新聞情緒
            dcf_res = PEG_Valuation_Engine.calculate(target, sentiment_score)

        hybrid = int((risk_score * 0.3) + (m_score * 0.7))
        curr_p = df_m['Close'].iloc[-1] if not df_m.empty else 0
        sl_p = curr_p - 2.5 * atr if not df_m.empty else 0
        size, risk_dets = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # --- 3. Verdict & Cards ---
    tag, comment, bg_color = Message_Generator.get_verdict(target, hybrid, m_score, chips)
    
    chip_html = ""
    if chips:
        bg = "#f44336" if chips['latest'] < 0 else "#4caf50"
        chip_html = f"<span class='chip-tag' style='background:{bg}; color:white;'>外資 {chips['latest']} 張</span>"
    
    st.markdown(f"<h1 style='color:white;'>{target} <span style='font-size:24px; color:#ffae00;'>${curr_p:.2f}</span> {chip_html}</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="verdict-box" style="background:{bg_color}30; border-color:{bg_color}">
        <h2 style="margin:0; color:{bg_color}; text-shadow:0 0 10px {bg_color}80;">{tag}</h2>
        <p style="margin-top:10px; font-size:18px; color:#e6edf3;">{comment}</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">技術評分 (Elder)</div><div class="highlight-val">{m_score}</div><div class="smart-text">{sigs[0] if sigs else '盤整'}</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">宏觀風險 (Macro)</div><div class="highlight-val">{risk_score}</div><div class="smart-text">{r_text}</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">新聞情緒 (News)</div><div class="highlight-val">{sentiment_score:+.2f}</div><div class="smart-text">PEG 修正係數</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="metric-card"><div class="highlight-lbl">建議倉位 %</div><div class="highlight-val">{risk_dets['pct']}%</div><div class="smart-text">${risk_dets['cap']:,}</div></div>""", unsafe_allow_html=True)

    # --- 4. Tabs ---
    st.markdown("#### 📊 全域戰術圖表")
    tab1, tab2, tab3 = st.tabs(["🕯️ 技術主圖", "🧬 PEG 估值矩陣", "📰 情報中心"])
    
    with tab1:
        if not df_m.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e6edf3', lw=1.5, label='Price')
            ax.plot(df_m.index, df_m['EMA22'], color='#ffae00', lw=1.5, label='EMA 22')
            ax.fill_between(df_m.index, df_m['K_Upper'], df_m['K_Lower'], color='#00f2ff', alpha=0.1)
            ax.axhline(sl_p, color='#f85149', ls='-', label='SL')
            ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e'); ax.grid(True, color='#30363d', alpha=0.3)
            st.pyplot(fig)
            
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols, alpha=0.8); ax1.set_title("MACD Momentum", color='white', fontsize=10)
            ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='#8b949e')
            
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_title("Force Index", color='white', fontsize=10)
            ax2.axhline(0, color='gray', ls='--')
            ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='#8b949e')
            fig2.patch.set_facecolor('#0d1117'); st.pyplot(fig2)
        else: st.warning("無 K 線數據")

    with tab2:
        c_fac, c_val = st.columns(2)
        with c_fac:
            st.markdown("##### 因子屬性 (Factor)")
            if factor_data:
                rows = ""
                for name, score in factor_data['scores'].items():
                    color = "#4caf50" if score >= 60 else ("#ff9800" if score >= 40 else "#f44336")
                    rows += f"<tr><td>{name}</td><td style='width:60%'><div class='factor-bar-bg'><div class='factor-bar-fill' style='width:{score}%; background:{color};'></div></div></td><td style='color:{color}'>{score}</td></tr>"
                st.markdown(f"<table class='factor-table' style='width:100%'>{rows}</table>", unsafe_allow_html=True)
                
        with c_val:
            st.markdown("##### PEG 動態估值 (News Adjusted)")
            if dcf_res and curr_p > 0:
                fair = dcf_res['fair']
                upside = (fair - curr_p) / curr_p * 100
                uc = "#4caf50" if upside > 0 else "#f44336"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span style="color:#aaa;">PEG Fair Value</span>
                        <span style="color:{uc}; font-weight:bold;">{upside:+.1f}% Upside</span>
                    </div>
                    <div style="font-size:32px; font-weight:bold; color:white; font-family:'JetBrains Mono';">${fair:.2f}</div>
                    <div style="font-size:12px; color:#666; margin-top:5px;">Target PEG: {dcf_res['peg_used']} | Impact: {dcf_res['sentiment_impact']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⚠️ 無法計算 PEG (可能因虧損或缺乏成長預測)")

    with tab3:
        if news_items:
            n_cols = st.columns(3)
            for i, item in enumerate(news_items):
                bd = "#4caf50" if item['sent']=="pos" else ("#f44336" if item['sent']=="neg" else "#444")
                with n_cols[i%3]:
                    st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd}; margin-bottom:10px;"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta">{item['date']}</div></div>""", unsafe_allow_html=True)
        else:
            st.info("暫無精準情報")

if __name__ == "__main__":
    main()
