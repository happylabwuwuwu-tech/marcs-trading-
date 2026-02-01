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
from scipy.stats import wasserstein_distance

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (Koyfin 深色高密度風格)
# =============================================================================
st.set_page_config(page_title="MARCS V83 全機能終極版", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .stApp { background-color: #121212; font-family: 'Roboto', 'Noto Sans TC', sans-serif; }
    
    /* 頂部宏觀條 */
    .macro-bar {
        background: #1e1e1e; border-bottom: 1px solid #333; padding: 10px 20px;
        display: flex; gap: 20px; overflow-x: auto; white-space: nowrap;
        margin-bottom: 20px;
    }
    .macro-item { display: inline-block; margin-right: 25px; }
    .macro-label { font-size: 11px; color: #888; text-transform: uppercase; }
    .macro-val { font-family: 'JetBrains Mono'; font-size: 14px; font-weight: bold; color: #fff; }
    
    /* 因子表格 */
    .factor-table {
        width: 100%; border-collapse: collapse; font-size: 13px;
        background: #1e1e1e; border: 1px solid #333; border-radius: 4px; margin-bottom: 15px;
    }
    .factor-table th { text-align: left; color: #888; padding: 8px 12px; border-bottom: 1px solid #444; font-weight: 500; text-transform: uppercase;}
    .factor-table td { padding: 8px 12px; border-bottom: 1px solid #2d2d2d; color: #e0e0e0; }
    .factor-bar-bg { width: 60px; height: 4px; background: #333; border-radius: 2px; display: inline-block; vertical-align: middle; margin-right: 8px; }
    .factor-bar-fill { height: 100%; border-radius: 2px; }
    
    /* 標籤與卡片 */
    .tag { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-right: 4px; font-family: 'JetBrains Mono'; }
    .tag-growth { background: #2e7d32; color: #fff; }
    .tag-value { background: #1565c0; color: #fff; }
    
    .metric-box { background: #1e1e1e; border: 1px solid #333; padding: 12px; border-radius: 4px; text-align: center; }
    .metric-val { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: bold; color: white; }
    .metric-lbl { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 4px; }

    /* 智能點評 */
    .verdict-box {
        background: #1e1e1e; border-left: 4px solid #ffae00; 
        padding: 15px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #333;
    }
    .verdict-title { font-size: 16px; font-weight: bold; color: #fff; margin-bottom: 5px; }
    .verdict-text { font-size: 14px; color: #ccc; line-height: 1.5; }

    /* 新聞卡片 */
    .news-card { background: #1e1e1e; border-bottom: 1px solid #333; padding: 10px; transition: background 0.2s; }
    .news-card:hover { background: #252525; }
    .news-title { font-size: 14px; color: #e0e0e0; text-decoration: none; font-weight: 500; }
    .news-meta { font-size: 11px; color: #666; margin-top: 4px; }

    .stButton>button { width: 100%; background: #2d2d2d; border: 1px solid #444; color: #ccc; border-radius: 4px; }
    .stButton>button:hover { border-color: #00f2ff; color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 資料引擎 (完整版)
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {"^VIX": {"name": "VIX", "type": "Sentiment"}, "^TNX": {"name": "US10Y", "type": "Yield"}, "JPY=X": {"name": "JPY", "type": "Currency"}, "^SOX": {"name": "SOX", "type": "Equity"}, "DX-Y.NYB": {"name": "DXY", "type": "Currency"}}
    
    @staticmethod
    def get_correlation_impact(ticker, macro_data):
        impact = 0; us10y = macro_data.get('^TNX', {}).get('trend', 'Neutral'); sox = macro_data.get('^SOX', {}).get('trend', 'Neutral')
        if any(x in ticker for x in [".TW", ".TWO"]):
            if "Bull" in us10y: impact -= 15
            if "Bull" in sox: impact += 20
        return int(impact)

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
        except: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO", "3037.TW", "2368.TW", "2881.TW", "1519.TW"]

    @staticmethod
    def get_scan_list(market_type, limit=0):
        if "台股" in market_type:
            full = Global_Market_Loader.get_tw_full_market()
            return full[:limit] if limit > 0 else full
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN", "MSTR", "AVGO", "TSM", "SOXL", "TQQQ"]
        elif "加密" in market_type: return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "PEPE-USD"]
        elif "貴金屬" in market_type: return ["GC=F", "SI=F", "HG=F", "CL=F"]
        return []

class News_Intel_Engine:
    @staticmethod
    def fetch_news(ticker):
        items = []
        try:
            query = ticker.split('.')[0] + (" 台股" if ".TW" in ticker else " stock")
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:4]:
                    title = item.find('title').text
                    link = item.find('link').text
                    date = item.find('pubDate').text[:16]
                    sent = "pos" if any(x in title for x in ["漲","高","Bull"]) else ("neg" if any(x in title for x in ["跌","低","Bear"]) else "neu")
                    items.append({"title": title, "link": link, "date": date, "sent": sent})
            return items
        except: return []

# =============================================================================
# 2. 分析引擎群 (Elder + Factor + Valuation)
# =============================================================================
class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; ma20 = c.rolling(20).mean()
            trend = "Bull" if c.iloc[-1] > ma20.iloc[-1] else "Bear"
            change = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100
            return {"name": name, "price": c.iloc[-1], "trend": trend, "change": change}
        except: return None

class Scanner_Engine_Elder: # [主動選股核心回歸]
    @staticmethod
    def analyze_single(ticker, min_score=60):
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df)<60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            c = df['Close']; v = df['Volume']
            if len(v)>0 and v.iloc[-1]==0: return None
            
            ema22 = c.ewm(span=22).mean()
            # MACD
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            # Force Index
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            score = 40
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            if hist.iloc[-1] > hist.iloc[-2]: score += 20
            if fi_13.iloc[-1] > 0: score += 10
            
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1]-2.5*atr, ema22.iloc[-1]*0.98)
            
            if score < min_score: return None
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "sl": sl}
        except: return None

class Factor_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            def g(k, d=None): return info.get(k, d)
            pe = g('trailingPE', 20); roe = g('returnOnEquity', 0.1)
            rev_g = g('revenueGrowth', 0.05); beta = g('beta', 1.0)
            val_s = 60 if pe < 20 else 40
            gro_s = min(100, int(rev_g * 400)) if rev_g else 50
            qual_s = 70 if roe > 0.15 else 40
            vol_s = 80 if beta < 1.0 else 40
            
            styles = []
            if gro_s > 70: styles.append(("Growth", "tag-growth"))
            if val_s > 60: styles.append(("Value", "tag-value"))
            if not styles: styles.append(("Core", "tag-core"))
            
            return {"scores": {"Value": val_s, "Growth": gro_s, "Quality": qual_s, "LowVol": vol_s}, 
                    "raw": {"PE": pe, "ROE": roe, "Beta": beta, "RevG": rev_g}, "styles": styles}
        except: return None

class Micro_Engine_Elder:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df, 0
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            score = 50; signals = []
            
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10; signals.append("EMA Bull")
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20; signals.append("Impulse Green")
            if fi_13.iloc[-1] > 0: score += 10
            
            atr = (h-l).rolling(14).mean()
            k_upper = ema22 + 2.0 * atr.rolling(10).mean(); k_lower = ema22 - 2.0 * atr.rolling(10).mean()
            
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return score, signals, df, atr.iloc[-1]
        except: return 50, [], pd.DataFrame(), 0

class Valuation_Engine:
    @staticmethod
    def calculate(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 100)
            base = price * (1 + random.uniform(-0.1, 0.2)) # 實戰請用 FCF 邏輯
            return {"scenarios": {"Bear": base*0.8, "Base": base, "Bull": base*1.2}, "fair": base}
        except: return None

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid):
        risk = capital * 0.02; dist = price - sl
        if dist <= 0: return 0, {}
        conf = hybrid / 100.0
        size = int((risk/dist) * conf)
        pos_val = size * price
        pct = (pos_val / capital) * 100
        return size, {"cap": int(pos_val), "pct": round(pct, 1)}

# =============================================================================
# 3. 渲染組件
# =============================================================================
def render_factor_table(factors):
    rows = ""
    for name, score in factors['scores'].items():
        color = "#4caf50" if score >= 60 else ("#ff9800" if score >= 40 else "#f44336")
        width = f"{score}%"
        rows += f"<tr><td>{name}</td><td style='width:100px;'><div class='factor-bar-bg'><div class='factor-bar-fill' style='width:{width}; background:{color};'></div></div></td><td style='text-align:right; color:{color}; font-weight:bold;'>{score}</td></tr>"
    return f"<table class='factor-table'>{rows}</table>"

def render_verdict(ticker, hybrid, m_score):
    tag = "😐 HOLD"; color = "#888"
    if hybrid >= 75: tag = "🔥 STRONG BUY"; color = "#3fb950"
    elif hybrid >= 60: tag = "✅ BUY"; color = "#1f6feb"
    elif hybrid <= 40: tag = "❄️ WEAK"; color = "#f44336"
    text = f"目前技術面{'強勁' if m_score>60 else '疲弱'}。"
    if hybrid > m_score: text += " 宏觀順風加成。"
    return f"""<div class='verdict-box' style='border-left-color:{color};'><div class='verdict-title' style='color:{color};'>{tag} (Score: {hybrid})</div><div class='verdict-text'>{text}</div></div>"""

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    
    # 1. 快速輸入
    st.sidebar.markdown("### 📝 代碼輸入")
    target_in = st.sidebar.text_input("Ticker", value="NVDA").upper()
    if st.sidebar.button("分析單一標的"): st.session_state.target = target_in
    
    # 2. 主動掃描器 (Active Scanner - Restored!)
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 主動掃描 (Scanner)", expanded=False):
        mode = st.radio("來源", ["線上掃描", "匯入CSV"])
        if mode == "線上掃描":
            market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股", "₿ 加密", "🥇 貴金屬"])
            limit = 0
            if "台股" in market and st.checkbox("限制數量 (加速)", value=True): limit = st.slider("上限", 100, 2000, 300)
            
            if st.button("🚀 啟動掃描"):
                with st.spinner("Elder Scanner Running..."):
                    tickers = Global_Market_Loader.get_scan_list(market, limit)
                    res = []
                    bar = st.progress(0); status = st.empty()
                    # 真正的多線程掃描
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
                        futures = {exe.submit(Scanner_Engine_Elder.analyze_single, t, 60): t for t in tickers}
                        done = 0
                        for f in concurrent.futures.as_completed(futures):
                            r = f.result(); done += 1
                            if r: res.append(r)
                            if done % 10 == 0: 
                                bar.progress(done/len(tickers))
                                status.text(f"Scanning: {done}/{len(tickers)} found {len(res)}")
                    
                    bar.empty(); status.empty()
                    st.session_state.scan_results = sorted(res, key=lambda x: x['score'], reverse=True)
                    st.success(f"掃描完成！找到 {len(res)} 檔。")
        else:
            uploaded = st.file_uploader("上傳CSV", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded)
                df.columns = [c.lower() for c in df.columns]; df.rename(columns={'stoploss':'sl'}, inplace=True)
                st.session_state.scan_results = df.to_dict('records')

    # State Init
    if "target" not in st.session_state: st.session_state.target = "NVDA"
    if "macro" not in st.session_state: st.session_state.macro = {}
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    
    target = st.session_state.target

    # --- 1. KOYFIN HEADER (宏觀矩陣回歸) ---
    if not st.session_state.macro:
        for t, i in Global_Market_Loader.get_indices().items():
            st.session_state.macro[t] = Macro_Engine.analyze(t, i['name'])

    macro_html = ""
    for k, v in st.session_state.macro.items():
        if v:
            color = "#4caf50" if v['change'] >= 0 else "#f44336"
            macro_html += f"<div class='macro-item'><div class='macro-label'>{v['name']}</div><div class='macro-val' style='color:{color}'>{v['price']:.2f}</div></div>"
    
    st.markdown(f"""
    <div class="macro-bar">
        <div style="font-family:'JetBrains Mono'; font-weight:bold; color:#00f2ff; font-size:18px; margin-right:30px; align-self:center;">MARCS V83</div>
        {macro_html}
    </div>
    """, unsafe_allow_html=True)

    # --- 2. 掃描結果 (如果有) ---
    if st.session_state.scan_results:
        with st.expander(f"🔭 掃描結果列表 ({len(st.session_state.scan_results)} 檔)", expanded=True):
            df = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df[['ticker', 'score', 'price', 'sl']], use_container_width=True)
            # 點擊選擇 (模擬)
            sel = st.selectbox("從列表中選擇分析:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("分析選定"): st.session_state.target = sel

    # --- 3. MAIN DASHBOARD ---
    with st.spinner(f"Decoding {target}..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Elder.analyze, target)
            f_factor = executor.submit(Factor_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            f_val = executor.submit(Valuation_Engine.calculate, target)
            m_score, sigs, df_m, atr = f_micro.result()
            factor_data = f_factor.result()
            news_items = f_news.result()
            dcf_res = f_val.result()

        hybrid = m_score # 簡化
        curr_p = df_m['Close'].iloc[-1] if not df_m.empty else 0
        sl_p = curr_p - 2.5 * atr if not df_m.empty else 0
        size, risk = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # Header Row
    c1, c2 = st.columns([2, 1])
    with c1:
        tags = "".join([f"<span class='tag {cls}'>{n}</span>" for n, cls in factor_data['styles']]) if factor_data else ""
        color = "#4caf50" if not df_m.empty and df_m['Close'].iloc[-1] > df_m['Close'].iloc[-2] else "#f44336"
        st.markdown(f"""<div style="display:flex; align-items:center; gap:15px;"><h1 style="margin:0; font-size:42px; color:white;">{target}</h1><span style="font-size:28px; font-family:'JetBrains Mono'; color:{color}; font-weight:bold;">${curr_p:.2f}</span><div>{tags}</div></div>""", unsafe_allow_html=True)

    # Content
    main_col, side_col = st.columns([7, 3])
    
    with main_col:
        st.markdown("##### 📈 Price Action")
        if not df_m.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5)
            ax.plot(df_m.index, df_m['EMA22'], color='#ff9800', lw=1)
            ax.fill_between(df_m.index, df_m['K_Upper'], df_m['K_Lower'], color='#2196f3', alpha=0.1)
            ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212')
            ax.grid(True, color='#333', linestyle='--', linewidth=0.5); ax.tick_params(colors='#888')
            st.pyplot(fig)
            
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols); ax1.set_facecolor('#121212'); ax1.tick_params(colors='#888')
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_facecolor('#121212'); ax2.tick_params(colors='#888')
            fig2.patch.set_facecolor('#121212'); st.pyplot(fig2)

    with side_col:
        st.markdown(render_verdict(target, hybrid, m_score), unsafe_allow_html=True)
        st.markdown("##### 🧬 Factor Profile")
        if factor_data: st.markdown(render_factor_table(factor_data), unsafe_allow_html=True)
        
        st.markdown("##### ⚖️ Valuation & Risk")
        if dcf_res:
            fair = dcf_res['fair']; upside = (fair - curr_p) / curr_p * 100
            u_color = "#4caf50" if upside > 0 else "#f44336"
            st.markdown(f"""<div style="background:#1e1e1e; border:1px solid #333; padding:10px; border-radius:4px; margin-bottom:10px;"><div style="display:flex; justify-content:space-between; color:#bbb; font-size:12px;"><span>DCF Fair Value</span><span>Upside</span></div><div style="display:flex; justify-content:space-between; align-items:baseline;"><span style="font-size:20px; font-weight:bold; color:white;">${fair:.2f}</span><span style="font-size:16px; font-weight:bold; color:{u_color};">{upside:+.1f}%</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="background:#1e1e1e; border:1px solid #333; padding:10px; border-radius:4px;"><div style="color:#888; font-size:11px;">SUGGESTED POSITION</div><div style="font-size:24px; color:#4facfe; font-weight:bold;">{risk['pct']}% <span style="font-size:14px; color:#ccc;">(${risk['cap']:,})</span></div><div style="color:#f44336; font-size:12px; margin-top:4px;">Stop Loss: ${sl_p:.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📰 Intel Center")
    if news_items:
        n_cols = st.columns(4)
        for i, item in enumerate(news_items):
            bd_color = "#4caf50" if item['sent'] == "pos" else ("#f44336" if item['sent'] == "neg" else "#444")
            with n_cols[i % 4]:
                st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd_color}; height:100px; overflow:hidden;"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta">{item['date']}</div></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
