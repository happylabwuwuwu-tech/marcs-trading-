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
import xml.etree.ElementTree as ET # 用於新聞解析
from scipy.stats import wasserstein_distance

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (Koyfin 深色高密度風格)
# =============================================================================
st.set_page_config(page_title="MARCS V82 究極融合版", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .stApp { background-color: #121212; font-family: 'Roboto', 'Noto Sans TC', sans-serif; }
    
    /* Koyfin 風格表格 */
    .factor-table {
        width: 100%; border-collapse: collapse; font-size: 13px;
        background: #1e1e1e; border: 1px solid #333; border-radius: 4px; margin-bottom: 15px;
    }
    .factor-table th { text-align: left; color: #888; padding: 8px 12px; border-bottom: 1px solid #444; font-weight: 500; text-transform: uppercase;}
    .factor-table td { padding: 8px 12px; border-bottom: 1px solid #2d2d2d; color: #e0e0e0; }
    .factor-bar-bg { width: 60px; height: 4px; background: #333; border-radius: 2px; display: inline-block; vertical-align: middle; margin-right: 8px; }
    .factor-bar-fill { height: 100%; border-radius: 2px; }
    
    /* 標籤與數據 */
    .tag { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-right: 4px; font-family: 'JetBrains Mono'; }
    .tag-growth { background: #2e7d32; color: #fff; }
    .tag-value { background: #1565c0; color: #fff; }
    .tag-mom { background: #c62828; color: #fff; }
    .tag-core { background: #444; color: #ccc; }
    
    .metric-box { background: #1e1e1e; border: 1px solid #333; padding: 12px; border-radius: 4px; text-align: center; }
    .metric-val { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: bold; color: white; }
    .metric-lbl { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 4px; }

    /* 智能點評區塊 (V75 Style adapted for V81) */
    .verdict-box {
        background: #1e1e1e; border-left: 4px solid #ffae00; 
        padding: 15px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #333;
    }
    .verdict-title { font-size: 16px; font-weight: bold; color: #fff; margin-bottom: 5px; }
    .verdict-text { font-size: 14px; color: #ccc; line-height: 1.5; }

    /* 新聞卡片 (V77 Style) */
    .news-card {
        background: #1e1e1e; border-bottom: 1px solid #333; padding: 10px; 
        transition: background 0.2s;
    }
    .news-card:hover { background: #252525; }
    .news-title { font-size: 14px; color: #e0e0e0; text-decoration: none; font-weight: 500; }
    .news-meta { font-size: 11px; color: #666; margin-top: 4px; }

    .stButton>button { width: 100%; background: #2d2d2d; border: 1px solid #444; color: #ccc; border-radius: 4px; }
    .stButton>button:hover { border-color: #00f2ff; color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 資料獲取引擎群 (Data Engines)
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
# 2. 核心分析引擎群 (Analysis Engines)
# =============================================================================
class Factor_Engine: # [V81]
    @staticmethod
    def analyze(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            def g(k, d=None): return info.get(k, d)
            
            # 因子數據
            pe = g('trailingPE', 20); pb = g('priceToBook', 3); roe = g('returnOnEquity', 0.1)
            rev_g = g('revenueGrowth', 0.05); beta = g('beta', 1.0)
            
            # 評分 (簡易版)
            val_s = 60 if pe < 20 else 40
            gro_s = min(100, int(rev_g * 400)) if rev_g else 50
            qual_s = 70 if roe > 0.15 else 40
            vol_s = 80 if beta < 1.0 else 40
            mom_s = 60 # 需技術面補充
            
            styles = []
            if gro_s > 70: styles.append(("Growth", "tag-growth"))
            if val_s > 60: styles.append(("Value", "tag-value"))
            if not styles: styles.append(("Core", "tag-core"))
            
            return {"scores": {"Value": val_s, "Growth": gro_s, "Quality": qual_s, "LowVol": vol_s}, 
                    "raw": {"PE": pe, "ROE": roe, "Beta": beta, "RevG": rev_g}, "styles": styles}
        except: return None

class Micro_Engine_Elder: # [V74 Elder 邏輯]
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
            
            # MACD
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            
            # Force Index
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            # Impulse System
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): 
                score += 20; signals.append("Impulse Green")
            if fi_13.iloc[-1] > 0: score += 10
            
            atr = (h-l).rolling(14).mean()
            k_upper = ema22 + 2.0 * atr.rolling(10).mean(); k_lower = ema22 - 2.0 * atr.rolling(10).mean()
            
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return score, signals, df, atr.iloc[-1]
        except: return 50, [], pd.DataFrame(), 0

class Valuation_Engine: # [V79 DCF]
    @staticmethod
    def calculate(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 100)
            # 簡易模擬 DCF 區間 (實戰請接真實財報)
            base = price * (1 + random.uniform(-0.1, 0.2))
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
# 3. 渲染組件 (UI Helpers)
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
    
    text = f"目前技術面動能{'強勁' if m_score>60 else '疲弱'}。"
    if hybrid > m_score: text += " 受惠於宏觀順風，評分獲得加成。"
    
    return f"""<div class='verdict-box' style='border-left-color:{color};'><div class='verdict-title' style='color:{color};'>{tag} (Score: {hybrid})</div><div class='verdict-text'>{text}</div></div>"""

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "NVDA").upper()
    if st.sidebar.button("分析"): st.session_state.target = target_in
    if "target" not in st.session_state: st.session_state.target = "NVDA"
    target = st.session_state.target

    # --- Header ---
    st.markdown("""<div style="padding:10px 0; border-bottom:1px solid #333; margin-bottom:20px;">
        <span style="font-family:'JetBrains Mono'; font-weight:bold; color:#00f2ff; font-size:20px;">MARCS V82</span> 
        <span style="color:#666; font-size:12px; margin-left:10px;">ULTIMATE FUSION TERMINAL</span>
    </div>""", unsafe_allow_html=True)

    # --- Data Fetching ---
    with st.spinner(f"Decoding {target}..."):
        # Parallel Fetching
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Elder.analyze, target)
            f_factor = executor.submit(Factor_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            f_val = executor.submit(Valuation_Engine.calculate, target)
            
            m_score, sigs, df_m, atr = f_micro.result()
            factor_data = f_factor.result()
            news_items = f_news.result()
            dcf_res = f_val.result()

        # Logic Sync
        hybrid = m_score # 簡化宏觀疊加
        curr_p = df_m['Close'].iloc[-1] if not df_m.empty else 0
        sl_p = curr_p - 2.5 * atr if not df_m.empty else 0
        size, risk = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # --- LAYOUT: Koyfin Style Grid ---
    
    # 1. Title Row
    c1, c2 = st.columns([2, 1])
    with c1:
        tags = "".join([f"<span class='tag {cls}'>{n}</span>" for n, cls in factor_data['styles']]) if factor_data else ""
        color = "#4caf50" if not df_m.empty and df_m['Close'].iloc[-1] > df_m['Close'].iloc[-2] else "#f44336"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:15px;">
            <h1 style="margin:0; font-size:42px; color:white;">{target}</h1>
            <span style="font-size:28px; font-family:'JetBrains Mono'; color:{color}; font-weight:bold;">${curr_p:.2f}</span>
            <div>{tags}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Main Content (7:3 Split)
    main_col, side_col = st.columns([7, 3])
    
    with main_col:
        # A. Chart (V74 Elder Style)
        st.markdown("##### 📈 價格與趨勢 (Price & Trend)")
        if not df_m.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5)
            ax.plot(df_m.index, df_m['EMA22'], color='#ff9800', lw=1, alpha=0.8, label='EMA22')
            ax.fill_between(df_m.index, df_m['K_Upper'], df_m['K_Lower'], color='#2196f3', alpha=0.1)
            ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212')
            ax.grid(True, color='#333', linestyle='--', linewidth=0.5)
            ax.tick_params(colors='#888')
            st.pyplot(fig)
            
            # B. Subplots (MACD/Force)
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols); ax1.set_facecolor('#121212')
            ax1.tick_params(colors='#888', labelsize=8); ax1.set_ylabel("MACD", color='#888')
            
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_facecolor('#121212')
            ax2.axhline(0, color='#555', ls='--')
            ax2.tick_params(colors='#888', labelsize=8); ax2.set_ylabel("Force", color='#888')
            fig2.patch.set_facecolor('#121212')
            st.pyplot(fig2)

    with side_col:
        # C. Verdict (V75)
        st.markdown(render_verdict(target, hybrid, m_score), unsafe_allow_html=True)
        
        # D. Factor Table (V81)
        st.markdown("##### 🧬 因子分析")
        if factor_data:
            st.markdown(render_factor_table(factor_data), unsafe_allow_html=True)
            
            # E. Fundamentals
            raw = factor_data['raw']
            c_a, c_b = st.columns(2)
            with c_a: st.markdown(f"<div class='metric-box'><div class='metric-lbl'>P/E</div><div class='metric-val'>{raw['PE']:.1f}</div></div>", unsafe_allow_html=True)
            with c_b: st.markdown(f"<div class='metric-box'><div class='metric-lbl'>ROE</div><div class='metric-val'>{raw['ROE']:.1%}</div></div>", unsafe_allow_html=True)

        # F. Valuation & Risk (V79/V73)
        st.markdown("##### ⚖️ 估值與風控")
        if dcf_res:
            fair = dcf_res['fair']
            upside = (fair - curr_p) / curr_p * 100
            u_color = "#4caf50" if upside > 0 else "#f44336"
            st.markdown(f"""
            <div style="background:#1e1e1e; border:1px solid #333; padding:10px; border-radius:4px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; color:#bbb; font-size:12px;"><span>DCF Fair Value</span><span>Upside</span></div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span style="font-size:20px; font-weight:bold; color:white;">${fair:.2f}</span>
                    <span style="font-size:16px; font-weight:bold; color:{u_color};">{upside:+.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div style="background:#1e1e1e; border:1px solid #333; padding:10px; border-radius:4px;">
            <div style="color:#888; font-size:11px;">SUGGESTED POSITION</div>
            <div style="font-size:24px; color:#4facfe; font-weight:bold;">{risk['pct']}% <span style="font-size:14px; color:#ccc;">(${risk['cap']:,})</span></div>
            <div style="color:#f44336; font-size:12px; margin-top:4px;">Stop Loss: ${sl_p:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. News Feed (V77 - Grid Layout)
    st.markdown("---")
    st.markdown("### 📰 戰場情報中心 (Latest Intel)")
    if news_items:
        n_cols = st.columns(4)
        for i, item in enumerate(news_items):
            bd_color = "#4caf50" if item['sent'] == "pos" else ("#f44336" if item['sent'] == "neg" else "#444")
            with n_cols[i % 4]:
                st.markdown(f"""
                <div class="news-card" style="border-left:3px solid {bd_color}; height:100px; overflow:hidden;">
                    <a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a>
                    <div class="news-meta">{item['date']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent news found.")

if __name__ == "__main__":
    main()
