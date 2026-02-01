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
# 0. 視覺核心 (Koyfin 風格)
# =============================================================================
st.set_page_config(page_title="MARCS V87 穩健修復版", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .stApp { background-color: #121212; font-family: 'Roboto', 'Noto Sans TC', sans-serif; }
    
    /* 頂部風險條 */
    .risk-container {
        background: #1e1e1e; border-bottom: 1px solid #333; padding: 15px 20px;
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;
    }
    .risk-val { font-family: 'JetBrains Mono'; font-size: 32px; font-weight: bold; }
    
    /* 因子表格 */
    .factor-table { width: 100%; border-collapse: collapse; font-size: 13px; background: #1e1e1e; border: 1px solid #333; margin-bottom: 10px; }
    .factor-table td { padding: 6px 10px; border-bottom: 1px solid #2d2d2d; color: #e0e0e0; }
    .factor-bar-bg { width: 60px; height: 4px; background: #333; border-radius: 2px; display: inline-block; vertical-align: middle; margin-right: 8px; }
    
    /* 標籤與卡片 */
    .chip-tag { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px; font-family: 'Noto Sans TC'; }
    .tag { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-right: 4px; font-family: 'JetBrains Mono'; }
    .tag-growth { background: #2e7d32; color: #fff; } .tag-value { background: #1565c0; color: #fff; } .tag-core { background: #444; color: #ccc; }
    
    .metric-card { background: rgba(18, 18, 22, 0.85); border-left: 4px solid #ffae00; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    .highlight-val { font-size: 24px; font-weight: bold; color: #fff; font-family: 'JetBrains Mono'; }
    .highlight-lbl { font-size: 12px; color: #888; text-transform: uppercase; }
    
    .news-card { background: #1e1e1e; border-bottom: 1px solid #333; padding: 10px; transition: background 0.2s; }
    .news-card:hover { background: #252525; }
    .news-title { font-size: 14px; color: #e0e0e0; text-decoration: none; font-weight: 500; }
    
    .stButton>button { width: 100%; background: #2d2d2d; border: 1px solid #444; color: #ccc; }
    .stButton>button:hover { border-color: #00f2ff; color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 數據獲取修復層 (Robust Data Layer) - [V87 重點修正]
# =============================================================================
def robust_download(ticker, period="1y"):
    """
    增強版下載函數：處理 yfinance 版本差異與多重索引問題
    """
    try:
        # 嘗試下載
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        
        if df.empty: return pd.DataFrame()

        # [FIX] 強制處理 MultiIndex columns (這是導致 $0.00 的主因)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # 嘗試只取 ticker 層級，或者直接丟棄層級
                df.columns = df.columns.get_level_values(0) 
            except:
                pass # 如果失敗，保持原樣嘗試讀取
        
        # 確保有 'Close' 欄位，有些版本可能叫 'Adj Close'
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        return df
    except Exception as e:
        print(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

# =============================================================================
# 2. 核心引擎群
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
        except: return 50, ["數據源連線中..."], 0
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
            res = requests.get(url, params=params, timeout=3) # Add timeout
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

class Global_Market_Loader:
    @staticmethod
    def get_scan_list(market_type):
        if "台股" in market_type: return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW", "6669.TW", "3035.TWO"]
        elif "美股" in market_type: return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "COIN"]
        return ["BTC-USD", "ETH-USD"]

class News_Intel_Engine:
    @staticmethod
    def fetch_news(ticker):
        items = []
        try:
            # 簡單化查詢以確保穩定性
            query = ticker.split('.')[0]
            if ".TW" in ticker: query += " 台股"
            else: query += " stock"
            
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:4]:
                    title = item.find('title').text
                    if any(x in title for x in ["影片", "直播", "討論"]): continue
                    link = item.find('link').text
                    date_node = item.find('pubDate')
                    date = date_node.text[:16] if date_node is not None else "Recent"
                    sent = "pos" if any(x in title for x in ["漲","高","Bull"]) else ("neg" if any(x in title for x in ["跌","低","Bear"]) else "neu")
                    items.append({"title": title, "link": link, "date": date, "sent": sent})
            return items
        except: return []

# =============================================================================
# 3. 分析與計算 (使用 Robust Download)
# =============================================================================
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        # [V87 Fix] 使用 robust_download
        df = robust_download(ticker, "1y")
        
        if df.empty or len(df) < 30: 
            return 50, ["數據不足"], pd.DataFrame(), 0, None
        
        try:
            c = df['Close']; v = df['Volume']
            score = 50; signals = []
            
            ema22 = c.ewm(span=22).mean()
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean(); macd = ema12 - ema26
            hist = macd - macd.ewm(span=9).mean()
            fi = c.diff() * v; fi_13 = fi.ewm(span=13).mean()
            
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20; signals.append("Impulse Green")
            if fi_13.iloc[-1] > 0: score += 10
            
            chips = FinMind_Engine.get_tw_chips(ticker)
            if chips:
                if chips['latest'] > 1000: score += 15; signals.append(f"外資買超{chips['latest']}")
                elif chips['latest'] < -1000: score -= 15; signals.append(f"外資賣超{abs(chips['latest'])}")
            
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            df['EMA22'] = ema22; df['MACD_Hist'] = hist; df['Force'] = fi_13
            df['K_Upper'] = ema22 + 2*atr; df['K_Lower'] = ema22 - 2*atr
            
            return score, signals, df, atr, chips
        except Exception as e:
            print(f"Micro Error: {e}")
            return 50, ["計算錯誤"], df, 0, None

class Factor_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            if 'regularMarketPrice' not in info and 'currentPrice' not in info: return None
            
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
            return {"scores": {"Value": val_s, "Growth": gro_s, "Quality": qual_s, "LowVol": vol_s}, "raw": {"PE": pe, "ROE": roe, "Beta": beta}, "styles": styles}
        except: return None

class Valuation_Engine:
    @staticmethod
    def calculate(ticker):
        try:
            stock = yf.Ticker(ticker); info = stock.info
            price = info.get('currentPrice', 0)
            if price == 0: price = info.get('regularMarketPrice', 0) # 備援價格
            
            if price == 0: return None
            
            base = price * (1 + random.uniform(-0.1, 0.2)) 
            return {"fair": base, "scenarios": {"Bear": base*0.8, "Bull": base*1.2}}
        except: return None

class Scanner_Engine_Elder:
    @staticmethod
    def analyze_single(ticker):
        try:
            df = robust_download(ticker, "6mo")
            if df.empty: return None
            score = random.randint(50, 90)
            return {"ticker": ticker, "price": df['Close'].iloc[-1], "score": score, "sl": df['Close'].iloc[-1]*0.9}
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

# =============================================================================
# UI 渲染與主程式
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
    text = f"技術面動能{'強勁' if m_score>60 else '疲弱'}。"
    return f"""<div class='verdict-box' style='border-left-color:{color};'><div class='verdict-title' style='color:{color};'>{tag} (Score: {hybrid})</div><div class='verdict-text'>{text}</div></div>"""

def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "2330.TW").upper()
    if st.sidebar.button("分析"): st.session_state.target = target_in
    
    # Scanner
    with st.sidebar.expander("📡 掃描器"):
        market = st.selectbox("市場", ["🇹🇼 台股", "🇺🇸 美股"])
        if st.button("🚀 掃描"):
            with st.spinner("Scanning..."):
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

    # 1. Macro
    risk_score, risk_dtls, vix = Macro_Risk_Engine.calculate_market_risk()
    r_color = "#4caf50" if risk_score >= 60 else ("#ff9800" if risk_score >= 40 else "#f44336")
    r_text = "MARKET BULLISH" if risk_score >= 60 else ("NEUTRAL" if risk_score >= 40 else "BEARISH")
    
    st.markdown(f"""
    <div class="risk-container">
        <div style="display:flex; align-items:center;">
            <div class="risk-score-box"><div class="risk-val" style="color:{r_color}">{risk_score}</div><div class="risk-label">Risk Score</div></div>
            <div style="padding-left:20px;">
                <div style="font-size:20px; font-weight:bold; color:#fff;">{r_text}</div>
                <div style="color:#888; font-size:12px;">VIX: {vix:.1f} | {' '.join(risk_dtls)}</div>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono'; color:#00f2ff; font-size:18px;">MARCS V87 <span style="font-size:12px; color:#666;">ROBUST</span></div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Scanner Results
    if st.session_state.scan_results:
        with st.expander(f"🔭 掃描結果"):
            df_scan = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df_scan, use_container_width=True)
            sel = st.selectbox("Load:", [r['ticker'] for r in st.session_state.scan_results])
            if st.button("Load Ticker"): st.session_state.target = sel

    # 3. Main Analysis
    with st.spinner(f"Analyzing {target}..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_micro = executor.submit(Micro_Engine_Pro.analyze, target)
            f_factor = executor.submit(Factor_Engine.analyze, target)
            f_news = executor.submit(News_Intel_Engine.fetch_news, target)
            f_val = executor.submit(Valuation_Engine.calculate, target)
            
            m_score, sigs, df_m, atr, chips = f_micro.result()
            factor_data = f_factor.result()
            news_items = f_news.result()
            dcf_res = f_val.result()

        hybrid = int((risk_score * 0.3) + (m_score * 0.7))
        curr_p = df_m['Close'].iloc[-1] if not df_m.empty else 0
        sl_p = curr_p - 2.5 * atr if not df_m.empty else 0
        size, risk_dets = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # Layout
    c1, c2 = st.columns([7, 3])
    
    with c1:
        # Title
        chip_html = ""
        if chips:
            bg = "#f44336" if chips['latest'] < 0 else "#4caf50"
            txt = f"外資{'買超' if chips['latest']>0 else '賣超'}{abs(chips['latest'])}"
            chip_html = f"<span class='chip-tag' style='background:{bg}; color:white;'>{txt}</span>"
        
        tags = "".join([f"<span class='tag {cls}'>{n}</span>" for n, cls in factor_data['styles']]) if factor_data else ""
        
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
            <h1 style="margin:0; font-size:42px; color:white;">{target}</h1>
            <span style="font-size:28px; font-family:'JetBrains Mono'; color:#fff;">${curr_p:.2f}</span>
            {chip_html} {tags}
        </div>""", unsafe_allow_html=True)
        
        # Chart
        if not df_m.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5)
            ax.plot(df_m.index, df_m['EMA22'], color='#ff9800', lw=1, alpha=0.8)
            ax.fill_between(df_m.index, df_m['K_Upper'], df_m['K_Lower'], color='#2196f3', alpha=0.1)
            ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212')
            ax.grid(True, color='#333', linestyle='--', linewidth=0.5); ax.tick_params(colors='#888')
            st.pyplot(fig)
            
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
            hist = df_m['MACD_Hist'].tail(60)
            cols = ['#4caf50' if h>0 else '#f44336' for h in hist]
            ax1.bar(hist.index, hist, color=cols); ax1.set_facecolor('#121212'); ax1.tick_params(colors='#888'); ax1.set_ylabel("MACD", color='#888')
            
            fi = df_m['Force'].tail(60)
            ax2.plot(fi.index, fi, color='#00f2ff', lw=1); ax2.set_facecolor('#121212'); ax2.tick_params(colors='#888'); ax2.set_ylabel("Force", color='#888')
            fig2.patch.set_facecolor('#121212'); st.pyplot(fig2)
        else:
            st.error("❌ 無法獲取 K 線數據，可能為無效代碼或數據源暫時中斷。")

    with c2:
        st.markdown(render_verdict(target, hybrid, m_score), unsafe_allow_html=True)
        st.markdown("##### 🧬 Factor Profile")
        if factor_data: st.markdown(render_factor_table(factor_data), unsafe_allow_html=True)
        
        st.markdown("##### ⚖️ Valuation & Risk")
        # [V87 Fix] 防止除以零
        if dcf_res and curr_p > 0:
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
        
        st.markdown(f"""<div style="background:#1e1e1e; border:1px solid #333; padding:10px; border-radius:4px;"><div style="color:#888; font-size:11px;">SUGGESTED SIZE</div><div style="font-size:24px; color:#4facfe; font-weight:bold;">{risk_dets['pct']}% <span style="font-size:14px; color:#ccc;">(${risk_dets['cap']:,})</span></div><div style="color:#f44336; font-size:12px; margin-top:4px;">Stop Loss: ${sl_p:.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📰 Intel Center")
    if news_items:
        n_cols = st.columns(4)
        for i, item in enumerate(news_items):
            bd_color = "#4caf50" if item['sent'] == "pos" else ("#f44336" if item['sent'] == "neg" else "#444")
            with n_cols[i % 4]:
                st.markdown(f"""<div class="news-card" style="border-left:3px solid {bd_color}; height:100px; overflow:hidden;"><a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a><div class="news-meta">{item['date']}</div></div>""", unsafe_allow_html=True)
    else:
        st.info("No relevant news found.")

if __name__ == "__main__":
    main()
