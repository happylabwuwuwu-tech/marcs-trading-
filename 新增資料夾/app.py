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

# =============================================================================
# 0. 視覺核心 (完全保留 V91 的星際風格與 CSS)
# =============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="MARCS V100 Sentinel Full-Spec", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; }
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px), radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px; animation: stars 120s linear infinite; z-index: -1; opacity: 0.7;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }
    .risk-container { background: rgba(30, 30, 35, 0.6); border: 1px solid #333; padding: 15px 20px; border-radius: 10px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; backdrop-filter: blur(10px); }
    .risk-score-box { text-align: center; padding: 0 20px; border-right: 1px solid #444; }
    .risk-val { font-family: 'JetBrains Mono'; font-size: 32px; font-weight: bold; }
    .metric-card { background: rgba(18, 18, 22, 0.85); border-left: 4px solid #ffae00; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    .highlight-val { font-size: 24px; font-weight: bold; color: #fff; font-family: 'JetBrains Mono'; }
    .highlight-lbl { font-size: 12px; color: #8b949e; text-transform: uppercase;}
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .factor-table { width: 100%; border-collapse: collapse; font-size: 13px; background: rgba(30,30,30,0.5); border: 1px solid #444; }
    .factor-bar-bg { width: 100%; height: 4px; background: #333; border-radius: 2px; }
    .factor-bar-fill { height: 100%; border-radius: 2px; }
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 數據獲取層 (V91 強化版：加入 MultiIndex 雲端修復)
# =============================================================================
def fetch_single_stock(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        # 雲端修復：確保欄位扁平化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

def fetch_batch_scanner(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, level=1, axis=1)
            except: df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# =============================================================================
# 2. 核心引擎組件 (全數保留 V91 邏輯)
# =============================================================================
class Macro_Risk_Engine:
    @staticmethod
    def calculate_market_risk():
        score = 50; details = []
        try:
            vix = fetch_single_stock("^VIX", "5d")
            vix_val = vix['Close'].iloc[-1] if not vix.empty else 20
            tnx = fetch_single_stock("^TNX", "5d")
            tnx_val = tnx['Close'].iloc[-1] if not tnx.empty else 4.0
            if vix_val < 15: score += 15; details.append("VIX低檔")
            elif vix_val > 25: score -= 20; details.append("VIX恐慌")
            if tnx_val > 4.5: score -= 10; details.append("美債高利")
            return max(0, min(100, score)), details, vix_val
        except: return 50, ["數據連線中..."], 20

class FinMind_Engine:
    @staticmethod
    def get_tw_chips(ticker):
        if ".TW" not in ticker and ".TWO" not in ticker: return None
        try:
            stock_id = ticker.split('.')[0]
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')}
            res = requests.get(url, params=params, timeout=3).json()
            if res['msg'] == 'success' and res['data']:
                df = pd.DataFrame(res['data'])
                foreign = df[df['name'] == 'Foreign_Investor']
                if not foreign.empty:
                    latest = foreign.iloc[-1]['buy'] - foreign.iloc[-1]['sell']
                    return {"latest": int(latest/1000), "date": foreign.iloc[-1]['date']}
            return None
        except: return None

class Factor_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', 20); roe = info.get('returnOnEquity', 0.1)
            rev_g = info.get('revenueGrowth', 0.05); beta = info.get('beta', 1.0)
            return {"scores": {"Value": 60 if pe<25 else 40, "Growth": min(100, int(rev_g*400)) if rev_g else 50, "Quality": 70 if roe>0.15 else 40, "LowVol": 80 if beta<1.0 else 40}}
        except: return None

# [新增] SMC 結構偵測
def detect_smc_structure(df):
    if len(df) < 5: return None
    if df['Low'].iloc[-1] > df['High'].iloc[-3]: return {'type': 'Bullish', 'top': df['Low'].iloc[-1], 'bottom': df['High'].iloc[-3]}
    elif df['High'].iloc[-1] < df['Low'].iloc[-3]: return {'type': 'Bearish', 'top': df['Low'].iloc[-3], 'bottom': df['High'].iloc[-1]}
    return None

# =============================================================================
# 3. 主 UI 流程 (整合分頁與三層圖表)
# =============================================================================
def main():
    # Sidebar 控制台 (保留 V91 所有功能)
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target = st.sidebar.text_input("代碼", "2330.TW").upper()
    
    # 宏觀風險
    risk_score, risk_dtls, vix = Macro_Risk_Engine.calculate_market_risk()
    r_color = "#4caf50" if risk_score >= 60 else ("#ff9800" if risk_score >= 40 else "#f44336")
    
    # 頂部 Risk Gauge
    st.markdown(f"""
    <div class="risk-container">
        <div style="display:flex; align-items:center;">
            <div class="risk-score-box"><div class="risk-val" style="color:{r_color}">{risk_score}</div><div class="risk-label">Risk Score</div></div>
            <div style="padding-left:20px;"><div style="font-size:20px; font-weight:bold; color:#fff;">{risk_dtls[0] if risk_dtls else '穩定'}</div><div style="color:#aaa; font-size:12px;">VIX: {vix:.1f}</div></div>
        </div>
        <div style="font-family:'Rajdhani'; color:#ffae00; font-size:24px; font-weight:bold;">MARCS V100 FULL</div>
    </div>
    """, unsafe_allow_html=True)

    # 數據加載
    df = fetch_single_stock(target)
    if df.empty: return st.error("數據獲取失敗")
    
    # 指標計算
    ema22 = df['Close'].ewm(span=22).mean()
    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    hist = macd - macd.ewm(span=9).mean()
    fi = (df['Close'].diff() * df['Volume']).ewm(span=13).mean()
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    curr_p = df['Close'].iloc[-1]
    sl_p = curr_p - 2.5 * atr
    smc = detect_smc_structure(df)
    chips = FinMind_Engine.get_tw_chips(target)

    # 判斷與 Verdict
    st.markdown(f"<h1 style='color:white;'>{target} <span style='color:#ffae00;'>${curr_p:.2f}</span></h1>", unsafe_allow_html=True)

    # --- 四大分頁 (對齊您的圖片) ---
    tab1, tab2, tab3, tab4 = st.tabs(["🕯️ 技術主圖", "🧬 PEG 估值", "📰 情報中心", "🔄 策略回測"])

    with tab1:
        # [升級：三層疊加圖表]
        fig, (ax_p, ax_m, ax_f) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.05)
        
        # 價格層
        ax_p.plot(df.index, df['Close'], color='white', lw=1.5, label='Price')
        ax_p.plot(df.index, ema22, color='#ffae00', ls='--', alpha=0.7, label='EMA22')
        ax_p.axhline(sl_p, color='#f85149', ls='-', lw=2, label='Stop Loss')
        if smc:
            ax_p.axhspan(smc['bottom'], smc['top'], color='green' if smc['type']=='Bullish' else 'red', alpha=0.2)
        ax_p.set_facecolor('#0d1117')
        ax_p.legend(loc='upper left')

        # MACD 層
        ax_m.bar(df.index, hist, color=['#4caf50' if x > 0 else '#f44336' for x in hist], alpha=0.8)
        ax_m.set_facecolor('#0d1117'); ax_m.set_ylabel("MACD")

        # Force Index 層
        ax_f.plot(df.index, fi, color='#00f2ff', lw=1)
        ax_f.axhline(0, color='gray', ls='--')
        ax_f.set_facecolor('#0d1117'); ax_f.set_ylabel("Force")
        
        fig.patch.set_facecolor('#050505')
        st.pyplot(fig)

    with tab2:
        st.markdown("### 🧬 因子與 PEG 估值")
        f_data = Factor_Engine.analyze(target)
        if f_data:
            c_a, c_b = st.columns(2)
            with c_a:
                for name, score in f_data['scores'].items():
                    st.markdown(f"**{name}**: {score} pts")
                    st.progress(score/100)
            with c_b:
                st.metric("Fair Value Estimate", f"${curr_p * 1.1:.2f}") # 簡化估值邏輯
        else: st.info("此標的暫無完整因子數據")

    with tab3:
        st.markdown("### 📰 最新情報")
        try:
            news = yf.Ticker(target).news
            for n in news[:5]:
                st.markdown(f"""<div class="news-card"><a href="{n['link']}" target="_blank" style="color:#ffae00; text-decoration:none; font-weight:bold;">{n['title']}</a><br/><small style="color:#888;">{n['publisher']}</small></div>""", unsafe_allow_html=True)
        except: st.write("情報獲取中...")

    with tab4:
        st.markdown("### 🔄 策略回測 (EMA22)")
        df['Ret'] = df['Close'].pct_change()
        df['Strat'] = np.where(df['Close'] > ema22, 1, 0)
        df['CumRet'] = (1 + df['Strat'].shift(1) * df['Ret']).cumprod()
        st.line_chart(df['CumRet'])
        st.metric("累積回報", f"{(df['CumRet'].iloc[-1]-1):.2%}")

if __name__ == "__main__":
    main()
