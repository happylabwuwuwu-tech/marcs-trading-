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
# 0. 視覺核心 (維持 V83 Koyfin 風格，新增 Risk Bar 樣式)
# =============================================================================
st.set_page_config(page_title="MARCS V85 籌碼戰神版", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .stApp { background-color: #121212; font-family: 'Roboto', 'Noto Sans TC', sans-serif; }
    
    /* 風險儀表板 */
    .risk-container {
        background: #1e1e1e; border-bottom: 1px solid #333; padding: 15px 20px;
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;
    }
    .risk-score-box {
        text-align: center; padding: 0 20px; border-right: 1px solid #444;
    }
    .risk-val { font-family: 'JetBrains Mono'; font-size: 32px; font-weight: bold; }
    .risk-label { font-size: 12px; color: #888; text-transform: uppercase; }
    
    /* 籌碼標籤 */
    .chip-tag { 
        padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; 
        margin-right: 5px; font-family: 'Noto Sans TC'; 
    }
    
    /* ... (保留原本 V83 的 Koyfin CSS，如 .metric-card, .factor-table 等) ... */
    .metric-card { background: rgba(18, 18, 22, 0.85); border-left: 4px solid #ffae00; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    .highlight-val { font-size: 24px; font-weight: bold; color: #fff; }
    .highlight-lbl { font-size: 12px; color: #888; }
    .news-card { background: #1e1e1e; border-bottom: 1px solid #333; padding: 10px; }
    .stButton>button { width: 100%; background: #2d2d2d; border: 1px solid #444; color: #ccc; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. [V85 New] 宏觀風險運算引擎 (Macro Risk Engine)
# =============================================================================
class Macro_Risk_Engine:
    @staticmethod
    def calculate_market_risk():
        """
        計算市場操作風險分 (0-100)
        100 = 極度安全 (Risk On)
        0 = 極度危險 (Risk Off)
        """
        score = 50 # 基礎分
        details = []
        
        try:
            # 1. VIX (恐慌指數)
            vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            if vix < 15: score += 15; details.append("VIX 低檔安穩")
            elif vix > 25: score -= 20; details.append("VIX 恐慌飆升")
            else: details.append("VIX 正常區間")
            
            # 2. 美債殖利率 (TNX)
            tnx = yf.Ticker("^TNX").history(period="5d")['Close']
            tnx_now = tnx.iloc[-1]
            tnx_trend = tnx.iloc[-1] - tnx.iloc[-5]
            if tnx_now > 4.5: score -= 10; details.append("美債利率過高")
            if tnx_trend > 0.1: score -= 10; details.append("利率急速攀升")
            
            # 3. 費半指數 (SOX) - 科技股風向球
            sox = yf.Ticker("^SOX").history(period="20d")['Close']
            ma20 = sox.mean()
            if sox.iloc[-1] > ma20: score += 15; details.append("半導體多頭排列")
            else: score -= 15; details.append("半導體跌破月線")
            
            # 4. 匯率 (DXY)
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]
            if dxy > 106: score -= 10; details.append("美元過強吸金")
            
        except:
            return 50, ["數據連線異常"], 50
            
        final_score = max(0, min(100, score))
        return final_score, details, vix

# =============================================================================
# 2. [V85 New] FinMind 台股籌碼引擎
# =============================================================================
class FinMind_Engine:
    @staticmethod
    def get_tw_chips(ticker):
        """
        使用 FinMind 開源 API 抓取外資買賣超
        不需要 API Key (但在高頻使用下建議申請)
        """
        if ".TW" not in ticker and ".TWO" not in ticker:
            return None # 美股不適用
            
        stock_id = ticker.split('.')[0]
        try:
            # 抓取最近 10 天的三大法人數據
            start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
            url = f"https://api.finmindtrade.com/api/v4/data"
            params = {
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                "data_id": stock_id,
                "start_date": start_date,
            }
            res = requests.get(url, params=params)
            data = res.json()
            
            if data['msg'] == 'success' and data['data']:
                df = pd.DataFrame(data['data'])
                # 篩選外資 (Foreign_Investor)
                foreign = df[df['name'] == 'Foreign_Investor']
                if not foreign.empty:
                    latest_buy = foreign.iloc[-1]['buy'] - foreign.iloc[-1]['sell']
                    cum_5d = (foreign.tail(5)['buy'] - foreign.tail(5)['sell']).sum()
                    return {
                        "latest_foreign": int(latest_buy / 1000), # 換算張數
                        "5d_foreign": int(cum_5d / 1000),
                        "date": foreign.iloc[-1]['date']
                    }
            return None
        except:
            return None

# =============================================================================
# 3. 其他核心引擎 (保留 V84 精華)
# =============================================================================
# ... (Global_Market_Loader, Micro_Engine_Elder, Scanner_Engine_Elder, News_Intel_Engine 保持 V84 狀態) ...
# ... (為了代碼長度，這裡隱藏未修改部分，請合併 V84 的代碼) ...

# 這裡為了完整運行，我必須把必要的 Micro Engine 放進來，並加入籌碼整合
class Micro_Engine_Pro:
    @staticmethod
    def analyze(ticker):
        # 1. 技術面 (Elder)
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df, 0, None
            
            c = df['Close']; ema22 = c.ewm(span=22).mean()
            score = 50
            signals = []
            
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            
            # 2. 籌碼面 (FinMind Integration)
            chips_data = FinMind_Engine.get_tw_chips(ticker)
            if chips_data:
                f_buy = chips_data['latest_foreign']
                f_5d = chips_data['5d_foreign']
                
                if f_buy > 1000: # 外資大買 1000 張
                    score += 15
                    signals.append(f"💰 外資大買 {f_buy} 張")
                elif f_buy < -1000:
                    score -= 15
                    signals.append(f"💸 外資提款 {abs(f_buy)} 張")
                
                if f_5d > 3000: signals.append("🔥 外資連買")
            
            # Keltner & ATR
            atr = (df['High']-df['Low']).rolling(14).mean().iloc[-1]
            df['EMA22'] = ema22
            df['K_Upper'] = ema22 + 2*atr
            df['K_Lower'] = ema22 - 2*atr
            
            return score, signals, df, atr, chips_data
        except: return 50, [], pd.DataFrame(), 0, None

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, sl, ticker, hybrid_score):
        risk_per_trade = capital * 0.02
        dist = price - sl
        if dist <= 0: return 0, {}
        
        # 根據分數調整曝險
        confidence = hybrid_score / 100.0
        size = int((risk_per_trade / dist) * confidence)
        pos_val = size * price
        pct = (pos_val / capital) * 100
        return size, {"cap": int(pos_val), "pct": round(pct, 1)}

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼 (如 2330.TW)", "2330.TW").upper()
    if st.sidebar.button("分析"): st.session_state.target = target_in
    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    target = st.session_state.target

    # --- 1. [V85] 風險儀表板 (Risk Gauge) ---
    risk_score, risk_reasons, vix_val = Macro_Risk_Engine.calculate_market_risk()
    
    # 決定顏色
    r_color = "#4caf50" if risk_score >= 60 else ("#ff9800" if risk_score >= 40 else "#f44336")
    r_text = "MARKET BULLISH" if risk_score >= 60 else ("MARKET NEUTRAL" if risk_score >= 40 else "MARKET BEARISH")
    
    st.markdown(f"""
    <div class="risk-container">
        <div style="display:flex; align-items:center;">
            <div class="risk-score-box">
                <div class="risk-val" style="color:{r_color}">{risk_score}</div>
                <div class="risk-label">Risk Score</div>
            </div>
            <div style="padding-left:20px;">
                <div style="font-size:20px; font-weight:bold; color:#fff;">{r_text}</div>
                <div style="color:#888; font-size:12px;">VIX: {vix_val:.1f} | {' | '.join(risk_reasons)}</div>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono'; color:#00f2ff; font-size:18px;">MARCS V85 <span style="font-size:12px; color:#666;">CHIPS & RISK</span></div>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. 核心分析 ---
    with st.spinner(f"正在分析 {target} 的籌碼與結構..."):
        m_score, sigs, df_m, atr, chips = Micro_Engine_Pro.analyze(target)
        
        # 綜合評分 (Macro Risk 權重 30% + Micro 權重 70%)
        hybrid = int((risk_score * 0.3) + (m_score * 0.7))
        
        curr_p = df_m['Close'].iloc[-1] if not df_m.empty else 0
        sl_p = curr_p - 2.5 * atr if not df_m.empty else 0
        size, risk_dets = Risk_Manager.calculate(capital, curr_p, sl_p, target, hybrid)

    # --- 3. 儀表板內容 ---
    c1, c2 = st.columns([7, 3])
    
    with c1:
        # Title Row
        chip_html = ""
        if chips:
            bg = "#f44336" if chips['latest_foreign'] < 0 else "#4caf50"
            txt = f"外資 {'買超' if chips['latest_foreign']>0 else '賣超'} {abs(chips['latest_foreign'])} 張"
            chip_html = f"<span class='chip-tag' style='background:{bg}; color:white;'>{txt}</span>"
            
        st.markdown(f"""<div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
            <h1 style="margin:0; font-size:42px; color:white;">{target}</h1>
            <span style="font-size:28px; font-family:'JetBrains Mono'; color:#fff;">${curr_p:.2f}</span>
            {chip_html}
        </div>""", unsafe_allow_html=True)
        
        # Chart
        if not df_m.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_m.index, df_m['Close'], color='#e0e0e0', lw=1.5, label='Price')
            ax.plot(df_m.index, df_m['EMA22'], color='#ff9800', lw=1, alpha=0.8, label='EMA22')
            ax.fill_between(df_m.index, df_m['K_Upper'], df_m['K_Lower'], color='#2196f3', alpha=0.1)
            ax.axhline(sl_p, color='#f44336', ls='--', label='StopLoss')
            
            ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212')
            ax.grid(True, color='#333', linestyle='--', linewidth=0.5)
            ax.tick_params(colors='#888')
            st.pyplot(fig)

    with c2:
        # 評分卡
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#4caf50' if hybrid>=60 else '#f44336'};">
            <div class="highlight-lbl">MARCS HYBRID SCORE</div>
            <div class="highlight-val">{hybrid}</div>
            <div style="font-size:12px; color:#aaa; margin-top:5px;">結合宏觀風險與外資籌碼</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 籌碼卡 (如果有的話)
        if chips:
            f_color = "#4caf50" if chips['5d_foreign'] > 0 else "#f44336"
            st.markdown(f"""
            <div class="metric-card">
                <div class="highlight-lbl">外資 5日累計 (CHIPS)</div>
                <div class="highlight-val" style="color:{f_color}">{chips['5d_foreign']:,} <span style="font-size:14px">張</span></div>
                <div style="font-size:12px; color:#aaa;">資料日期: {chips['date']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 倉位建議
        st.markdown(f"""
        <div class="metric-card">
            <div class="highlight-lbl">SUGGESTED SIZE</div>
            <div class="highlight-val" style="color:#4facfe">{risk_dets['pct']}%</div>
            <div style="font-size:12px; color:#aaa;">{size} shares (${risk_dets['cap']:,})</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
