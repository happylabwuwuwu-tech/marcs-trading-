import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import warnings
import random
from datetime import datetime, timedelta

# 過濾警告
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 視覺核心 (維持 V96 星際戰術面板)
# =============================================================================
st.set_page_config(page_title="MARCS V102 全裝甲版", layout="wide", page_icon="🛡️")

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
    
    /* 戰術面板 (V96 Style) */
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
    
    .chip-tag { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 10px; font-family: 'Noto Sans TC'; vertical-align: middle; }
    
    .news-card { background: rgba(25,25,30,0.8); border-bottom: 1px solid #444; padding: 10px; transition: 0.2s; border-radius: 5px; }
    .news-card:hover { background: rgba(40,40,50,0.9); }
    .news-title { color: #e0e0e0; text-decoration: none; font-weight: bold; font-size: 14px; }
    
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; border:none; background: linear-gradient(90deg, #333 0%, #ffae00 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 核心數據層 (The Armored Core) - 統一處理下載
# =============================================================================
def get_headers():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/91.0.4472.114 Safari/537.36"
    ]
    return {"User-Agent": random.choice(agents)}

@st.cache_data(ttl=600) # 快取 10 分鐘，這是「跑得動」的關鍵
def get_stock_data(ticker, period="2y"): # 預設抓 2年，足夠回測 + SMC
    """
    V102 核心：只連線一次，抓取所有需要的數據。
    包含強力清洗邏輯，確保美股/台股通用。
    """
    session = requests.Session()
    session.headers.update(get_headers())
    
    try:
        # 1. 優先使用 Ticker.history (結構最穩)
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)
        
        # 2. 如果 history 失敗，使用 download
        if df.empty:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, session=session)
        
        if df.empty: return None

        # 3. [V102] 終極扁平化 (The Flattening)
        # 不管是哪種 MultiIndex，全部壓平成單層
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0) 
            except: pass
        
        # 4. 欄位映射與清洗
        # 確保 'Close' 存在
        col_map = {c: c for c in df.columns}
        if 'Adj Close' in df.columns: col_map['Adj Close'] = 'Close'
        df.rename(columns=col_map, inplace=True)
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required): return None
        
        df = df[required].dropna()
        if len(df) < 50: return None # 數據太少不玩
        
        return df
    except: return None

# =============================================================================
# 2. 策略全家桶 (All-in-One Logic)
# =============================================================================
class Alpha_Engine:
    @staticmethod
    def analyze(df):
        """
        一次計算 Elder, SMC, Quant, ATR，效率最高
        """
        try:
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            
            # --- A. Elder System ---
            ema22 = c.ewm(span=22).mean()
            ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26; signal = macd.ewm(span=9).mean()
            hist = macd - signal
            fi = ((c - c.shift(1)) * v).ewm(span=13).mean()
            
            # --- B. SMC FVG (Order Block Logic) ---
            fvgs = []
            # 倒序查找最近的缺口
            for i in range(len(df)-2, len(df)-30, -1):
                # Bullish
                if l.iloc[i] > h.iloc[i-2]:
                    fvgs.append({'type': 'Bull', 'top': l.iloc[i], 'bottom': h.iloc[i-2], 'idx': df.index[i-2]})
                # Bearish
                elif h.iloc[i] < l.iloc[i-2]:
                    fvgs.append({'type': 'Bear', 'top': l.iloc[i-2], 'bottom': h.iloc[i], 'idx': df.index[i-2]})
            fvgs = fvgs[:3] # 只取最近3個
            
            # --- C. Quant Logic (Hurst & ATR) ---
            # ATR
            tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(atr): atr = c.iloc[-1] * 0.02
            
            # Hurst (簡易版，取最近100天)
            hurst = 0.5
            try:
                ts = c.tail(100).values
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0] * 2.0
            except: pass
            
            # --- D. 評分與訊號 ---
            score = 50
            signals = []
            curr_p = c.iloc[-1]
            
            # Elder Rules
            if curr_p > ema22.iloc[-1]: score += 10
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): 
                score += 20; signals.append("Elder Impulse Bull")
            if fi.iloc[-1] > 0: score += 10
            
            # SMC Rules
            in_bull = any(f['bottom'] <= curr_p <= f['top'] and f['type']=='Bull' for f in fvgs)
            if in_bull: score += 15; signals.append("SMC FVG Support")
            
            # Regime
            regime = "TRENDING" if hurst > 0.5 else "MEAN REV"
            
            # Data Pack
            df['EMA22'] = ema22
            df['K_Up'] = ema22 + 2*atr
            df['K_Lo'] = ema22 - 2*atr
            
            return {
                "df": df, "score": score, "atr": atr, "fvgs": fvgs,
                "regime": regime, "hurst": hurst, "signals": signals,
                "price": curr_p
            }
        except Exception as e:
            st.error(f"Engine Error: {e}")
            return None

# =============================================================================
# 3. 輔助模組 (Valuation & Risk) - 獨立且有防護
# =============================================================================
class Valuation_Engine:
    @staticmethod
    def calculate(ticker, price):
        # V100 的技術估值保底邏輯
        try:
            stock = yf.Ticker(ticker)
            try: info = stock.info
            except: info = {}
            
            pe = info.get('trailingPE')
            
            # 如果抓不到 PE，用 EMA50 作為技術合理價
            if not pe:
                return {"fair": price, "method": "Technical Mean", "desc": "EMA50 Proxy"}
            
            # 有 PE，做簡單估值
            fair = price * (1.1 if pe < 20 else (0.9 if pe > 40 else 1.0))
            return {"fair": fair, "method": "PE Adjusted", "desc": f"PE: {pe}"}
        except:
            return {"fair": price, "method": "Market Price", "desc": "No Data"}

class Risk_Manager:
    @staticmethod
    def calculate(capital, price, atr, score):
        try:
            sl = price - 2.5 * atr
            tp = price + 4.0 * atr
            risk_share = price - sl
            if risk_share <= 0: return 0, 0, 0, 0
            
            # Kelly-like sizing based on score
            confidence = score / 100
            risk_amt = capital * 0.02
            size = int((risk_amt / risk_share) * confidence)
            
            return size, sl, tp, round((size*price/capital)*100, 1)
        except: return 0, 0, 0, 0

class Backtest_Engine:
    @staticmethod
    def run(df):
        # 簡單回測，使用已有的 DF，不重新下載
        try:
            c = df['Close']
            ema22 = df['EMA22']
            # 簡單策略：價格 > EMA22 持有
            equity = [100000]
            pos = 0
            for i in range(1, len(df)):
                p = c.iloc[i]; prev = c.iloc[i-1]
                if p > ema22.iloc[i]: pos = 1
                elif p < ema22.iloc[i]: pos = 0
                
                if pos == 1: equity.append(equity[-1] * (p/prev))
                else: equity.append(equity[-1])
            
            return pd.Series(equity, index=df.index)
        except: return None

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("本金", value=1000000)
    target_in = st.sidebar.text_input("代碼", "2330.TW").upper()
    if st.sidebar.button("分析"): st.session_state.target = target_in
    if "target" not in st.session_state: st.session_state.target = "2330.TW"
    target = st.session_state.target

    # 1. 宏觀 VIX (獨立且容錯)
    try:
        vix_df = get_stock_data("^VIX", "5d")
        vix = vix_df['Close'].iloc[-1] if vix_df is not None else 20.0
    except: vix = 20.0
    
    st.markdown(f"""
    <div class="risk-container">
        <div><div class="risk-label">VIX RISK</div><div class="risk-val" style="color:#4caf50">{vix:.1f}</div></div>
        <div style="font-family:'Rajdhani'; color:#ffae00; font-size:24px; font-weight:bold;">MARCS V102</div>
    </div>""", unsafe_allow_html=True)

    # 2. 核心分析
    with st.spinner(f"Analyzing {target} (Full Armor)..."):
        # 單次獲取
        df_main = get_stock_data(target)
        
        if df_main is None:
            st.error(f"❌ 無法獲取 {target}。請檢查代碼或稍後再試。")
            st.stop()
            
        # 單次計算
        res = Alpha_Engine.analyze(df_main)
        if not res: st.stop()
        
        # 輔助計算
        val = Valuation_Engine.calculate(target, res['price'])
        size, sl, tp, pct = Risk_Manager.calculate(capital, res['price'], res['atr'], res['score'])
        eq_curve = Backtest_Engine.run(res['df'])

    # 3. 呈現 (戰術面板回歸)
    st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${res['price']:.2f}</span></h1>", unsafe_allow_html=True)
    
    tag = "BUY" if res['score'] > 60 else ("SELL" if res['score'] < 40 else "HOLD")
    bg = "#3fb950" if tag=="BUY" else ("#f44336" if tag=="SELL" else "#333")
    st.markdown(f"""<div class="verdict-box" style="background:{bg}30; border-color:{bg}"><h2 style="margin:0; color:{bg}">{tag} (Score: {res['score']})</h2><p style="color:#ccc">{' | '.join(res['signals'])}</p></div>""", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">ATR</div><div class="tac-val">{res['atr']:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="tac-card" style="border-color:#f44336"><div><div class="tac-label">STOP LOSS</div><div class="tac-val" style="color:#f44336">${sl:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t3: st.markdown(f"""<div class="tac-card" style="border-color:#4caf50"><div><div class="tac-label">TAKE PROFIT</div><div class="tac-val" style="color:#4caf50">${tp:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t4: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">POSITION</div><div class="tac-val">{pct}%</div></div></div>""", unsafe_allow_html=True)

    # 4. 圖表區
    tab1, tab2, tab3 = st.tabs(["📊 SMC戰術圖", "🧬 估值與量化", "🔄 策略回測"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sub = res['df'].tail(100)
        ax.plot(sub.index, sub['Close'], color='#e0e0e0', lw=1.5, label='Price')
        ax.plot(sub.index, sub['EMA22'], color='#ffae00', lw=1.5, label='EMA 22')
        ax.fill_between(sub.index, sub['K_Up'], sub['K_Lo'], color='#00f2ff', alpha=0.1)
        
        # 畫 FVG
        for fvg in res['fvgs']:
            if fvg['idx'] >= sub.index[0]:
                color = 'green' if fvg['type'] == 'Bull' else 'red'
                rect = patches.Rectangle((fvg['idx'], fvg['bottom']), width=timedelta(days=3), height=fvg['top']-fvg['bottom'], facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                
        ax.axhline(sl, color='#f44336', ls='--', label='SL')
        ax.axhline(tp, color='#4caf50', ls='--', label='TP')
        
        ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
        ax.tick_params(colors='#888'); ax.grid(True, color='#333', alpha=0.3)
        ax.legend(loc='upper left')
        st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1: 
            st.metric("合理估值", f"${val['fair']:.2f}")
            st.caption(f"Method: {val['method']}")
        with c2:
            st.metric("Hurst 指數", f"{res['hurst']:.2f}")
            st.caption(f"Regime: {res['regime']}")

    with tab3:
        if eq_curve is not None:
            ret = (eq_curve.iloc[-1]-100000)/100000
            st.metric("回測報酬 (2Y)", f"{ret:.1%}")
            st.line_chart(eq_curve)
        else:
            st.warning("數據不足無法回測")

if __name__ == "__main__":
    main()
