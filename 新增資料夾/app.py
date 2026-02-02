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
# 0. 視覺核心 (維持 V88 星際風格)
# =============================================================================
st.set_page_config(page_title="MARCS V101 歸零重構版", layout="wide", page_icon="🛡️")

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
    
    /* 戰術面板 */
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
# 1. 核心數據層 (Robust Data Layer)
# =============================================================================
def get_headers():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) Chrome/91.0.4472.101 Safari/537.36"
    ]
    return {"User-Agent": random.choice(agents)}

@st.cache_data(ttl=300) # 快取 5 分鐘，避免頻繁請求
def get_stock_data(ticker, period="1y"):
    """
    V101 核心下載器：只做一件事，就是把數據抓下來並清洗乾淨。
    失敗就回傳 None，絕對不讓錯誤擴散。
    """
    try:
        session = requests.Session()
        session.headers.update(get_headers())
        
        # 1. 優先使用 Ticker.history (結構最穩)
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)
        
        # 2. 數據清洗
        if df.empty: return None
        
        # 確保欄位存在且為數值
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols): return None
        
        df = df[required_cols] # 只留需要的
        df = df.dropna()
        
        if len(df) < 20: return None # 數據太少不分析
        
        return df
    except:
        return None

def get_macro_vix():
    try:
        df = get_stock_data("^VIX", "5d")
        if df is not None: return df['Close'].iloc[-1]
    except: pass
    return 20.0 # Default fallback

# =============================================================================
# 2. 計算引擎 (Calc Engine) - 包含 Elder, SMC, Risk
# =============================================================================
class Calc_Engine:
    @staticmethod
    def run_analysis(df):
        try:
            # --- 1. 指標計算 ---
            c = df['Close']
            ema22 = c.ewm(span=22).mean()
            ema12 = c.ewm(span=12).mean()
            ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            hist = macd - signal
            
            # ATR (14)
            h, l, c_prev = df['High'], df['Low'], c.shift(1)
            tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            # Force Index
            fi = (c - c.shift(1)) * df['Volume']
            fi_13 = fi.ewm(span=13).mean()
            
            # --- 2. 評分邏輯 (Elder) ---
            score = 50
            if c.iloc[-1] > ema22.iloc[-1]: score += 10
            if (ema22.iloc[-1] > ema22.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]): score += 20
            if fi_13.iloc[-1] > 0: score += 10
            
            # --- 3. SMC FVG 識別 ---
            fvgs = []
            for i in range(len(df)-2, len(df)-30, -1): # 看最近30根
                if df['Low'].iloc[i] > df['High'].iloc[i-2]: # Bullish
                    fvgs.append({'type': 'Bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'idx': df.index[i-2]})
                elif df['High'].iloc[i] < df['Low'].iloc[i-2]: # Bearish
                    fvgs.append({'type': 'Bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'idx': df.index[i-2]})
            
            # --- 4. 數據整合 ---
            df['EMA22'] = ema22
            df['K_Up'] = ema22 + 2*atr
            df['K_Lo'] = ema22 - 2*atr
            
            return {
                "df": df, "score": score, "atr": atr.iloc[-1], 
                "fvgs": fvgs[:3], "price": c.iloc[-1]
            }
        except: return None

class Risk_Engine:
    @staticmethod
    def calculate(capital, price, atr, score):
        try:
            sl = price - 2.5 * atr
            tp = price + 4.0 * atr
            
            risk_amt = capital * 0.02
            risk_per_share = price - sl
            
            if risk_per_share <= 0: return 0, 0, 0, 0
            
            size = int((risk_amt / risk_per_share) * (score/100))
            pos_val = size * price
            pct = (pos_val / capital) * 100
            
            return size, pos_val, pct, sl, tp
        except: return 0, 0, 0, 0, 0

# =============================================================================
# 3. 輔助數據 (News, Chips, PEG) - 獨立模組，失敗不影響主程式
# =============================================================================
def get_chips(ticker):
    if ".TW" not in ticker: return None
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": ticker.split('.')[0],
            "start_date": (datetime.now()-timedelta(days=10)).strftime('%Y-%m-%d')
        }
        r = requests.get(url, params=params, timeout=2)
        data = r.json()
        if data['data']:
            df = pd.DataFrame(data['data'])
            f = df[df['name']=='Foreign_Investor']
            if not f.empty: return int((f.iloc[-1]['buy']-f.iloc[-1]['sell'])/1000)
    except: pass
    return None

def get_valuation(ticker, price):
    # [V101] 極簡估值：只用技術面與簡單PE，避免抓不到 info 崩潰
    try:
        stock = yf.Ticker(ticker)
        # 嘗試抓 PE，抓不到就算了
        try: pe = stock.info.get('trailingPE', 20)
        except: pe = 20
        
        # 技術面估值保底
        fair = price # 默認當前價格合理
        if pe < 15: fair = price * 1.1 # 低估
        elif pe > 50: fair = price * 0.9 # 高估
        
        return {"fair": fair, "pe": pe}
    except: return {"fair": price, "pe": "N/A"}

def get_news(ticker):
    items = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            for n in news[:3]:
                items.append({"title": n.get('title'), "link": n.get('link'), "date": "Recent"})
    except: pass
    return items

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

    # 1. 宏觀 (獨立加載)
    vix = get_macro_vix()
    risk_level = "HIGH" if vix > 25 else ("LOW" if vix < 15 else "NORMAL")
    risk_color = "#f44336" if vix > 25 else ("#4caf50" if vix < 15 else "#ff9800")
    
    st.markdown(f"""
    <div class="risk-container">
        <div class="risk-score-box">
            <div class="risk-val" style="color:{risk_color}">{vix:.1f}</div>
            <div class="risk-label">VIX RISK</div>
        </div>
        <div style="font-family:'JetBrains Mono'; color:#fff; font-size:20px; align-self:center;">
            MARKET IS <span style="color:{risk_color}">{risk_level}</span>
        </div>
        <div style="font-family:'Rajdhani'; color:#ffae00; font-size:24px; font-weight:bold; align-self:center;">
            MARCS <span style="font-size:14px; color:#888;">V101</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. 主分析 (V101: 單線程，保證穩定)
    with st.spinner(f"Analyzing {target}..."):
        df = get_stock_data(target)
        
        if df is None:
            st.error(f"❌ 無法獲取 {target} 數據。可能原因：代碼錯誤、剛上市、或被 Yahoo 暫時阻擋。")
            st.stop() # 停止執行，避免後續報錯
            
        # 計算核心指標
        res = Calc_Engine.run_analysis(df)
        if not res:
            st.error("❌ 數據計算失敗 (長度不足)。")
            st.stop()

        # 輔助數據 (失敗不影響主程式)
        chips = get_chips(target)
        val = get_valuation(target, res['price'])
        news = get_news(target)
        
        # 風控計算
        size, pos_val, pct, sl, tp = Risk_Engine.calculate(capital, res['price'], res['atr'], res['score'])

    # 3. 呈現結果
    chip_tag = f"<span class='chip-tag' style='background:#f44336'>外資 {chips}</span>" if chips else ""
    st.markdown(f"<h1 style='color:white'>{target} <span style='color:#ffae00'>${res['price']:.2f}</span> {chip_tag}</h1>", unsafe_allow_html=True)
    
    # 戰術面板 (V96+ 回歸)
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">ATR</div><div class="tac-val">{res['atr']:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="tac-card" style="border-color:#f44336"><div><div class="tac-label">STOP LOSS</div><div class="tac-val" style="color:#f44336">${sl:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t3: st.markdown(f"""<div class="tac-card" style="border-color:#4caf50"><div><div class="tac-label">TAKE PROFIT</div><div class="tac-val" style="color:#4caf50">${tp:.2f}</div></div></div>""", unsafe_allow_html=True)
    with t4: st.markdown(f"""<div class="tac-card"><div><div class="tac-label">POSITION</div><div class="tac-val">{pct}%</div></div></div>""", unsafe_allow_html=True)

    # 圖表 Tabs
    tab1, tab2, tab3 = st.tabs(["📊 SMC戰術圖表", "🧬 估值", "📰 新聞"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sub = res['df'].tail(100) # 只畫最後 100 根
        ax.plot(sub.index, sub['Close'], color='#e0e0e0', lw=1.5, label='Price')
        ax.plot(sub.index, sub['EMA22'], color='#ffae00', lw=1.5, label='EMA22')
        ax.fill_between(sub.index, sub['K_Up'], sub['K_Lo'], color='#00f2ff', alpha=0.1)
        
        # 畫 FVG
        for fvg in res['fvgs']:
            if fvg['idx'] >= sub.index[0]: # 只畫範圍內的
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
        with c1: st.metric("技術/PE 合理價", f"${val['fair']:.2f}")
        with c2: st.metric("參考 PE", val['pe'])
        st.info("註：若無法獲取成長數據，此估值基於 PE 與技術面推算。")

    with tab3:
        if news:
            for n in news:
                st.markdown(f"- [{n['title']}]({n['link']})")
        else: st.info("無新聞數據")

if __name__ == "__main__":
    main()
