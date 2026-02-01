import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import random
import warnings
import os
from scipy.stats import wasserstein_distance

# 過濾警告
warnings.filterwarnings('ignore')

# 設定網頁配置
st.set_page_config(
    page_title="MARCS V60 全能戰情室",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# CSS 美化 (星際風格)
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 8px; padding: 15px; text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {color: #ffffff; font-size: 22px; font-weight: bold;}
    .metric-label {color: #8b949e; font-size: 12px;}
    .scan-row {
        padding: 10px; border-radius: 5px; margin-bottom: 5px;
        border-left: 4px solid #3fb950; background: #161b22;
    }
    /* 影片區塊樣式 */
    .stVideo {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PART 1: 宏觀引擎 (Step 1)
# =============================================================================
class Global_Index_List:
    @staticmethod
    def get_macro_indices():
        return {
            "^VIX": {"name": "VIX 恐慌指數", "type": "Sentiment"},
            "DX-Y.NYB": {"name": "DXY 美元指數", "type": "Currency"},
            "TLT": {"name": "TLT 美債20年", "type": "Rates"},
            "JPY=X": {"name": "JPY 日圓", "type": "Currency"}
        }

class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            c = df['Close']
            # RSI
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Chaos
            returns = np.log(c).diff().dropna()
            if len(returns) < 40: return None
            curr_w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            hist_std = returns.rolling(40).std().mean() * 0.1
            chaos = curr_w2 / (hist_std + 1e-9)
            
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {"ticker": ticker, "name": name, "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend}
        except: return None

    @staticmethod
    def calculate_mmi(results):
        score = 50.0
        data_map = {r['ticker']: r for r in results if r}
        
        vix = data_map.get('^VIX')
        if vix:
            if vix['trend'] == 'Overbought': score += 15
            elif vix['trend'] == 'Oversold': score -= 15
            
        dxy = data_map.get('DX-Y.NYB')
        if dxy:
            if dxy['trend'] == 'Overbought': score -= 12
            elif dxy['trend'] == 'Oversold': score += 12
            
        return min(100, max(0, score))

# =============================================================================
# PART 2: 選股雷達 (Step 2 - V38 Engine)
# =============================================================================
class TW_Market_Crawler:
    @staticmethod
    def get_tickers(mode='demo'):
        if mode == 'demo':
            return [
                "2330.TW", "2317.TW", "2454.TW", "2382.TW", "3231.TW", "2603.TW", "3035.TWO", 
                "8069.TWO", "3293.TWO", "2376.TW", "2356.TW", "3017.TW", "3044.TW", "2303.TW", 
                "6274.TWO", "3529.TWO", "3005.TW", "2368.TW", "3037.TW", "3443.TW", "3008.TW",
                "2498.TW", "3034.TW", "6669.TW", "2383.TW", "6213.TW", "6285.TW", "6112.TWO",
                "3563.TW", "3025.TW", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "COIN"
            ]
        else:
            return ["2330.TW", "2317.TW"] # 簡化示意

class Scanner_Engine_V38:
    def __init__(self, ticker):
        self.ticker = ticker

    def analyze(self):
        try:
            df = yf.download(self.ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
            c = df['Close']; v = df['Volume']
            if c.iloc[-1] < 10: return None
            
            # V38 核心動能邏輯
            ma20 = c.rolling(20).mean().iloc[-1]
            ma60 = c.rolling(60).mean().iloc[-1]
            
            # 必須是多頭排列
            if not (c.iloc[-1] > ma20 > ma60): return None
            
            # RSI
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            score = 40
            if 55 <= rsi <= 75: score += 20
            elif rsi > 75: score += 10
            elif rsi < 50: score -= 10
            
            vol_ma5 = v.rolling(5).mean().iloc[-1]
            if v.iloc[-1] > vol_ma5 * 1.5: score += 15 # 爆量
            
            # 停損計算 (ATR)
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1] - 2.5 * atr, ma20 * 0.98)
            
            return {
                "ticker": self.ticker, "price": c.iloc[-1], "score": score,
                "rsi": rsi, "sl": sl, "atr": atr
            }
        except: return None

# =============================================================================
# PART 3: 風控與微觀引擎 (Step 3 - V57 Engine)
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
            score = 50; signals = []
            
            # Keltner
            ema20 = c.ewm(span=20).mean()
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr10 = tr.rolling(10).mean()
            k_upper = ema20 + 2.0 * atr10
            k_lower = ema20 - 2.0 * atr10
            
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner 突破")
            
            # R-Breaker Trend
            if c.iloc[-1] > c.iloc[-2] * 1.02: score += 5; signals.append("強勢紅K")
            
            # OBV
            obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
            if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV 多方")
            
            # 準備繪圖數據
            df['K_Upper'] = k_upper
            df['K_Lower'] = k_lower
            df['EMA20'] = ema20
            
            return min(100, max(0, score)), signals, df
        except: return 50, [], pd.DataFrame()

class Antifragile_Position_Sizing:
    @staticmethod
    def calculate(capital, price, sl, chaos_level=0.5, vol_cap=0.5):
        risk_per_trade = capital * 0.02
        risk_per_share = price - sl
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        
        # Taleb 風控
        taleb_multiplier = 1.0
        if chaos_level > 1.2: taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
        
        # Vol Cap (Crypto vs Stock)
        vol_adj = 0.5 if vol_cap > 0.8 else 1.0
        
        final_size = int(base_size * taleb_multiplier * vol_adj)
        final_capital = final_size * price
        
        return final_size, {
            "risk_money": int(risk_per_trade), "taleb_factor": round(taleb_multiplier, 2),
            "final_capital": int(final_capital)
        }

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # --- 側邊欄 ---
    st.sidebar.title("⚙️ 控制台")
    capital = st.sidebar.number_input("總操作本金", value=1000000, step=100000)
    scan_mode = st.sidebar.selectbox("選股範圍", ["Demo (熱門股)", "Full (全市場-需等待)"])
    min_score = st.sidebar.slider("選股門檻 (V38 Score)", 50, 90, 65)

    # [V60 Plus] 影片演示區 (整合回來了！)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎥 系統架構演示")
    # 這裡請確保你的影片檔名正確
    video_file = "demo.mp4"
    if os.path.exists(video_file):
        st.sidebar.video(video_file)
    else:
        st.sidebar.info("⚠️ 找不到影片檔 (model_arch.mp4.mp4)")

    # 主標題
    st.markdown("<h1 style='color:#00f2ff; text-align:center;'>🛡️ MARCS V60 全能戰情室</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b949e;'>天 (Macro) + 地 (Scan) + 人 (Risk) 一站式解決方案</p>", unsafe_allow_html=True)
    
    # =================================================
    # STEP 1: 宏觀天候 (The Weather)
    # =================================================
    st.markdown("### 📡 Step 1: 宏觀風向 (Macro View)")
    if st.button("更新宏觀數據"):
        with st.spinner("同步全球指數..."):
            macro_res = []
            cols = st.columns(4)
            for idx, (t, info) in enumerate(Global_Index_List.get_macro_indices().items()):
                res = Macro_Engine.analyze(t, info['name'])
                macro_res.append(res)
                if res:
                    color = "#f85149" if res['trend']=='Overbought' else ("#3fb950" if res['trend']=='Oversold' else "#8b949e")
                    with cols[idx%4]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top:2px solid {color}">
                            <div class="metric-label">{res['name']}</div>
                            <div class="metric-value">{res['price']:.2f}</div>
                            <div class="metric-label" style="color:{color}">{res['trend']} (Chaos: {res['chaos']:.2f})</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            mmi = Macro_Engine.calculate_mmi(macro_res)
            mmi_color = "#3fb950" if mmi > 60 else ("#f85149" if mmi < 40 else "#d2a8ff")
            st.markdown(f"""
            <div style='background:#161b22; padding:10px; border-radius:5px; text-align:center; margin-top:10px;'>
                <span style='color:#8b949e'>MMI 宏觀風險偏好:</span> 
                <span style='font-size:24px; font-weight:bold; color:{mmi_color}'>{mmi:.1f}</span>
            </div>""", unsafe_allow_html=True)

    # =================================================
    # STEP 2: 智能選股 (The Scanner)
    # =================================================
    st.markdown("---")
    st.markdown("### 🔭 Step 2: 主動選股 (Scanner)")
    
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = []

    if st.button("🚀 啟動掃描 (Scan Market)"):
        tickers = TW_Market_Crawler.get_tickers('demo' if 'Demo' in scan_mode else 'full')
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"Scanning {t}...")
            eng = Scanner_Engine_V38(t)
            res = eng.analyze()
            if res and res['score'] >= min_score:
                results.append(res)
            progress_bar.progress((i+1)/len(tickers))
            
        st.session_state.scan_results = sorted(results, key=lambda x: x['score'], reverse=True)
        status_text.text(f"掃描完成！找到 {len(results)} 檔標的。")
        progress_bar.empty()

    # 顯示掃描結果表格
    if st.session_state.scan_results:
        df_scan = pd.DataFrame(st.session_state.scan_results)
        st.dataframe(
            df_scan[['ticker', 'score', 'price', 'rsi', 'sl']],
            column_config={
                "ticker": "代碼", "score": "動能評分", "price": "現價",
                "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                "sl": st.column_config.NumberColumn("建議停損", format="%.2f")
            },
            use_container_width=True
        )
        
        # =================================================
        # STEP 3: 精密打擊 (Deep Dive)
        # =================================================
        st.markdown("---")
        st.markdown("### 🎯 Step 3: 精密打擊 (Risk & Sizing)")
        
        selected_ticker = st.selectbox(
            "選擇一檔股票進行微觀與風控分析:", 
            options=[r['ticker'] for r in st.session_state.scan_results]
        )
        
        if st.button("🔍 執行深度分析 (Deep Dive)"):
            with st.spinner(f"正在分析 {selected_ticker} 的微觀結構與風控參數..."):
                micro_score, signals, df_micro = Micro_Structure_Engine.analyze(selected_ticker)
                target_info = next((r for r in st.session_state.scan_results if r['ticker'] == selected_ticker), None)
                
                if target_info and not df_micro.empty:
                    curr_p = target_info['price']
                    sl_p = target_info['sl']
                    vol_cap = 1.0 if "BTC" in selected_ticker else 0.5
                    chaos_sim = 0.6 
                    
                    size, details = Antifragile_Position_Sizing.calculate(capital, curr_p, sl_p, chaos_sim, vol_cap)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">微觀評分</div>
                            <div class="metric-value" style="color:#3fb950">{micro_score}</div>
                            <div class="metric-label">{', '.join(signals) if signals else 'N/A'}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Taleb 建議部位</div>
                            <div class="metric-value">{size} 股/單位</div>
                            <div class="metric-label" style="color:#d2a8ff">本金: ${details['final_capital']:,}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">智能停損 (SL)</div>
                            <div class="metric-value" style="color:#f85149">{sl_p:.2f}</div>
                            <div class="metric-label">風險: -${details['risk_money']}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    st.markdown("#### 技術分析圖表 (Keltner Channel)")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    plot_df = df_micro.tail(120)
                    ax.plot(plot_df.index, plot_df['Close'], color='white', lw=1.5, label='Price')
                    ax.plot(plot_df.index, plot_df['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                    ax.plot(plot_df.index, plot_df['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                    ax.fill_between(plot_df.index, plot_df['K_Upper'], plot_df['K_Lower'], color='#00f2ff', alpha=0.1)
                    ax.axhline(sl_p, color='#f85149', ls='-', lw=1, label=f'SL: {sl_p:.2f}')
                    
                    ax.set_facecolor('#0e1117'); fig.patch.set_facecolor('#0e1117')
                    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.1)
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.error("數據獲取失敗，請重試。")

if __name__ == "__main__":
    main()
