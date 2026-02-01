import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import concurrent.futures
from datetime import datetime, timedelta

# =============================================================================
# 0. 系統配置與極致視覺 (V96 Sentinel Ultimate)
# =============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="MARCS V96 Sentinel", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Noto+Sans+TC:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', 'Noto Sans TC', sans-serif; color: #e6edf3; }
    /* 星空背景動畫 */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px), radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px; animation: stars 120s linear infinite; z-index: -1; opacity: 0.3;
    }
    @keyframes stars { from {transform: translateY(0);} to {transform: translateY(-1000px);} }
    .metric-card { background: rgba(20, 20, 25, 0.9); border-left: 4px solid #ffae00; padding: 15px; border-radius: 8px; margin-bottom: 10px; backdrop-filter: blur(10px); border: 1px solid #333; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 4px 4px 0 0; padding: 10px 20px; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 量化核心引擎 (Hurst, SMC, Indicators)
# =============================================================================
class Quant_Engine:
    @staticmethod
    def calculate_hurst(series):
        """優化效能：限制 120 樣本計算 Hurst"""
        if len(series) < 50: return 0.5
        y = series.tail(120).values
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(y[lag:], y[:-lag]))) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return max(0.0, min(1.0, reg[0] * 2.0))

    @staticmethod
    def detect_smc(df):
        """偵測 FVG 與 OB 結構"""
        if len(df) < 5: return None
        # Bullish FVG
        if df['Low'].iloc[-1] > df['High'].iloc[-3]:
            return {'type': 'Bullish', 'top': df['Low'].iloc[-1], 'bottom': df['High'].iloc[-3], 'ob': df['Close'].iloc[-3]}
        # Bearish FVG
        elif df['High'].iloc[-1] < df['Low'].iloc[-3]:
            return {'type': 'Bearish', 'top': df['Low'].iloc[-3], 'bottom': df['High'].iloc[-1], 'ob': df['Close'].iloc[-3]}
        return None

    @staticmethod
    def get_indicators(df):
        """計算 MACD, Force Index, ATR"""
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - signal_line
        # Force Index (13-day EMA of Price Change * Volume)
        force_index = (df['Close'].diff() * df['Volume']).ewm(span=13).mean()
        # ATR for Stop Loss
        atr = (df['High'] - df['Low']).rolling(14).mean()
        return macd_hist, force_index, atr

# =============================================================================
# 2. 多線程掃描與回測引擎
# =============================================================================
class Scanner_Sentinel:
    @staticmethod
    def analyze_ticker(ticker):
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: return None
            h = Quant_Engine.calculate_hurst(df['Close'])
            smc = Quant_Engine.detect_smc(df)
            score = 50 + (20 if h > 0.55 else -10) + (20 if smc and smc['type'] == 'Bullish' else 0)
            return {"Ticker": ticker, "Price": round(df['Close'].iloc[-1], 2), "Hurst": round(h, 3), "SMC": smc['type'] if smc else "None", "Score": score}
        except: return None

    @staticmethod
    def run_scan(market_type):
        tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "GOOG", "AMZN"] if market_type == "美股" else ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2382.TW"]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(Scanner_Sentinel.analyze_ticker, t) for t in tickers]
            for f in concurrent.futures.as_completed(futures):
                res = f.result(); 
                if res: results.append(res)
        return sorted(results, key=lambda x: x['Score'], reverse=True)

class Backtest_Lab:
    @staticmethod
    def run_strategy(ticker):
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df.empty: return None, 0
        df['Hurst'] = df['Close'].rolling(60).apply(Quant_Engine.calculate_hurst)
        df['EMA22'] = df['Close'].ewm(span=22).mean()
        capital = 100000.0; pos = 0; equity = []
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            if pos == 0 and df['Hurst'].iloc[i] > 0.52 and price > df['EMA22'].iloc[i]:
                pos = capital / price; capital = 0
            elif pos > 0 and (df['Hurst'].iloc[i] < 0.48 or price < df['EMA22'].iloc[i]):
                capital = pos * price; pos = 0
            equity.append(capital + (pos * price if pos > 0 else 0))
        return pd.Series(equity, index=df.index), (equity[-1]-100000)/100000

# =============================================================================
# 3. 主介面 UI 整合
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.title("🛡️ Sentinel V96")
    market_choice = st.sidebar.selectbox("掃描市場", ["美股", "台股"])
    if st.sidebar.button("🚀 啟動多線程掃描"):
        st.session_state.scan_results = Scanner_Sentinel.run_scan(market_choice)
    
    target = st.sidebar.text_input("分析代碼", "NVDA").upper()
    capital_base = st.sidebar.number_input("模擬本金", value=100000)

    # --- Data Fetching ---
    df = yf.download(target, period="1y", progress=False, auto_adjust=True)
    if df.empty: return st.error("數據獲取失敗")

    # --- Calculations ---
    h_val = Quant_Engine.calculate_hurst(df['Close'])
    smc_info = Quant_Engine.detect_smc(df)
    macd_hist, force_index, atr_series = Quant_Engine.get_indicators(df)
    
    # 最新數值
    curr_p = df['Close'].iloc[-1]
    sl_p = curr_p - (2.5 * atr_series.iloc[-1])
    
    # --- Header Dashboard ---
    st.title(f"MARCS V96: {target} 戰情室")
    if "scan_results" in st.session_state:
        with st.expander("📡 實時掃描情報 (Top Picks)", expanded=False):
            st.table(pd.DataFrame(st.session_state.scan_results))

    # --- Verdict Box ---
    score = int(h_val * 100)
    verdict = "✅ 建議佈局" if h_val > 0.52 else "😐 觀望震盪"
    color = "#1f6feb" if h_val > 0.52 else "#30363d"
    st.markdown(f"""<div class="verdict-box" style="background:{color}40; border-color:{color};">
        <h2 style="margin:0; color:{color};">{verdict}</h2>
        <p style="color:#8b949e; margin:5px 0 0 0;">Hurst: {h_val:.3f} | SMC: {smc_info['type'] if smc_info else 'Neutral'} | SL: ${sl_p:.2f}</p>
    </div>""", unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["🕯️ 技術疊加圖 (MACD/Force)", "🧬 SMC 結構詳解", "🔄 策略回測實驗室"])
    
    with tab1:
        # 三層繪圖邏輯
        fig, (ax_p, ax_m, ax_f) = plt.subplots(3, 1, figsize=(12, 10), 
                                               gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # [上層：價格/SMC/停損]
        ax_p.plot(df.index, df['Close'], color='white', lw=1.5, label="Price")
        ax_p.plot(df.index, df['Close'].ewm(span=22).mean(), color='#ffae00', ls='--', alpha=0.7, label="EMA22")
        ax_p.axhline(sl_p, color='#f85149', ls='-', lw=2, alpha=0.8) # 停損紅線
        ax_p.text(df.index[0], sl_p, f'  STOP LOSS: {sl_p:.2f}', color='#f85149', fontweight='bold')
        if smc_info:
            ax_p.axhspan(smc_info['bottom'], smc_info['top'], color='green' if smc_info['type']=='Bullish' else 'red', alpha=0.2)
        ax_p.legend(loc='upper left', facecolor='#111', edgecolor='#333')

        # [中層：MACD]
        m_colors = ['#3fb950' if x > 0 else '#f85149' for x in macd_hist]
        ax_m.bar(df.index, macd_hist, color=m_colors, alpha=0.8, width=1.0)
        ax_m.set_ylabel("MACD Hist", color='#888')

        # [下層：Force Index]
        ax_f.plot(df.index, force_index, color='#00f2ff', lw=1.2)
        ax_f.axhline(0, color='gray', ls='--', lw=0.8)
        ax_f.fill_between(df.index, force_index, 0, where=(force_index > 0), color='#00f2ff', alpha=0.1)
        ax_f.set_ylabel("Force Index", color='#888')

        # 視覺美化
        for ax in [ax_p, ax_m, ax_f]:
            ax.set_facecolor('#050505'); ax.tick_params(colors='#666', labelsize=8)
            ax.grid(alpha=0.05, color='white')
        fig.patch.set_facecolor('#050505')
        st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏛️ SMC 結構分析")
            if smc_info:
                st.success(f"偵測到 {smc_info['type']} 缺口 (FVG)")
                st.write(f"**區間上限:** {smc_info['top']:.2f}")
                st.write(f"**區間下限:** {smc_info['bottom']:.2f}")
                st.write(f"**機構成本區 (OB):** {smc_info['ob']:.2f}")
            else: st.write("目前市場結構完整，無明顯流動性缺口。")
        with c2:
            st.markdown("### 📊 波動率診斷")
            st.metric("ATR (14)", f"{atr_series.iloc[-1]:.2f}")
            st.metric("建議停損位", f"{sl_p:.2f}")

    with tab3:
        st.markdown("### 🔄 歷史回測報告 (2-Year Horizon)")
        if st.button("執行回測模擬"):
            equity_curve, ret = Backtest_Lab.run_strategy(target)
            st.metric("累積報酬率", f"{ret:.2%}")
            st.line_chart(equity_curve)

if __name__ == "__main__":
    main()
