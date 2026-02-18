import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, lfilter, hilbert
from datetime import datetime

# =============================================================================
# 1. 系統核心配置
# =============================================================================
st.set_page_config(page_title="MARCS NEO-LEVIATHAN V2", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #0E1117; font-family: 'Rajdhani', sans-serif; color: #C9D1D9; }
    .metric-card { background: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 15px; margin-bottom: 10px; }
    .highlight-lbl { font-size: 12px; color: #8B949E; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; }
    .highlight-val { font-size: 24px; font-weight: 700; color: #E6EDF3; font-family: 'JetBrains Mono'; }
    .signal-box { background: linear-gradient(180deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,1) 100%); border: 1px solid #30363D; border-radius: 12px; padding: 25px; text-align: center; }
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# =============================================================================
# 2. 數據與物理引擎 (Data & Physics) - [CORE KEPT INTACT]
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]: df = df.xs(ticker, axis=1, level=0)
                else: df.columns = df.columns.get_level_values(0)
            except:
                if len(df.columns) >= 5:
                    df = df.iloc[:, :5]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

class Signal_Processor:
    @staticmethod
    def causal_bandpass(data, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    @staticmethod
    def calc_adx(df, n=14):
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(lower=0) # Logic fix for calc
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/n).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/n).mean().abs() / atr)
        dx = 100 * (abs(plus_di - minus_di) / abs(plus_di + minus_di).replace(0, 1))
        return dx.rolling(n).mean().fillna(0)

    @staticmethod
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, 1e-9))
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calc_atr(df, n=14):
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    @staticmethod
    def engineer_features(df):
        if len(df) < 200: return df
        df = df.copy()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = Signal_Processor.calc_rsi(df['Close'])
        df['ADX'] = Signal_Processor.calc_adx(df)
        df['ATR'] = Signal_Processor.calc_atr(df)
        
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        cycle_component = Signal_Processor.causal_bandpass(log_ret.values, 1/40, 1/10, 1, order=2)
        analytic_signal = hilbert(cycle_component)
        price_phase = np.angle(analytic_signal)
        
        vol_change = df['Volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        vol_cycle = Signal_Processor.causal_bandpass(vol_change.values, 1/40, 1/10, 1, order=2)
        vol_phase = np.angle(hilbert(vol_cycle))
        
        df['Sync'] = np.cos(price_phase - vol_phase)
        df['Sync_Smooth'] = df['Sync'].rolling(3).mean()
        return df.iloc[100:]

class Strategy_Engine:
    @staticmethod
    def evaluate(df, capital=100000, risk_pct=0.02):
        last = df.iloc[-1]
        
        # Regime
        regime = "NEUTRAL"
        if last['ADX'] > 25: regime = "TRENDING"
        elif last['ADX'] < 20: regime = "RANGING"
            
        # Score
        score = 50
        if last['Sync_Smooth'] > 0.6: score += 25
        elif last['Sync_Smooth'] < -0.6: score -= 25
        if last['Close'] > last['EMA20']: score += 15
        else: score -= 15
        
        if regime == "TRENDING" and last['RSI'] > 70: score += 5
        elif regime == "RANGING":
            if last['RSI'] > 70: score -= 20
            if last['RSI'] < 30: score += 20
            
        score = min(max(score, 0), 100)
        
        # Position Sizing (Risk Management)
        # Volatility Sizing: Risk Amount / (ATR * Multiplier)
        risk_amount = capital * risk_pct
        stop_loss_dist = last['ATR'] * 2.5 # 2.5 ATR Stop
        if stop_loss_dist == 0: position_size = 0
        else: position_size = risk_amount / stop_loss_dist
        
        # Kelly Criterion Estimation (Simplified)
        # Assuming win_rate=0.45, reward_risk=2.0 for Trend Following
        kelly_f = 0.45 - (0.55 / 2.0) # ~ 17%
        kelly_size = (capital * max(kelly_f, 0)) / last['Close']
        
        suggested_pos = min(position_size, kelly_size) if score >= 70 else 0
        
        return {
            "score": score,
            "regime": regime,
            "last": last,
            "stop_loss": last['Close'] - stop_loss_dist,
            "take_profit": last['Close'] + (stop_loss_dist * 2.0),
            "suggested_shares": int(suggested_pos),
            "risk_exposure": risk_amount
        }

# =============================================================================
# 3. 擴展模組 (Expansion Modules)
# =============================================================================
def render_macro():
    st.markdown("### 🌍 Macro Oracle (宏觀風向)")
    c1, c2, c3 = st.columns(3)
    
    # 抓取 VIX 和 美元指數
    vix = fetch_data("^VIX", period="1mo")
    dxy = fetch_data("DX-Y.NYB", period="1mo")
    
    if not vix.empty:
        curr_vix = vix['Close'].iloc[-1]
        delta_vix = curr_vix - vix['Close'].iloc[-2]
        c_vix = "#F85149" if curr_vix > 20 else "#3FB950"
        c1.metric("VIX (恐慌指數)", f"{curr_vix:.2f}", f"{delta_vix:.2f}", delta_color="inverse")
        
    if not dxy.empty:
        curr_dxy = dxy['Close'].iloc[-1]
        c2.metric("DXY (美元指數)", f"{curr_dxy:.2f}", f"{curr_dxy - dxy['Close'].iloc[-2]:.2f}")
        
    # 簡易風向判斷
    status = "NEUTRAL"
    if not vix.empty:
        if curr_vix > 25: status = "RISK OFF (避險)"
        elif curr_vix < 15: status = "RISK ON (積極)"
    
    c3.markdown(f"""
    <div class='metric-card'>
        <div class='highlight-lbl'>MARKET REGIME</div>
        <div class='highlight-val' style='font-size:20px'>{status}</div>
    </div>
    """, unsafe_allow_html=True)

def render_scanner():
    st.markdown("### 🔭 Quantum Scanner (批量掃描)")
    st.caption("使用因果物理引擎掃描觀察清單")
    
    # 預設觀察清單
    default_tickers = "2330.TW, 2317.TW, 2454.TW, 2603.TW, NVDA, TSLA, BTC-USD, ETH-USD"
    user_input = st.text_area("輸入代碼 (逗號分隔)", default_tickers)
    
    if st.button("🚀 開始掃描"):
        tickers = [t.strip() for t in user_input.split(",")]
        results = []
        
        progress = st.progress(0)
        for i, t in enumerate(tickers):
            df = fetch_data(t)
            if not df.empty and len(df) > 200:
                df_eng = Signal_Processor.engineer_features(df)
                res = Strategy_Engine.evaluate(df_eng)
                results.append({
                    "Ticker": t,
                    "Score": res['score'],
                    "Regime": res['regime'],
                    "Sync": res['last']['Sync_Smooth'],
                    "Price": res['last']['Close']
                })
            progress.progress((i+1)/len(tickers))
            
        if results:
            res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(
                res_df.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100)
                      .format({"Price": "{:.2f}", "Sync": "{:.2f}"}),
                use_container_width=True
            )
        else:
            st.warning("無有效數據。")

def render_analysis():
    st.markdown("### 🛡️ Single Asset Deep Dive")
    c_side, c_main = st.columns([1, 3])
    
    with c_side:
        ticker = st.text_input("Ticker", "2330.TW")
        capital = st.number_input("Account Capital ($)", value=100000, step=10000)
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100
        run = st.button("ANALYZE", type="primary")
        
    if run:
        with st.spinner("Calculating Physics..."):
            raw_df = fetch_data(ticker)
            if raw_df.empty or len(raw_df) < 200:
                st.error("數據不足 (需 > 200 K棒)")
                return
                
            df = Signal_Processor.engineer_features(raw_df)
            res = Strategy_Engine.evaluate(df, capital, risk)
            
            # --- Signal Section ---
            score = res['score']
            color = "#3FB950" if score >= 70 else "#F85149" if score <= 30 else "#8B949E"
            action = "LONG" if score >= 70 else "SHORT/CASH" if score <= 30 else "WAIT"
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="signal-box" style="border-top: 4px solid {color}">
                    <div class="highlight-lbl">ACTION</div>
                    <div class="highlight-val" style="color:{color}">{action}</div>
                    <div>Score: {score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                # Position Sizing Card
                shares = res['suggested_shares']
                pos_val = shares * res['last']['Close']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="highlight-lbl">SUGGESTED SIZE</div>
                    <div class="highlight-val">{shares:,} <span style="font-size:14px">shares</span></div>
                    <div style="font-size:12px; color:#888">Val: ${pos_val:,.0f} (Kelly/ATR)</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                # Risk Card
                st.markdown(f"""
                <div class="metric-card">
                    <div class="highlight-lbl">RISK PLAN</div>
                    <div style="color:#F85149">SL: ${res['stop_loss']:.2f}</div>
                    <div style="color:#3FB950">TP: ${res['take_profit']:.2f}</div>
                    <div style="font-size:12px; color:#888">Risk: ${res['risk_exposure']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- Chart ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#FFAE00', width=1), name='EMA20'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Sync_Smooth'], line=dict(color='#D2A8FF', width=2), name='Sync'), row=2, col=1)
            fig.add_hrect(y0=0.6, y1=1.1, row=2, col=1, fillcolor="#3FB950", opacity=0.1, line_width=0)
            fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="#555")
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 4. 主程序入口
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ SYSTEM MODE")
    mode = st.sidebar.radio("Select Module", ["1. Single Analysis", "2. Market Scanner", "3. Macro Dashboard"])
    
    if "Single" in mode: render_analysis()
    elif "Scanner" in mode: render_scanner()
    elif "Macro" in mode: render_macro()

if __name__ == "__main__":
    main()
