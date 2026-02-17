import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from scipy.stats import norm
import warnings
from datetime import datetime

# 兼容性處理 (Wasserstein Distance)
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        u_values = np.sort(u_values)
        v_values = np.sort(v_values)
        return np.mean(np.abs(u_values - v_values))

# =============================================================================
# 0. 系統核心配置 & CSS (The Skin)
# =============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="MARCS ULTIMATE", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* Global Dark Theme */
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; color: #c9d1d9; }
    
    /* Bento Grid Card */
    .metric-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px;
    }
    .metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-family: 'Roboto Mono'; }
    .metric-value { color: #e6edf3; font-size: 24px; font-weight: 700; margin: 4px 0; }
    .metric-sub { font-size: 11px; font-family: 'Roboto Mono'; }
    
    /* Custom Tags */
    .tag-physics { background: rgba(210, 168, 255, 0.15); color: #d2a8ff; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(210, 168, 255, 0.3); }
    .tag-macro { background: rgba(56, 139, 253, 0.15); color: #58a6ff; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(56, 139, 253, 0.3); }
    
    /* Colors */
    .c-green { color: #3fb950; } .c-red { color: #f85149; } .c-gold { color: #d29922; } .c-purple { color: #d2a8ff; }
</style>
""", unsafe_allow_html=True)

# 工具函式
def smart_format(value, is_currency=True):
    if value is None or pd.isna(value): return "N/A"
    val = float(value)
    prefix = "$" if is_currency else ""
    if abs(val) < 1 and abs(val) > 0: return f"{prefix}{val:.4f}"
    return f"{prefix}{val:,.2f}"

# =============================================================================
# 1. 物理引擎核心 (V40 Physics Engine)
# =============================================================================
class Physics_Engine:
    """計算 Hilbert Transform, Hurst, VPIN, Chaos"""
    
    @staticmethod
    def calc_metrics(df):
        if df.empty or len(df) < 50: return df
        df = df.copy()
        
        c = df['Close'].values
        v = df['Volume'].values
        
        # 1. Hilbert Transform (Phase Sync)
        # 去除趨勢 (Detrend)
        ema = pd.Series(c).ewm(span=20).mean()
        detrended_c = c - ema
        detrended_v = v - pd.Series(v).rolling(20).mean()
        
        # 填充 NaN
        detrended_c = detrended_c.fillna(0).values
        detrended_v = detrended_v.fillna(0).values
        
        # 計算相位
        analytic_c = hilbert(detrended_c)
        analytic_v = hilbert(detrended_v)
        phase_c = np.angle(analytic_c)
        phase_v = np.angle(analytic_v)
        
        # Sync: 1 = 共振, -1 = 背離
        df['Sync'] = np.cos(phase_c - phase_v)
        df['Sync_Smooth'] = df['Sync'].rolling(5).mean()
        
        # 2. VPIN Proxy (Order Flow Toxicity)
        delta_p = np.diff(c, prepend=c[0])
        sigma = np.std(delta_p) + 1e-9
        cdf = norm.cdf(delta_p / sigma)
        buy_vol = v * cdf
        sell_vol = v * (1 - cdf)
        oi = np.abs(buy_vol - sell_vol)
        df['VPIN'] = pd.Series(oi).rolling(20).sum() / (pd.Series(v).rolling(20).sum() + 1e-9)
        
        # 3. Chaos (Wasserstein)
        log_ret = np.log(df['Close']).diff().fillna(0)
        # 滾動計算 Chaos (比較 t與t-1的分布)
        chaos_list = [0]*40
        for i in range(40, len(df)):
            w2 = wasserstein_distance(log_ret[i-20:i], log_ret[i-40:i-20])
            chaos_list.append(w2 * 1000) # Scale up
        df['Chaos'] = pd.Series(chaos_list, index=df.index).fillna(0)
        
        # 4. Technicals (ATR/RSI) for Risk Mgmt
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def get_signal_score(row):
        """用於掃描器的快速評分"""
        score = 50
        # 物理共振加分
        if row['Sync_Smooth'] > 0.5: score += 25
        if row['Sync_Smooth'] < -0.5: score -= 15
        
        # 籌碼安定加分
        if row['VPIN'] < 0.3: score += 10
        if row['VPIN'] > 0.6: score -= 20
        
        # 動能加分
        if row['RSI'] > 50 and row['RSI'] < 80: score += 15
        
        return min(max(score, 0), 100)

# =============================================================================
# 2. 宏觀模組 (V100 Macro Oracle)
# =============================================================================
class Macro_Brain:
    INDICES = {
        "RISK": ["^NDX", "^SOX", "BTC-USD", "^TWII"],
        "SAFE": ["TLT", "GLD", "DX-Y.NYB"],
        "SENTIMENT": ["^VIX"]
    }
    ALL_TICKERS = [t for cat in INDICES.values() for t in cat]

    @staticmethod
    @st.cache_data(ttl=3600) # 緩存1小時
    def fetch_macro_data():
        df = yf.download(Macro_Brain.ALL_TICKERS, period="6mo", progress=False)
        return df

    @staticmethod
    def analyze_macro(df):
        metrics = {}
        df = df.ffill().bfill()
        # 相關性矩陣
        corr = df['Close'].pct_change().tail(60).corr()
        
        for ticker in df['Close'].columns:
            close = df['Close'][ticker]
            # 簡單物理計算
            log_ret = np.log(close).diff().dropna()
            w2 = wasserstein_distance(log_ret.tail(20), log_ret.iloc[-40:-20])
            chaos = w2 * 1000
            
            ma50 = close.rolling(50).mean().iloc[-1]
            trend = (close.iloc[-1] - ma50) / ma50
            
            metrics[ticker] = {
                "Price": close.iloc[-1],
                "Trend%": trend * 100,
                "Chaos": chaos
            }
            
        # 判斷體制
        ndx = metrics.get('^NDX', {}).get('Trend%', 0)
        tlt = metrics.get('TLT', {}).get('Trend%', 0)
        vix = metrics.get('^VIX', {}).get('Price', 20)
        
        regime = "NEUTRAL"
        color = "#8b949e"
        if ndx > 0 and tlt > -2: regime, color = "GOLDILOCKS (舒適)", "#3fb950"
        elif ndx < 0 and tlt > 0 and vix > 25: regime, color = "RECESSION (衰退)", "#f85149"
        elif ndx < 0 and tlt < -2: regime, color = "INFLATION (通膨)", "#d2a8ff"
        
        # 計算總分
        score = 50
        if metrics.get('^VIX',{}).get('Price',20) < 20: score += 10
        if metrics.get('^NDX',{}).get('Trend%',0) > 0: score += 10
        if metrics.get('DX-Y.NYB',{}).get('Trend%',0) < 0: score += 10
        
        return metrics, corr, regime, color, score

# =============================================================================
# 3. 回測與策略模組 (V90 + V40 Fusion)
# =============================================================================
class Unified_Backtester:
    def __init__(self, ticker, capital, fee_rate, tax_rate, use_physics=True):
        self.ticker = ticker
        self.capital = capital
        self.fee_rate = fee_rate
        self.tax_rate = tax_rate
        self.use_physics = use_physics

    def run(self):
        # 下載數據 (Physics 需要連續數據，推薦 1h，但日線也可以跑)
        df = yf.download(self.ticker, period="1y", interval="1h", progress=False, auto_adjust=True)
        if df.empty: 
            # Fallback to Daily if 1h fails (e.g. non-US stocks often fail on 1h)
            df = yf.download(self.ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
        if df.empty: return None, None, None

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 計算物理指標
        df = Physics_Engine.calc_metrics(df)
        df = df.dropna()
        
        # 初始化回測變數
        cash = self.capital
        position = 0
        trades = []
        equity_curve = []
        
        # 基準 (Buy & Hold)
        start_price = df['Close'].iloc[0]
        bh_shares = self.capital // start_price
        
        total_fees = 0

        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            sync = df['Sync_Smooth'].iloc[i]
            vpin = df['VPIN'].iloc[i]
            
            # --- 策略核心 (Physics Fusion) ---
            # 進場: 能量共振 (Sync > 0.5) & 籌碼安定 (VPIN < 0.5)
            buy_signal = (sync > 0.5) and (vpin < 0.5)
            
            # 出場: 能量背離 (Sync < 0) 或 籌碼劇毒 (VPIN > 0.7)
            sell_signal = (sync < -0.2) or (vpin > 0.7)
            
            # 執行交易
            if position > 0 and sell_signal:
                gross = position * price
                fee = gross * self.fee_rate
                tax = gross * self.tax_rate
                cash += (gross - fee - tax)
                total_fees += (fee + tax)
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Reason': 'Physics Exit'})
                position = 0
            
            elif position == 0 and buy_signal:
                cost = cash * 0.98 # Buffer
                fee = cost * self.fee_rate
                shares = (cost - fee) // price
                if shares > 0:
                    cash -= (shares * price + fee)
                    total_fees += fee
                    position = shares
                    trades.append({'Date': date, 'Type': 'BUY', 'Price': price})

            # 計算淨值
            strat_val = cash + (position * price)
            bh_val = bh_shares * price # 簡單對照
            
            equity_curve.append({'Date': date, 'Strategy': strat_val, 'BuyHold': bh_val})
            
        return pd.DataFrame(equity_curve), pd.DataFrame(trades), df

# =============================================================================
# 4. 介面渲染函式 (UI Renderers)
# =============================================================================

def render_macro_oracle():
    st.markdown("### 🌍 MACRO ORACLE (宏觀預言機)")
    
    with st.spinner("Connecting to Global Markets..."):
        df = Macro_Brain.fetch_macro_data()
        if df.empty: st.error("Data connection failed."); return
        
        metrics, corr, regime, reg_color, score = Macro_Brain.analyze_macro(df)
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-label'>MACRO SCORE</div><div class='metric-value'>{score}</div><div class='metric-sub'>Risk Appetite</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-label'>REGIME</div><div style='font-size:20px; font-weight:bold; color:{reg_color}'>{regime}</div><div class='metric-sub'>Current State</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>VIX</div><div class='metric-value'>{metrics['^VIX']['Price']:.2f}</div><div class='metric-sub'>Fear Index</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-label'>LIQUIDITY (TLT)</div><div class='metric-value' style='color:{'#3fb950' if metrics['TLT']['Trend%']>0 else '#f85149'}'>{metrics['TLT']['Trend%']:+.1f}%</div><div class='metric-sub'>Bond Trend</div></div>", unsafe_allow_html=True)

        # Correlation Matrix
        st.markdown("#### 🔗 Global Correlation Heatmap (60D)")
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(paper_bgcolor="#050505", plot_bgcolor="#050505", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

def render_market_scanner():
    st.markdown("### 🔭 QUANTUM SCANNER (物理掃描器)")
    
    default_list = "BTC-USD, ETH-USD, SOL-USD, NVDA, TSLA, AAPL, COIN, MSTR"
    user_input = st.text_area("Watchlist (Comma Separated)", default_list)
    tickers = [x.strip() for x in user_input.split(",")]
    
    if st.button("🚀 INITIATE SCAN", type="primary"):
        results = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            try:
                # 掃描模式用日線即可，速度較快
                df = yf.download(t, period="6mo", progress=False, auto_adjust=True)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = Physics_Engine.calc_metrics(df)
                    row = df.iloc[-1]
                    score = Physics_Engine.get_signal_score(row)
                    
                    if score > 0:
                        results.append({
                            "Ticker": t,
                            "Price": row['Close'],
                            "Score": score,
                            "Sync": row['Sync_Smooth'],
                            "VPIN": row['VPIN'],
                            "RSI": row['RSI']
                        })
            except: pass
            progress_bar.progress((i+1)/len(tickers))
            
        if results:
            res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(
                res_df.style.background_gradient(subset=['Score'], cmap='Greens')
                      .format({'Price': "{:.2f}", 'Sync': "{:.2f}", 'VPIN': "{:.2f}", 'RSI': "{:.0f}"}),
                use_container_width=True
            )
        else:
            st.info("No tickers found or data error.")

def render_strategy_lab():
    st.markdown("### 🧪 PHYSICS LAB (深度策略實驗室)")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    ticker = c1.text_input("Target Ticker", "BTC-USD")
    capital = c2.number_input("Capital", 10000)
    fee = c3.number_input("Fee Rate", 0.001, format="%.4f")
    
    if st.button("🔬 RUN SIMULATION", type="primary"):
        with st.spinner("Running Physics Simulation (Hilbert/Hurst)..."):
            ub = Unified_Backtester(ticker, capital, fee, 0.0) # Tax設為0簡化，可自行添加
            df_eq, df_tr, df_raw = ub.run()
            
            if df_eq is None: st.error("Simulation Failed."); return
            
            # 1. 績效總結
            final_ret = (df_eq['Strategy'].iloc[-1] - capital) / capital * 100
            bh_ret = (df_eq['BuyHold'].iloc[-1] - capital) / capital * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-card'><div class='metric-label'>NET RETURN</div><div class='metric-value' style='color:{'#3fb950' if final_ret>0 else '#f85149'}'>{final_ret:+.2f}%</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><div class='metric-label'>ALPHA vs HOLD</div><div class='metric-value' style='color:#d2a8ff'>{(final_ret - bh_ret):+.2f}%</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><div class='metric-label'>TRADES</div><div class='metric-value'>{len(df_tr)}</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-card'><div class='metric-label'>LAST SYNC</div><div class='metric-value'>{df_raw['Sync_Smooth'].iloc[-1]:.2f}</div></div>", unsafe_allow_html=True)
            
            # 2. 雙圖表 (Price + Physics)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Price
            fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Price', line=dict(color='#8b949e', width=1)), row=1, col=1)
            # Buy/Sell Signals
            if not df_tr.empty:
                buys = df_tr[df_tr['Type']=='BUY']
                sells = df_tr[df_tr['Type']=='SELL']
                fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00f2ff'), name='Buy'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#f85149'), name='Sell'), row=1, col=1)
            
            # Physics (Sync)
            fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Sync_Smooth'], name='Phase Sync', line=dict(color='#d2a8ff', width=2)), row=2, col=1)
            fig.add_hrect(y0=0.5, y1=1.0, row=2, col=1, fillcolor="#d2a8ff", opacity=0.1, line_width=0)
            fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="#333")
            
            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#050505", plot_bgcolor="#050505")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 5. 主程式入口 (Main Loop)
# =============================================================================
def main():
    with st.sidebar:
        st.title("⚡ MARCS ULT")
        st.caption("v100.2 | Physics Core")
        st.markdown("---")
        
        mode = st.radio("SYSTEM MODULE", [
            "🌍 MACRO ORACLE", 
            "🔭 QUANTUM SCANNER", 
            "🧪 PHYSICS LAB"
        ])
        
        st.markdown("---")
        st.info("System Status: ONLINE")
    
    if mode == "🌍 MACRO ORACLE":
        render_macro_oracle()
    elif mode == "🔭 QUANTUM SCANNER":
        render_market_scanner()
    elif mode == "🧪 PHYSICS LAB":
        render_strategy_lab()

if __name__ == "__main__":
    main()
