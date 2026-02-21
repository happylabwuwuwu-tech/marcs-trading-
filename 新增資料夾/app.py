import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
import cvxpy as cp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =============================================================================
# 1. 系統核心配置 (Core Config)
# =============================================================================
st.set_page_config(page_title="MARCS NEO-LEVIATHAN V4", layout="wide", page_icon="🛡️")

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
# 2. 數據與物理引擎 (Data & Physics Engine)
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    try:
        # Auto adjust ensures we get split/dividend adjusted prices
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        
        # Handle MultiIndex columns if present (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]: 
                    df = df.xs(ticker, axis=1, level=0)
                else: 
                    # Fallback if structure is different
                    df.columns = df.columns.get_level_values(0)
            except:
                pass
        
        # Ensure standard columns exist
        if len(df.columns) >= 5:
            # Simple check to rename if needed, though auto_adjust usually gives correct names
            pass 
            
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

class Signal_Processor:
    @staticmethod
    def calc_atr(df, n=14):
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    @staticmethod
    def rolling_hurst(series, window=50):
        diff10 = series.diff(10)
        diff1 = series.diff(1)
        var_diff1 = diff1.rolling(window).var()
        var_diff10 = diff10.rolling(window).var()
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = np.log(np.sqrt(var_diff10 / var_diff1.replace(0, np.nan))) / np.log(10)
            hurst = 0.5 + (tau / 2)
        return hurst.fillna(0.5)

    @staticmethod
    def engineer_features(df):
        if len(df) < 100: return df
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['ATR'] = Signal_Processor.calc_atr(df)
        df['Hurst'] = Signal_Processor.rolling_hurst(df['Close'], window=50)
        df['EMA20'] = df['SMA20'] # UI Compatibility
        return df.iloc[50:]

class GAMMA_Engine:
    """
    機構級量化核心：整合 SVD 正交投影、Ledoit-Wolf 共變異數收縮與帶換手率懲罰的凸優化器。
    專為 Long-Only (純做多) 實盤環境打造。
    """
    
    @staticmethod
    def robust_orthogonal_projection(alpha: np.ndarray, X: np.ndarray, variance_threshold: float = 0.85) -> np.ndarray:
        """模組 1: SVD Alpha 中性化 (剝離系統性 Beta)"""
        X_demean = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_demean, full_matrices=False)
        
        explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        k_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        k_components = min(k_components, X.shape[1] - 1)
        
        U_k = U[:, :k_components]
        N_assets = X.shape[0]
        
        P = np.eye(N_assets) - U_k @ U_k.T
        alpha_neutral = P @ alpha
        
        if np.std(alpha_neutral) > 0:
            alpha_neutral = (alpha_neutral - np.mean(alpha_neutral)) / np.std(alpha_neutral)
            
        return alpha_neutral

    @staticmethod
    def linear_covariance_shrinkage(returns: np.ndarray, delta: float = 0.5) -> np.ndarray:
        """模組 2: ESL 線性共變異數收縮"""
        S = np.cov(returns, rowvar=False)
        F = np.diag(np.diag(S))
        Sigma_shrink = delta * F + (1 - delta) * S
        Sigma_shrink = (Sigma_shrink + Sigma_shrink.T) / 2
        Sigma_shrink += np.eye(Sigma_shrink.shape[0]) * 1e-8
        return Sigma_shrink

    @staticmethod
    def optimize_long_only_portfolio(
        alpha: np.ndarray, 
        Sigma: np.ndarray, 
        w_prev: np.ndarray, 
        risk_aversion: float = 2.0, 
        turnover_penalty: float = 0.003, 
        max_position: float = 0.3
    ) -> np.ndarray:
        """模組 3: 帶換手率懲罰的凸優化"""
        N = len(alpha)
        w = cp.Variable(N)
        
        expected_return = alpha.T @ w
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))
        turnover = cp.norm(w - w_prev, 1)
        
        objective = cp.Maximize(expected_return - (risk_aversion * risk) - (turnover_penalty * turnover))
        
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= max_position
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.ECOS)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return w_prev
            
            w_opt = np.array(w.value)
            w_opt[w_opt < 1e-4] = 0.0
            return w_opt / np.sum(w_opt)
            
        except Exception as e:
            print(f"Optimizer failed: {e}")
            return w_prev

    @classmethod
    def run_pipeline(cls, returns_df: pd.DataFrame, raw_alpha: pd.Series, w_prev: dict = None) -> dict:
        tickers = returns_df.columns.tolist()
        N = len(tickers)
        
        X = returns_df.values.T 
        ret_matrix = returns_df.values 
        alpha_vec = raw_alpha.reindex(tickers).fillna(0).values
        
        if w_prev is None:
            w_p = np.zeros(N)
        else:
            w_p = np.array([w_prev.get(t, 0.0) for t in tickers])
            
        alpha_neutral = cls.robust_orthogonal_projection(alpha_vec, X)
        Sigma_shrink = cls.linear_covariance_shrinkage(ret_matrix)
        w_opt = cls.optimize_long_only_portfolio(alpha_neutral, Sigma_shrink, w_p)
        
        return dict(zip(tickers, np.round(w_opt, 4)))

class Strategy_Engine:
    """
    單一資產策略評估引擎 (用於 Scanner 和 Single Analysis)
    """
    @staticmethod
    def evaluate(df, capital=100000, risk_pct=0.02):
        if df.empty: return {}
        last = df.iloc[-1]
        
        # 簡單策略邏輯
        is_trending = last['Hurst'] > 0.55
        is_bullish = last['SMA20'] > last['SMA50']
        
        regime = "TRENDING" if is_trending else "CHOP/NOISE"
        
        # 評分系統
        score = 50
        if is_trending:
            score += 20
            if is_bullish: score += 20
        else:
            score = 10
        
        # 倉位計算
        risk_amount = capital * risk_pct
        stop_loss_dist = last['ATR'] * 2.5 
        
        if stop_loss_dist == 0:
            position_size = 0
        else:
            position_size = risk_amount / stop_loss_dist
            
        suggested_pos = position_size if score >= 70 else 0
        
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
# 3. 市場爬蟲與過濾器 (Market Crawlers & Filters)
# =============================================================================
class Market_List_Provider:
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_full_tw_tickers():
        try:
            url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
            df_twse = pd.read_html(requests.get(url_twse, timeout=10).text)[0]
            df_twse.columns = df_twse.iloc[0]
            
            url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
            df_tpex = pd.read_html(requests.get(url_tpex, timeout=10).text)[0]
            df_tpex.columns = df_tpex.iloc[0]
            
            df_all = pd.concat([df_twse.iloc[1:], df_tpex.iloc[1:]])
            df_all['Code'] = df_all['有價證券代號及名稱'].apply(lambda x: x.split()[0] if type(x)==str else "")
            stocks = df_all[df_all['Code'].str.len() == 4]
            return [f"{c}.TW" for c in stocks['Code'].tolist()]
        except:
            return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2881.TW"]

    @staticmethod
    def get_crypto_list():
        return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD"]

class Batch_Pre_Filter:
    @staticmethod
    def filter_by_volume(tickers, min_volume_shares=1000000):
        survivors = []
        batch_size = 20 # Reduced batch size for stability
        status_text, bar = st.empty(), st.progress(0)
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i+batch_size]
            status_text.text(f"🚀 Filtering Volume: Batch {i//batch_size + 1}...")
            bar.progress(min((i + batch_size) / len(tickers), 1.0))
            try:
                # yfinance download structure handling
                df = yf.download(batch, period="5d", group_by='ticker', progress=False, threads=True)
                
                for t in batch:
                    try:
                        if len(batch) == 1:
                            vol = df['Volume'].iloc[-1]
                            price = df['Close'].iloc[-1]
                        else:
                            # Check if ticker exists in columns
                            if t not in df.columns.levels[0]: continue
                            vol = df[t]['Volume'].iloc[-1]
                            price = df[t]['Close'].iloc[-1]
                            
                        if pd.notna(vol) and pd.notna(price) and vol > min_volume_shares and price > 10:
                            survivors.append(t)
                    except: continue
            except: pass
        _ = bar.empty()
        _ = status_text.empty()
        return survivors

# =============================================================================
# 4. 渲染模組 (Render Modules)
# =============================================================================
def render_macro():
    st.markdown("### 🌍 Macro Oracle (宏觀風向)")
    c1, c2, c3 = st.columns(3)
    vix = fetch_data("^VIX", period="1mo")
    dxy = fetch_data("DX-Y.NYB", period="1mo")
    
    if not vix.empty:
        delta = vix['Close'].iloc[-1] - vix['Close'].iloc[-2]
        c1.metric("VIX (恐慌指數)", f"{vix['Close'].iloc[-1]:.2f}", f"{delta:.2f}", delta_color="inverse")
    if not dxy.empty:
        delta_dxy = dxy['Close'].iloc[-1] - dxy['Close'].iloc[-2]
        c2.metric("DXY (美元指數)", f"{dxy['Close'].iloc[-1]:.2f}", f"{delta_dxy:.2f}")
    
    status = "NEUTRAL"
    if not vix.empty:
        val = vix['Close'].iloc[-1]
        status = "RISK OFF (避險)" if val > 25 else "RISK ON (積極)" if val < 15 else "NEUTRAL"
        
    color = "#F85149" if "OFF" in status else "#3FB950" if "ON" in status else "#8B949E"
    c3.markdown(f"<div class='metric-card'><div class='highlight-lbl'>MARKET REGIME</div><div class='highlight-val' style='color:{color}'>{status}</div></div>", unsafe_allow_html=True)

def render_scanner():
    st.markdown("### 🔭 Quantum Scanner (Leviathan Mode)")
    col1, col2 = st.columns([2, 1])
    with col1: market = st.selectbox("Market Target", ["🇹🇼 台股全市場 (Full TW)", "🪙 Crypto Top 20"])
    with col2: vol_lots = st.slider("Min Volume (張/Lots)", 500, 10000, 2000, step=500) if "TW" in market else 0
    
    if st.button("🚀 Start Deep Scan"):
        if "TW" in market:
            raw_tickers = Market_List_Provider.get_full_tw_tickers()
            # Limit to first 50 for demo speed, remove [:50] for full scan
            targets = Batch_Pre_Filter.filter_by_volume(raw_tickers[:100], min_volume_shares=vol_lots*1000)
        else:
            raw_tickers = Market_List_Provider.get_crypto_list()
            targets = raw_tickers
            
        if not targets:
            st.warning("No liquid assets found.")
            return
        
        res_list = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, t in enumerate(targets):
            status.text(f"🔬 Physics Engine: {t}...")
            try:
                df = fetch_data(t, period="1y")
                if len(df) > 100:
                    df = Signal_Processor.engineer_features(df)
                    res = Strategy_Engine.evaluate(df)
                    if res.get('score', 0) >= 50: # Show more results
                        res_list.append({
                            "Ticker": t, 
                            "Price": res['last']['Close'], 
                            "Score": res['score'], 
                            "Hurst": res['last']['Hurst'],
                            "Regime": res['regime']
                        })
            except Exception as e: 
                pass
            bar.progress(min((i+1)/len(targets), 1.0))
            
        _ = bar.empty()
        _ = status.empty()
        
        if res_list:
            df_res = pd.DataFrame(res_list).sort_values(by=["Score", "Hurst"], ascending=[False, False])
            st.dataframe(
                df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn')
                      .background_gradient(subset=['Hurst'], cmap='Purples', vmin=0.4, vmax=0.7)
                      .format({"Price": "{:.2f}", "Hurst": "{:.3f}"}), 
                use_container_width=True
            )
        else:
            st.info("No assets matched the criteria.")

def render_analysis():
    st.markdown("### 🛡️ Single Asset Deep Dive")
    c_side, c_main = st.columns([1, 3])
    with c_side:
        ticker = st.text_input("Ticker", "2330.TW")
        capital = st.number_input("Capital ($)", value=100000, step=10000)
        risk = st.slider("Risk/Trade (%)", 0.5, 5.0, 2.0) / 100
        run = st.button("ANALYZE", type="primary")
        
    if run:
        df = fetch_data(ticker)
        if df.empty:
            st.error("Ticker not found or no data.")
            return
            
        df = Signal_Processor.engineer_features(df)
        res = Strategy_Engine.evaluate(df, capital, risk)
        
        if not res:
            st.error("Not enough data to calculate indicators.")
            return

        color = "#3FB950" if res['score'] >= 70 else "#F85149" if res['score'] <= 30 else "#8B949E"
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='signal-box' style='border-top: 4px solid {color}'><div class='highlight-lbl'>ACTION</div><div class='highlight-val' style='color:{color}'>{'LONG' if res['score']>=70 else 'WAIT'}</div><div style='color:#888'>{res['regime']} (H={res['last']['Hurst']:.2f})</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='highlight-lbl'>SIZE</div><div class='highlight-val'>{res['suggested_shares']:,}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='highlight-lbl'>RISK (2.5x ATR)</div><div style='color:#F85149'>SL: ${res['stop_loss']:.2f}</div><div style='color:#3FB950'>TP: ${res['take_profit']:.2f}</div></div>", unsafe_allow_html=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#FFAE00'), name='SMA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#00BFFF', dash='dot'), name='SMA50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], line=dict(color='#D2A8FF'), name='Hurst'), row=2, col=1)
        fig.add_hrect(y0=0.55, y1=1.0, row=2, col=1, fillcolor="#3FB950", opacity=0.15)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 5. 歷史回測模組 (Backtest Engine Module)
# =============================================================================
def render_backtest():
    st.markdown("### 📊 Historical Backtest Engine")
    st.caption("驗證物理邏輯：Hurst > 0.55 且 MA多頭 + 2.5x ATR 動態止損")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("Ticker to Backtest", "2330.TW")
    with col2: years = st.slider("History (Years)", 1, 10, 4)
    with col3: friction = st.number_input("Friction per Trade (%)", value=0.1, step=0.05) / 100

    if st.button("⚙️ Execute Backtest"):
        with st.spinner("Processing Time Series..."):
            df = fetch_data(ticker, period=f"{years}y")
            if df.empty or len(df) < 100:
                st.error("Insufficient historical data.")
                return
                
            df = Signal_Processor.engineer_features(df).reset_index()
            # Rename index column to Date if needed
            if 'Date' not in df.columns and 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            
            # --- Vectorized Simulation & State Machine ---
            signal = np.zeros(len(df))
            in_pos = False
            highest_high = 0
            
            # Simple loop backtest (can be vectorized further but loop is readable)
            for i in range(1, len(df)):
                current_close = df['Close'].iloc[i-1]
                current_sma20 = df['SMA20'].iloc[i-1]
                current_sma50 = df['SMA50'].iloc[i-1]
                current_hurst = df['Hurst'].iloc[i-1]
                current_atr = df['ATR'].iloc[i-1]
                
                if pd.isna(current_sma50): continue

                if not in_pos:
                    if current_sma20 > current_sma50 and current_hurst > 0.55:
                        in_pos = True
                        highest_high = df['High'].iloc[i]
                        signal[i] = 1
                else:
                    highest_high = max(highest_high, df['High'].iloc[i])
                    stop_price = highest_high - (2.5 * current_atr)
                    
                    if current_close < stop_price or current_sma20 < current_sma50:
                        in_pos = False
                        signal[i] = 0
                    else:
                        signal[i] = 1
                        
            # --- Performance Metrics ---
            base_ret = pd.Series(signal).shift(1) * df['Close'].pct_change()
            base_ret = base_ret.fillna(0)
            trade_events = pd.Series(signal).diff().abs().fillna(0) / 2 # entry+exit = 1 cycle
            net_ret = base_ret - (trade_events * friction)
            
            df['Strategy_Equity'] = (1 + net_ret).cumprod() * 100
            df['Hold_Equity'] = (1 + df['Close'].pct_change().fillna(0)).cumprod() * 100
            
            str_cum_ret = df['Strategy_Equity'].iloc[-1] - 100
            
            # Max Drawdown
            peak = df['Strategy_Equity'].cummax()
            dd = (df['Strategy_Equity'] - peak) / peak
            str_max_dd = dd.min() * 100
            
            # Sharpe
            sharpe = np.sqrt(252) * net_ret.mean() / net_ret.std() if net_ret.std() != 0 else 0
            
            # --- Render Dashboard ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Return", f"{str_cum_ret:.2f}%")
            c2.metric("Max Drawdown", f"{str_max_dd:.2f}%", delta_color="inverse")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c4.metric("Total Trades", int(trade_events.sum()))
            
            # --- Equity Curve Chart ---
            fig = go.Figure()
            # Ensure Date column exists for plotting
            x_axis = df['Date'] if 'Date' in df.columns else df.index
            
            fig.add_trace(go.Scatter(x=x_axis, y=df['Strategy_Equity'], line=dict(color='#3FB950', width=2), name='Strategy (Net)'))
            fig.add_trace(go.Scatter(x=x_axis, y=df['Hold_Equity'], line=dict(color='#888', dash='dot'), name='Buy & Hold'))
            fig.update_layout(template="plotly_dark", height=400, title="Equity Curve Comparison")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 主程序入口 (Main Router)
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ SYSTEM MODE")
    mode = st.sidebar.radio("Navigation", [
        "1. Single Analysis (即時分析)", 
        "2. Market Scanner (批次掃描)", 
        "3. Historical Backtest (歷史回測)",
        "4. Macro Dashboard (巨集儀表板)"
    ])
    
    if "Single" in mode: render_analysis()
    elif "Scanner" in mode: render_scanner()
    elif "Backtest" in mode: render_backtest()
    elif "Macro" in mode: render_macro()

if __name__ == "__main__":
    main()
