import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

# 過濾警告
warnings.filterwarnings('ignore')

# 設定網頁配置
st.set_page_config(
    page_title="MARCS V57 GAMMA",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 0. 核心工具函數 (智能格式化) [NEW]
# =============================================================================
def smart_format(value, is_currency=True):
    """
    智能精度適配：解決小幣種顯示為 0.00 的問題
    """
    if value is None or pd.isna(value) or value == 0:
        return "$0.00" if is_currency else "0.00"
        
    val = float(value)
    abs_val = abs(val)
    prefix = "$" if is_currency else ""
    
    if abs_val < 0.000001:  # 極小數值 (如 SHIB)
        return f"{prefix}{val:.8f}".rstrip('0')
    elif abs_val < 0.01:    # 微小數值
        return f"{prefix}{val:.6f}"
    elif abs_val < 1:       # 小數值
        return f"{prefix}{val:.4f}"
    else:                   # 正常數值
        return f"{prefix}{val:,.2f}"

# =============================================================================
# 1. CSS 視覺魔法
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* 動態星空背景 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 3px);
        background-size: 550px 550px, 350px 350px, 250px 250px;
        background-position: 0 0, 40px 60px, 130px 270px;
        animation: stars 120s linear infinite;
        z-index: -1;
        opacity: 0.8;
    }
    @keyframes stars {
        from {transform: translateY(0);}
        to {transform: translateY(-1000px);}
    }

    /* 科技感卡片 */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    .metric-label { color: #8b949e; font-size: 14px; letter-spacing: 1px; font-family: 'Roboto Mono'; }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: 700; text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); }
    .metric-sub { font-size: 12px; margin-top: 8px; font-family: 'Roboto Mono'; }
    
    [data-testid="stSidebar"] { background-color: rgba(13, 17, 23, 0.9); border-right: 1px solid rgba(48, 54, 61, 0.5); }
    div.stButton > button { background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%); color: black; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# 兼容性處理
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        return np.mean(np.abs(np.sort(u_values) - np.sort(v_values)))

# =============================================================================
# 2. 資產與引擎定義
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
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
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

class Micro_Structure_Engine:
    @staticmethod
    def attach_indicators(df):
        if df.empty: return df
        c, h, l = df['Close'], df['High'], df['Low']
        
        df['EMA20'] = c.ewm(span=20).mean()
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        atr10 = tr.rolling(10).mean()
        df['K_Upper'] = df['EMA20'] + 2.0 * atr10
        df['K_Lower'] = df['EMA20'] - 2.0 * atr10
        
        plus_dm = (h - h.shift()).clip(lower=0)
        minus_dm = (l.shift() - l).clip(lower=0)
        tr_smooth = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df['ADX'] = dx.rolling(14).mean().fillna(0)
        return df

    @staticmethod
    def get_signals(df_row):
        score = 50
        signals = []
        c = df_row['Close']
        is_trending = df_row['ADX'] > 20
        
        if is_trending:
            if c > df_row['K_Upper']: 
                score += 15; signals.append("Keltner Breakout")
            elif c < df_row['K_Lower']: 
                score -= 15; signals.append("Keltner Breakdown")
            if c > df_row['EMA20']: score += 10
        else:
            signals.append("Low Trend")
        return score, signals

class Antifragile_Position_Sizing:
    @staticmethod
    def calculate_size(account_balance, current_price, stop_loss_price, chaos_level, vol_cap=0.5):
        risk_per_trade = account_balance * 0.02 
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        taleb_multiplier = 1.0
        if chaos_level > 1.2: taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
            
        final_size = int(base_size * taleb_multiplier)
        suggested_capital = final_size * current_price
        
        return final_size, {
            "risk_money": risk_per_trade,
            "final_capital": suggested_capital
        }

class Risk_Entropy_Engine:
    @staticmethod
    def run_monte_carlo_historical(trades_df, initial_capital, simulations=1000):
        if trades_df.empty or len(trades_df) < 5: return None
        if 'Return_Pct' not in trades_df.columns:
            returns = trades_df.sort_values('Date')['Price'].pct_change().dropna().values
        else:
            returns = trades_df['Return_Pct'].values

        results = []
        for _ in range(simulations):
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            equity_curve = initial_capital * np.cumprod(1 + simulated_returns)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            results.append({'final_equity': equity_curve[-1], 'max_dd': np.max(drawdown)})
        return pd.DataFrame(results)

    @staticmethod
    def run_monte_carlo_theoretical(n_simulations, n_trades, win_rate, risk_reward, risk_per_trade, start_capital):
        results_final_equity = []
        max_drawdowns = []
        ruin_count = 0
        all_curves = []
        
        for i in range(n_simulations):
            outcomes = np.random.choice([0, 1], size=n_trades, p=[1-win_rate, win_rate])
            risk_amt = start_capital * risk_per_trade
            pnl_seq = np.where(outcomes == 1, risk_amt * risk_reward, -risk_amt)
            equity_curve = np.cumsum(pnl_seq) + start_capital
            equity_curve = np.insert(equity_curve, 0, start_capital)
            
            results_final_equity.append(equity_curve[-1])
            peak = np.maximum.accumulate(equity_curve)
            max_drawdowns.append(np.max((peak - equity_curve) / peak))
            if np.min(equity_curve) < start_capital * 0.5: ruin_count += 1
            if i < 50: all_curves.append(equity_curve)
                
        return {
            "final_equities": results_final_equity, "max_drawdowns": max_drawdowns,
            "ruin_count": ruin_count, "curves": all_curves
        }

class MARCS_Backtester:
    def __init__(self, ticker, initial_capital):
        self.ticker = ticker; self.initial_capital = initial_capital
        self.df = pd.DataFrame()

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if self.df.empty: return False
            if isinstance(self.df.columns, pd.MultiIndex): self.df.columns = self.df.columns.get_level_values(0)
            return True
        except: return False

    def run(self):
        self.df = Micro_Structure_Engine.attach_indicators(self.df)
        cash = self.initial_capital; position = 0; stop_loss = 0
        trades = []; equity = []
        entry_price = 0

        for i in range(60, len(self.df)):
            curr_date = self.df.index[i]
            row = self.df.iloc[i]
            curr_price = row['Close']
            micro_score, signals = Micro_Structure_Engine.get_signals(row)
            
            if position > 0:
                if curr_price < stop_loss:
                    cash += position * curr_price
                    ret_pct = (curr_price - entry_price) / entry_price
                    trades.append({'Date': curr_date, 'Type': 'SELL', 'Price': curr_price, 'Return_Pct': ret_pct})
                    position = 0
                else:
                    new_sl = curr_price - 2.5 * row['ATR']
                    if new_sl > stop_loss: stop_loss = new_sl
            
            if position == 0 and micro_score >= 65 and "Low Trend" not in str(signals):
                sl_price = curr_price - 2.5 * row['ATR']
                size, _ = Antifragile_Position_Sizing.calculate_size(cash, curr_price, sl_price, 0.5)
                cost = size * curr_price
                if size > 0 and cost <= cash:
                    cash -= cost; position = size; stop_loss = sl_price
                    entry_price = curr_price
                    trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price})

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
        return pd.DataFrame(equity), pd.DataFrame(trades)

# =============================================================================
# 3. 主程序
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ SYSTEM CORE")
    mode = st.sidebar.radio("MODE SELECT", ["LIVE MARKET MONITOR", "SIMULATION LAB"], index=0)
    st.sidebar.markdown("---")
    
    if mode == "LIVE MARKET MONITOR":
        ticker_input = st.sidebar.text_input("TARGET", value="BTC-USD")
        capital = st.sidebar.number_input("CAPITAL", value=1000000, step=100000)
        st.sidebar.info("GAMMA KERNEL: ACTIVE\nPRECISION: ADAPTIVE")
        
        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>🛡️ MARCS V57 GAMMA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e;'>QUANTUM MACRO INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
        
        if st.sidebar.button("🚀 INITIATE SCAN", type="primary"):
            # 1. Macro Dashboard
            st.markdown("### 📡 MACRO METRICS")
            macro_indices = Global_Index_List.get_macro_indices()
            cols = st.columns(4)
            for idx, (sym, info) in enumerate(macro_indices.items()):
                res = Macro_Engine.analyze(sym, info['name'])
                if res:
                    col = cols[idx % 4]
                    color = "#f85149" if res['trend'] == 'Overbought' else ("#3fb950" if res['trend'] == 'Oversold' else "#8b949e")
                    with col:
                        # 這裡使用 smart_format
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 2px solid {color}">
                            <div class="metric-label">{res['name']}</div>
                            <div class="metric-value">{smart_format(res['price'])}</div>
                            <div class="metric-sub" style="color:{color}">{res['trend']}</div>
                        </div>""", unsafe_allow_html=True)

            # 2. Target Analysis
            st.markdown(f"### 🔭 TARGET ANALYSIS: {ticker_input}")
            bt = MARCS_Backtester(ticker_input, capital)
            with st.spinner("Decoding Market Structure..."):
                if bt.fetch_data():
                    df_equity, df_trades = bt.run()
                    last_row = bt.df.iloc[-1]
                    score, signals = Micro_Structure_Engine.get_signals(last_row)
                    
                    # 計算動態止損與 ATR
                    curr_price = last_row['Close']
                    atr_val = last_row['ATR']
                    sl_val = curr_price - (2.5 * atr_val)
                    tp_val = curr_price + (2.5 * atr_val * 2) # 假設 1:2
                    
                    # 第一排：核心信號
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("MICRO SCORE", f"{score}", delta="Bullish" if score>60 else "Bearish")
                    with c2: st.metric("ADX STRENGTH", f"{last_row['ADX']:.1f}", delta="Trending" if last_row['ADX']>20 else "Choppy")
                    
                    ret = 0
                    if not df_equity.empty:
                        ret = (df_equity['Equity'].iloc[-1] - df_equity['Equity'].iloc[0]) / df_equity['Equity'].iloc[0] * 100
                    with c3: st.metric("2Y RETURN", f"{ret:.1f}%", f"{len(df_trades)} Trades")
                    
                    mc_dd = 0.0
                    if not df_trades.empty:
                        sell_trades = df_trades[df_trades['Type']=='SELL']
                        mc_res = Risk_Entropy_Engine.run_monte_carlo_historical(sell_trades, capital, simulations=100)
                        if mc_res is not None: mc_dd = mc_res['max_dd'].quantile(0.95) * 100
                    with c4: st.metric("VAR (95%)", f"-{mc_dd:.1f}%", "Monte Carlo Est.")

                    # 第二排：價格與風控 (修復顯示問題)
                    st.markdown("#### 🛡️ RISK PARAMETERS")
                    r1, r2, r3, r4 = st.columns(4)
                    with r1: 
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">CURRENT PRICE</div>
                        <div class="metric-value">{smart_format(curr_price)}</div></div>""", unsafe_allow_html=True)
                    with r2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">ATR (VOLATILITY)</div>
                        <div class="metric-value">{smart_format(atr_val, is_currency=False)}</div></div>""", unsafe_allow_html=True)
                    with r3:
                        st.markdown(f"""<div class="metric-card" style="border-bottom: 2px solid #f85149"><div class="metric-label">STOP LOSS</div>
                        <div class="metric-value" style="color:#f85149">{smart_format(sl_val)}</div></div>""", unsafe_allow_html=True)
                    with r4:
                        st.markdown(f"""<div class="metric-card" style="border-bottom: 2px solid #3fb950"><div class="metric-label">TAKE PROFIT</div>
                        <div class="metric-value" style="color:#3fb950">{smart_format(tp_val)}</div></div>""", unsafe_allow_html=True)

                    # Visuals
                    tab1, tab2 = st.tabs(["CHART", "EQUITY"])
                    with tab1:
                        fig1, ax1 = plt.subplots(figsize=(12, 5))
                        p_df = bt.df.tail(150)
                        ax1.plot(p_df.index, p_df['Close'], color='#e6edf3', lw=1.5)
                        ax1.plot(p_df.index, p_df['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                        ax1.plot(p_df.index, p_df['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                        ax1.fill_between(p_df.index, p_df['K_Upper'], p_df['K_Lower'], color='#00f2ff', alpha=0.05)
                        
                        # [FIX] Y軸動態精度設定
                        if curr_price < 1:
                            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
                        else:
                            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                            
                        if not df_trades.empty:
                            bs = df_trades[(df_trades['Type']=='BUY') & (df_trades['Date']>=p_df.index[0])]
                            ss = df_trades[(df_trades['Type']=='SELL') & (df_trades['Date']>=p_df.index[0])]
                            ax1.scatter(bs['Date'], bs['Price'], marker='^', color='#3fb950', s=100, zorder=5)
                            ax1.scatter(ss['Date'], ss['Price'], marker='v', color='#f85149', s=100, zorder=5)
                        
                        ax1.set_facecolor('#0d1117'); fig1.patch.set_facecolor('#0d1117')
                        ax1.tick_params(colors='#8b949e'); ax1.grid(True, color='#30363d', alpha=0.3)
                        st.pyplot(fig1)

                    with tab2:
                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        if not df_equity.empty:
                            ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                        ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
                        ax2.tick_params(colors='#8b949e'); ax2.grid(True, color='#30363d', alpha=0.3)
                        st.pyplot(fig2)
                else:
                    st.error("Data Unavailable")

    elif mode == "SIMULATION LAB":
        st.markdown("<h1 style='text-align: center; color: #f85149;'>🧪 STRESS TEST LAB</h1>", unsafe_allow_html=True)
        # (保留之前的實驗室代碼，這裡簡化顯示以節省空間，功能完全相同)
        with st.expander("⚙️ LAB PARAMETERS", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                lab_win_rate = st.slider("Win Rate (%)", 10, 90, 45) / 100
                lab_n_trades = st.slider("Trades per Run", 100, 1000, 500)
            with c2:
                lab_rr = st.slider("Reward/Risk Ratio", 0.5, 5.0, 2.0, 0.1)
                lab_sims = st.slider("Simulations", 100, 2000, 1000)
            with c3:
                lab_risk_pct = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
                lab_capital = st.number_input("Start Capital", value=100000)

        if st.button("🧬 RUN SIMULATION", type="primary"):
            with st.spinner(f"Simulating {lab_sims} universes..."):
                res = Risk_Entropy_Engine.run_monte_carlo_theoretical(
                    lab_sims, lab_n_trades, lab_win_rate, lab_rr, lab_risk_pct, lab_capital
                )
                final_eqs = np.array(res['final_equities'])
                max_dds = np.array(res['max_drawdowns'])
                ruin_prob = (res['ruin_count'] / lab_sims) * 100
                avg_final = np.mean(final_eqs)
                p95_dd = np.percentile(max_dds, 95) * 100
                
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("SURVIVAL PROB", f"{100-ruin_prob:.1f}%")
                with m2: st.metric("AVG FINAL EQUITY", f"${avg_final:,.0f}")
                with m3: st.metric("P95 DRAWDOWN", f"-{p95_dd:.1f}%")

                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    fig_lab1, ax_lab1 = plt.subplots(figsize=(6, 4))
                    for curve in res['curves']: ax_lab1.plot(curve, color='#00f2ff', alpha=0.1, lw=1)
                    ax_lab1.set_facecolor('#0d1117'); fig_lab1.patch.set_facecolor('#0d1117')
                    ax_lab1.tick_params(colors='#8b949e'); ax_lab1.grid(True, color='#30363d', alpha=0.3)
                    st.pyplot(fig_lab1)
                with c_chart2:
                    fig_lab2, ax_lab2 = plt.subplots(figsize=(6, 4))
                    ax_lab2.hist(max_dds * 100, bins=40, color='#f85149', alpha=0.7)
                    ax_lab2.set_facecolor('#0d1117'); fig_lab2.patch.set_facecolor('#0d1117')
                    ax_lab2.tick_params(colors='#8b949e'); ax_lab2.grid(True, color='#30363d', alpha=0.3)
                    st.pyplot(fig_lab2)

if __name__ == "__main__":
    main()
