import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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
# 0. CSS 視覺魔法 (保留原版星空 + 科技感)
# =============================================================================
st.markdown("""
<style>
    /* 1. 全局字體與背景設置 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Rajdhani', sans-serif;
    }

    /* 2. 動態星空背景 */
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

    /* 3. 科技感毛玻璃卡片 */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.4);
        border-color: rgba(88, 166, 255, 0.8);
    }

    /* 4. 文字霓虹特效 */
    .metric-label {
        color: #8b949e; 
        font-size: 14px; 
        letter-spacing: 1px;
        text-transform: uppercase;
        font-family: 'Roboto Mono', monospace;
    }
    .metric-value {
        color: #ffffff; 
        font-size: 28px; 
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    .metric-sub {
        font-size: 12px; 
        margin-top: 8px;
        font-family: 'Roboto Mono', monospace;
    }

    /* 5. UI 元件優化 */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 17, 23, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        transition: all 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.6);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 兼容性處理
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        u_values = np.sort(u_values)
        v_values = np.sort(v_values)
        return np.mean(np.abs(u_values - v_values))

# =============================================================================
# 1. 資產定義 (Global Assets)
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

    @staticmethod
    def get_tradable_indices():
        return {
            "^TWII": {"name": "台股加權", "vol_cap": 0.5},
            "^NDX": {"name": "那斯達克", "vol_cap": 0.6},
            "BTC-USD": {"name": "比特幣", "vol_cap": 1.0},
            "GC=F": {"name": "黃金", "vol_cap": 0.4},
            "NVDA": {"name": "輝達", "vol_cap": 0.8},
            "TSLA": {"name": "特斯拉", "vol_cap": 0.9}
        }

# =============================================================================
# 2. 宏觀引擎 (Macro Engine)
# =============================================================================
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
            # Wasserstein Chaos Metric
            curr_w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            hist_std = returns.rolling(40).std().mean() * 0.1
            chaos = curr_w2 / (hist_std + 1e-9)
            
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {"ticker": ticker, "name": name, "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend}
        except: return None

# =============================================================================
# 3. 微觀結構引擎 (Micro Structure Engine)
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def attach_indicators(df):
        if df.empty: return df
        c, h, l = df['Close'], df['High'], df['Low']
        
        # 基礎指標
        df['EMA20'] = c.ewm(span=20).mean()
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        atr10 = tr.rolling(10).mean()
        df['K_Upper'] = df['EMA20'] + 2.0 * atr10
        df['K_Lower'] = df['EMA20'] - 2.0 * atr10
        
        # ADX 計算 (向量化)
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
        k_up = df_row['K_Upper']
        k_low = df_row['K_Lower']
        ma20 = df_row['EMA20']
        adx = df_row['ADX']
        
        is_trending = adx > 20
        
        if is_trending:
            if c > k_up: 
                score += 15
                signals.append("Keltner Breakout")
            elif c < k_low: 
                score -= 15
                signals.append("Keltner Breakdown")
            
            if c > ma20: score += 10
        else:
            signals.append("Low Trend (ADX < 20)")
            
        return score, signals

# =============================================================================
# 4. 反脆弱倉位管理 (Antifragile Sizing)
# =============================================================================
class Antifragile_Position_Sizing:
    @staticmethod
    def calculate_size(account_balance, current_price, stop_loss_price, chaos_level, vol_cap):
        risk_per_trade = account_balance * 0.02 
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        taleb_multiplier = 1.0
        if chaos_level > 1.2: taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
            
        vol_adj = 0.5 if vol_cap > 0.8 else 1.0
        final_size = int(base_size * taleb_multiplier * vol_adj)
        suggested_capital = final_size * current_price
        
        return final_size, {
            "risk_money": int(risk_per_trade), "taleb_factor": round(taleb_multiplier, 2),
            "final_capital": int(suggested_capital)
        }

# =============================================================================
# 5. 蒙地卡羅風控引擎 (Risk Entropy Engine) - 增強版
# =============================================================================
class Risk_Entropy_Engine:
    @staticmethod
    def run_monte_carlo_historical(trades_df, initial_capital, simulations=1000):
        """基於歷史交易進行 Bootstrap 模擬"""
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
            max_dd = np.max(drawdown)
            results.append({'final_equity': equity_curve[-1], 'max_dd': max_dd})
            
        return pd.DataFrame(results)

    @staticmethod
    def run_monte_carlo_theoretical(n_simulations, n_trades, win_rate, risk_reward, risk_per_trade, start_capital):
        """
        [NEW] 基於參數的理論壓力測試 (Snippet 1 的邏輯)
        """
        results_final_equity = []
        max_drawdowns = []
        ruin_count = 0
        all_equity_curves = []
        
        # 為了視覺化，只存前 50 條曲線
        save_curves_limit = 50
        
        for i in range(n_simulations):
            # 0 = Loss, 1 = Win
            outcomes = np.random.choice([0, 1], size=n_trades, p=[1-win_rate, win_rate])
            
            # 這裡為了展示風險，使用「固定金額風險」 (更符合多數人習慣)
            # 若要測試複利爆炸，可改為 risk_amt = capital * risk_per_trade
            risk_amt_fixed = start_capital * risk_per_trade
            
            # 使用 numpy 向量化加速計算
            # 贏 = risk_amt * RR, 輸 = -risk_amt
            pnl_seq = np.where(outcomes == 1, risk_amt_fixed * risk_reward, -risk_amt_fixed)
            
            # 計算資金曲線
            equity_curve = np.cumsum(pnl_seq) + start_capital
            equity_curve = np.insert(equity_curve, 0, start_capital) # 加入起點
            
            # 記錄最終結果
            final_eq = equity_curve[-1]
            results_final_equity.append(final_eq)
            
            # Drawdown 計算
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_dd = np.max(drawdown)
            max_drawdowns.append(max_dd)
            
            # 破產判定 (-50% 視為技術性破產)
            if np.min(equity_curve) < start_capital * 0.5:
                ruin_count += 1
                
            if i < save_curves_limit:
                all_equity_curves.append(equity_curve)
                
        return {
            "final_equities": results_final_equity,
            "max_drawdowns": max_drawdowns,
            "ruin_count": ruin_count,
            "curves": all_equity_curves,
            "n_sims": n_simulations
        }

# =============================================================================
# 6. 回測引擎 (Backtester)
# =============================================================================
class MARCS_Backtester:
    def __init__(self, ticker, initial_capital):
        self.ticker = ticker; self.initial_capital = initial_capital
        self.df = pd.DataFrame()
        self.vol_cap = Global_Index_List.get_tradable_indices().get(ticker, {}).get('vol_cap', 0.5)

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
            chaos_sim = 0.5
            
            if position > 0:
                if curr_price < stop_loss:
                    cash += position * curr_price
                    ret_pct = (curr_price - entry_price) / entry_price
                    trades.append({'Date': curr_date, 'Type': 'SELL', 'Price': curr_price, 'Reason': 'SL', 'Return_Pct': ret_pct})
                    position = 0
                else:
                    new_sl = curr_price - 2.5 * row['ATR']
                    if new_sl > stop_loss: stop_loss = new_sl
            
            if position == 0:
                if micro_score >= 65 and "Low Trend" not in str(signals):
                    sl_price = curr_price - 2.5 * row['ATR']
                    size, _ = Antifragile_Position_Sizing.calculate_size(cash, curr_price, sl_price, chaos_sim, self.vol_cap)
                    cost = size * curr_price
                    if size > 0 and cost <= cash:
                        cash -= cost; position = size; stop_loss = sl_price
                        entry_price = curr_price
                        trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price})

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
            
        return pd.DataFrame(equity), pd.DataFrame(trades)

# =============================================================================
# 7. 主介面 (V57 Starfield Edition)
# =============================================================================
def main():
    # Sidebar
    st.sidebar.markdown("## ⚙️ SYSTEM CORE")
    mode = st.sidebar.radio("MODE SELECT", ["LIVE MARKET MONITOR", "SIMULATION LAB"], index=0)
    
    st.sidebar.markdown("---")
    
    if mode == "LIVE MARKET MONITOR":
        ticker = st.sidebar.text_input("TARGET", value="BTC-USD")
        capital = st.sidebar.number_input("CAPITAL", value=1000000, step=100000)
        st.sidebar.info("GAMMA KERNEL: ACTIVE\nADX FILTER: ON")
        
        # 標題區
        st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 0 0 10px #00f2ff;'>🛡️ MARCS V57 GAMMA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e; letter-spacing: 2px;'>QUANTUM MACRO INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
        
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
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 2px solid {color}">
                            <div class="metric-label">{res['name']}</div>
                            <div class="metric-value">{res['price']:.2f}</div>
                            <div class="metric-sub" style="color:{color}">{res['trend']}</div>
                        </div>""", unsafe_allow_html=True)

            # 2. Target Analysis
            st.markdown(f"### 🔭 TARGET ANALYSIS: {ticker}")
            bt = MARCS_Backtester(ticker, capital)
            with st.spinner("Decoding Market Structure..."):
                if bt.fetch_data():
                    df_equity, df_trades = bt.run()
                    last_row = bt.df.iloc[-1]
                    score, signals = Micro_Structure_Engine.get_signals(last_row)
                    
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("MICRO SCORE", f"{score}", delta="Bullish" if score>60 else "Bearish")
                    with c2: st.metric("ADX STRENGTH", f"{last_row['ADX']:.1f}", delta="Trending" if last_row['ADX']>20 else "Choppy")
                    
                    ret = 0
                    if not df_equity.empty:
                        ret = (df_equity['Equity'].iloc[-1] - df_equity['Equity'].iloc[0]) / df_equity['Equity'].iloc[0] * 100
                    with c3: st.metric("2Y RETURN", f"{ret:.1f}%", f"{len(df_trades)} Trades")

                    # Historical MC
                    mc_dd = 0
                    if not df_trades.empty:
                        sell_trades = df_trades[df_trades['Type']=='SELL']
                        mc_res = Risk_Entropy_Engine.run_monte_carlo_historical(sell_trades, capital, simulations=100)
                        if mc_res is not None: mc_dd = mc_res['max_dd'].quantile(0.95) * 100
                    with c4: st.metric("VAR (95%)", f"-{mc_dd:.1f}%", "Monte Carlo Est.")

                    # Visuals
                    tab1, tab2 = st.tabs(["CHART", "EQUITY"])
                    with tab1:
                        fig1, ax1 = plt.subplots(figsize=(12, 5))
                        p_df = bt.df.tail(150)
                        ax1.plot(p_df.index, p_df['Close'], color='#e6edf3', lw=1.5)
                        ax1.plot(p_df.index, p_df['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                        ax1.plot(p_df.index, p_df['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                        ax1.fill_between(p_df.index, p_df['K_Upper'], p_df['K_Lower'], color='#00f2ff', alpha=0.05)
                        
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

    # =========================================================================
    # 實驗室模式 (The Merged Feature)
    # =========================================================================
    elif mode == "SIMULATION LAB":
        st.markdown("<h1 style='text-align: center; color: #f85149; text-shadow: 0 0 10px #f85149;'>🧪 STRESS TEST LAB</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e;'>MONTE CARLO THEORETICAL VERIFICATION</p>", unsafe_allow_html=True)
        
        # 參數控制台
        with st.expander("⚙️ LAB PARAMETERS", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                lab_win_rate = st.slider("Win Rate (%)", 10, 90, 45) / 100
                lab_n_trades = st.slider("Trades per Run", 100, 1000, 500)
            with c2:
                lab_rr = st.slider("Reward/Risk Ratio", 0.5, 5.0, 2.0, 0.1)
                lab_sims = st.slider("Simulations (Universes)", 100, 2000, 1000)
            with c3:
                lab_risk_pct = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
                lab_capital = st.number_input("Start Capital", value=100000)

        if st.button("🧬 RUN SIMULATION", type="primary"):
            with st.spinner(f"Simulating {lab_sims} universes..."):
                res = Risk_Entropy_Engine.run_monte_carlo_theoretical(
                    lab_sims, lab_n_trades, lab_win_rate, lab_rr, lab_risk_pct, lab_capital
                )
                
                # 分析數據
                final_eqs = np.array(res['final_equities'])
                max_dds = np.array(res['max_drawdowns'])
                ruin_prob = (res['ruin_count'] / lab_sims) * 100
                avg_final = np.mean(final_eqs)
                p95_dd = np.percentile(max_dds, 95) * 100
                p99_dd = np.percentile(max_dds, 99) * 100
                
                # 顯示核心結果
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("SURVIVAL PROB", f"{100-ruin_prob:.1f}%", f"Ruin: {ruin_prob:.1f}%")
                with m2: st.metric("AVG FINAL EQUITY", f"${avg_final:,.0f}", f"Expectancy")
                with m3: st.metric("P95 DRAWDOWN", f"-{p95_dd:.1f}%", "Sleep Well Limit")
                with m4: st.metric("P99 DRAWDOWN", f"-{p99_dd:.1f}%", "Black Swan")

                if p95_dd > 25:
                    st.error(f"⚠️ CRITICAL RISK: P95 Drawdown is {p95_dd:.1f}%. This strategy is psychologically unplayable.")
                else:
                    st.success("✅ SYSTEM STABLE: Risk parameters are within acceptable limits.")

                # 圖表視覺化
                c_chart1, c_chart2 = st.columns(2)
                
                with c_chart1:
                    # 資金曲線雲圖
                    fig_lab1, ax_lab1 = plt.subplots(figsize=(6, 4))
                    for curve in res['curves']:
                        ax_lab1.plot(curve, color='#00f2ff', alpha=0.1, lw=1)
                    # 畫平均線
                    # ax_lab1.plot(np.mean(res['curves'], axis=0), color='white', lw=2, ls='--')
                    
                    ax_lab1.set_title("Monte Carlo Paths (First 50)", color='white')
                    ax_lab1.set_facecolor('#0d1117'); fig_lab1.patch.set_facecolor('#0d1117')
                    ax_lab1.tick_params(colors='#8b949e'); ax_lab1.grid(True, color='#30363d', alpha=0.3)
                    ax_lab1.axhline(y=lab_capital, color='#f85149', linestyle='--', alpha=0.5)
                    st.pyplot(fig_lab1)
                
                with c_chart2:
                    # 回撤分佈圖
                    fig_lab2, ax_lab2 = plt.subplots(figsize=(6, 4))
                    ax_lab2.hist(max_dds * 100, bins=40, color='#f85149', alpha=0.7, edgecolor='#0d1117')
                    ax_lab2.set_title("Max Drawdown Distribution (%)", color='white')
                    ax_lab2.set_facecolor('#0d1117'); fig_lab2.patch.set_facecolor('#0d1117')
                    ax_lab2.tick_params(colors='#8b949e'); ax_lab2.grid(True, color='#30363d', alpha=0.3)
                    ax_lab2.axvline(x=p95_dd, color='white', linestyle='--', label='P95')
                    st.pyplot(fig_lab2)

if __name__ == "__main__":
    main()
