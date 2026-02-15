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
# 3. 微觀結構引擎 (Micro Structure Engine) - REFACTORED
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def attach_indicators(df):
        """
        計算所有技術指標，包括 ADX 過濾器所需的數據
        """
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
        # 避免除以零
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df['ADX'] = dx.rolling(14).mean().fillna(0)
        
        return df

    @staticmethod
    def get_signals(df_row):
        """
        單一真相來源：統一回測與實盤的信號邏輯
        """
        score = 50
        signals = []
        
        c = df_row['Close']
        k_up = df_row['K_Upper']
        k_low = df_row['K_Lower']
        ma20 = df_row['EMA20']
        adx = df_row['ADX']
        
        # 核心過濾：ADX > 20 才視為有趨勢
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
        # Taleb Multiplier: Chaos 越高，倉位越小 (Sigmoid)
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
# 5. 蒙地卡羅風控引擎 (Risk Entropy Engine) - NEW
# =============================================================================
class Risk_Entropy_Engine:
    @staticmethod
    def run_monte_carlo(trades_df, initial_capital, simulations=1000):
        """
        模擬 1000 種可能的未來，計算最大回撤分佈
        """
        if trades_df.empty or len(trades_df) < 5:
            return None

        # 計算每筆交易的收益率 (PnL %)
        if 'Return_Pct' not in trades_df.columns:
            # 兼容性處理，如果回測沒有存 Return_Pct，嘗試用 Price 推算 (較不準確)
            returns = trades_df.sort_values('Date')['Price'].pct_change().dropna().values
        else:
            returns = trades_df['Return_Pct'].values

        results = []
        for _ in range(simulations):
            # 隨機重排交易結果 (Bootstrap)
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            equity_curve = initial_capital * np.cumprod(1 + simulated_returns)
            
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_dd = np.max(drawdown)
            
            results.append({
                'final_equity': equity_curve[-1],
                'max_dd': max_dd
            })
            
        return pd.DataFrame(results)

# =============================================================================
# 6. 回測引擎 (Backtester) - REFACTORED
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
        # 1. 計算全量指標
        self.df = Micro_Structure_Engine.attach_indicators(self.df)
        
        cash = self.initial_capital; position = 0; stop_loss = 0
        trades = []; equity = []
        entry_price = 0

        # 開始回測循環
        for i in range(60, len(self.df)):
            curr_date = self.df.index[i]
            row = self.df.iloc[i]
            curr_price = row['Close']
            
            # 2. 調用統一邏輯引擎
            micro_score, signals = Micro_Structure_Engine.get_signals(row)
            
            chaos_sim = 0.5 # 簡化模擬
            
            if position > 0:
                # 止損觸發
                if curr_price < stop_loss:
                    cash += position * curr_price
                    # 記錄收益率以便蒙地卡羅分析
                    ret_pct = (curr_price - entry_price) / entry_price
                    trades.append({
                        'Date': curr_date, 'Type': 'SELL', 
                        'Price': curr_price, 'Reason': 'SL', 
                        'Return_Pct': ret_pct
                    })
                    position = 0
                else:
                    # Trailing Stop: 隨著價格上漲提高止損
                    new_sl = curr_price - 2.5 * row['ATR']
                    if new_sl > stop_loss: stop_loss = new_sl
            
            if position == 0:
                # 開倉條件: Score 高 + ADX 有趨勢 (非 Low Trend)
                if micro_score >= 65 and "Low Trend" not in str(signals):
                    sl_price = curr_price - 2.5 * row['ATR']
                    size, _ = Antifragile_Position_Sizing.calculate_size(
                        cash, curr_price, sl_price, chaos_sim, self.vol_cap
                    )
                    cost = size * curr_price
                    if size > 0 and cost <= cash:
                        cash -= cost; position = size; stop_loss = sl_price
                        entry_price = curr_price
                        trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price})

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
            
        return pd.DataFrame(equity), pd.DataFrame(trades)

# =============================================================================
# 7. 主介面 (V57 Starfield Edition) - INTEGRATED
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 系統控制台")
    ticker = st.sidebar.text_input("TARGET", value="BTC-USD")
    capital = st.sidebar.number_input("CAPITAL", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    st.sidebar.info("GAMMA KERNEL: ACTIVE\nADX FILTER: ON\nMONTE CARLO: READY")

    # 標題區
    st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 0 0 10px #00f2ff;'>🛡️ MARCS V57 GAMMA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; letter-spacing: 2px;'>QUANTUM MACRO INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
    
    if st.sidebar.button("🚀 INITIATE SCAN", type="primary"):
        # 1. 宏觀儀表板
        st.markdown("### 📡 MACRO METRICS")
        macro_indices = Global_Index_List.get_macro_indices()
        cols = st.columns(4)
        
        for idx, (sym, info) in enumerate(macro_indices.items()):
            res = Macro_Engine.analyze(sym, info['name'])
            if res:
                col = cols[idx % 4]
                color = "#f85149" if res['trend'] == 'Overbought' else ("#3fb950" if res['trend'] == 'Oversold' else "#8b949e")
                chaos_mk = "⚡" if res['chaos'] > 1.2 else ""
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 2px solid {color}">
                        <div class="metric-label">{res['name']}</div>
                        <div class="metric-value">{res['price']:.2f}</div>
                        <div class="metric-sub" style="color:{color}">{res['trend']}</div>
                        <div class="metric-sub">Chaos: {res['chaos']:.2f} {chaos_mk}</div>
                    </div>""", unsafe_allow_html=True)

        # 2. 個股分析
        st.markdown(f"### 🔭 TARGET ANALYSIS: {ticker}")
        bt = MARCS_Backtester(ticker, capital)
        
        with st.spinner("Decodin Market Structure (ADX Filter Applied)..."):
            if bt.fetch_data():
                df_equity, df_trades = bt.run()
                
                # 取得最新一根 Bar 的狀態
                last_row = bt.df.iloc[-1]
                score, signals = Micro_Structure_Engine.get_signals(last_row)
                
                # 顯示核心指標
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">MICRO SCORE</div>
                        <div class="metric-value" style="color:{'#3fb950' if score>60 else '#f85149'}">{score}</div>
                        <div class="metric-sub">{', '.join(signals) if signals else 'NEUTRAL'}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">ADX STRENGTH</div>
                        <div class="metric-value" style="color:{'#3fb950' if last_row['ADX']>20 else '#8b949e'}">{last_row['ADX']:.1f}</div>
                        <div class="metric-sub">{'TRENDING' if last_row['ADX']>20 else 'CHOPPY'}</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    ret = 0
                    if not df_equity.empty:
                        ret = (df_equity['Equity'].iloc[-1] - df_equity['Equity'].iloc[0]) / df_equity['Equity'].iloc[0] * 100
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">2Y RETURN</div>
                        <div class="metric-value" style="color:{'#3fb950' if ret>0 else '#f85149'}">{ret:.1f}%</div>
                        <div class="metric-sub">Trades: {len(df_trades)}</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    # 蒙地卡羅快速預覽
                    mc_dd = 0
                    if not df_trades.empty:
                        sell_trades = df_trades[df_trades['Type']=='SELL']
                        mc_res = Risk_Entropy_Engine.run_monte_carlo(sell_trades, capital, simulations=100)
                        if mc_res is not None:
                            mc_dd = mc_res['max_dd'].quantile(0.95) * 100
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">VAR (95%)</div>
                        <div class="metric-value" style="color:#f85149">-{mc_dd:.1f}%</div>
                        <div class="metric-sub">Monte Carlo Est.</div>
                    </div>""", unsafe_allow_html=True)

                # 圖表區域
                st.markdown("#### 📊 TACTICAL VISUALIZATION")
                tab1, tab2, tab3 = st.tabs(["CHART", "EQUITY", "MONTE CARLO"])
                
                with tab1:
                    fig1, ax1 = plt.subplots(figsize=(12, 5))
                    p_df = bt.df.tail(150)
                    ax1.plot(p_df.index, p_df['Close'], color='#e6edf3', lw=1.5, label='Price')
                    ax1.plot(p_df.index, p_df['K_Upper'], color='#00f2ff', ls='--', alpha=0.5, label='Keltner Up')
                    ax1.plot(p_df.index, p_df['K_Lower'], color='#00f2ff', ls='--', alpha=0.5, label='Keltner Low')
                    ax1.fill_between(p_df.index, p_df['K_Upper'], p_df['K_Lower'], color='#00f2ff', alpha=0.05)
                    
                    if not df_trades.empty:
                        bs = df_trades[df_trades['Type']=='BUY']
                        ss = df_trades[df_trades['Type']=='SELL']
                        bs = bs[bs['Date']>=p_df.index[0]]
                        ss = ss[ss['Date']>=p_df.index[0]]
                        ax1.scatter(bs['Date'], bs['Price'], marker='^', color='#3fb950', s=100, zorder=5, label='Buy')
                        ax1.scatter(ss['Date'], ss['Price'], marker='v', color='#f85149', s=100, zorder=5, label='Sell')
                    
                    ax1.set_facecolor('#0d1117'); fig1.patch.set_facecolor('#0d1117')
                    ax1.tick_params(colors='#8b949e'); ax1.grid(True, color='#30363d', alpha=0.3)
                    ax1.legend(facecolor='#0d1117', labelcolor='#8b949e')
                    st.pyplot(fig1)

                with tab2:
                    if not df_equity.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                        ax2.set_title("Equity Curve", color='white')
                        ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
                        ax2.tick_params(colors='#8b949e'); ax2.grid(True, color='#30363d', alpha=0.3)
                        st.pyplot(fig2)
                
                with tab3:
                    if not df_trades.empty:
                        sell_trades = df_trades[df_trades['Type']=='SELL']
                        mc_results = Risk_Entropy_Engine.run_monte_carlo(sell_trades, capital, simulations=1000)
                        
                        if mc_results is not None:
                            col_m1, col_m2 = st.columns(2)
                            
                            # DD 分佈圖
                            fig3, ax3 = plt.subplots(figsize=(6, 4))
                            ax3.hist(mc_results['max_dd'] * 100, bins=30, color='#f85149', alpha=0.7)
                            ax3.set_title("Max Drawdown Distribution (%)", color='white')
                            ax3.set_facecolor('#0d1117'); fig3.patch.set_facecolor('#0d1117')
                            ax3.tick_params(colors='#8b949e')
                            col_m1.pyplot(fig3)
                            
                            # 最終淨值分佈圖
                            fig4, ax4 = plt.subplots(figsize=(6, 4))
                            ax4.hist(mc_results['final_equity'], bins=30, color='#3fb950', alpha=0.7)
                            ax4.set_title("Final Equity Distribution ($)", color='white')
                            ax4.set_facecolor('#0d1117'); fig4.patch.set_facecolor('#0d1117')
                            ax4.tick_params(colors='#8b949e')
                            col_m2.pyplot(fig4)
                            
                            st.caption("Simulation runs: 1000 | Resampling method: Bootstrap with replacement")

            else:
                st.error("Connection Failed: Data Unavailable or Ticker Invalid")

if __name__ == "__main__":
    main()
