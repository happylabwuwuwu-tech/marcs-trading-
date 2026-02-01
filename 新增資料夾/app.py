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
    page_title="MARCS V55 全景戰情室",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center;}
    .metric-label {color: #8b949e; font-size: 12px; margin-bottom: 5px;}
    .metric-value {color: #ffffff; font-size: 20px; font-weight: bold;}
    .metric-sub {font-size: 11px; margin-top: 5px;}
    .macro-box {padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 5px;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif;}
</style>
""", unsafe_allow_html=True)

# 兼容性處理 (Wasserstein Distance)
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        u_values = np.sort(u_values)
        v_values = np.sort(v_values)
        return np.mean(np.abs(u_values - v_values))

# =============================================================================
# 1. 資產定義 (宏觀 + 交易標的)
# =============================================================================
class Global_Index_List:
    @staticmethod
    def get_macro_indices():
        return {
            "^VIX": {"name": "恐慌指數 (VIX)", "type": "Sentiment"},
            "DX-Y.NYB": {"name": "美元指數 (DXY)", "type": "Currency"},
            "TLT": {"name": "美債20年 (TLT)", "type": "Rates"},
            "JPY=X": {"name": "日圓 (JPY)", "type": "Currency"}
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
# 2. 宏觀引擎 (處理 VIX, DXY 等)
# =============================================================================
class Macro_Engine:
    @staticmethod
    def analyze(ticker, name):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            c = df['Close']
            
            # 1. RSI
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 2. Chaos (Wasserstein)
            returns = np.log(c).diff().dropna()
            if len(returns) < 40: return None
            curr_w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            hist_std = returns.rolling(40).std().mean() * 0.1
            chaos = curr_w2 / (hist_std + 1e-9)
            
            # 3. Trend Status
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {
                "ticker": ticker, "name": name, 
                "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend
            }
        except: return None

    @staticmethod
    def calculate_macro_score(results):
        score = 50.0
        data_map = {r['ticker']: r for r in results if r}
        
        # VIX: 高=恐慌(加分), 低=貪婪(扣分)
        vix = data_map.get('^VIX')
        if vix:
            if vix['trend'] == 'Overbought': score += 15
            elif vix['trend'] == 'Oversold': score -= 15
            
        # DXY: 高=資金緊縮(扣分), 低=寬鬆(加分)
        dxy = data_map.get('DX-Y.NYB')
        if dxy:
            if dxy['trend'] == 'Overbought': score -= 12
            elif dxy['trend'] == 'Oversold': score += 12
            
        # TLT: 低=利率高(扣分)
        tlt = data_map.get('TLT')
        if tlt and tlt['trend'] == 'Oversold': score -= 8
            
        return min(100, max(0, score))

# =============================================================================
# 3. 微觀引擎 & 反脆弱資金管理 (V54 核心)
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def analyze(df):
        if df.empty or len(df) < 60: return 50, [], pd.DataFrame()
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        score = 50; signals = []
        
        # Keltner
        ema20 = c.ewm(span=20).mean()
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr10 = tr.rolling(10).mean()
        k_upper = ema20 + 2.0 * atr10
        k_lower = ema20 - 2.0 * atr10
        
        if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("肯特納突破")
        elif c.iloc[-1] < k_lower.iloc[-1]: score -= 15; signals.append("肯特納跌破")

        # R-Breaker
        if c.iloc[-1] > c.iloc[-2] * 1.015: score += 5; signals.append("強勢紅K")
        
        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV多方")

        indicators = pd.DataFrame({'EMA20': ema20, 'K_Upper': k_upper, 'K_Lower': k_lower}, index=df.index)
        return min(100, max(0, score)), signals, indicators

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
# 4. 回測引擎
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
            h, l, c = self.df['High'], self.df['Low'], self.df['Close']
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            self.df['ATR'] = tr.rolling(14).mean()
            return True
        except: return False

    def run(self):
        cash = self.initial_capital; position = 0; stop_loss = 0
        trades = []; equity = []
        _, _, indicators = Micro_Structure_Engine.analyze(self.df)
        self.df = pd.concat([self.df, indicators], axis=1)

        for i in range(60, len(self.df)):
            curr_date = self.df.index[i]
            curr_price = self.df['Close'].iloc[i]
            curr_atr = self.df['ATR'].iloc[i]
            k_upper = self.df['K_Upper'].iloc[i]
            ma20 = self.df['EMA20'].iloc[i]
            
            micro_score = 50
            if curr_price > k_upper: micro_score += 15
            if curr_price > ma20: micro_score += 10
            
            chaos_sim = 0.5
            if pd.notna(curr_atr):
                chaos_sim = max(0, (curr_atr / self.df['ATR'].iloc[i-20:i].mean() - 1.0) + 0.5)

            if position > 0:
                if curr_price < stop_loss:
                    cash += position * curr_price
                    trades.append({'Date': curr_date, 'Type': 'SELL', 'Price': curr_price, 'Reason': 'SL'})
                    position = 0
                else:
                    new_sl = curr_price - 2.5 * curr_atr
                    if new_sl > stop_loss: stop_loss = new_sl
            
            if position == 0:
                if micro_score >= 65:
                    sl_price = curr_price - 2.5 * curr_atr
                    size, _ = Antifragile_Position_Sizing.calculate_size(cash, curr_price, sl_price, chaos_sim, self.vol_cap)
                    cost = size * curr_price
                    if size > 0 and cost <= cash:
                        cash -= cost; position = size; stop_loss = sl_price
                        trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price, 'Size': size})

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
        return pd.DataFrame(equity), pd.DataFrame(trades)

# =============================================================================
# 5. 主程式 (Streamlit UI)
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 參數設定")
    ticker = st.sidebar.text_input("交易代碼 (Ticker)", value="BTC-USD")
    capital = st.sidebar.number_input("初始本金 (Capital)", value=1000000, step=100000)
    
    st.title("🛡️ MARCS V55 全景戰情室")
    
    if st.sidebar.button("🚀 啟動全域掃描", type="primary"):
        # --- PART 1: 宏觀儀表板 (Macro Dashboard) ---
        st.markdown("### 1. 全球宏觀天候 (The Weather)")
        
        macro_indices = Global_Index_List.get_macro_indices()
        macro_results = []
        
        # 使用 4 列佈局
        cols = st.columns(4)
        
        for idx, (sym, info) in enumerate(macro_indices.items()):
            res = Macro_Engine.analyze(sym, info['name'])
            macro_results.append(res)
            
            if res:
                col = cols[idx % 4]
                # 顏色邏輯
                status_color = "#8b949e"
                if res['trend'] == 'Overbought': status_color = "#f85149" # 紅
                elif res['trend'] == 'Oversold': status_color = "#3fb950" # 綠
                
                chaos_mark = "⚡" if res['chaos'] > 1.2 else ""
                
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid {status_color}">
                        <div class="metric-label">{res['name']}</div>
                        <div class="metric-value">{res['price']:.2f}</div>
                        <div class="metric-sub" style="color:{status_color}">{res['trend']} (RSI: {res['rsi']:.0f})</div>
                        <div class="metric-sub">Chaos: {res['chaos']:.2f} {chaos_mark}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 計算宏觀總分
        mmi_score = Macro_Engine.calculate_macro_score(macro_results)
        mmi_color = "#3fb950" if mmi_score > 60 else ("#f85149" if mmi_score < 40 else "#d2a8ff")
        
        st.markdown(f"""
        <div style="background:#161b22; padding:10px; border-radius:5px; margin: 15px 0; text-align:center;">
            <span style="color:#8b949e">MARCS 宏觀風險偏好指數 (MMI): </span>
            <span style="font-size:24px; font-weight:bold; color:{mmi_color}">{mmi_score:.1f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # --- PART 2: 個股微觀與回測 (Individual Analysis) ---
        st.markdown(f"### 2. 標的深度分析: {ticker} (The Ship)")
        
        bt = MARCS_Backtester(ticker, capital)
        with st.spinner(f"正在分析 {ticker} 微觀結構..."):
            if bt.fetch_data():
                df_equity, df_trades = bt.run()
                score_now, signals_now, indicators = Micro_Structure_Engine.analyze(bt.df)
                
                # 計算即時建議
                last_row = bt.df.iloc[-1]
                curr_price = last_row['Close']
                sl_price = curr_price - 2.5 * last_row['ATR']
                size_now, details_now = Antifragile_Position_Sizing.calculate_size(
                    capital, curr_price, sl_price, 0.8, bt.vol_cap
                )
                
                # 顯示三欄位資訊
                c1, c2, c3 = st.columns(3)
                with c1:
                     st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">微觀評分 (Micro)</div>
                        <div class="metric-value" style="color:{'#3fb950' if score_now>60 else '#f85149'}">{score_now}</div>
                        <div class="metric-sub">{', '.join(signals_now) if signals_now else '盤整'}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Taleb 建議倉位</div>
                        <div class="metric-value">{details_now.get('final_capital', 0)//int(curr_price) if curr_price else 0} 單位</div>
                        <div class="metric-sub">Taleb 係數: {details_now.get('taleb_factor', 1.0)}x</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    ret = 0
                    if not df_equity.empty:
                        ret = (df_equity['Equity'].iloc[-1] - df_equity['Equity'].iloc[0]) / df_equity['Equity'].iloc[0] * 100
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">2年回測報酬</div>
                        <div class="metric-value" style="color:{'#3fb950' if ret>0 else '#f85149'}">{ret:.1f}%</div>
                        <div class="metric-sub">交易次數: {len(df_trades)}</div>
                    </div>""", unsafe_allow_html=True)
                
                # 圖表區
                st.markdown("#### 策略回測圖表")
                tab1, tab2 = st.tabs(["🕯️ Keltner 通道訊號", "📈 資金權益曲線"])
                
                with tab1:
                    fig1, ax1 = plt.subplots(figsize=(12, 5))
                    plot_df = bt.df.tail(150); plot_ind = indicators.tail(150)
                    ax1.plot(plot_df.index, plot_df['Close'], color='white', lw=1, label='Price')
                    ax1.plot(plot_ind.index, plot_ind['K_Upper'], color='#00f2ff', ls='--', alpha=0.5, label='Upper')
                    ax1.plot(plot_ind.index, plot_ind['K_Lower'], color='#00f2ff', ls='--', alpha=0.5, label='Lower')
                    ax1.fill_between(plot_ind.index, plot_ind['K_Upper'], plot_ind['K_Lower'], color='#00f2ff', alpha=0.05)
                    
                    if not df_trades.empty:
                        buys = df_trades[df_trades['Type']=='BUY']
                        sells = df_trades[df_trades['Type']=='SELL']
                        buys = buys[buys['Date'] >= plot_df.index[0]]
                        sells = sells[sells['Date'] >= plot_df.index[0]]
                        ax1.scatter(buys['Date'], buys['Price'], marker='^', color='#3fb950', s=80, zorder=5)
                        ax1.scatter(sells['Date'], sells['Price'], marker='v', color='#f85149', s=80, zorder=5)
                        
                    ax1.set_facecolor('#0e1117'); fig1.patch.set_facecolor('#0e1117')
                    ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.1)
                    st.pyplot(fig1)

                with tab2:
                    if not df_equity.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                        ax2.set_facecolor('#0e1117'); fig2.patch.set_facecolor('#0e1117')
                        ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.1)
                        st.pyplot(fig2)
                
                with st.expander("查看詳細交易紀錄"):
                    st.dataframe(df_trades, use_container_width=True)

            else:
                st.error("❌ 無法獲取標的數據")

if __name__ == "__main__":
    main()
