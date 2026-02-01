import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import warnings

# 過濾警告
warnings.filterwarnings('ignore')

# 設定網頁配置
st.set_page_config(
    page_title="MARCS V54 量化戰情室",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 美化
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center;}
    .metric-label {color: #8b949e; font-size: 12px; margin-bottom: 5px;}
    .metric-value {color: #ffffff; font-size: 24px; font-weight: bold;}
    .metric-sub {font-size: 12px; margin-top: 5px;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 資產定義 (含 Vol Cap)
# =============================================================================
class Global_Index_List:
    @staticmethod
    def get_indices():
        return {
            "^TWII": {"name": "台股加權", "vol_cap": 0.5},
            "^NDX": {"name": "那斯達克", "vol_cap": 0.6},
            "BTC-USD": {"name": "比特幣", "vol_cap": 1.0},
            "GC=F": {"name": "黃金", "vol_cap": 0.4},
            "NVDA": {"name": "輝達", "vol_cap": 0.8},
            "TSLA": {"name": "特斯拉", "vol_cap": 0.9}
        }

# =============================================================================
# 2. 反脆弱資金管理 (Taleb + Elder) - 核心風控
# =============================================================================
class Antifragile_Position_Sizing:
    @staticmethod
    def calculate_size(account_balance, current_price, stop_loss_price, chaos_level, vol_cap):
        # 1. Elder 邏輯：單筆風險 2%
        risk_per_trade = account_balance * 0.02 
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        
        # 2. Taleb 邏輯：混沌懲罰
        taleb_multiplier = 1.0
        if chaos_level > 1.2:
            taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
            
        # 3. 波動率上限修正
        vol_adjustment = 1.0
        if vol_cap > 0.8: # 高波動資產強制減倉
            vol_adjustment = 0.5
            
        final_size = int(base_size * taleb_multiplier * vol_adjustment)
        suggested_capital = final_size * current_price
        
        # 創建詳細資訊字典，確保所有鍵值都存在
        details = {
            "risk_money": int(risk_per_trade),
            "taleb_factor": round(taleb_multiplier, 2),
            "elder_size": int(base_size),
            "final_capital": int(suggested_capital)
        }
        
        return final_size, details

# =============================================================================
# 3. 微觀引擎 V52 (含書中 R-Breaker + Keltner 策略)
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def analyze(df):
        if df.empty or len(df) < 60: return 50, [], pd.DataFrame()
        
        c = df['Close']
        h = df['High']
        l = df['Low']
        v = df['Volume']
        
        score = 50
        signals = [] # 紀錄觸發的訊號
        
        # --- A. 肯特納通道 (Keltner Channel) ---
        ema20 = c.ewm(span=20).mean()
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr10 = tr.rolling(10).mean()
        k_upper = ema20 + 2.0 * atr10
        k_lower = ema20 - 2.0 * atr10
        
        if c.iloc[-1] > k_upper.iloc[-1]: 
            score += 15
            signals.append("肯特納突破")
        elif c.iloc[-1] < k_lower.iloc[-1]: 
            score -= 15
            signals.append("肯特納跌破")

        # --- B. R-Breaker 趨勢確認 ---
        prev_c = c.iloc[-2]
        if c.iloc[-1] > prev_c * 1.015: 
            score += 5
            signals.append("強勢紅K")

        # --- C. OBV ---
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_ma = obv.rolling(20).mean()
        if obv.iloc[-1] > obv_ma.iloc[-1]: 
            score += 5
            signals.append("OBV多方")

        # 返回計算指標供繪圖用
        indicators = pd.DataFrame({
            'EMA20': ema20,
            'K_Upper': k_upper,
            'K_Lower': k_lower
        }, index=df.index)

        return min(100, max(0, score)), signals, indicators

# =============================================================================
# 4. 回測引擎
# =============================================================================
class MARCS_Backtester:
    def __init__(self, ticker, initial_capital):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.df = pd.DataFrame()
        
        indices = Global_Index_List.get_indices()
        # 預設 vol_cap 為 0.5
        self.vol_cap = indices.get(ticker, {}).get('vol_cap', 0.5)

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if self.df.empty: return False
            if isinstance(self.df.columns, pd.MultiIndex): self.df.columns = self.df.columns.get_level_values(0)
            
            # 計算 ATR
            h, l, c = self.df['High'], self.df['Low'], self.df['Close']
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            self.df['ATR'] = tr.rolling(14).mean()
            return True
        except: return False

    def run(self):
        cash = self.initial_capital
        position = 0
        stop_loss = 0
        trades = []
        equity = []
        
        # 預計算指標 (加速回測)
        _, _, indicators = Micro_Structure_Engine.analyze(self.df)
        self.df = pd.concat([self.df, indicators], axis=1)

        # 確保回測數據足夠
        start_idx = 60
        if len(self.df) <= start_idx:
            return pd.DataFrame(), pd.DataFrame()

        for i in range(start_idx, len(self.df)):
            curr_date = self.df.index[i]
            curr_price = self.df['Close'].iloc[i]
            curr_atr = self.df['ATR'].iloc[i]
            k_upper = self.df['K_Upper'].iloc[i]
            ma20 = self.df['EMA20'].iloc[i]
            
            # 模擬微觀分數 (簡化邏輯以加速迴圈)
            micro_score = 50
            if curr_price > k_upper: micro_score += 15 # 肯特納突破
            if curr_price > ma20: micro_score += 10    # 均線之上
            
            # 模擬混沌值
            if pd.notna(curr_atr) and pd.notna(self.df['ATR'].iloc[i-20:i].mean()):
                chaos_sim = (curr_atr / self.df['ATR'].iloc[i-20:i].mean()) - 1.0
                chaos_sim = max(0, chaos_sim + 0.5)
            else:
                chaos_sim = 0.5 # 默認值

            # --- 交易執行 ---
            if position > 0:
                # 停損出場
                if curr_price < stop_loss:
                    cash += position * curr_price
                    trades.append({'Date': curr_date, 'Type': 'SELL', 'Price': curr_price, 'Reason': 'SL'})
                    position = 0
                else:
                    # 移動停利
                    new_sl = curr_price - 2.5 * curr_atr
                    if new_sl > stop_loss: stop_loss = new_sl
            
            if position == 0:
                # 進場條件: 微觀強勢 + 價格站上通道
                if micro_score >= 65:
                    sl_price = curr_price - 2.5 * curr_atr
                    size, _ = Antifragile_Position_Sizing.calculate_size(
                        cash, curr_price, sl_price, chaos_sim, self.vol_cap
                    )
                    
                    cost = size * curr_price
                    if size > 0 and cost <= cash:
                        cash -= cost
                        position = size
                        stop_loss = sl_price
                        trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price, 'Size': size})

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
            
        return pd.DataFrame(equity), pd.DataFrame(trades)

# =============================================================================
# 5. UI 介面
# =============================================================================
def main():
    # --- 側邊欄 ---
    st.sidebar.markdown("## ⚙️ 參數控制台")
    ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
    capital = st.sidebar.number_input("Capital", value=1000000, step=100000)
    
    st.title("🛡️ MARCS V54: 量化戰情室")
    st.markdown("##### Book Strategy (Keltner/R-Breaker) + Taleb Risk Control")
    
    if st.sidebar.button("🚀 啟動全系統分析", type="primary"):
        bt = MARCS_Backtester(ticker, capital)
        
        with st.spinner("正在連線全球交易所數據..."):
            if bt.fetch_data():
                # 1. 執行運算
                df_equity, df_trades = bt.run()
                
                # 檢查回測是否成功
                if df_equity.empty:
                    st.warning("⚠️ 數據不足或無交易產生，請嘗試其他代碼。")
                    return

                score_now, signals_now, indicators = Micro_Structure_Engine.analyze(bt.df)
                
                # 計算即時建議
                last_row = bt.df.iloc[-1]
                curr_price = last_row['Close']
                sl_price = curr_price - 2.5 * last_row['ATR']
                size_now, details_now = Antifragile_Position_Sizing.calculate_size(
                    capital, curr_price, sl_price, 0.8, bt.vol_cap
                )
                
                # --- A. 儀表板區域 ---
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">微觀評分</div>
                        <div class="metric-value" style="color:{'#3fb950' if score_now>60 else '#f85149'}">{score_now}</div>
                        <div class="metric-sub">{', '.join(signals_now) if signals_now else '盤整中'}</div>
                    </div>""", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">建議部位 (股/顆)</div>
                        <div class="metric-value">{details_now.get('final_capital', 0)//int(curr_price) if curr_price else 0}</div>
                        <div class="metric-sub" style="color:#d2a8ff">Taleb 係數: {details_now.get('taleb_factor', 1.0)}x</div>
                    </div>""", unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">智能停損 (SL)</div>
                        <div class="metric-value" style="color:#f85149">{sl_price:.2f}</div>
                        <div class="metric-sub">Risk: -${details_now.get('risk_money', 0)}</div>
                    </div>""", unsafe_allow_html=True)

                with col4:
                    ret = (df_equity['Equity'].iloc[-1] - df_equity['Equity'].iloc[0]) / df_equity['Equity'].iloc[0] * 100
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">2年總報酬</div>
                        <div class="metric-value" style="color:{'#3fb950' if ret>0 else '#f85149'}">{ret:.1f}%</div>
                        <div class="metric-sub">交易次數: {len(df_trades)}</div>
                    </div>""", unsafe_allow_html=True)
                
                # --- B. 視覺化圖表 (分頁) ---
                st.markdown("### 📊 策略透視")
                tab1, tab2 = st.tabs(["🕯️ 技術分析 (Keltner)", "📈 資金權益曲線"])
                
                with tab1: # 這是書中策略的視覺化
                    st.caption("展示書中「肯特納通道」策略邏輯與進出場點")
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # 取最近 150 天數據繪圖
                    plot_df = bt.df.tail(150)
                    plot_ind = indicators.tail(150)
                    
                    # 畫 K 線 (簡化為收盤線) 與 通道
                    ax1.plot(plot_df.index, plot_df['Close'], color='white', lw=1.5, label='Price')
                    ax1.plot(plot_ind.index, plot_ind['K_Upper'], color='#00f2ff', ls='--', alpha=0.6, label='Keltner Upper')
                    ax1.plot(plot_ind.index, plot_ind['K_Lower'], color='#00f2ff', ls='--', alpha=0.6, label='Keltner Lower')
                    ax1.fill_between(plot_ind.index, plot_ind['K_Upper'], plot_ind['K_Lower'], color='#00f2ff', alpha=0.05)
                    
                    # 標記買賣點
                    if not df_trades.empty:
                        buy_signals = df_trades[df_trades['Type'] == 'BUY']
                        sell_signals = df_trades[df_trades['Type'] == 'SELL']
                        
                        # 過濾出在繪圖範圍內的交易
                        buy_signals = buy_signals[buy_signals['Date'] >= plot_df.index[0]]
                        sell_signals = sell_signals[sell_signals['Date'] >= plot_df.index[0]]
                        
                        ax1.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='#3fb950', s=100, label='Buy', zorder=5)
                        ax1.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='#f85149', s=100, label='Sell', zorder=5)
                    
                    ax1.set_facecolor('#0e1117')
                    fig1.patch.set_facecolor('#0e1117')
                    ax1.tick_params(colors='gray')
                    ax1.grid(True, alpha=0.1)
                    ax1.legend(loc='upper left', frameon=False, labelcolor='white')
                    st.pyplot(fig1)
                    
                with tab2:
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                    ax2.set_facecolor('#0e1117')
                    fig2.patch.set_facecolor('#0e1117')
                    ax2.tick_params(colors='gray')
                    ax2.set_title('Account Equity Curve', color='white')
                    ax2.grid(True, alpha=0.1)
                    st.pyplot(fig2)
                
                # --- C. 交易明細 ---
                with st.expander("查看詳細交易清單"):
                    st.dataframe(df_trades, use_container_width=True)
                    
            else:
                st.error("❌ 無法獲取數據，請檢查代碼或網路連線。")

if __name__ == "__main__":
    main()
