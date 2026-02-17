import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

# =============================================================================
# 0. 系統配置與 CSS (The Skin)
# =============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="MARCS V90 FUSION", layout="wide", page_icon="⚡")

# 引入 "Bento Grid" 暗黑風格與專業排版
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* 全局背景 */
    .stApp { background-color: #0d1117; font-family: 'Rajdhani', sans-serif; }
    
    /* Bento Card 風格 */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    .metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-family: 'Roboto Mono'; }
    .metric-value { color: #e6edf3; font-size: 24px; font-weight: 700; margin: 4px 0; }
    .metric-sub { font-size: 11px; font-family: 'Roboto Mono'; }
    
    /* 顏色定義 */
    .c-green { color: #3fb950; }
    .c-red { color: #f85149; }
    .c-gold { color: #d29922; }
    .c-blue { color: #2f81f7; }
    
    /* 表格樣式優化 */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# 智能格式化工具
def smart_format(value, is_currency=True, include_sign=False):
    if value is None or pd.isna(value): return "N/A"
    val = float(value)
    prefix = "$" if is_currency else ""
    sign = "+" if include_sign and val > 0 else ("-" if val < 0 else "")
    val = abs(val)
    if val < 1 and val > 0: return f"{sign}{prefix}{val:.4f}"
    return f"{sign}{prefix}{val:,.0f}" if val > 100 else f"{sign}{prefix}{val:,.2f}"

# =============================================================================
# 1. 核心引擎群 (The Brains)
# =============================================================================

class Micro_Structure_Engine:
    """計算技術指標 (ATR, RSI, Keltner)"""
    @staticmethod
    def attach_indicators(df):
        if df.empty: return df
        df = df.copy()
        c = df['Close']
        h = df['High']
        l = df['Low']
        
        # EMA
        df['EMA20'] = c.ewm(span=20, adjust=False).mean()
        
        # ATR
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean().fillna(tr.mean())
        
        # Keltner Channels (用於突破策略)
        atr10 = tr.rolling(10).mean().fillna(tr.mean())
        df['K_Upper'] = df['EMA20'] + 2.0 * atr10
        df['K_Lower'] = df['EMA20'] - 2.0 * atr10
        
        # RSI
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.fillna(method='bfill')

    @staticmethod
    def get_signal_score(row):
        """簡單評分用於掃描"""
        score = 50
        if row['Close'] > row['EMA20']: score += 20
        if row['RSI'] > 50: score += 10
        if row['RSI'] > 70: score -= 10 # 過熱
        if row['Close'] > row['K_Upper']: score += 20 # 突破
        return min(max(score, 0), 100)

class Backtester_Pro:
    """專業回測引擎：含手續費、稅金、基準比較"""
    def __init__(self, ticker, initial_capital, fee_rate=0.001425*0.6, tax_rate=0.003):
        self.ticker = ticker
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.tax_rate = tax_rate
        self.df = pd.DataFrame()

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if self.df.empty: return False
            if isinstance(self.df.columns, pd.MultiIndex): self.df.columns = self.df.columns.get_level_values(0)
            self.df = Micro_Structure_Engine.attach_indicators(self.df)
            return True
        except: return False

    def run(self):
        cash = self.capital
        position = 0
        trades = []
        equity_curve = []
        
        # 基準設定 (Buy & Hold)
        start_price = self.df.iloc[20]['Close']
        bh_shares = self.capital // start_price
        bh_cash = self.capital - (bh_shares * start_price)
        
        # DCA 設定 (每月定投)
        dca_cash = 0
        dca_shares = 0
        dca_invested = 0
        monthly_budget = self.capital / 12 

        total_fees = 0
        
        # 模擬回測 loop
        for i in range(20, len(self.df)):
            date = self.df.index[i]
            row = self.df.iloc[i]
            price = row['Close']
            
            # --- 策略邏輯 (MARCS Lite) ---
            # 進場: 突破上軌 & RSI 健康
            buy_signal = (price > row['K_Upper']) and (row['RSI'] > 50) and (row['RSI'] < 80)
            # 出場: 跌破 EMA20
            sell_signal = (price < row['EMA20'])

            # 執行交易
            if position > 0 and sell_signal:
                # 賣出 (扣稅+費)
                gross = position * price
                fee = gross * self.fee_rate
                tax = gross * self.tax_rate
                cash += (gross - fee - tax)
                total_fees += (fee + tax)
                
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Fee': fee+tax})
                position = 0
                
            elif position == 0 and buy_signal:
                # 買入 (扣費)
                cost = cash * 0.98 # 留一點現金buffer
                fee = cost * self.fee_rate
                shares = (cost - fee) // price
                if shares > 0:
                    cash -= (shares * price + fee)
                    total_fees += fee
                    position = shares
                    trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Fee': fee})

            # --- 計算權益 ---
            # 1. 策略
            strat_val = cash + (position * price)
            
            # 2. Buy & Hold
            bh_val = bh_cash + (bh_shares * price)
            
            # 3. DCA (簡化: 每月1號買入)
            if date.day == 1 and i > 20:
                new_shares = monthly_budget // price
                dca_shares += new_shares
                dca_invested += (new_shares * price)
            
            dca_current_val = (dca_shares * price) + (self.capital - dca_invested) # 簡單計算

            equity_curve.append({
                'Date': date,
                'Strategy': strat_val,
                'BuyHold': bh_val,
                'DCA': dca_current_val if dca_invested > 0 else self.capital
            })
            
        return pd.DataFrame(equity_curve), pd.DataFrame(trades), total_fees

# =============================================================================
# 2. UI 組件 (The View)
# =============================================================================

def render_strategy_lab(ticker, capital):
    """被動分析模式：專業儀表板"""
    st.markdown(f"### 🧪 STRATEGY LAB: {ticker}")
    
    bt = Backtester_Pro(ticker, capital)
    
    with st.spinner("Simulating Market Replay..."):
        if bt.fetch_data():
            df_eq, df_tr, fees = bt.run()
            
            if df_eq.empty:
                st.error("Insufficient data for simulation.")
                return

            # --- Row 1: The Arena (Plotly) ---
            st.markdown("#### ⚔️ PERFORMANCE ARENA")
            fig = go.Figure()
            
            # 繪製三條曲線
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['BuyHold'], name='Buy & Hold', line=dict(color='#2f81f7', width=2), opacity=0.5))
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['DCA'], name='DCA (Safe)', line=dict(color='#3fb950', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['Strategy'], name='MARCS Alpha', line=dict(color='#d29922', width=3), fill='tonexty', fillcolor='rgba(210, 153, 34, 0.1)'))
            
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=450, hovermode="x unified", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Row 2: Metrics (Bento Grid) ---
            final_val = df_eq.iloc[-1]['Strategy']
            pnl = final_val - capital
            pnl_pct = (pnl / capital) * 100
            bh_pnl_pct = ((df_eq.iloc[-1]['BuyHold'] - capital) / capital) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: 
                st.markdown(f"""<div class="metric-card"><div class="metric-label">NET PROFIT</div>
                <div class="metric-value {('c-green' if pnl>0 else 'c-red')}">{smart_format(pnl)}</div>
                <div class="metric-sub">{pnl_pct:+.2f}% Return</div></div>""", unsafe_allow_html=True)
            with c2:
                alpha = pnl_pct - bh_pnl_pct
                st.markdown(f"""<div class="metric-card"><div class="metric-label">ALPHA vs B&H</div>
                <div class="metric-value {('c-gold' if alpha>0 else 'c-red')}">{alpha:+.2f}%</div>
                <div class="metric-sub">Strategy Edge</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">FRICTION COST</div>
                <div class="metric-value c-red">{smart_format(fees)}</div>
                <div class="metric-sub">Fees & Tax Paid</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">TRADE COUNT</div>
                <div class="metric-value">{len(df_tr)}</div>
                <div class="metric-sub">Signals Executed</div></div>""", unsafe_allow_html=True)
            
            # --- Row 3: Technicals ---
            with st.expander("📊 Technical Analysis & Signals", expanded=False):
                fig_t = go.Figure()
                p_df = bt.df.tail(100)
                fig_t.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='Price'))
                fig_t.add_trace(go.Scatter(x=p_df.index, y=p_df['K_Upper'], line=dict(color='cyan', width=1, dash='dot'), name='Breakout Line'))
                
                # 標記買賣點
                if not df_tr.empty:
                    buys = df_tr[df_tr['Type']=='BUY']
                    sells = df_tr[df_tr['Type']=='SELL']
                    fig_t.add_trace(go.Scatter(x=buys['Date'], y=buys['Price']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='#3fb950', size=10), name='BUY'))
                    fig_t.add_trace(go.Scatter(x=sells['Date'], y=sells['Price']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='#f85149', size=10), name='SELL'))
                
                fig_t.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_t, use_container_width=True)

        else:
            st.error(f"Failed to load data for {ticker}")

def render_scanner_mode(watchlist):
    """主動選股模式：掃描觀察清單"""
    st.markdown("### 🔭 ACTIVE MARKET SCANNER")
    st.info(f"Scanning {len(watchlist)} targets for 'Phoenix' setups...")
    
    results = []
    
    # Progress Bar
    my_bar = st.progress(0)
    
    for i, ticker in enumerate(watchlist):
        try:
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = Micro_Structure_Engine.attach_indicators(df)
                last_row = df.iloc[-1]
                
                score = Micro_Structure_Engine.get_signal_score(last_row)
                
                # 只有分數 > 60 才顯示
                if score >= 60:
                    results.append({
                        "Ticker": ticker.replace(".TW", ""),
                        "Price": last_row['Close'],
                        "Score": score,
                        "RSI": round(last_row['RSI'], 1),
                        "Trend": "Bull" if last_row['Close'] > last_row['EMA20'] else "Bear"
                    })
        except:
            pass
        my_bar.progress((i + 1) / len(watchlist))
    
    if results:
        res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.markdown("#### 🔥 Potential Targets")
        # 使用 Styler 進行著色
        st.dataframe(
            res_df.style.background_gradient(subset=['Score'], cmap='Greens'),
            use_container_width=True
        )
    else:
        st.warning("No high-probability setups found today.")

# =============================================================================
# 3. 主程序 (Main Loop)
# =============================================================================
def main():
    # 側邊欄控制
    st.sidebar.title("⚡ MARCS V90")
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio("SYSTEM MODE", ["🔭 Market Scanner (Active)", "🧪 Strategy Lab (Passive)"])
    
    st.sidebar.markdown("---")
    
    if "Scanner" in mode:
        # 主動模式
        # 這裡你可以放入你關注的 20-30 檔股票，不建議放 1700 檔以免跑太久
        default_list = "2330, 2317, 2454, 2603, 2609, 2618, 3035, 3037, 2382, 3231"
        user_list = st.sidebar.text_area("Watchlist (Comma separated)", default_list)
        targets = [f"{x.strip()}.TW" for x in user_list.split(",")]
        
        if st.sidebar.button("🚀 RUN SCAN", type="primary"):
            render_scanner_mode(targets)
            
    else:
        # 被動模式
        ticker = st.sidebar.text_input("TARGET TICKER", "2330.TW")
        capital = st.sidebar.number_input("CAPITAL", 1000000, step=100000)
        
        st.sidebar.markdown("##### Friction Settings")
        fee = st.sidebar.number_input("Fee (%)", 0.0, 1.0, 0.1425, format="%.4f")
        tax = st.sidebar.number_input("Tax (%)", 0.0, 1.0, 0.3, format="%.2f")
        
        if st.sidebar.button("🔬 ANALYZE", type="primary"):
            render_strategy_lab(ticker, capital)

if __name__ == "__main__":
    main()
