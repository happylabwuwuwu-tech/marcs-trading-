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
    page_title="MARCS V57 星際戰情室",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 0. CSS 視覺魔法 (星空 + 科技感)
# =============================================================================
st.markdown("""
<style>
    /* 1. 全局字體與背景設置 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Rajdhani', sans-serif;
    }

    /* 2. 動態星空背景 (使用 CSS 徑向漸層模擬星星) */
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

    /* 3. 科技感毛玻璃卡片 (Glassmorphism) */
    .metric-card {
        background: rgba(22, 27, 34, 0.6); /* 半透明黑 */
        backdrop-filter: blur(12px);         /* 毛玻璃模糊 */
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2); /* 科技藍邊框 */
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.4); /* 懸浮發光 */
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

    /* 5. 側邊欄優化 */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 17, 23, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
    }
    
    /* 6. 按鈕科技化 */
    div.stButton > button {
        background: linear-gradient(90deg, #1f6feb 0%, #00f2ff 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        transition: all 0.3s;
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
# 1. 資產定義
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
# 2. 宏觀引擎
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
            curr_w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            hist_std = returns.rolling(40).std().mean() * 0.1
            chaos = curr_w2 / (hist_std + 1e-9)
            
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {"ticker": ticker, "name": name, "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend}
        except: return None

    @staticmethod
    def calculate_macro_score(results):
        score = 50.0
        data_map = {r['ticker']: r for r in results if r}
        
        vix = data_map.get('^VIX')
        if vix:
            if vix['trend'] == 'Overbought': score += 15
            elif vix['trend'] == 'Oversold': score -= 15
            
        dxy = data_map.get('DX-Y.NYB')
        if dxy:
            if dxy['trend'] == 'Overbought': score -= 12
            elif dxy['trend'] == 'Oversold': score += 12
            
        return min(100, max(0, score))

# =============================================================================
# 3. 微觀引擎 & 風控
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def analyze(df):
        if df.empty or len(df) < 60: return 50, [], pd.DataFrame()
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        score = 50; signals = []
        
        ema20 = c.ewm(span=20).mean()
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr10 = tr.rolling(10).mean()
        k_upper = ema20 + 2.0 * atr10
        k_lower = ema20 - 2.0 * atr10
        
        if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner Breakout")
        elif c.iloc[-1] < k_lower.iloc[-1]: score -= 15; signals.append("Keltner Breakdown")

        if c.iloc[-1] > c.iloc[-2] * 1.015: score += 5; signals.append("Power Candle")
        
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]: score += 5; signals.append("OBV Bullish")

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
# 5. 主介面 (V57 Starfield Edition)
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ 系統控制台")
    ticker = st.sidebar.text_input("TARGET", value="BTC-USD")
    capital = st.sidebar.number_input("CAPITAL", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    # 請替換為你的 GitHub Raw Video URL
    video_url = "https://raw.githubusercontent.com/YOUR_NAME/YOUR_REPO/main/model_arch.mp4.mp4" 
    st.sidebar.markdown("### 🎥 系統架構演示")
    try: st.sidebar.video(video_url)
    except: st.sidebar.info("請配置影片 URL")

    # 標題區
    st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 0 0 10px #00f2ff;'>🛡️ MARCS V57 INTERSTELLAR</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; letter-spacing: 2px;'>QUANTUM MACRO INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
    
    if st.sidebar.button("🚀 INITIATE SCAN", type="primary"):
        # 1. 宏觀儀表板
        st.markdown("### 📡 MACRO METRICS")
        macro_indices = Global_Index_List.get_macro_indices()
        macro_results = []
        cols = st.columns(4)
        
        for idx, (sym, info) in enumerate(macro_indices.items()):
            res = Macro_Engine.analyze(sym, info['name'])
            macro_results.append(res)
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
        
        with st.spinner("Decodin Market Structure..."):
            if bt.fetch_data():
                df_equity, df_trades = bt.run()
                score, signals, indicators = Micro_Structure_Engine.analyze(bt.df)
                
                last = bt.df.iloc[-1]
                curr_p = last['Close']
                sl_p = curr_p - 2.5 * last['ATR']
                size, details = Antifragile_Position_Sizing.calculate_size(capital, curr_p, sl_p, 0.8, bt.vol_cap)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">MICRO SCORE</div>
                        <div class="metric-value" style="color:{'#3fb950' if score>60 else '#f85149'}">{score}</div>
                        <div class="metric-sub">{', '.join(signals) if signals else 'NEUTRAL'}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">TALEB SIZE</div>
                        <div class="metric-value">{details.get('final_capital', 0)//int(curr_p) if curr_p else 0}</div>
                        <div class="metric-sub" style="color:#00f2ff">Factor: {details.get('taleb_factor', 1.0)}x</div>
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

                # 圖表
                st.markdown("#### 📊 TACTICAL VISUALIZATION")
                tab1, tab2 = st.tabs(["CHART", "EQUITY"])
                
                with tab1:
                    fig1, ax1 = plt.subplots(figsize=(12, 5))
                    p_df = bt.df.tail(150); p_ind = indicators.tail(150)
                    ax1.plot(p_df.index, p_df['Close'], color='#e6edf3', lw=1.5)
                    ax1.plot(p_ind.index, p_ind['K_Upper'], color='#00f2ff', ls='--', alpha=0.7)
                    ax1.plot(p_ind.index, p_ind['K_Lower'], color='#00f2ff', ls='--', alpha=0.7)
                    ax1.fill_between(p_ind.index, p_ind['K_Upper'], p_ind['K_Lower'], color='#00f2ff', alpha=0.1)
                    
                    if not df_trades.empty:
                        bs = df_trades[df_trades['Type']=='BUY']
                        ss = df_trades[df_trades['Type']=='SELL']
                        bs = bs[bs['Date']>=p_df.index[0]]
                        ss = ss[ss['Date']>=p_df.index[0]]
                        ax1.scatter(bs['Date'], bs['Price'], marker='^', color='#3fb950', s=100, zorder=5)
                        ax1.scatter(ss['Date'], ss['Price'], marker='v', color='#f85149', s=100, zorder=5)
                    
                    ax1.set_facecolor('#0d1117'); fig1.patch.set_facecolor('#0d1117')
                    ax1.tick_params(colors='#8b949e'); ax1.grid(True, color='#30363d', alpha=0.5)
                    st.pyplot(fig1)

                with tab2:
                    if not df_equity.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                        ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
                        ax2.tick_params(colors='#8b949e'); ax2.grid(True, color='#30363d', alpha=0.5)
                        st.pyplot(fig2)

            else:
                st.error("Connection Failed: Data Unavailable")

if __name__ == "__main__":
    main()
