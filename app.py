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
    page_title="MARCS V81 OMEGA",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 0. 核心工具函數 (智能格式化)
# =============================================================================
def smart_format(value, is_currency=True, is_percent=False, include_sign=False):
    """
    全能格式化引擎：自動適配精度
    """
    if value is None or pd.isna(value) or value == 0:
        return "$0.00" if is_currency else "0.00"
    
    val = float(value)
    if is_percent: return f"{val*100:.2f}%"

    abs_val = abs(val)
    prefix = "$" if is_currency else ""
    sign = ""
    if include_sign and val > 0: sign = "+"
    elif val < 0: sign = "-"
    
    if abs_val < 0.000001: return f"{sign}{prefix}{abs_val:.9f}".rstrip('0')
    elif abs_val < 0.001: return f"{sign}{prefix}{abs_val:.7f}".rstrip('0')
    elif abs_val < 1: return f"{sign}{prefix}{abs_val:.5f}"
    else: return f"{sign}{prefix}{abs_val:,.2f}"

# =============================================================================
# 1. CSS 視覺魔法
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px);
        background-size: 550px 550px, 350px 350px; opacity: 0.8; z-index: -1;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.6); backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px;
        padding: 20px; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .metric-label { color: #8b949e; font-size: 14px; letter-spacing: 1px; font-family: 'Roboto Mono'; text-transform: uppercase; }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: 700; margin: 5px 0; }
    .metric-sub { font-size: 12px; margin-top: 8px; font-family: 'Roboto Mono'; color: #8b949e; }
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
# 2. 引擎定義
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

class Macro_Regime_Engine:
    @staticmethod
    def get_market_regime():
        try:
            tickers = ["^VIX", "DX-Y.NYB"]
            df = yf.download(tickers, period="1mo", progress=False, auto_adjust=True)
            if df.empty: return 0.5, "Data Missing"
            
            if isinstance(df.columns, pd.MultiIndex): 
                vix = df['Close']['^VIX'].iloc[-1]
                dxy = df['Close']['DX-Y.NYB'].iloc[-1]
            else:
                return 0.5, "Format Error"

            risk_score = 0
            if vix < 15: risk_score += 0
            elif vix < 20: risk_score += 0.2
            elif vix < 30: risk_score += 0.5
            else: risk_score += 1.0
            
            if dxy > 105: risk_score += 0.3
            final_risk = min(risk_score, 1.0)
            
            if final_risk >= 0.8: regime_desc = "CRASH MODE (High Risk)"
            elif final_risk >= 0.5: regime_desc = "CAUTION (Volatile)"
            else: regime_desc = "RISK ON (Safe)"
            return final_risk, regime_desc
        except Exception as e:
            return 0.5, f"Error: {str(e)}"

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

# [FIXED] 六因子引擎 (加強魯棒性)
class Factor_DNA_Engine:
    @staticmethod
    def analyze_factors(df):
        """
        計算技術面六因子分數 (0-100)
        注意：必須先執行 attach_indicators 確保指標存在
        """
        if df.empty or len(df) < 60: return None, 0
        
        last = df.iloc[-1]
        c = df['Close']
        v = df['Volume'] if 'Volume' in df.columns else pd.Series(np.ones(len(df)))
        scores = {}
        
        # 1. Momentum (RSI)
        # 使用 .get() 避免 crash，預設 50
        rsi = last['rsi'] if 'rsi' in last else 50 
        scores['Momentum'] = min(max(rsi, 0), 100)
        
        # 2. Trend (Price vs EMA20 & ADX)
        trend_raw = 50
        ema = last['EMA20'] if 'EMA20' in last else c.mean()
        if c.iloc[-1] > ema: trend_raw += 30
        
        adx = last['ADX'] if 'ADX' in last else 0
        if adx > 25: trend_raw += 20
        scores['Trend'] = min(trend_raw, 100)
        
        # 3. Volatility (ATR)
        atr = last['ATR'] if 'ATR' in last else (c.iloc[-1]*0.02)
        atr_pct = (atr / c.iloc[-1]) * 100
        vol_score = 100 - (atr_pct * 10)
        scores['Volatility'] = min(max(vol_score, 0), 100)
        
        # 4. Volume
        vol_ma = v.rolling(20).mean().iloc[-1]
        # 避免除以零或極小量
        vol_score = 50
        if v.iloc[-1] > vol_ma: vol_score = 80
        scores['Volume'] = vol_score
        
        # 5. Strength (ROC)
        roc = ((c.iloc[-1] - c.iloc[-20]) / c.iloc[-20]) * 100
        scores['Strength'] = min(max(50 + roc * 2, 0), 100)
        
        # 6. Consistency
        returns = c.pct_change().tail(20)
        pos_days = returns[returns > 0].count()
        scores['Consistency'] = (pos_days / 20) * 100
        
        avg_score = sum(scores.values()) / 6
        return scores, avg_score

class Micro_Structure_Engine:
    @staticmethod
    def attach_indicators(df):
        """
        [CRITICAL] 這是所有分析的基礎，必須最先被執行
        """
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
        
        # 計算 RSI 供六因子使用
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def get_signals(df_row):
        score = 50
        signals = []
        c = df_row['Close']
        is_trending = df_row.get('ADX', 0) > 20
        
        if is_trending:
            if c > df_row.get('K_Upper', c*1.1): score += 15; signals.append("Keltner Breakout")
            elif c < df_row.get('K_Lower', c*0.9): score -= 15; signals.append("Keltner Breakdown")
            if c > df_row.get('EMA20', c): score += 10
        else:
            signals.append("Low Trend")
        return score, signals

class Multi_Timeframe_Engine:
    @staticmethod
    def fetch_and_process_weekly(ticker):
        try:
            df_wk = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
            if df_wk.empty: return None
            if isinstance(df_wk.columns, pd.MultiIndex): df_wk.columns = df_wk.columns.get_level_values(0)
            
            c = df_wk['Close']
            df_wk['W_EMA20'] = c.ewm(span=20).mean()
            
            h, l = df_wk['High'], df_wk['Low']
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            plus_dm = (h - h.shift()).clip(lower=0)
            minus_dm = (l.shift() - l).clip(lower=0)
            tr_smooth = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            df_wk['W_ADX'] = dx.rolling(14).mean().fillna(0)
            
            last = df_wk.iloc[-1]
            trend = "BULL" if last['Close'] > last['W_EMA20'] else "BEAR"
            strength = "STRONG" if last['W_ADX'] > 20 else "WEAK"
            
            return {"df": df_wk, "trend": trend, "strength": strength, "ema": last['W_EMA20'], "adx": last['W_ADX']}
        except: return None

    @staticmethod
    def merge_mtf_data(daily_df, weekly_df):
        weekly_resampled = weekly_df[['W_EMA20', 'W_ADX']].resample('D').ffill()
        merged = daily_df.join(weekly_resampled, how='left')
        merged[['W_EMA20', 'W_ADX']] = merged[['W_EMA20', 'W_ADX']].ffill()
        return merged

class Antifragile_Position_Sizing:
    @staticmethod
    def calculate_size(account_balance, current_price, stop_loss_price, chaos_level, macro_penalty=0.0):
        risk_per_trade = account_balance * 0.02 
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        
        # Taleb Multiplier
        taleb_multiplier = 1.0
        if chaos_level > 1.2: taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
        
        # Macro Penalty
        macro_multiplier = 1.0 - macro_penalty
        
        final_size = int(base_size * taleb_multiplier * macro_multiplier)
        suggested_capital = final_size * current_price
        
        return final_size, {
            "risk_money": risk_per_trade,
            "final_capital": suggested_capital,
            "macro_mult": macro_multiplier
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
    def __init__(self, ticker, initial_capital, use_mtf=False, macro_risk=0.0):
        self.ticker = ticker; self.initial_capital = initial_capital
        self.df = pd.DataFrame()
        self.use_mtf = use_mtf
        self.macro_risk = macro_risk

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if self.df.empty: return False
            if isinstance(self.df.columns, pd.MultiIndex): self.df.columns = self.df.columns.get_level_values(0)
            return True
        except: return False

    def run(self):
        # 注意：指標已經在 main 中 pre-calculate 了，這裡只是確保萬一
        if 'EMA20' not in self.df.columns:
            self.df = Micro_Structure_Engine.attach_indicators(self.df)
        
        if self.use_mtf:
            mtf_data = Multi_Timeframe_Engine.fetch_and_process_weekly(self.ticker)
            if mtf_data:
                self.df = Multi_Timeframe_Engine.merge_mtf_data(self.df, mtf_data['df'])
            else:
                self.df['W_EMA20'] = 0; self.df['W_ADX'] = 0

        cash = self.initial_capital; position = 0; stop_loss = 0
        trades = []; equity = []
        entry_price = 0
        filtered_count = 0

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
            
            if position == 0:
                base_signal = micro_score >= 65 and "Low Trend" not in str(signals)
                mtf_approved = True
                if self.use_mtf and 'W_EMA20' in row and row['W_EMA20'] > 0:
                    if curr_price < row['W_EMA20']: mtf_approved = False
                
                if base_signal:
                    if mtf_approved:
                        sl_price = curr_price - 2.5 * row['ATR']
                        size, _ = Antifragile_Position_Sizing.calculate_size(cash, curr_price, sl_price, 0.5, self.macro_risk)
                        cost = size * curr_price
                        if size > 0 and cost <= cash:
                            cash -= cost; position = size; stop_loss = sl_price
                            entry_price = curr_price
                            trades.append({'Date': curr_date, 'Type': 'BUY', 'Price': curr_price})
                    else:
                        filtered_count += 1

            equity.append({'Date': curr_date, 'Equity': cash + (position * curr_price)})
        return pd.DataFrame(equity), pd.DataFrame(trades), filtered_count

# =============================================================================
# 3. 主程序
# =============================================================================
def main():
    st.sidebar.markdown("## ⚙️ SYSTEM CORE")
    mode = st.sidebar.radio("MODE SELECT", ["LIVE MARKET MONITOR", "SIMULATION LAB"], index=0)
    st.sidebar.markdown("---")
    use_mtf = st.sidebar.toggle("启用 MTF 週線過濾", value=True)
    
    if mode == "LIVE MARKET MONITOR":
        ticker_input = st.sidebar.text_input("TARGET", value="BTC-USD")
        capital = st.sidebar.number_input("CAPITAL", value=1000000, step=100000)
        
        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>🛡️ MARCS V81 OMEGA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e;'>ULTIMATE MACRO-QUANT INTELLIGENCE</p>", unsafe_allow_html=True)
        
        if st.sidebar.button("🚀 INITIATE SCAN", type="primary"):
            # 1. Macro Analysis
            risk_factor, regime_desc = Macro_Regime_Engine.get_market_regime()
            regime_color = "#3fb950" if risk_factor < 0.5 else ("#e6a23c" if risk_factor < 0.8 else "#f85149")

            st.markdown("### 🌍 MACRO REGIME SHIELD")
            m1, m2 = st.columns([1, 3])
            with m1:
                st.markdown(f"""<div class="metric-card" style="border: 2px solid {regime_color}">
                <div class="metric-label">MARKET STATUS</div>
                <div class="metric-value" style="color:{regime_color}">{regime_desc}</div>
                <div class="metric-sub">Risk Penalty: {risk_factor*100:.0f}%</div></div>""", unsafe_allow_html=True)
            with m2:
                if risk_factor > 0.5: st.warning(f"⚠️ 宏觀風險高。系統自動縮減倉位 {risk_factor*100:.0f}%。")
                else: st.success("✅ 宏觀環境安全。")

            st.markdown("### 📡 MACRO METRICS")
            macro_indices = Global_Index_List.get_macro_indices()
            cols = st.columns(4)
            for idx, (sym, info) in enumerate(macro_indices.items()):
                res = Macro_Engine.analyze(sym, info['name'])
                if res:
                    col = cols[idx % 4]
                    color = "#f85149" if res['trend'] == 'Overbought' else ("#3fb950" if res['trend'] == 'Oversold' else "#8b949e")
                    with col:
                        st.markdown(f"""<div class="metric-card" style="border-top: 2px solid {color}">
                            <div class="metric-label">{res['name']}</div>
                            <div class="metric-value">{smart_format(res['price'])}</div>
                            <div class="metric-sub" style="color:{color}">{res['trend']}</div></div>""", unsafe_allow_html=True)

            # 2. Target Analysis
            st.markdown(f"### 🔭 TARGET ANALYSIS: {ticker_input}")
            bt = MARCS_Backtester(ticker_input, capital, use_mtf=use_mtf, macro_risk=risk_factor)
            
            with st.spinner("Compiling Matrix..."):
                if bt.fetch_data():
                    # [CRITICAL FIX] 必須先計算指標，才能讓因子引擎分析
                    bt.df = Micro_Structure_Engine.attach_indicators(bt.df)
                    
                    mtf_info = Multi_Timeframe_Engine.fetch_and_process_weekly(ticker_input)
                    factors, total_score = Factor_DNA_Engine.analyze_factors(bt.df)
                    
                    df_equity, df_trades, filtered_num = bt.run()
                    last_row = bt.df.iloc[-1]
                    score, signals = Micro_Structure_Engine.get_signals(last_row)
                    
                    # [FACTOR RADAR]
                    st.markdown("#### 🧬 SIX-FACTOR DNA")
                    if factors:
                        f1, f2, f3, f4, f5, f6 = st.columns(6)
                        def render_factor(col, label, s):
                            c = "#3fb950" if s >= 70 else ("#e6a23c" if s >= 40 else "#f85149")
                            with col:
                                st.markdown(f"""<div style="text-align:center">
                                <div style="font-size:12px; color:#8b949e">{label}</div>
                                <div style="font-size:20px; font-weight:bold; color:{c}">{s:.0f}</div>
                                <div style="height:4px; width:100%; background:#30363d; margin-top:5px; border-radius:2px;">
                                    <div style="height:100%; width:{s}%; background:{c}; border-radius:2px;"></div>
                                </div></div>""", unsafe_allow_html=True)
                        render_factor(f1, "MOMENTUM", factors['Momentum'])
                        render_factor(f2, "TREND", factors['Trend'])
                        render_factor(f3, "VOLATILITY", factors['Volatility'])
                        render_factor(f4, "VOLUME", factors['Volume'])
                        render_factor(f5, "STRENGTH", factors['Strength'])
                        render_factor(f6, "CONSISTENCY", factors['Consistency'])
                    
                    # [MTF INFO]
                    if mtf_info:
                        w_trend = mtf_info['trend']
                        trend_color = "#3fb950" if w_trend == "BULL" else "#f85149"
                        st.markdown("#### 🌊 WEEKLY TIDE")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {trend_color}">
                            <div class="metric-label">WEEKLY TREND</div>
                            <div class="metric-value" style="color:{trend_color}">{w_trend}</div></div>""", unsafe_allow_html=True)
                        with m2: st.metric("WEEKLY MOMENTUM", mtf_info['strength'])
                        with m3: st.metric("WEEKLY SUPPORT", smart_format(mtf_info['ema']))
                        with m4: st.metric("NOISE FILTERED", filtered_num)

                    # [TACTICAL BOARD]
                    st.markdown("#### 🎯 TACTICAL BOARD")
                    
                    curr_price = last_row['Close']
                    atr_val = last_row['ATR']
                    sl_val = curr_price - (2.5 * atr_val)
                    tp_val = curr_price + (5.0 * atr_val)
                    fair_value = last_row['EMA20']

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                         st.markdown(f"""<div class="metric-card"><div class="metric-label">ATR (Volatility)</div>
                        <div class="metric-value">{smart_format(atr_val, is_currency=False)}</div>
                        <div class="metric-sub">Risk Unit</div></div>""", unsafe_allow_html=True)
                    with r2:
                         st.markdown(f"""<div class="metric-card" style="border-bottom: 2px solid #f85149">
                        <div class="metric-label">STOP LOSS</div>
                        <div class="metric-value" style="color:#f85149">{smart_format(sl_val)}</div>
                        <div class="metric-sub">-2.5 ATR</div></div>""", unsafe_allow_html=True)
                    with r3:
                         st.markdown(f"""<div class="metric-card" style="border-bottom: 2px solid #3fb950">
                        <div class="metric-label">TAKE PROFIT</div>
                        <div class="metric-value" style="color:#3fb950">{smart_format(tp_val)}</div>
                        <div class="metric-sub">+5.0 ATR</div></div>""", unsafe_allow_html=True)
                    with r4:
                        fair_gap = (curr_price - fair_value) / fair_value
                        gap_label = "Fair"
                        if fair_gap > 0.05: gap_label = "Overvalued"
                        elif fair_gap < -0.05: gap_label = "Undervalued"
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Fair Value Gauge</div>
                        <div class="metric-value">{smart_format(fair_value)}</div>
                        <div class="metric-sub" style="color: {'#f85149' if fair_gap > 0 else '#3fb950'}">
                            {gap_label} ({fair_gap*100:+.1f}%)
                        </div></div>""", unsafe_allow_html=True)

                    # Visuals
                    tab1, tab2 = st.tabs(["CHART", "EQUITY"])
                    with tab1:
                        fig1, ax1 = plt.subplots(figsize=(12, 5))
                        p_df = bt.df.tail(150)
                        
                        ax1.plot(p_df.index, p_df['Close'], color='#e6edf3', lw=1.5, label='Price')
                        ax1.plot(p_df.index, p_df['K_Upper'], color='#00f2ff', ls='--', alpha=0.3)
                        ax1.plot(p_df.index, p_df['K_Lower'], color='#00f2ff', ls='--', alpha=0.3)
                        ax1.fill_between(p_df.index, p_df['K_Upper'], p_df['K_Lower'], color='#00f2ff', alpha=0.05)
                        
                        if 'W_EMA20' in p_df.columns:
                             ax1.plot(p_df.index, p_df['W_EMA20'], color='#d2a8ff', lw=2, label='Weekly Trend')

                        ax1.axhline(y=sl_val, color='#f85149', linestyle='--', alpha=0.8, lw=1, label=f'SL: {smart_format(sl_val, is_currency=False)}')
                        ax1.axhline(y=tp_val, color='#3fb950', linestyle='--', alpha=0.8, lw=1, label=f'TP: {smart_format(tp_val, is_currency=False)}')

                        if curr_price < 0.0001: ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))
                        elif curr_price < 1: ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
                        else: ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                            
                        if not df_trades.empty:
                            bs = df_trades[(df_trades['Type']=='BUY') & (df_trades['Date']>=p_df.index[0])]
                            ss = df_trades[(df_trades['Type']=='SELL') & (df_trades['Date']>=p_df.index[0])]
                            ax1.scatter(bs['Date'], bs['Price'], marker='^', color='#3fb950', s=100, zorder=5)
                            ax1.scatter(ss['Date'], ss['Price'], marker='v', color='#f85149', s=100, zorder=5)
                        
                        ax1.set_facecolor('#0d1117'); fig1.patch.set_facecolor('#0d1117')
                        ax1.tick_params(colors='#8b949e'); ax1.grid(True, color='#30363d', alpha=0.3)
                        ax1.legend(facecolor='#0d1117', labelcolor='#8b949e', loc='upper left')
                        st.pyplot(fig1)

                    with tab2:
                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        if not df_equity.empty:
                            ax2.plot(pd.to_datetime(df_equity['Date']), df_equity['Equity'], color='#238636', lw=2)
                        ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
                        ax2.tick_params(colors='#8b949e'); ax2.grid(True, color='#30363d', alpha=0.3)
                        st.pyplot(fig2)
                else:
                    st.error("Data Unavailable: Please check the ticker symbol.")

    elif mode == "SIMULATION LAB":
        st.markdown("<h1 style='text-align: center; color: #f85149;'>🧪 STRESS TEST LAB</h1>", unsafe_allow_html=True)
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
                res = Risk_Entropy_Engine.run_monte_carlo_theoretical(lab_sims, lab_n_trades, lab_win_rate, lab_rr, lab_risk_pct, lab_capital)
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
