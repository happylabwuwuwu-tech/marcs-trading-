import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, lfilter, hilbert
from datetime import datetime

# =============================================================================
# 1. 系統核心配置 (System Configuration)
# =============================================================================
st.set_page_config(page_title="MARCS NEO-LEVIATHAN", layout="wide", page_icon="🛡️")

# 注入高科技儀表板 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    .stApp { background-color: #0E1117; font-family: 'Rajdhani', sans-serif; color: #C9D1D9; }
    
    /* 數據卡片 */
    .metric-card { 
        background: #161B22; 
        border: 1px solid #30363D; 
        border-radius: 6px; 
        padding: 15px; 
        margin-bottom: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* 標籤與數值 */
    .highlight-lbl { 
        font-size: 12px; 
        color: #8B949E; 
        letter-spacing: 1.5px; 
        text-transform: uppercase; 
        margin-bottom: 5px;
    }
    .highlight-val { 
        font-size: 28px; 
        font-weight: 700; 
        color: #E6EDF3; 
        font-family: 'JetBrains Mono'; 
    }
    
    /* 訊號箱 */
    .signal-box { 
        background: linear-gradient(180deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,1) 100%); 
        border: 1px solid #30363D; 
        border-radius: 12px; 
        padding: 25px; 
        text-align: center; 
        backdrop-filter: blur(5px);
    }
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# =============================================================================
# 2. 數據獲取層 (Robust Data Layer)
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    """
    獲取市場數據並進行標準化清洗。
    """
    try:
        # 抓取足夠的歷史數據以供濾波器穩定 (Warm-up period)
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if df.empty: return pd.DataFrame()
        
        # 處理 yfinance v0.2+ 的 MultiIndex Column 問題
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]:
                    df = df.xs(ticker, axis=1, level=0)
                else:
                    # 如果找不到 Ticker Key，嘗試直接取第一層
                    df.columns = df.columns.get_level_values(0)
            except:
                # 最終手段：強制重命名
                if len(df.columns) >= 5:
                    df = df.iloc[:, :5]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # 移除時區資訊，避免 Plotly 報錯
        if df.index.tz is not None: 
            df.index = df.index.tz_localize(None)
            
        # 確保數據列名正確
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# =============================================================================
# 3. 物理引擎 (Causal Signal Processing)
# =============================================================================
class Signal_Processor:
    @staticmethod
    def causal_bandpass(data, lowcut, highcut, fs, order=2):
        """
        [CRITICAL] 因果帶通濾波器
        使用 lfilter (單向) 代替 filtfilt (雙向)，確保不使用未來數據。
        代價：訊號會有相位延遲 (Phase Lag)，這是真實交易必須面對的物理定律。
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # 獲取濾波器係數
        b, a = butter(order, [low, high], btype='band')
        
        # 單向濾波
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def calc_adx(df, n=14):
        """ 計算平均趨向指標 (ADX) 用於判斷市場狀態 """
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['High'] - df['Low'])
        tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
        atr = tr.rolling(n).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/n).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/n).mean().abs() / atr)
        
        # 避免除以零
        denominator = abs(plus_di + minus_di)
        dx = 100 * (abs(plus_di - minus_di) / denominator.replace(0, 1))
        
        adx = dx.rolling(n).mean()
        return adx.fillna(0)

    @staticmethod
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, 1e-9))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def engineer_features(df):
        # 數據長度檢查，濾波器需要足夠的樣本來收斂
        if len(df) < 200: return df
        df = df.copy()
        
        # --- A. 基礎指標 (因果) ---
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = Signal_Processor.calc_rsi(df['Close'])
        df['ADX'] = Signal_Processor.calc_adx(df)
        
        # --- B. 物理指標 (The Physics) ---
        
        # 1. 對數收益率 (Log Returns) - 使數據分佈更接近常態
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # 2. 因果帶通濾波 (Causal Bandpass Filter)
        # 提取 10~40 天的市場週期 (月線至季線級別)
        # fs=1 (日數據), low=1/40, high=1/10
        # 使用 lfilter 會導致訊號向右偏移 (Lag)，這是正常的
        cycle_component = Signal_Processor.causal_bandpass(log_ret.values, 1/40, 1/10, 1, order=2)
        
        # 3. 解析訊號與相位 (Analytic Signal)
        # 使用 Hilbert 轉換計算瞬時相位
        analytic_signal = hilbert(cycle_component)
        price_phase = np.angle(analytic_signal)
        
        # 4. 成交量相位 (Volume Phase)
        # 處理成交量變化率，並提取相同頻段的週期
        vol_change = df['Volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        vol_cycle = Signal_Processor.causal_bandpass(vol_change.values, 1/40, 1/10, 1, order=2)
        vol_phase = np.angle(hilbert(vol_cycle))
        
        # 5. 相位同步率 (Phase Sync)
        # Cosine Similarity: 1 = 完全同步 (共振), -1 = 完全背離
        df['Sync'] = np.cos(price_phase - vol_phase)
        
        # 6. 因果平滑 (Causal Smoothing)
        # 使用向後 Rolling (window=3) 以減少噪音，但會稍微增加延遲
        df['Sync_Smooth'] = df['Sync'].rolling(3).mean()
        
        # 截斷濾波器初始化階段的不穩定數據 (前100天)
        return df.iloc[100:]

# =============================================================================
# 4. 策略邏輯 (Strategy Logic Core)
# =============================================================================
class Strategy_Engine:
    @staticmethod
    def evaluate(df):
        last = df.iloc[-1]
        
        # --- 1. 市場狀態識別 (Regime Identification) ---
        regime = "NEUTRAL"
        regime_color = "#8B949E"
        
        if last['ADX'] > 25: 
            regime = "TRENDING (趨勢)"
            regime_color = "#D2A8FF" # 紫色
        elif last['ADX'] < 20: 
            regime = "RANGING (震盪)"
            regime_color = "#8B949E" # 灰色
            
        # --- 2. 評分系統 ---
        score = 50
        reasons = []
        
        # A. 物理層 (Sync) - 權重最高
        # Sync > 0.6 代表價量週期共振，這是真金白銀在推動
        if last['Sync_Smooth'] > 0.6: 
            score += 25
            reasons.append("🌊 物理共振 (Phase Sync > 0.6)")
        elif last['Sync_Smooth'] < -0.6:
            score -= 25
            reasons.append("⚠️ 結構背離 (Phase Divergence)")
            
        # B. 趨勢層 (EMA)
        if last['Close'] > last['EMA20']:
            score += 15
            reasons.append("📈 價格位於均線之上")
        else:
            score -= 15
            reasons.append("📉 價格位於均線之下")
            
        # C. 動能層 (RSI) - 根據狀態調整邏輯
        if "TRENDING" in regime:
            if last['RSI'] > 70: 
                score += 5 
                reasons.append("🚀 強勢鈍化 (RSI > 70)")
        elif "RANGING" in regime:
            if last['RSI'] > 70: 
                score -= 20
                reasons.append("🛑 震盪超買 (RSI > 70)")
            if last['RSI'] < 30: 
                score += 20
                reasons.append("🟢 震盪超賣 (RSI < 30)")
        
        # 邊界限制
        score = min(max(score, 0), 100)
        
        return {
            "score": score,
            "regime": regime,
            "regime_color": regime_color,
            "reasons": reasons,
            "last": last
        }

# =============================================================================
# 5. UI 主程式 (Frontend)
# =============================================================================
def main():
    st.sidebar.markdown("## 🛡️ NEO-LEVIATHAN")
    st.sidebar.caption("Causal Physics Trading Engine")
    st.sidebar.markdown("---")
    
    ticker = st.sidebar.text_input("輸入代碼 (Ticker)", "2330.TW")
    st.sidebar.caption("例如: 2330.TW, NVDA, BTC-USD")
    
    run_btn = st.sidebar.button("INITIALIZE SYSTEM", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 分析師筆記")
    st.sidebar.info(
        "此系統已啟用「因果濾波器」(Causal Filter)。\n"
        "訊號會有 3-5 天的物理延遲，這是正常的。\n"
        "請勿用於極短線交易。"
    )

    if run_btn:
        with st.spinner("Processing Signal Physics..."):
            # 1. 獲取數據
            raw_df = fetch_data(ticker)
            
            if raw_df.empty or len(raw_df) < 200:
                st.error("❌ 數據不足或下載失敗。至少需要 200 根 K 棒以供物理引擎運算。")
                return
            
            # 2. 特徵工程
            df = Signal_Processor.engineer_features(raw_df)
            
            # 3. 策略評估
            result = Strategy_Engine.evaluate(df)
            
            # --- 儀表板顯示 ---
            col_main, col_info = st.columns([2, 1])
            
            with col_main:
                st.markdown("### 📊 Market Physics Chart")
                
                # 決定訊號顏色與文字
                final_score = result['score']
                if final_score >= 70:
                    sig_color = "#3FB950" # 綠
                    action = "ACCUMULATE (做多)"
                elif final_score <= 30:
                    sig_color = "#F85149" # 紅
                    action = "DISTRIBUTE (減碼/空)"
                else:
                    sig_color = "#8B949E" # 灰
                    action = "HOLD / WATCH (觀望)"
                
                # 渲染訊號箱
                st.markdown(f"""
                <div class="signal-box" style="border-top: 4px solid {sig_color}">
                    <div style="color:#8B949E; font-size:14px; margin-bottom:5px">SYSTEM OUTPUT</div>
                    <div class="highlight-val" style="color:{sig_color}; font-size:42px">{action}</div>
                    <div style="margin-top:15px; font-size:18px">
                        Confidence Score: <span style="color:#E6EDF3; font-weight:bold">{final_score:.0f}</span> / 100
                    </div>
                    <div style="color:{result['regime_color']}; font-size:14px; margin-top:5px">
                        Regime: {result['regime']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 繪製圖表
                fig = make_subplots(
                    rows=3, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=("Price Action", "Phase Sync (Physics)", "Trend Strength (ADX)")
                )
                
                # Row 1: Price & EMA
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                    name='Price'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA20'], 
                    line=dict(color='#FFAE00', width=1.5), name='EMA 20'
                ), row=1, col=1)
                
                # Row 2: Sync (Physics)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Sync_Smooth'], 
                    line=dict(color='#D2A8FF', width=2), name='Phase Sync'
                ), row=2, col=1)
                # 繪製共振區域
                fig.add_hrect(y0=0.6, y1=1.1, row=2, col=1, fillcolor="#3FB950", opacity=0.1, line_width=0)
                fig.add_hline(y=0, line_dash="dot", row=2, col=1, line_color="#555")
                
                # Row 3: ADX (Regime)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['ADX'], 
                    line=dict(color='#E6EDF3', width=1), name='ADX', fill='tozeroy', fillcolor='rgba(230, 237, 243, 0.1)'
                ), row=3, col=1)
                fig.add_hline(y=25, line_dash="dot", row=3, col=1, line_color="#F85149", annotation_text="Trend")
                fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="#8B949E", annotation_text="Range")
                
                # Layout Config
                fig.update_layout(
                    template="plotly_dark", 
                    height=800, 
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13,17,23,0.5)',
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_info:
                st.markdown("### 🧬 Logic Decode")
                
                # 基本數據
                last = result['last']
                curr_price = last['Close']
                
                m1, m2 = st.columns(2)
                m1.markdown(f"<div class='metric-card'><div class='highlight-lbl'>PRICE</div><div class='highlight-val'>${curr_price:,.2f}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='highlight-lbl'>VOLUME</div><div class='highlight-val'>{int(last['Volume']/1000):,}K</div></div>", unsafe_allow_html=True)
                
                # 關鍵指標
                st.markdown("#### Core Metrics")
                k1, k2, k3 = st.columns(3)
                k1.metric("Sync (Physics)", f"{last['Sync_Smooth']:.2f}", delta_color="off")
                k2.metric("ADX (Trend)", f"{last['ADX']:.1f}")
                k3.metric("RSI (Mom)", f"{last['RSI']:.0f}")
                
                st.markdown("---")
                st.markdown("#### 🎯 Decision Factors")
                if not result['reasons']:
                    st.info("無顯著訊號，建議觀望。")
                else:
                    for r in result['reasons']:
                        st.success(r) if "共振" in r or "強勢" in r or "之上" in r or "超賣" in r else st.warning(r)

if __name__ == "__main__":
    main()
