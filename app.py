import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import io
import base64

# --- 1. 頁面與 CSS 設定 ---
st.set_page_config(
    page_title="MARCS Pro Terminal",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            font-family: 'Roboto Mono', monospace, sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        /* 優化後的卡片樣式 */
        .metric-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%; /* 讓卡片等高 */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin: 5px 0;
        }
        .metric-label {
            color: #8b949e;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        /* 新增：實務建議專用樣式 */
        .metric-advice {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #30363d;
            font-size: 13px;
            color: #c9d1d9;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
            padding: 8px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
		    gap: 10px;
            background-color: #0e1117;
	    }
	    .stTabs [data-baseweb="tab"] {
		    height: 45px;
            border-radius: 8px;
            color: #c9d1d9;
            font-weight: 600;
	    }
        .stTabs [aria-selected="true"] {
            background-color: #1f6feb !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --- 2. 核心引擎 (無更動) ---
class MARCS_V34_2_Engine:
    def __init__(self, ticker, period='1y'):
        self.ticker = ticker
        try:
            self.df = yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=True)
        except Exception:
            self.df = pd.DataFrame()

    def get_features(self):
        if self.df.empty or len(self.df) < 60: return None
        close = self.df['Close'].values.flatten()
        vol = self.df['Volume'].values.flatten()
        local_mean = pd.Series(close).rolling(window=5, center=True).mean().bfill().ffill().values
        imf1 = close - local_mean
        trend = pd.Series(close).rolling(window=20, center=True).mean().bfill().ffill().values
        al_p = hilbert(imf1)
        al_v = hilbert(vol - np.mean(vol))
        sync = np.cos(np.angle(al_p) - np.angle(al_v))
        returns = np.diff(np.log(close))
        d_alpha = np.std(returns[-20:]) * 15 if len(returns) > 20 else 0
        noise_std = np.std(imf1[-15:])
        sl = close[-1] - (1.8 * noise_std)
        sr = close[-1] + (2.2 * noise_std)
        return {'price': close[-1], 'trend': trend, 'imf1': imf1, 'sync': sync, 'd_alpha': d_alpha, 'sl': sl, 'sr': sr, 'df': self.df, 'noise_std': noise_std}

# --- 3. 繪圖功能 ---
def generate_plots(res):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14), facecolor='#0d1117')
    gs = fig.add_gridspec(4, 2)
    
    ax1 = fig.add_subplot(gs[0, :]); ax1.set_facecolor('#0d1117')
    ax1.plot(res['trend'][-60:], color='#58a6ff', lw=3)
    ax1.set_title("1. CEEMD Trend: 機構資金主趨勢", color='#58a6ff', loc='left', fontsize=14)
    
    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor('#0d1117')
    sync_data = res['sync'][-30:]
    colors = ['#3fb950' if s > 0 else '#f85149' for s in sync_data]
    ax2.bar(range(30), sync_data, color=colors)
    ax2.set_title("2. WCA Sync: 動能同步狀態", color='#3fb950', loc='left', fontsize=14)
    
    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor('#0d1117')
    x = np.linspace(0, 1, 100); y = -(x-0.5)**2 + res['d_alpha']
    ax3.plot(x, y, color='#a371f7', lw=3); ax3.fill_between(x, y, color='#a371f7', alpha=0.2)
    ax3.set_title(f"3. MF Risk: {res['d_alpha']:.2f} (市場複雜度)", color='#a371f7', loc='left', fontsize=14)
    
    ax4 = fig.add_subplot(gs[2:, :]); ax4.set_facecolor('#0d1117')
    df_p = res['df'].tail(60)
    ax4.plot(df_p.index, df_p['Close'], color='#00f2ff', lw=2, label='Price')
    ax4.axhline(res['sl'], color='#f85149', ls='--', lw=2, label=f'SL: {res["sl"]:.1f}')
    ax4.fill_between(df_p.index, res['sl'], res['price'], color='#f85149', alpha=0.05)
    ax4.set_title("4. Action Boundary: 實務執行邊界", color='#00f2ff', loc='left', fontsize=14)
    ax4.legend(loc='upper left', frameon=False)
    
    plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor='#0d1117'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 4. 主 UI 邏輯 ---
def main():
    with st.sidebar:
        st.markdown("<h1 style='color:#58a6ff; margin-bottom:0;'>🛡️ MARCS Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8b949e; font-size:12px;'>物理特徵交易終端 V34.2</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("🛠️ 參數設定")
        ticker_input = st.text_input("股票代碼", value="2330.TW", help="例如: 2330.TW, NVDA")
        run_btn = st.button("⚡ 執行實務診斷", type="primary", use_container_width=True)

    st.markdown(f"""
        <div>
            <h1 style='color:#c9d1d9; display: inline-block;'>前哨站主控台</h1>
            <span style='background:#1f6feb; color:white; padding: 4px 12px; border-radius: 20px; font-size: 14px; vertical-align: middle; margin-left: 10px;'>Target: {ticker_input.upper()}</span>
        </div>
    """, unsafe_allow_html=True)

    if run_btn:
        with st.spinner('🔄 正在運算物理特徵...'):
            engine = MARCS_V34_2_Engine(ticker_input.upper())
            res = engine.get_features()

            if res:
                # --- A. 準備邏輯與建議文字 ---
                
                # 1. WCA 同步邏輯
                sync_val = res['sync'][-1]
                is_sync = sync_val > 0
                sync_color = '#3fb950' if is_sync else '#f85149'
                sync_status = '✅ 能量同步' if is_sync else '⚠️ 能量背離'
                # 這裡定義 WCA 的實務建議
                sync_advice = "能量健康，趨勢有支撐，適合續抱。" if is_sync else "小心虛假突破，上漲無量，不宜追高加碼。"

                # 2. MFA 風險邏輯
                risk_val = res['d_alpha']
                is_stable = risk_val < 0.15
                risk_color = '#a371f7' if is_stable else '#d2a8ff'
                risk_status = '穩定結構' if is_stable else '波動劇增'
                # 這裡定義 MFA 的實務建議
                risk_advice = "市場結構穩定，可按標準倉位操作。" if is_stable else "混沌風險升高，建議降低槓桿或縮小部位。"

                # 3. SL 止損邏輯
                sl_price = res['sl']
                sl_dist = (1 - sl_price/res['price']) * 100
                # 這裡定義 SL 的實務建議
                sl_advice = f"距離現價 {sl_dist:.1f}%，觸價應果斷離場，嚴守紀律。"

                # --- B. 顯示卡片 (包含新的建議區塊) ---
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card" style="border-bottom: 4px solid {sync_color};">
                            <div class="metric-label">WCA 價量相位</div>
                            <div class="metric-value" style="color:{sync_color}">{sync_val:.2f}</div>
                            <div style="font-weight:bold; color:{sync_color}; margin-bottom:5px;">{sync_status}</div>
                            <div class="metric-advice">💡 {sync_advice}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card" style="border-bottom: 4px solid {risk_color};">
                            <div class="metric-label">MFA 市場複雜度</div>
                            <div class="metric-value" style="color:{risk_color}">{risk_val:.2f}</div>
                            <div style="font-weight:bold; color:{risk_color}; margin-bottom:5px;">{risk_status}</div>
                            <div class="metric-advice">💡 {risk_advice}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card" style="border-bottom: 4px solid #f85149;">
                            <div class="metric-label">Tight SL 實務止損</div>
                            <div class="metric-value" style="color:#f85149">{sl_price:.2f}</div>
                             <div style="font-weight:bold; color:#f85149; margin-bottom:5px;">距離 -{sl_dist:.1f}%</div>
                            <div class="metric-advice">💡 {sl_advice}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.write("") 

                # --- C. 分頁內容 ---
                tab_plots, tab_report = st.tabs(["📊 核心圖譜", "📋 詳細決策報告"])

                with tab_plots:
                    img = generate_plots(res)
                    st.markdown(f"""
                        <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid #30363d;">
                            <img src="data:image/png;base64,{img}" style="width:100%;">
                        </div>
                    """, unsafe_allow_html=True)

                with tab_report:
                    action_main = "✅ 持有 / 追蹤" if is_sync else "⚠️ 減碼 / 嚴禁追高"
                    main_color = sync_color
                    
                    html_report = f"""
                    <div style="background:#161b22; color:#c9d1d9; padding:30px; border-radius:12px; border:1px solid #30363d;">
                        <h2 style="margin-top:0; color:{main_color};">🚀 最終綜合決策</h2>
                        <div style="background: {main_color}22; padding: 20px; border-left: 6px solid {main_color}; border-radius: 4px; margin-bottom: 25px;">
                            <div style="font-size: 24px; font-weight: bold; color: {main_color}; margin-bottom: 10px;">{action_main}</div>
                            <div style="font-size: 16px;">{sync_advice} {risk_advice}</div>
                        </div>
                        <h3 style="color:#8b949e; border-bottom:1px solid #30363d; padding-bottom:10px;">數據細節</h3>
                        <table style="width:100%; border-collapse: collapse; margin-top: 15px;">
                            <tr style="border-bottom: 1px solid #30363d;">
                                <td style="padding: 10px; color:#8b949e;">當前價格</td>
                                <td style="padding: 10px; text-align:right;">{res['price']:.2f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #30363d;">
                                <td style="padding: 10px; color:#f85149;">止損價位 (SL)</td>
                                <td style="padding: 10px; text-align:right; color:#f85149; font-weight:bold;">{res['sl']:.2f}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    st.markdown(html_report, unsafe_allow_html=True)

            else:
                st.error(f"❌ 無法獲取 {ticker_input} 的數據。")

if __name__ == "__main__":
    main()
