import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import io
import base64
import time
import os

# --- 1. 頁面與 CSS 設定 ---
st.set_page_config(
    page_title="MARCS Pro Terminal",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- 影片讀取小工具 ---
def get_video_base64(file_name):
    """讀取本地影片並轉為 Base64，若找不到檔案則回傳 None"""
    if os.path.exists(file_name):
        try:
            with open(file_name, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode()
            return f"data:video/mp4;base64,{encoded}"
        except Exception as e:
            st.warning(f"影片讀取錯誤: {e}")
            return None
    return None

def load_tech_ui():
    # ==============================
    # 🎥 影片設定區
    # ==============================
    
    # 1. 背景影片
    my_bg_file = "background.mp4"  
    default_bg = "https://cdn.pixabay.com/video/2020/04/18/36465-412239632_large.mp4"

    # 2. 架構影片
    my_arch_file = "model_arch.mp4"
    default_arch = "https://cdn.pixabay.com/video/2016/09/21/5398-183786499_tiny.mp4"

    # 自動偵測邏輯
    local_bg = get_video_base64(my_bg_file)
    bg_url = local_bg if local_bg else default_bg

    local_arch = get_video_base64(my_arch_file)
    arch_url = local_arch if local_arch else default_arch
    
    st.session_state['arch_video_url'] = arch_url

    # --- CSS 樣式注入 ---
    st.markdown(f"""
        <style>
        /* 背景影片 */
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%; 
            min-height: 100%;
            z-index: -1;
            opacity: 0.4;
            filter: hue-rotate(180deg) contrast(1.2);
            object-fit: cover;
        }}
        
        .stApp {{
            background: transparent;
            font-family: 'Roboto Mono', monospace, sans-serif;
        }}
        
        /* 毛玻璃卡片 */
        .metric-card {{
            background: rgba(13, 17, 23, 0.75);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(88, 166, 255, 0.3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px) scale(1.02);
            border-color: #00f2ff;
            box-shadow: 0 0 25px rgba(0, 242, 255, 0.4);
        }}

        /* 側邊欄 */
        [data-testid="stSidebar"] {{
            background-color: rgba(22, 27, 34, 0.9);
            border-right: 1px solid rgba(48, 54, 61, 0.8);
            backdrop-filter: blur(5px);
        }}

        /* 建議文字 */
        .metric-advice {{
            margin-top: 10px;
            padding: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 12px;
            color: #c9d1d9;
            background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
            border-radius: 4px;
            text-align: left;
        }}

        /* 標題 */
        .tech-header {{
            display: flex; 
            align-items: center; 
            background: rgba(13, 17, 23, 0.6); 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #00f2ff;
            backdrop-filter: blur(5px);
            margin-bottom: 20px;
        }}

        /* 左下角影片容器 */
        .arch-video-container {{
            border: 1px solid #00f2ff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0, 242, 255, 0.2);
            margin-top: 10px;
            position: relative;
        }}
        
        .scan-line {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: rgba(0, 242, 255, 0.5);
            animation: scan 3s linear infinite;
            z-index: 2;
        }}
        
        @keyframes scan {{
            0% {{ top: 0%; }}
            100% {{ top: 100%; }}
        }}
        </style>
        
        <video autoplay muted loop id="myVideo">
            <source src="{bg_url}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

load_tech_ui()

# --- 核心引擎 ---
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

# --- 繪圖功能 ---
def generate_plots(res):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14), facecolor='none') 
    gs = fig.add_gridspec(4, 2)
    chart_bg = '#0d1117' 
    alpha_val = 0.7
    
    ax1 = fig.add_subplot(gs[0, :]); ax1.set_facecolor(chart_bg); ax1.patch.set_alpha(alpha_val)
    ax1.plot(res['trend'][-60:], color='#00f2ff', lw=3,  path_effects=[])
    ax1.set_title("1. CEEMD Trend: 機構資金主趨勢", color='#00f2ff', loc='left', fontsize=14)
    ax1.grid(True, color='#30363d', linestyle='--', linewidth=0.5)
    
    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor(chart_bg); ax2.patch.set_alpha(alpha_val)
    sync_data = res['sync'][-30:]
    colors = ['#3fb950' if s > 0 else '#f85149' for s in sync_data]
    ax2.bar(range(30), sync_data, color=colors, alpha=0.9)
    ax2.set_title("2. WCA Sync: 動能同步狀態", color='#3fb950', loc='left', fontsize=14)
    ax2.grid(True, color='#30363d', linestyle='--', linewidth=0.5)
    
    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor(chart_bg); ax3.patch.set_alpha(alpha_val)
    x = np.linspace(0, 1, 100); y = -(x-0.5)**2 + res['d_alpha']
    ax3.plot(x, y, color='#a371f7', lw=3); ax3.fill_between(x, y, color='#a371f7', alpha=0.2)
    ax3.set_title(f"3. MF Risk: {res['d_alpha']:.2f} (市場複雜度)", color='#a371f7', loc='left', fontsize=14)
    ax3.grid(True, color='#30363d', linestyle='--', linewidth=0.5)
    
    ax4 = fig.add_subplot(gs[2:, :]); ax4.set_facecolor(chart_bg); ax4.patch.set_alpha(alpha_val)
    df_p = res['df'].tail(60)
    ax4.plot(df_p.index, df_p['Close'], color='#e6edf3', lw=2, label='Price')
    ax4.axhline(res['sl'], color='#f85149', ls='--', lw=2, label=f'SL: {res["sl"]:.1f}')
    ax4.fill_between(df_p.index, res['sl'], res['price'], color='#f85149', alpha=0.1)
    ax4.set_title("4. Action Boundary: 實務執行邊界", color='#e6edf3', loc='left', fontsize=14)
    ax4.legend(loc='upper left', frameon=False)
    ax4.grid(True, color='#30363d', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor='none', transparent=True); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 主 UI 邏輯 ---
def main():
    with st.sidebar:
        st.markdown("<h1 style='color:#00f2ff; text-shadow: 0 0 10px #00f2ff;'>🛡️ MARCS Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8b949e; font-size:12px;'>SYSTEM V34.2 // ONLINE</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("🛠️ CONTROL PANEL")
        ticker_input = st.text_input("TARGET TICKER", value="2330.TW")
        run_btn = st.button("INITIATE SCAN ⚡", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("<h4 style='color:#00f2ff; margin-bottom:5px; font-size:14px;'>🏗️ SYSTEM ARCHITECTURE</h4>", unsafe_allow_html=True)
        
        arch_url = st.session_state.get('arch_video_url', "")
        
        if arch_url:
            st.markdown(f"""
                <div class="arch-video-container">
                    <div class="scan-line"></div>
                    <video autoplay muted loop style="width:100%; opacity:0.8;">
                        <source src="{arch_url}" type="video/mp4">
                    </video>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
            <div style="font-size:10px; color:#58a6ff; text-align:right; margin-top:5px;">
                >> MODEL VISUALIZATION LIVE
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="tech-header">
            <div>
                <h1 style='color:#ffffff; margin:0; text-shadow: 0 0 5px rgba(255,255,255,0.5);'>MARKET RECONNAISSANCE</h1>
                <div style='color:#00f2ff; font-size:14px; letter-spacing:2px;'>QUANTITATIVE PHYSICS ENGINE ENGAGED</div>
            </div>
            <div style="flex:1;"></div>
            <div style="text-align:right;">
                <span style='background:rgba(31, 111, 235, 0.3); color:#58a6ff; padding: 5px 15px; border:1px solid #1f6feb; border-radius: 4px; font-family:"Roboto Mono";'>TARGET: {ticker_input.upper()}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if run_btn:
        progress_text = "Establishing Secure Link to Market Data..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text="Analyzing Quantum Fluctuations..." if percent_complete > 50 else progress_text)
        my_bar.empty()

        engine = MARCS_V34_2_Engine(ticker_input.upper())
        res = engine.get_features()

        if res:
            sync_val = res['sync'][-1]
            is_sync = sync_val > 0
            sync_color = '#3fb950' if is_sync else '#f85149'
            sync_status = 'SYNC ESTABLISHED' if is_sync else 'DIVERGENCE DETECTED'
            sync_advice = "ENERGY FLOW STABLE. HOLD POSITION." if is_sync else "WARNING: ENERGY LEAK. REDUCE EXPOSURE."

            risk_val = res['d_alpha']
            is_stable = risk_val < 0.15
            risk_color = '#a371f7' if is_stable else '#d2a8ff'
            risk_status = 'STRUCTURE STABLE' if is_stable else 'CHAOS DETECTED'
            risk_advice = "LOW ENTROPY. SYSTEM GREEN." if is_stable else "HIGH VOLATILITY. SYSTEM ALERT."

            sl_price = res['sl']
            sl_dist = (1 - sl_price/res['price']) * 100
            sl_advice = f"CRITICAL BOUNDARY: -{sl_dist:.1f}% DELTA"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid {sync_color};">
                        <div class="metric-label" style="color:{sync_color}">WCA PHASE</div>
                        <div class="metric-value" style="color:#ffffff; text-shadow: 0 0 10px {sync_color};">{sync_val:.2f}</div>
                        <div style="font-weight:bold; color:{sync_color}; font-size:14px;">[{sync_status}]</div>
                        <div class="metric-advice">>> {sync_advice}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid {risk_color};">
                        <div class="metric-label" style="color:{risk_color}">MFA ENTROPY</div>
                        <div class="metric-value" style="color:#ffffff; text-shadow: 0 0 10px {risk_color};">{risk_val:.2f}</div>
                        <div style="font-weight:bold; color:{risk_color}; font-size:14px;">[{risk_status}]</div>
                        <div class="metric-advice">>> {risk_advice}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid #f85149;">
                        <div class="metric-label" style="color:#f85149">TIGHT STOP-LOSS</div>
                        <div class="metric-value" style="color:#ffffff; text-shadow: 0 0 10px #f85149;">{sl_price:.2f}</div>
                         <div style="font-weight:bold; color:#f85149; font-size:14px;">DIST: -{sl_dist:.1f}%</div>
                        <div class="metric-advice">>> {sl_advice}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("") 

            tab_plots, tab_report = st.tabs(["📊 VISUALIZATION", "📋 TACTICAL REPORT"])

            with tab_plots:
                img = generate_plots(res)
                st.markdown(f"""
                    <div style="background: rgba(13, 17, 23, 0.8); backdrop-filter: blur(10px); border-radius: 12px; overflow: hidden; border: 1px solid #30363d;">
                        <img src="data:image/png;base64,{img}" style="width:100%;">
                    </div>
                """, unsafe_allow_html=True)

            with tab_report:
                action_main = "MAINTAIN POSITIONS" if is_sync else "ABORT / REDUCE SIZE"
                main_color = sync_color
                
                html_report = f"""
                <div style="background:rgba(22, 27, 34, 0.85); backdrop-filter: blur(10px); color:#c9d1d9; padding:30px; border-radius:12px; border:1px solid #30363d;">
                    <h2 style="margin-top:0; color:{main_color}; border-bottom:1px dashed {main_color}; padding-bottom:10px;">
                        COMMAND DECISION
                    </h2>
                    <div style="margin: 20px 0;">
                        <div style="font-size: 28px; font-weight: bold; color: {main_color}; text-shadow: 0 0 15px {main_color};">
                            [{action_main}]
                        </div>
                        <div style="font-family: 'Roboto Mono'; margin-top:10px;">
                            >> SIGNAL INTEGRITY: {'100%' if is_sync else 'UNSTABLE'}<br>
                            >> RISK PROTOCOL: {'STANDARD' if is_stable else 'DEFENSIVE'}
                        </div>
                    </div>
                    <table style="width:100%; border-collapse: collapse; margin-top: 20px; font-family:'Roboto Mono';">
                        <tr style="border-bottom: 1px solid #30363d;">
                            <td style="padding: 10px; color:#8b949e;">CURRENT_PRICE</td>
                            <td style="padding: 10px; text-align:right;">{res['price']:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #30363d;">
                            <td style="padding: 10px; color:#f85149;">STOP_LOSS_LEVEL</td>
                            <td style="padding: 10px; text-align:right; color:#f85149; font-weight:bold;">{res['sl']:.2f}</td>
                        </tr>
                         <tr>
                            <td style="padding: 10px; color:#3fb950;">TARGET_RESISTANCE</td>
                            <td style="padding: 10px; text-align:right; color:#3fb950;">{res['sr']:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
                st.markdown(html_report, unsafe_allow_html=True)

        else:
            st.error(f"❌ CONNECTION FAILED: UNABLE TO RETRIEVE DATA FOR {ticker_input}")

if __name__ == "__main__":
    main()
