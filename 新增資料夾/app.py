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
    # 🎥 影片設定區 (在此更換你的影片)
    # ==============================
    
    # 1. 設定背景影片檔名 (請把你的 mp4 上傳並改成這個名字，或是直接修改這裡)
    my_bg_file = "background.mp4"  
    # 預設背景 (藍色粒子)
    default_bg = "https://cdn.pixabay.com/video/2020/04/18/36465-412239632_large.mp4"

    # 2. 設定左下角架構影片檔名
    my_arch_file = "model_arch.mp4"
    # 預設架構 (3D 網格)
    default_arch = "https://cdn.pixabay.com/video/2016/09/21/5398-183786499_tiny.mp4"

    # --- 自動偵測邏輯 ---
    # 如果找得到本地檔案就用本地的，找不到就用預設網址
    local_bg = get_video_base64(my_bg_file)
    bg_url = local_bg if local_bg else default_bg

    local_arch = get_video_base64(my_arch_file)
    arch_url = local_arch if local_arch else default_arch
    
    # 儲存到 session_state 傳給 main 使用
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

# --- 核心引擎 (完全保留) ---
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
    ax2.grid(True, color='#303
