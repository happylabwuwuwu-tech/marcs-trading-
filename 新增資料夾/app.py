import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import io
import base64

# --- 設定頁面 ---
st.set_page_config(page_title="MARCS V34.2 交易終端", layout="wide", page_icon="🛡️")

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

        return {
            'price': close[-1], 'trend': trend, 'imf1': imf1,
            'sync': sync, 'd_alpha': d_alpha, 'sl': sl, 'sr': sr, 'df': self.df,
            'noise_std': noise_std
        }

def generate_plots(res):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12), facecolor='#0d1117')
    gs = fig.add_gridspec(4, 2)

    ax1 = fig.add_subplot(gs[0, :]); ax1.set_facecolor('#0d1117')
    ax1.plot(res['trend'][-60:], color='#58a6ff', lw=3)
    ax1.set_title("1. CEEMD Trend: 機構資金主趨勢", color='#58a6ff', loc='left')

    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor('#0d1117')
    sync_data = res['sync'][-30:]
    colors = ['#3fb950' if s > 0 else '#f85149' for s in sync_data]
    ax2.bar(range(30), sync_data, color=colors)
    ax2.set_title("2. WCA Sync: 動能同步狀態", color='#3fb950', loc='left')

    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor('#0d1117')
    x = np.linspace(0, 1, 100); y = -(x-0.5)**2 + res['d_alpha']
    ax3.plot(x, y, color='#a371f7', lw=3); ax3.fill_between(x, y, color='#a371f7', alpha=0.2)
    ax3.set_title(f"3. MF Risk: {res['d_alpha']:.2f}", color='#a371f7', loc='left')

    ax4 = fig.add_subplot(gs[2:, :]); ax4.set_facecolor('#0d1117')
    df_p = res['df'].tail(60)
    ax4.plot(df_p.index, df_p['Close'], color='#00f2ff', lw=2)
    ax4.axhline(res['sl'], color='#f85149', ls='--', lw=2)
    ax4.fill_between(df_p.index, res['sl'], res['price'], color='#f85149', alpha=0.05)
    ax4.set_title("4. Action Boundary: 實務止損線", color='#00f2ff', loc='left')

    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor='#0d1117'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def main():
    st.sidebar.header("🛠️ 參數設定")
    ticker_input = st.sidebar.text_input("股票代碼", value="2330.TW")
    if st.sidebar.button("執行診斷", type="primary"):
        with st.spinner('計算中...'):
            engine = MARCS_V34_2_Engine(ticker_input.upper())
            res = engine.get_features()
            if res:
                img = generate_plots(res)
                sync_val = res['sync'][-1]
                action = "✅ 持有 / 追蹤" if sync_val > 0 else "⚠️ 減碼 / 觀望"
                html = f"""
                <div style="background:#0d1117; color:#c9d1d9; padding:20px; border-radius:10px;">
                    <h2 style="color:#ff7b72;">🚀 {ticker_input} 診斷報告</h2>
                    <div style="font-size:18px; margin-bottom:15px; border-left:4px solid #3fb950; padding-left:10px;">
                        決策建議：<b>{action}</b> (SL: {res['sl']:.2f})
                    </div>
                    <img src="data:image/png;base64,{img}" style="width:100%">
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.error("查無資料")

if __name__ == "__main__":
    main()
