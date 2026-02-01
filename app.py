import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import warnings
import os
import random
import concurrent.futures
from scipy.stats import wasserstein_distance

# 過濾警告
warnings.filterwarnings('ignore')

# 設定網頁配置
st.set_page_config(
    page_title="MARCS V64 全域戰情室",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# CSS 美化 (黑金極速版)
st.markdown("""
<style>
    .stApp {background-color: #000000;}
    .metric-card {
        background: rgba(20, 20, 20, 0.9);
        border: 1px solid #333;
        border-radius: 8px; padding: 15px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    .metric-value {color: #ffffff; font-size: 22px; font-weight: bold; font-family: 'Courier New';}
    .metric-label {color: #aaaaaa; font-size: 12px; text-transform: uppercase;}
    .stProgress > div > div > div > div { background-color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 資料庫與強力爬蟲
# =============================================================================
class Global_Market_Loader:
    @staticmethod
    def get_indices():
        return {
            "^VIX": {"name": "VIX 恐慌", "type": "Sentiment"},
            "DX-Y.NYB": {"name": "DXY 美元", "type": "Currency"},
            "TLT": {"name": "TLT 美債", "type": "Rates"},
            "JPY=X": {"name": "JPY 日圓", "type": "Currency"}
        }

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_tw_full_market():
        """
        [修復版] 強力爬取台股上市+上櫃完整清單
        """
        tickers = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            # 模式 2=上市, 4=上櫃
            for mode, suffix in [(2, '.TW'), (4, '.TWO')]:
                url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
                res = requests.get(url, headers=headers, timeout=10)
                
                # 簡單解析：尋找所有 4 位數代碼
                df = pd.read_html(res.text)[0]
                
                # 針對證交所表格特性進行過濾 (第0欄通常是 "有價證券代號及名稱")
                # 我們把這一欄轉成字串，並尋找 "四個數字" 開頭的
                raw_col = df.iloc[:, 0].astype(str)
                for item in raw_col:
                    parts = item.split() # 用空白切割 "2330 台積電" -> ["2330", "台積電"]
                    if len(parts) > 0:
                        code = parts[0]
                        # 嚴格篩選：必須是4位數字 (過濾掉權證、債券)
                        if len(code) == 4 and code.isdigit():
                            tickers.append(f"{code}{suffix}")
            
            # 如果爬到的太少，代表爬蟲失敗，觸發 Exception
            if len(tickers) < 100: raise Exception("Crawl result too small")
            
            return tickers
            
        except Exception as e:
            # [備用方案] 內建 300 檔涵蓋各產業龍頭與熱門股，確保永遠有東西跑
            fallback = [
                "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2603.TW", "2382.TW", "3231.TW", "2881.TW", "2882.TW",
                "1519.TW", "3035.TWO", "8069.TWO", "3293.TWO", "2376.TW", "2356.TW", "3017.TW", "3044.TW", "2308.TW",
                "2412.TW", "1301.TW", "1303.TW", "2002.TW", "2891.TW", "5871.TW", "2886.TW", "2884.TW", "1216.TW",
                "2912.TW", "1101.TW", "2609.TW", "2615.TW", "2344.TW", "2301.TW", "2327.TW", "2357.TW", "2353.TW",
                "3034.TW", "3037.TW", "2379.TW", "2408.TW", "3443.TW", "3661.TW", "6669.TW", "6515.TW", "5269.TW",
                "2383.TW", "6278.TW", "6214.TW", "6415.TW", "6147.TWO", "3529.TWO", "5347.TWO", "6182.TWO"
            ]
            # 這裡為了演示，我們就只列這些，實戰中這份列表可以更長
            return fallback

    @staticmethod
    def get_scan_list(market_type, limit=300):
        if "台股" in market_type:
            full_list = Global_Market_Loader.get_tw_full_market()
            # 隨機打亂以避免每次都只掃代號小的
            random.shuffle(full_list)
            return full_list[:limit]
        
        elif "美股" in market_type:
            return ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOG", "AMZN", "META", "SMCI", "PLTR", "COIN", "MSTR", "ARM", "AVGO", "QCOM", "INTC", "TSM", "SOXL", "TQQQ", "MRVL", "MU", "DELL", "SMH"]
        
        elif "加密貨幣" in market_type:
            return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "SHIB-USD", "PEPE-USD", "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD"]
        
        elif "貴金屬" in market_type:
            return ["GC=F", "SI=F", "HG=F", "CL=F", "PL=F", "NG=F", "PA=F"]
            
        return []

# =============================================================================
# 2. 宏觀與選股引擎 (支援多執行緒)
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
            try: curr_w2 = wasserstein_distance(returns.tail(20), returns.iloc[-40:-20])
            except: curr_w2 = 0.5
            
            hist_std = returns.rolling(40).std().mean() * 0.1
            chaos = curr_w2 / (hist_std + 1e-9)
            
            trend = "Neutral"
            if rsi > 70: trend = "Overbought"
            elif rsi < 30: trend = "Oversold"
            
            return {"ticker": ticker, "name": name, "price": c.iloc[-1], "rsi": rsi, "chaos": chaos, "trend": trend}
        except: return None

    @staticmethod
    def calculate_mmi(results):
        score = 50.0
        data_map = {r['ticker']: r for r in results if r}
        vix = data_map.get('^VIX')
        if vix: score += 15 if vix['trend']=='Overbought' else (-15 if vix['trend']=='Oversold' else 0)
        dxy = data_map.get('DX-Y.NYB')
        if dxy: score -= 12 if dxy['trend']=='Overbought' else (12 if dxy['trend']=='Oversold' else 0)
        return min(100, max(0, score))

class Scanner_Engine_V38:
    @staticmethod
    def analyze_single(ticker, min_score=60):
        """
        單一標的分析函數 (供多執行緒呼叫)
        """
        try:
            # 只下載必要數據，加快速度
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if 'Adj Close' in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
            c = df['Close']; v = df['Volume']
            if len(v) > 0 and v.iloc[-1] == 0: return None # 濾掉無量
            
            # 動能邏輯
            ma20 = c.rolling(20).mean().iloc[-1]
            ma60 = c.rolling(60).mean().iloc[-1]
            if not (c.iloc[-1] > ma20 > ma60): return None # 必須多頭
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            score = 40
            if 55 <= rsi <= 75: score += 20
            elif rsi > 75: score += 10
            
            vol_ma5 = v.rolling(5).mean().iloc[-1]
            if v.iloc[-1] > vol_ma5 * 1.3: score += 15
            
            tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl = max(c.iloc[-1] - 2.5 * atr, ma20 * 0.98)
            
            # 過濾低分
            if score < min_score: return None
            
            return {"ticker": ticker, "price": c.iloc[-1], "score": score, "rsi": rsi, "sl": sl}
        except: return None

# =============================================================================
# 3. 微觀與風控 (Step 3)
# =============================================================================
class Micro_Structure_Engine:
    @staticmethod
    def analyze(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty: return 50, [], df
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            score = 50; signals = []
            
            ema20 = c.ewm(span=20).mean()
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr10 = tr.rolling(10).mean()
            k_upper = ema20 + 2.0 * atr10
            k_lower = ema20 - 2.0 * atr10
            
            if c.iloc[-1] > k_upper.iloc[-1]: score += 15; signals.append("Keltner突破")
            if c.iloc[-1] > c.iloc[-2] * 1.015: score += 5; signals.append("強勢紅K")
            
            df['K_Upper'] = k_upper; df['K_Lower'] = k_lower
            return min(100, max(0, score)), signals, df
        except: return 50, [], pd.DataFrame()

class Antifragile_Position_Sizing:
    @staticmethod
    def calculate(capital, price, sl, ticker):
        if any(x in ticker for x in ["-USD", "BTC", "ETH"]): 
            vol_cap = 1.0; asset_type = "Crypto (高波)"
        elif "=F" in ticker: 
            vol_cap = 0.4; asset_type = "Metal (保守)"
        elif any(x in ticker for x in [".TW", ".TWO"]): 
            vol_cap = 0.5; asset_type = "TW Stock (標準)"
        else: 
            vol_cap = 0.6; asset_type = "US Stock (積極)"

        risk_per_trade = capital * 0.02
        risk_per_share = price - sl
        if risk_per_share <= 0: return 0, {}
        
        base_size = risk_per_trade / risk_per_share
        
        taleb_multiplier = 1.0
        # 假設宏觀 Chaos
        chaos_level = 0.6 
        if chaos_level > 1.2: taleb_multiplier = 1 / (1 + np.exp(chaos_level - 1.0))
        
        vol_adj = 0.5 if vol_cap > 0.8 else 1.0
        final_size = base_size * taleb_multiplier * vol_adj
        
        if vol_cap > 0.8: final_size = round(final_size, 4)
        else: final_size = int(final_size)
            
        final_capital = final_size * price
        
        return final_size, {
            "risk_money": int(risk_per_trade), 
            "taleb_factor": round(taleb_multiplier, 2),
            "final_capital": int(final_capital),
            "asset_type": asset_type
        }

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ 戰情控制台")
    capital = st.sidebar.number_input("總本金 (USD/TWD)", value=1000000, step=100000)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 被動輸入 (Manual)")
    manual_ticker = st.sidebar.text_input("輸入代碼 (如 2330.TW, NVDA)", value="").upper().strip()
    run_manual = st.sidebar.button("分析單一標的")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 主動掃描 (Scanner)")
    market_select = st.sidebar.radio("選擇戰場:", ["🇹🇼 台股 (全市場)", "🇺🇸 美股 (科技)", "₿ 加密貨幣", "🥇 貴金屬"])
    
    # 掃描限制
    scan_limit = 300
    if "台股" in market_select:
        st.sidebar.caption("提示: 台股掃描全市場較慢，建議限制數量")
        scan_limit = st.sidebar.slider("掃描數量上限", 100, 2000, 300, step=100)
    
    run_scan = st.sidebar.button(f"啟動 {market_select} 掃描")
    
    # Video
    st.sidebar.markdown("---")
    video_file = "demo.mp4"
    if os.path.exists(video_file): 
        with st.sidebar.expander("🎥 系統架構"):
            st.video(video_file)

    # --- Header ---
    st.markdown("<h1 style='color:#00f2ff; text-align:center;'>⚡ MARCS V64 極速戰情室</h1>", unsafe_allow_html=True)

    # Session State
    if "scan_results" not in st.session_state: st.session_state.scan_results = []
    if "analysis_target" not in st.session_state: st.session_state.analysis_target = None

    # Logic
    if run_manual and manual_ticker:
        st.session_state.analysis_target = manual_ticker
        st.session_state.scan_results = [] 

    if run_scan:
        st.session_state.analysis_target = None
        # 1. 獲取清單
        with st.spinner("正在獲取市場清單 (爬蟲啟動中)..."):
            tickers = Global_Market_Loader.get_scan_list(market_select, scan_limit)
        
        if not tickers:
            st.error("❌ 無法獲取標的清單，請稍後再試。")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 2. 多執行緒並行掃描 (關鍵加速)
            status_text.text(f"啟動多核心引擎，掃描 {len(tickers)} 檔標的...")
            
            # 使用 ThreadPoolExecutor 平行處理
            # max_workers=20 代表同時開 20 個線程下載
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                # 提交任務
                future_to_ticker = {executor.submit(Scanner_Engine_V38.analyze_single, t, 60): t for t in tickers}
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_ticker):
                    res = future.result()
                    if res:
                        results.append(res)
                    
                    completed += 1
                    # 更新進度條
                    progress_bar.progress(completed / len(tickers))
                    status_text.text(f"掃描進度: {completed}/{len(tickers)}... 命中: {len(results)} 檔")

            st.session_state.scan_results = sorted(results, key=lambda x: x['score'], reverse=True)
            status_text.text(f"✅ 掃描完成！共發現 {len(results)} 檔高動能標的。")
            progress_bar.empty()

    # =================================================
    # Step 1: Macro
    # =================================================
    with st.expander("📡 Step 1: 宏觀風向 (Macro View)", expanded=True):
        if st.button("更新宏觀數據"):
            with st.spinner("同步中..."):
                macro_res = []
                cols = st.columns(4)
                for idx, (t, info) in enumerate(Global_Market_Loader.get_indices().items()):
                    r = Macro_Engine.analyze(t, info['name'])
                    macro_res.append(r)
                    if r:
                        clr = "#f85149" if r['trend']=='Overbought' else ("#3fb950" if r['trend']=='Oversold' else "#8b949e")
                        with cols[idx%4]:
                            st.markdown(f"""<div class="metric-card" style="border-top:2px solid {clr}">
                                <div class="metric-label">{r['name']}</div>
                                <div class="metric-value">{r['price']:.2f}</div>
                                <div class="metric-label" style="color:{clr}">{r['trend']}</div>
                            </div>""", unsafe_allow_html=True)
                mmi = Macro_Engine.calculate_mmi(macro_res)
                st.info(f"MMI 宏觀風險偏好指數: {mmi:.1f}")

    # =================================================
    # Step 2: Scanner Results
    # =================================================
    if st.session_state.scan_results:
        st.markdown(f"### 🔭 Step 2: 掃描結果 ({len(st.session_state.scan_results)} 檔)")
        df_scan = pd.DataFrame(st.session_state.scan_results)
        st.dataframe(df_scan[['ticker', 'score', 'price', 'rsi', 'sl']], use_container_width=True)
        
        sel = st.selectbox("選擇要深度分析的標的:", [r['ticker'] for r in st.session_state.scan_results])
        if st.button("分析選定標的"):
            st.session_state.analysis_target = sel

    # =================================================
    # Step 3: Deep Dive
    # =================================================
    target = st.session_state.analysis_target
    
    if target:
        st.markdown("---")
        st.markdown(f"### 🎯 Step 3: 深度分析 & 風控 ({target})")
        
        with st.spinner(f"正在分析 {target}..."):
            m_score, sigs, df_m = Micro_Structure_Engine.analyze(target)
            
            # 從掃描結果獲取數據，或是被動輸入時重新計算
            scan_info = next((r for r in st.session_state.scan_results if r['ticker'] == target), None)
            
            if scan_info:
                curr_p = scan_info['price']; sl_p = scan_info['sl']
            elif not df_m.empty:
                curr_p = df_m['Close'].iloc[-1]
                tr = pd.concat([df_m['High']-df_m['Low'], (df_m['High']-df_m['Close'].shift()).abs(), (df_m['Low']-df_m['Close'].shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                ma20 = df_m['Close'].rolling(20).mean().iloc[-1]
                sl_p = max(curr_p - 2.5 * atr, ma20 * 0.98)
            else:
                curr_p = 0; sl_p = 0
            
            if curr_p > 0:
                size, dets = Antifragile_Position_Sizing.calculate(capital, curr_p, sl_p, target)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"""<div class="metric-card"><div class="metric-label">微觀評分</div><div class="metric-value" style="color:#3fb950">{m_score}</div><div class="metric-label">{', '.join(sigs)}</div></div>""", unsafe_allow_html=True)
                with c2: st.markdown(f"""<div class="metric-card"><div class="metric-label">建議倉位 ({dets['asset_type']})</div><div class="metric-value">{size}</div><div class="metric-label" style="color:#d2a8ff">${dets['final_capital']:,}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="metric-card"><div class="metric-label">停損價</div><div class="metric-value" style="color:#f85149">{sl_p:.2f}</div><div class="metric-label">Risk: -${dets['risk_money']}</div></div>""", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                sub = df_m.tail(100)
                ax.plot(sub.index, sub['Close'], color='white', lw=1)
                ax.plot(sub.index, sub['K_Upper'], color='#00f2ff', ls='--', alpha=0.5)
                ax.plot(sub.index, sub['K_Lower'], color='#00f2ff', ls='--', alpha=0.5)
                ax.fill_between(sub.index, sub['K_Upper'], sub['K_Lower'], color='#00f2ff', alpha=0.1)
                ax.axhline(sl_p, color='#f85149', ls='-', label=f'SL: {sl_p:.2f}')
                ax.legend()
                ax.set_facecolor('#0e1117'); fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='gray'); ax.grid(True, alpha=0.1)
                st.pyplot(fig)
            else:
                st.error(f"無法獲取 {target} 數據，請檢查代碼是否正確。")

if __name__ == "__main__":
    main()
