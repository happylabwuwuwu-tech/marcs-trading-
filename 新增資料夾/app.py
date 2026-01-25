<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MARCS Pro: Cyberpunk Protocol</title>
    <style>
        /* --- 0. 基礎環境設定 --- */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Segoe+UI:wght@400;600&display=swap');

        body {
            margin: 0;
            padding: 0;
            font-family: 'Roboto Mono', monospace;
            color: #c9d1d9;
            height: 100vh;
            overflow: hidden; /* 防止捲軸出現 (模擬 App) */
            display: flex;
        }

        /* --- 1. 動態背景影片 --- */
        #bg-video {
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            opacity: 0.4;
            filter: hue-rotate(180deg) contrast(1.2); /* 調整成冷色調 */
            object-fit: cover;
        }

        /* --- 2. 側邊欄 (Sidebar) --- */
        .sidebar {
            width: 320px;
            background: rgba(22, 27, 34, 0.85); /* 半透明黑 */
            backdrop-filter: blur(10px); /* 毛玻璃模糊 */
            border-right: 1px solid rgba(48, 54, 61, 0.6);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 20px;
            box-shadow: 5px 0 15px rgba(0,0,0,0.3);
            z-index: 10;
        }

        .brand-title {
            color: #00f2ff;
            text-shadow: 0 0 10px rgba(0, 242, 255, 0.6);
            font-size: 2rem;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 800;
        }

        .system-status {
            color: #8b949e;
            font-size: 12px;
            letter-spacing: 1px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.8);
            padding-bottom: 15px;
            margin-bottom: 10px;
        }

        /* 輸入框與按鈕模擬 */
        label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        input[type="text"] {
            background: rgba(13, 17, 23, 0.6);
            border: 1px solid #30363d;
            color: white;
            padding: 12px;
            border-radius: 4px;
            width: 100%;
            box-sizing: border-box; /* 確保 padding 不會撐大寬度 */
            font-family: 'Roboto Mono', monospace;
            margin-top: 5px;
        }

        .btn-scan {
            background: rgba(31, 111, 235, 0.9);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s;
            margin-top: 10px;
            box-shadow: 0 0 10px rgba(31, 111, 235, 0.4);
        }
        .btn-scan:hover {
            background: #00f2ff;
            color: black;
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        }

        /* --- 3. 主內容區 (Main Content) --- */
        .main-content {
            flex: 1;
            padding: 2rem 4rem;
            overflow-y: auto; /* 內容過長時捲動 */
            position: relative;
        }

        /* HUD Header */
        .tech-header {
            display: flex;
            align-items: center;
            background: rgba(13, 17, 23, 0.6);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00f2ff;
            backdrop-filter: blur(5px);
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .tag-target {
            background: rgba(31, 111, 235, 0.2);
            color: #58a6ff;
            padding: 5px 15px;
            border: 1px solid #1f6feb;
            border-radius: 4px;
            font-size: 14px;
        }

        /* 儀表板卡片 Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 25px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: rgba(13, 17, 23, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(88, 166, 255, 0.2);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #00f2ff;
            box-shadow: 0 0 25px rgba(0, 242, 255, 0.3);
        }

        /* 卡片頂部彩色線條 */
        .card-top-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
        }

        .metric-label {
            font-size: 12px;
            margin-bottom: 10px;
            opacity: 0.8;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 5px 0;
            color: white;
        }
        
        .metric-status {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        .metric-advice {
            background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            text-align: left;
            border-left: 2px solid rgba(255,255,255,0.2);
        }

        /* 頁籤模擬 */
        .tabs {
            display: flex;
            gap: 20px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.8);
            margin-bottom: 20px;
        }
        .tab-item {
            padding: 10px 0;
            cursor: pointer;
            color: #8b949e;
            position: relative;
        }
        .tab-item.active {
            color: #c9d1d9;
            font-weight: bold;
        }
        .tab-item.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 2px;
            background: #ff4b4b;
            box-shadow: 0 0 10px #ff4b4b;
        }

        /* 報告區域 */
        .report-panel {
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.8);
            border-radius: 12px;
            padding: 30px;
            margin-top: 20px;
        }

        .command-decision {
            font-size: 28px;
            font-weight: bold;
            color: #3fb950;
            text-shadow: 0 0 15px rgba(63, 185, 80, 0.6);
            margin-bottom: 5px;
        }

        /* 表格模擬 */
        .tech-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }
        .tech-table td {
            padding: 12px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.5);
        }
        .tech-table tr:last-child td {
            border-bottom: none;
        }

    </style>
</head>
<body>

    <video autoplay muted loop id="bg-video">
        <source src="https://cdn.pixabay.com/video/2020/04/18/36465-412239632_large.mp4" type="video/mp4">
    </video>

    <div class="sidebar">
        <div>
            <h1 class="brand-title">🛡️ MARCS Pro</h1>
            <div class="system-status">SYSTEM V34.2 // ONLINE</div>
        </div>

        <div>
            <div style="font-weight:bold; color:white; margin-bottom:15px;">🛠️ CONTROL PANEL</div>
            <label>Target Ticker</label>
            <input type="text" value="2330.TW">
            <button class="btn-scan">INITIATE SCAN ⚡</button>
        </div>

        <div style="margin-top:auto; font-size:11px; color:#586069;">
            PHYSICS ENGINE: CEEMD/MFA<br>
            SECURE CONNECTION: ESTABLISHED
        </div>
    </div>

    <div class="main-content">
        
        <div class="tech-header">
            <div>
                <h2 style="margin:0; color:white; text-shadow: 0 0 5px rgba(255,255,255,0.5);">MARKET RECONNAISSANCE</h2>
                <div style="color:#00f2ff; font-size:12px; letter-spacing:2px; margin-top:5px;">QUANTITATIVE PHYSICS ENGINE ENGAGED</div>
            </div>
            <div style="flex:1"></div>
            <div class="tag-target">TARGET: 2330.TW</div>
        </div>

        <div class="dashboard-grid">
            
            <div class="metric-card">
                <div class="card-top-line" style="background:#3fb950;"></div>
                <div class="metric-label" style="color:#3fb950;">WCA PHASE</div>
                <div class="metric-value" style="text-shadow: 0 0 15px rgba(63,185,80,0.5);">0.87</div>
                <div class="metric-status" style="color:#3fb950;">[ SYNC ESTABLISHED ]</div>
                <div class="metric-advice">>> ENERGY FLOW STABLE. HOLD POSITION.</div>
            </div>

            <div class="metric-card">
                <div class="card-top-line" style="background:#a371f7;"></div>
                <div class="metric-label" style="color:#a371f7;">MFA ENTROPY</div>
                <div class="metric-value" style="text-shadow: 0 0 15px rgba(163,113,247,0.5);">0.12</div>
                <div class="metric-status" style="color:#a371f7;">[ STRUCTURE STABLE ]</div>
                <div class="metric-advice">>> LOW ENTROPY. SYSTEM GREEN.</div>
            </div>

            <div class="metric-card">
                <div class="card-top-line" style="background:#f85149;"></div>
                <div class="metric-label" style="color:#f85149;">TIGHT STOP-LOSS</div>
                <div class="metric-value" style="text-shadow: 0 0 15px rgba(248,81,73,0.5);">1025.0</div>
                <div class="metric-status" style="color:#f85149;">DIST: -1.8% DELTA</div>
                <div class="metric-advice">>> CRITICAL BOUNDARY: MONITOR CLOSELY.</div>
            </div>

        </div>

        <div class="tabs">
            <div class="tab-item">📊 VISUALIZATION</div>
            <div class="tab-item active">📋 TACTICAL REPORT</div>
        </div>

        <div class="report-panel">
            <h3 style="margin:0 0 20px 0; border-bottom:1px dashed #3fb950; padding-bottom:10px; color:#3fb950;">COMMAND DECISION</h3>
            
            <div class="command-decision">[ MAINTAIN POSITIONS ]</div>
            
            <div style="font-size:14px; margin-bottom:20px; line-height:1.6;">
                >> SIGNAL INTEGRITY: 100%<br>
                >> RISK PROTOCOL: STANDARD<br>
                >> ACTION: Continue to hold. Add positions on pullback to support level.
            </div>

            <table class="tech-table">
                <tr>
                    <td style="color:#8b949e;">CURRENT_PRICE</td>
                    <td style="text-align:right; font-weight:bold; color:white;">1045.00</td>
                </tr>
                <tr>
                    <td style="color:#f85149;">STOP_LOSS_LEVEL</td>
                    <td style="text-align:right; font-weight:bold; color:#f85149;">1025.00</td>
                </tr>
                <tr>
                    <td style="color:#3fb950;">TARGET_RESISTANCE</td>
                    <td style="text-align:right; font-weight:bold; color:#3fb950;">1080.00</td>
                </tr>
            </table>
        </div>

    </div>

</body>
</html>
