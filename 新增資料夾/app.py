import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# 模擬一個隨機漫步數據 (0 到 100)
np.random.seed(42)
data_full = np.cumsum(np.random.randn(200))

# 情境 A: 我們站在第 100 天 (只看得到前 100 筆數據)
data_t100 = data_full[:100]
phase_A = np.angle(hilbert(data_t100))

# 情境 B: 我們站在第 200 天 (有了新數據，回頭看第 100 天以前的訊號)
phase_B = np.angle(hilbert(data_full))

# 檢查第 50 天的數值是否改變
diff = np.abs(phase_A[50] - phase_B[50])

print(f"🛑 CRITICAL FAILURE CHECK:")
print(f"Day 50 Phase (calculated at Day 100): {phase_A[50]:.4f}")
print(f"Day 50 Phase (calculated at Day 200): {phase_B[50]:.4f}")
print(f"⚠️ Signal Repainting Error: {diff:.4f}")

if diff > 0.0001:
    print("結論: 你的指標會重繪 (Repaint)。過去的買點在未來會消失。")
