"""
고압대 환자들의 채널별 안정기 압력 프로파일 시각화
직접 보고 LES 채널 판단용
"""
import numpy as np, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

onset = 1500
SR = 100

# 고압대 환자 목록
patients = [
    ("A01", "data/processed/01/swallows"),
    ("A03", "data/processed/03/swallows"),
    ("A13", "data/processed/13/swallows"),
    ("A22", "data/processed/22/swallows"),
    ("A25", "data/processed/25/swallows"),
    ("B04", "data/processed/B04/swallows"),
    ("B05", "data/processed/B05/swallows"),
    ("B08", "data/processed/B08/swallows"),
    ("B09", "data/processed/B09/swallows"),
    ("B11", "data/processed/B11/swallows"),
    ("B12", "data/processed/B12/swallows"),
    ("B14", "data/processed/B14/swallows"),
    ("B25", "data/processed/B25/swallows"),
]

n = len(patients)
fig, axes = plt.subplots(n, 1, figsize=(14, n * 2.5))
fig.suptitle("Channel Resting Pressure Profile (ch0~15) — 고압대 환자", fontsize=12)

for ax, (pid, sw_dir) in zip(axes, patients):
    fps = sorted(glob.glob(sw_dir + "/sw*.npy"))[:5]
    if not fps:
        ax.set_title(f"{pid} — no data")
        continue
    pre = np.mean([np.load(f)[500:1400, :].mean(axis=0) for f in fps], axis=0)
    post = np.mean([np.load(f)[onset:onset+500, :].mean(axis=0) for f in fps], axis=0)
    rise = post - pre

    chs = list(range(16))
    ax.bar(chs, pre[:16], color="steelblue", alpha=0.7, label="resting")
    ax.bar(chs, rise[:16], bottom=pre[:16], color="tomato", alpha=0.6, label="post-rise")
    ax.axhline(50, color="red", ls="--", lw=1, alpha=0.5, label="50 mmHg")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{pid}  (ch0~15 resting, red=after-contraction)", fontsize=9)
    ax.set_xticks(chs)
    ax.set_xticklabels([f"ch{c}" for c in chs], fontsize=7)
    ax.set_ylim(-20, max(150, pre[:16].max() + 20))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("preprocessing/channel_profiles_highpressure.png", dpi=120, bbox_inches="tight")
print("Saved: preprocessing/channel_profiles_highpressure.png")
plt.close()
