"""
고압대 환자들의 swallow topography 시각화 (sw01 기준)
"""
import numpy as np, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

onset = 1500
SR = 100
time_ax = np.arange(4000) / SR - 15.0

patients = [
    ("A01", "data/processed/01/swallows"),
    ("A25", "data/processed/25/swallows"),
    ("B04", "data/processed/B04/swallows"),
    ("B08", "data/processed/B08/swallows"),
    ("B09", "data/processed/B09/swallows"),
    ("B11", "data/processed/B11/swallows"),
    ("B12", "data/processed/B12/swallows"),
    ("B14", "data/processed/B14/swallows"),
    ("B25", "data/processed/B25/swallows"),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("Swallow Pressure Topography (sw01, ch0~20)", fontsize=13)

for ax, (pid, sw_dir) in zip(axes.flat, patients):
    fps = sorted(glob.glob(sw_dir + "/sw*.npy"))
    if not fps:
        continue
    crop = np.load(fps[0])  # sw01
    # ch0~20, t=-5~15s
    t_mask = (time_ax >= -5) & (time_ax <= 15)
    data = crop[t_mask, :21].T  # (21, n_t)
    im = ax.imshow(data[::-1], aspect="auto", origin="upper",
                   extent=[time_ax[t_mask][0], time_ax[t_mask][-1], 0, 20],
                   vmin=0, vmax=120, cmap="jet")
    ax.axvline(0, color="white", ls="--", lw=1.5)
    ax.set_title(pid, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Channel", fontsize=8)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_yticklabels(["ch20","ch18","ch16","ch14","ch12","ch10","ch8","ch6","ch4","ch2","ch0"], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)

plt.tight_layout()
plt.savefig("preprocessing/topography_highpressure.png", dpi=130, bbox_inches="tight")
print("Saved: preprocessing/topography_highpressure.png")
plt.close()
