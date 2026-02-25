"""
정상군 EDA (Exploratory Data Analysis)
========================================
1. 30명 평균 pressure topography (swallow별 평균)
2. 임상지표 분포 (DCI, LES pressure, UES peak, CFV, IBP)
3. 환자별 swallow 수 및 지표 요약 테이블
4. 채널별 평균 압력 프로파일
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROCESSED = Path("data/processed")
FIG_DIR   = Path("data/processed/figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 임상지표 로드
# ============================================================
with open(PROCESSED / "clinical_metrics.json", encoding="utf-8") as f:
    patients = json.load(f)

print(f"Patients: {len(patients)}")

# swallow별 지표 수집
all_sw = []
for pt in patients:
    for sw in pt["swallows"]:
        sw["patient_id"] = pt["patient_id"]
        all_sw.append(sw)

print(f"Total swallows: {len(all_sw)}")

def collect(key, filter_fn=None):
    vals = []
    for sw in all_sw:
        v = sw.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            if filter_fn is None or filter_fn(v):
                vals.append(v)
    return np.array(vals, dtype=float)

dci      = collect("dci",       lambda v: 0 <= v <= 10000)
les_p    = collect("les_pressure", lambda v: -10 <= v <= 100)
ues_peak = collect("ues_peak_p",   lambda v: 0 <= v <= 500)
cfv      = collect("cfv",       lambda v: 0 < v < 20)
ibp      = collect("ibp",       lambda v: -20 <= v <= 50)
dur      = collect("duration_s",lambda v: 10 <= v <= 60)

print(f"\n=== 임상지표 요약 ===")
for name, arr in [("DCI", dci), ("LES pressure", les_p),
                  ("UES peak", ues_peak), ("CFV", cfv), ("IBP", ibp)]:
    if len(arr):
        print(f"  {name:15s}: n={len(arr):3d}, mean={arr.mean():.1f}, "
              f"std={arr.std():.1f}, range=[{arr.min():.1f}, {arr.max():.1f}]")

# ============================================================
# 2. 임상지표 분포 시각화
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Normal Group Clinical Metrics Distribution (n=30)", fontsize=14)

metrics = [
    ("DCI (mmHg·s·cm)", dci,      "skyblue"),
    ("LES Resting Pressure (mmHg)", les_p,   "salmon"),
    ("UES Peak Pressure (mmHg)",    ues_peak, "lightgreen"),
    ("CFV (cm/s)",                  cfv,      "plum"),
    ("IBP (mmHg)",                  ibp,      "wheat"),
    ("Swallow Duration (s)",        dur,      "lightcoral"),
]

for ax, (label, arr, color) in zip(axes.flat, metrics):
    if len(arr) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(label)
        continue
    ax.hist(arr, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={arr.mean():.1f}")
    ax.axvline(np.median(arr), color="navy", linestyle=":", linewidth=1.5,
               label=f"median={np.median(arr):.1f}")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_clinical_metrics_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_clinical_metrics_dist.png")

# ============================================================
# 3. 평균 pressure topography (30명 × 21 swallows 평균)
# ============================================================
print("\nLoading swallow crops for mean topography...")
all_crops = []
for pt in patients:
    pid = f"{pt['patient_id']:02d}"
    sw_dir = PROCESSED / pid / "swallows"
    if not sw_dir.exists():
        continue
    for sw_file in sorted(sw_dir.glob("sw*.npy")):
        arr = np.load(sw_file)  # (4000, 36)
        if arr.shape == (4000, 36):
            all_crops.append(arr)

print(f"Total crops loaded: {len(all_crops)}")

if all_crops:
    stack = np.stack(all_crops, axis=0)  # (N, 4000, 36)
    mean_topo = stack.mean(axis=0)       # (4000, 36)
    std_topo  = stack.std(axis=0)        # (4000, 36)

    time_ax = np.arange(4000) * 0.01 - 15  # -15 ~ +25s (0=swallow begin)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Normal Group Mean Pressure Topography (N={len(all_crops)} swallows)",
                 fontsize=13)

    # 평균
    im1 = axes[0].imshow(mean_topo.T, aspect="auto", origin="upper",
                          vmin=0, vmax=150, cmap="jet",
                          extent=[time_ax[0], time_ax[-1], 0.5, 36.5])
    axes[0].axvline(0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[0].set_title("Mean Pressure (mmHg)")
    axes[0].set_xlabel("Time from swallow onset (s)")
    axes[0].set_ylabel("Channel (1=pharynx, 36=stomach)")
    plt.colorbar(im1, ax=axes[0], label="mmHg")

    # 표준편차
    im2 = axes[1].imshow(std_topo.T, aspect="auto", origin="upper",
                          vmin=0, vmax=50, cmap="hot",
                          extent=[time_ax[0], time_ax[-1], 0.5, 36.5])
    axes[1].axvline(0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Std Dev (mmHg)")
    axes[1].set_xlabel("Time from swallow onset (s)")
    plt.colorbar(im2, ax=axes[1], label="mmHg")

    # 채널별 평균 압력 프로파일 (시간 평균)
    ch_mean = mean_topo.mean(axis=0)   # (36,)
    ch_std  = std_topo.mean(axis=0)
    channels = np.arange(1, 37)
    axes[2].barh(channels, ch_mean, xerr=ch_std, color="steelblue",
                 alpha=0.7, ecolor="gray", capsize=2)
    axes[2].set_title("Channel Mean Pressure Profile")
    axes[2].set_xlabel("Mean Pressure (mmHg)")
    axes[2].set_ylabel("Channel")
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_mean_pressure_topography.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_mean_pressure_topography.png")

# ============================================================
# 4. 환자별 DCI 분포 (boxplot)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Per-Patient Clinical Metrics (Normal Group)", fontsize=13)

# DCI per patient
pt_ids = sorted(set(sw["patient_id"] for sw in all_sw))
dci_per_pt = []
les_per_pt = []
labels = []
for pid in pt_ids:
    sw_pt = [sw for sw in all_sw if sw["patient_id"] == pid]
    dci_vals = [sw["dci"] for sw in sw_pt if sw.get("dci") is not None and 0 <= sw["dci"] <= 10000]
    les_vals = [sw["les_pressure"] for sw in sw_pt
                if sw.get("les_pressure") is not None and -10 <= sw["les_pressure"] <= 100]
    dci_per_pt.append(dci_vals)
    les_per_pt.append(les_vals)
    labels.append(f"P{pid:02d}")

bp1 = axes[0].boxplot(dci_per_pt, labels=labels, patch_artist=True,
                       medianprops=dict(color="red", linewidth=2))
for patch in bp1["boxes"]:
    patch.set_facecolor("skyblue")
    patch.set_alpha(0.7)
axes[0].set_title("DCI per Patient (mmHg·s·cm)")
axes[0].set_xlabel("Patient")
axes[0].set_ylabel("DCI")
axes[0].tick_params(axis="x", rotation=90)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(450, color="red", linestyle="--", alpha=0.5, label="DCI=450 threshold")
axes[0].legend()

bp2 = axes[1].boxplot(les_per_pt, labels=labels, patch_artist=True,
                       medianprops=dict(color="red", linewidth=2))
for patch in bp2["boxes"]:
    patch.set_facecolor("salmon")
    patch.set_alpha(0.7)
axes[1].set_title("LES Resting Pressure per Patient (mmHg)")
axes[1].set_xlabel("Patient")
axes[1].set_ylabel("LES Pressure (mmHg)")
axes[1].tick_params(axis="x", rotation=90)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(10, color="navy", linestyle="--", alpha=0.5, label="LES=10 mmHg")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "03_per_patient_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_per_patient_metrics.png")

# ============================================================
# 5. 개별 swallow 샘플 모자이크 (환자 1, 처음 6개)
# ============================================================
sw_dir_p1 = PROCESSED / "01" / "swallows"
sw_files = sorted(sw_dir_p1.glob("sw*.npy"))[:6]
if sw_files:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Patient 01 - Individual Swallow Topographies", fontsize=13)
    time_ax = np.arange(4000) * 0.01 - 15

    for ax, fp in zip(axes.flat, sw_files):
        arr = np.load(fp)
        im = ax.imshow(arr.T, aspect="auto", origin="upper",
                       vmin=0, vmax=200, cmap="jet",
                       extent=[time_ax[0], time_ax[-1], 0.5, 36.5])
        ax.axvline(0, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(fp.stem)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax, label="mmHg")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_patient01_swallows.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_patient01_swallows.png")

print(f"\nAll figures saved to: {FIG_DIR}")
