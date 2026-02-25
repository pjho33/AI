"""
정상군 60명 합친 EDA
=====================
1. 임상지표 분포 (A그룹 vs B그룹 비교)
2. 60명 평균 pressure topography
3. 환자별 DCI/LES boxplot
4. A vs B 그룹 지표 비교 (t-test)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PROCESSED = Path("data/processed")
FIG_DIR   = PROCESSED / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── 데이터 로드 (compute_hrm_metrics.py 결과) ─────────────────────
with open(PROCESSED / "hrm_metrics_computed.json", encoding="utf-8") as f:
    results = json.load(f)

# patient-level 데이터를 swallow-level처럼 변환 (patient mean 사용)
grp_A = [r for r in results if r["patient_id"].startswith("A") and not r.get("excluded")]
grp_B = [r for r in results if r["patient_id"].startswith("B") and not r.get("excluded")]
print(f"Group A: {len(grp_A)} patients, Group B: {len(grp_B)} patients")

# clinical_metrics_all.json도 topography용으로 로드
with open(PROCESSED / "clinical_metrics_all.json", encoding="utf-8") as f:
    patients = json.load(f)

all_sw = []
for pt in patients:
    for sw in pt["swallows"]:
        sw = dict(sw)
        sw["patient_id"] = pt["patient_id"]
        sw["group"]      = pt["group"]
        all_sw.append(sw)

sw_A = [sw for sw in all_sw if sw["group"] == "A"]
sw_B = [sw for sw in all_sw if sw["group"] == "B"]
print(f"Swallows A: {len(sw_A)}, B: {len(sw_B)}, Total: {len(all_sw)}")


def collect(pt_list, key, lo=None, hi=None):
    vals = []
    for pt in pt_list:
        v = pt.get(key)
        if v is None: continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(v): continue
        if lo is not None and v < lo: continue
        if hi is not None and v > hi: continue
        vals.append(v)
    return np.array(vals)


# patient-level computed metrics (UES/IBP 제외)
metrics_cfg = [
    ("IRP4s_mean",   "IRP4s (mmHg)",               0,    20),
    ("DCI_mean",     "DCI (mmHg·s·cm)",             0,  8000),
    ("LESP_mean",    "LES Resting Pressure (mmHg)", -5,   40),
    ("CFV_mean",     "CFV (cm/s)",                   0,   20),
]

# ── 1. 임상지표 분포 (A vs B 오버레이) ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Normal Group Clinical Metrics: Group A (n={len(grp_A)}) vs B (n={len(grp_B)})", fontsize=13)

for ax, (key, label, lo, hi) in zip(axes.flat, metrics_cfg):
    a = collect(grp_A, key, lo, hi)
    b = collect(grp_B, key, lo, hi)
    bins = np.linspace(min(a.min() if len(a) else lo, b.min() if len(b) else lo),
                       max(a.max() if len(a) else hi, b.max() if len(b) else hi), 25)
    ax.hist(a, bins=bins, alpha=0.6, color="steelblue", label=f"A (n={len(a)})")
    ax.hist(b, bins=bins, alpha=0.6, color="tomato",    label=f"B (n={len(b)})")
    if len(a): ax.axvline(a.mean(), color="steelblue", linestyle="--", linewidth=1.5)
    if len(b): ax.axvline(b.mean(), color="tomato",    linestyle="--", linewidth=1.5)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Value"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "05_metrics_A_vs_B.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_metrics_A_vs_B.png")

# ── 2. 60명 평균 pressure topography ────────────────────────────
print("\nLoading all swallow crops...")
all_crops, crops_A, crops_B = [], [], []

for pt in patients:
    pid   = pt["patient_id"]
    group = pt["group"]
    # A그룹: 폴더명이 숫자 (01~31)
    if group == "A":
        num = pid.replace("A", "")
        sw_dir = PROCESSED / num / "swallows"
    else:
        sw_dir = PROCESSED / pid / "swallows"

    if not sw_dir.exists():
        continue
    for fp in sorted(sw_dir.glob("sw*.npy")):
        arr = np.load(fp)
        if arr.shape == (4000, 36):
            all_crops.append(arr)
            if group == "A": crops_A.append(arr)
            else:            crops_B.append(arr)

print(f"Crops loaded: A={len(crops_A)}, B={len(crops_B)}, total={len(all_crops)}")

time_ax = np.arange(4000) * 0.01 - 15

if all_crops:
    mean_all = np.stack(all_crops).mean(axis=0)
    mean_A   = np.stack(crops_A).mean(axis=0) if crops_A else None
    mean_B   = np.stack(crops_B).mean(axis=0) if crops_B else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Normal Group Mean Pressure Topography (N={len(all_crops)} swallows, 60 patients)",
                 fontsize=12)

    for ax, (data, title) in zip(axes, [
        (mean_all, f"All 60 patients (N={len(all_crops)})"),
        (mean_A,   f"Group A (N={len(crops_A)})"),
        (mean_B,   f"Group B (N={len(crops_B)})"),
    ]):
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        im = ax.imshow(data.T, aspect="auto", origin="upper",
                       vmin=0, vmax=150, cmap="jet",
                       extent=[time_ax[0], time_ax[-1], 35.5, -0.5])
        ax.axvline(0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time from swallow onset (s)")
        ax.set_ylabel("Channel (ch0=pharynx top, ch35=stomach bottom)")
        plt.colorbar(im, ax=ax, label="mmHg")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_mean_topography_60.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_mean_topography_60.png")

# ── 3. A vs B 통계 비교표 ────────────────────────────────────────
print("\n=== Group A vs B Statistical Comparison ===")
print(f"{'Metric':<25} {'A mean±std':>15} {'B mean±std':>15} {'p-value':>10}")
print("-" * 70)

for key, label, lo, hi in metrics_cfg:
    a = collect(grp_A, key, lo, hi)
    b = collect(grp_B, key, lo, hi)
    if len(a) < 5 or len(b) < 5:
        continue
    t, p = stats.ttest_ind(a, b)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"  {label:<23} {a.mean():6.1f}±{a.std():5.1f}   "
          f"{b.mean():6.1f}±{b.std():5.1f}   p={p:.4f} {sig}")

# ── 4. 환자별 DCI 전체 boxplot (60명) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Per-Patient DCI & LES Pressure — All 60 Normal Subjects", fontsize=12)

for ax, (key, label, lo, hi, color, thresh, thresh_label) in zip(axes, [
    ("dci",          "DCI (mmHg·s·cm)",          0, 10000, "steelblue", 450,  "DCI=450"),
    ("les_pressure", "LES Resting Pressure (mmHg)", -10, 50, "salmon",   10,   "LES=10"),
]):
    data_per_pt = []
    tick_labels = []
    colors_box  = []
    for pt in sorted(patients, key=lambda p: p["patient_id"]):
        vals = collect(pt["swallows"], key, lo, hi)
        if len(vals) == 0:
            continue
        data_per_pt.append(vals)
        tick_labels.append(pt["patient_id"])
        colors_box.append("steelblue" if pt["group"] == "A" else "tomato")

    bp = ax.boxplot(data_per_pt, tick_labels=tick_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(thresh, color="red", linestyle="--", alpha=0.5, label=thresh_label)
    ax.set_title(label); ax.set_xlabel("Patient ID"); ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # 범례 (A=blue, B=red)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Group A (named)"),
        Patch(facecolor="tomato",    alpha=0.7, label="Group B (numbered)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=thresh_label),
    ], fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "07_per_patient_60.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_per_patient_60.png")

print(f"\nAll figures saved to: {FIG_DIR}")
