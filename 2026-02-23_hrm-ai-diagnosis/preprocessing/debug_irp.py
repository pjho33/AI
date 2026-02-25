"""
IRP 계산 디버그: LES 이완기 시각화 + 최저 400 samples 선택
"""
import sys, numpy as np, json
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

fpath = "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"
blocks = extract_zlib_blocks(fpath)
xml_map = find_xml_blocks(blocks)
analysis = ET.fromstring(xml_map["analysis"])
sw_episodes = [ep for ep in analysis.find("episodes") if ep.tag == "swallow"]

with open("data/processed/01/meta.json") as f:
    meta = json.load(f)
block0_start_s = meta["block0_start_s"]

# sw2 (index 1) 기준
sw = sw_episodes[1]
sw_begin_ms  = float(sw.findtext("beginTime"))
les_elem     = sw.find("les")
les_begin_ms = float(les_elem.find("begin").findtext("time"))
les_end_ms   = float(les_elem.find("end").findtext("time"))
les_p_xml    = float(les_elem.findtext("pressure"))

# crop 내 sample index
crop_start_s     = sw_begin_ms / 1000.0 - 15.0
les_relax_sample = int((les_begin_ms / 1000.0 - crop_start_s) * 100)
les_end_sample   = int((les_end_ms   / 1000.0 - crop_start_s) * 100)
onset_sample     = 1500  # swallow begin

print(f"sw_begin:    {sw_begin_ms/1000:.2f}s")
print(f"LES relax:   {les_begin_ms/1000:.2f}s ~ {les_end_ms/1000:.2f}s")
print(f"XML LES pressure: {les_p_xml:.2f} mmHg")
print(f"crop sample: relax={les_relax_sample}, end={les_end_sample}, onset={onset_sample}")

# 실제 crop 로드
crop = np.load("data/processed/01/swallows/sw02.npy")  # (4000, 36)
les_ch = 32  # loc=-3cm (LES 고압대)

les_trace = crop[:, les_ch]
time_ax   = np.arange(len(les_trace)) / 100.0 - 15.0  # -15 ~ +25s

# IRP 구간: XML les.begin ~ les.end (이완기 정확히)
les_end_ms   = float(les_elem.find("end").findtext("time"))
les_end_sample = int((les_end_ms / 1000.0 - crop_start_s) * 100)

irp_start = les_relax_sample
irp_end   = max(les_end_sample, irp_start + 1000)  # 최소 10s 보장
irp_end   = min(irp_end, len(les_trace))
irp_seg   = les_trace[irp_start:irp_end]

# 가장 낮은 400 samples (비연속)
low400_idx = np.argsort(irp_seg)[:400]
irp4s      = float(irp_seg[low400_idx].mean())

print(f"\nIRP 구간 ({irp_start}~{irp_end}, {(irp_end-irp_start)/100:.0f}s):")
print(f"  min={irp_seg.min():.1f}, max={irp_seg.max():.1f}, mean={irp_seg.mean():.1f}")
print(f"  IRP4s (lowest 400 samples avg) = {irp4s:.2f} mmHg")
print(f"  XML LES pressure               = {les_p_xml:.2f} mmHg")

# ── 시각화 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 전체 LES 시계열
ax = axes[0]
ax.plot(time_ax, les_trace, "b-", linewidth=1, label=f"LES ch{les_ch} (loc=-3cm)")
ax.axvline(0,                    color="green",  linestyle="--", label="swallow begin")
ax.axvline(les_begin_ms/1000 - sw_begin_ms/1000,
           color="orange", linestyle="--", label="LES relax begin (XML)")
ax.axvline(les_end_ms/1000   - sw_begin_ms/1000,
           color="red",    linestyle="--", label="LES relax end (XML)")
ax.axhline(les_p_xml, color="purple", linestyle=":", linewidth=1.5,
           label=f"XML LES pressure={les_p_xml:.1f}")
ax.axhline(irp4s,     color="red",    linestyle="-", linewidth=2,
           label=f"IRP4s={irp4s:.1f} mmHg")
ax.set_title("LES Pressure Trace - sw02 (Patient 01)")
ax.set_xlabel("Time from swallow onset (s)")
ax.set_ylabel("Pressure (mmHg)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 15)

# IRP 구간 확대 + 선택된 400 samples 표시
ax2 = axes[1]
irp_time = time_ax[irp_start:irp_end]
ax2.plot(irp_time, irp_seg, "b-", linewidth=1.5, label="LES pressure (IRP window)")

# 낮은 400 samples 표시
low400_abs = low400_idx + irp_start
ax2.scatter(time_ax[low400_abs], les_trace[low400_abs],
            color="red", s=5, alpha=0.5, label="Lowest 400 samples (IRP4s)")
ax2.axhline(irp4s, color="red", linestyle="-", linewidth=2,
            label=f"IRP4s = {irp4s:.1f} mmHg")
ax2.axhline(les_p_xml, color="purple", linestyle=":", linewidth=1.5,
            label=f"XML LES pressure = {les_p_xml:.1f} mmHg")
ax2.set_title("IRP Window (10s from LES relax begin) — Lowest 400 samples highlighted")
ax2.set_xlabel("Time from swallow onset (s)")
ax2.set_ylabel("Pressure (mmHg)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("preprocessing/debug_irp.png", dpi=150, bbox_inches="tight")
print("\nSaved: preprocessing/debug_irp.png")
plt.close()
