"""
모든 채널 시계열을 한번에 그려서 LES 이완 채널 직접 확인
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

# sw2 XML 정보
sw = sw_episodes[1]
sw_begin_ms  = float(sw.findtext("beginTime"))
les_elem     = sw.find("les")
les_begin_ms = float(les_elem.find("begin").findtext("time"))
les_end_ms   = float(les_elem.find("end").findtext("time"))
les_p_xml    = float(les_elem.findtext("pressure"))
les_loc_xml  = float(les_elem.findtext("location"))

crop_start_s     = sw_begin_ms / 1000.0 - 15.0
les_relax_sample = int((les_begin_ms / 1000.0 - crop_start_s) * 100)
les_end_sample   = int((les_end_ms   / 1000.0 - crop_start_s) * 100)

print(f"XML LES location: {les_loc_xml} cm")
print(f"XML LES pressure (이완 최저): {les_p_xml:.2f} mmHg")
print(f"이완 구간 sample: {les_relax_sample}~{les_end_sample}")

crop = np.load("data/processed/01/swallows/sw02.npy")
time_ax = np.arange(4000) / 100.0 - 15.0

# 이완 구간에서 각 채널의 최저값 확인
print("\n이완 구간 내 채널별 최저값:")
mins = []
for ch in range(36):
    seg = crop[les_relax_sample:les_end_sample, ch]
    m = seg.min() if len(seg) > 0 else np.nan
    mins.append(m)
    if abs(m - les_p_xml) < 5:
        print(f"  *** ch{ch:2d}: min={m:.1f} mmHg  ← XML LES={les_p_xml:.1f}에 근접!")

# XML LES pressure에 가장 가까운 채널
closest = int(np.argmin(np.abs(np.array(mins) - les_p_xml)))
print(f"\n가장 가까운 채널: ch{closest}, min={mins[closest]:.1f} mmHg")

# 시각화: 채널 30~35 (LES 근처) 시계열
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 상단: ch28~35 시계열 (LES 고압대 주변)
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for i, ch in enumerate(range(28, 36)):
    from parse_mva import extract_zlib_blocks, find_xml_blocks
    probe_root = ET.fromstring(xml_map["probe"])
    sensors = sorted(probe_root.find("pressure").findall("sensor"),
                     key=lambda s: int(s.findtext("index")))
    locs_orig = [float(s.findtext("location") or 0) for s in sensors]
    locs_rev  = locs_orig[::-1]
    loc = locs_rev[ch]
    ax.plot(time_ax, crop[:, ch], color=colors[i], linewidth=1.2,
            label=f"ch{ch}(loc={loc:.0f}cm, relax_min={mins[ch]:.1f})")

ax.axvspan(les_begin_ms/1000 - sw_begin_ms/1000,
           les_end_ms/1000   - sw_begin_ms/1000,
           alpha=0.15, color="yellow", label="LES relax window (XML)")
ax.axhline(les_p_xml, color="red", linestyle="--", linewidth=2,
           label=f"XML LES pressure={les_p_xml:.1f} mmHg")
ax.axvline(0, color="green", linestyle="--", linewidth=1.5, label="swallow begin")
ax.set_xlim(-3, 15); ax.set_ylim(-10, 100)
ax.set_title("Channels 28~35 (LES region) — sw02 Patient 01")
ax.set_xlabel("Time from swallow onset (s)")
ax.set_ylabel("Pressure (mmHg)")
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

# 하단: 가장 가까운 채널 + 이완 구간 강조
ax2 = axes[1]
ax2.plot(time_ax, crop[:, closest], "b-", linewidth=2,
         label=f"ch{closest} (best match, min={mins[closest]:.1f})")
ax2.plot(time_ax[les_relax_sample:les_end_sample],
         crop[les_relax_sample:les_end_sample, closest],
         "r-", linewidth=2.5, label="LES relax window")

# 최저 400 samples
irp_seg = crop[les_relax_sample:les_end_sample, closest]
if len(irp_seg) >= 400:
    low400 = np.argsort(irp_seg)[:400]
    irp4s  = float(irp_seg[low400].mean())
    abs_idx = low400 + les_relax_sample
    ax2.scatter(time_ax[abs_idx], crop[abs_idx, closest],
                color="orange", s=8, zorder=5, label=f"Lowest 400 → IRP4s={irp4s:.1f}")
else:
    irp4s = float(irp_seg.mean()) if len(irp_seg) > 0 else np.nan

ax2.axhline(les_p_xml, color="purple", linestyle="--", linewidth=2,
            label=f"XML LES pressure={les_p_xml:.1f}")
ax2.axhline(irp4s, color="orange", linestyle="-", linewidth=2,
            label=f"Computed IRP4s={irp4s:.1f}")
ax2.axvline(0, color="green", linestyle="--")
ax2.set_xlim(-3, 15)
ax2.set_title(f"Best LES channel (ch{closest}) — IRP4s vs XML")
ax2.set_xlabel("Time from swallow onset (s)")
ax2.set_ylabel("Pressure (mmHg)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("preprocessing/find_les.png", dpi=150, bbox_inches="tight")
print(f"\nIRP4s computed = {irp4s:.2f} mmHg")
print(f"XML LES pressure = {les_p_xml:.2f} mmHg")
print("Saved: preprocessing/find_les.png")
plt.close()
