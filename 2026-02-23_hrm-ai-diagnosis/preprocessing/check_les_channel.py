"""
LES 채널 매핑 확인 - 어느 채널이 실제 LES인지
"""
import sys, numpy as np, json
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

fpath = "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"
blocks = extract_zlib_blocks(fpath)
xml_map = find_xml_blocks(blocks)
probe_root = ET.fromstring(xml_map["probe"])

# 원본 센서 순서 (역순 적용 전)
sensors = sorted(probe_root.find("pressure").findall("sensor"),
                 key=lambda s: int(s.findtext("index")))
print("원본 센서 index/name/location (역순 전):")
for s in sensors:
    idx = s.findtext("index")
    name = s.findtext("name")
    loc = s.findtext("location")
    print(f"  idx={idx:>2}, name={name:>3}, loc={loc} cm")

# 역순 후 채널 위치
locs_orig = [float(s.findtext("location") or 0) for s in sensors]
locs_rev  = locs_orig[::-1]
print("\n역순 후 txt 채널 위치:")
for i, l in enumerate(locs_rev):
    print(f"  txt ch{i:2d}: {l:6.1f} cm")

# XML LES location = -3cm → 역순 후 어느 채널?
les_loc_xml = -3.0
# 원본에서 -3cm는 어느 index?
orig_idx = [i for i, l in enumerate(locs_orig) if abs(l - les_loc_xml) < 0.5]
print(f"\nXML LES loc={les_loc_xml}cm → 원본 index={orig_idx}")
# 역순 후 해당 채널
rev_idx = [35 - i for i in orig_idx]
print(f"역순 후 txt 채널 index={rev_idx}")

# 실제 crop에서 모든 채널의 안정기 평균 압력 확인
crop = np.load("data/processed/01/swallows/sw02.npy")
print("\n각 채널 안정기 평균 (pre 5~14s = sample 500~1400):")
for ch in range(36):
    m = crop[500:1400, ch].mean()
    loc = locs_rev[ch]
    marker = " ← LES?" if abs(loc - les_loc_xml) < 2 else ""
    if abs(m - 4.14) < 10 or abs(loc - les_loc_xml) < 3:
        print(f"  ch{ch:2d} (loc={loc:6.1f}cm): mean={m:.1f} mmHg{marker}")

# XML LES pressure = 4.14 mmHg에 가장 가까운 채널
pre_means = [crop[500:1400, ch].mean() for ch in range(36)]
closest_ch = int(np.argmin(np.abs(np.array(pre_means) - 4.14)))
print(f"\nXML LES pressure=4.14에 가장 가까운 채널: ch{closest_ch} "
      f"(loc={locs_rev[closest_ch]:.1f}cm, mean={pre_means[closest_ch]:.1f})")

# 해당 채널의 IRP 계산
analysis = ET.fromstring(xml_map["analysis"])
sw_episodes = [ep for ep in analysis.find("episodes") if ep.tag == "swallow"]
sw = sw_episodes[1]
sw_begin_ms  = float(sw.findtext("beginTime"))
les_begin_ms = float(sw.find("les").find("begin").findtext("time"))
crop_start_s = sw_begin_ms / 1000.0 - 15.0
relax_sample = int((les_begin_ms / 1000.0 - crop_start_s) * 100)

les_trace = crop[:, closest_ch]
irp_seg   = les_trace[relax_sample: relax_sample + 1000]
irp4s     = float(np.sort(irp_seg)[:400].mean())
print(f"IRP4s (ch{closest_ch}): {irp4s:.2f} mmHg")
print(f"XML LES pressure:       4.14 mmHg")

# 시각화: 모든 채널 안정기 압력 vs 위치
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(range(36), pre_means, color="steelblue", alpha=0.7)
axes[0].axvline(4.14, color="red", linestyle="--", label="XML LES=4.14")
axes[0].set_yticks(range(36))
axes[0].set_yticklabels([f"ch{i}({locs_rev[i]:.0f}cm)" for i in range(36)], fontsize=6)
axes[0].set_xlabel("Pre-swallow mean pressure (mmHg)")
axes[0].set_title("All channels - pre-swallow baseline")
axes[0].legend()

# 가장 유력한 LES 채널들 시계열
time_ax = np.arange(4000) / 100.0 - 15.0
for ch in [closest_ch, closest_ch-1, closest_ch+1]:
    if 0 <= ch < 36:
        axes[1].plot(time_ax, crop[:, ch],
                     label=f"ch{ch}(loc={locs_rev[ch]:.0f}cm, mean={pre_means[ch]:.1f})")
axes[1].axvline(0, color="green", linestyle="--", label="swallow begin")
axes[1].axhline(4.14, color="red", linestyle=":", label="XML LES=4.14")
axes[1].set_xlim(-5, 15)
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Pressure (mmHg)")
axes[1].set_title("LES candidate channels")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("preprocessing/check_les.png", dpi=150, bbox_inches="tight")
print("\nSaved: preprocessing/check_les.png")
plt.close()
