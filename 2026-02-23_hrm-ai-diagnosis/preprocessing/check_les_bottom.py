import numpy as np, matplotlib.pyplot as plt, json, sys
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

mva_path = "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"
proc_dir  = "data/processed/01"

blocks   = extract_zlib_blocks(mva_path)
xml_map  = find_xml_blocks(blocks)
analysis = ET.fromstring(xml_map["analysis"])
sw_eps   = [ep for ep in analysis.find("episodes") if ep.tag == "swallow"]

probe_root = ET.fromstring(xml_map["probe"])
sensors    = sorted(probe_root.find("pressure").findall("sensor"),
                    key=lambda s: int(s.findtext("index")))
locs_rev   = [float(s.findtext("location") or 0) for s in sensors][::-1]

with open(proc_dir + "/meta.json") as f:
    meta = json.load(f)

# sw02~sw05 4개
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle("A01 — ch3~12 (LES 후보 영역) swallow별 압력\n"
             "노란 음영=XML LES 이완구간, 빨간 점선=XML LES pressure", fontsize=11)

for si in range(4):
    sw_meta = meta["swallows"][si]
    crop    = np.load(proc_dir + "/swallows/" + sw_meta["file"])
    ep      = sw_eps[si]
    sw_begin_ms  = float(ep.findtext("beginTime"))
    les          = ep.find("les")
    les_p_xml    = float(les.findtext("pressure"))
    les_bt       = float(les.find("begin").findtext("time"))
    les_et       = float(les.find("end").findtext("time"))
    crop_start_s = sw_begin_ms / 1000.0 - 15.0
    rs = int((les_bt / 1000.0 - crop_start_s) * 100)
    re = int((les_et / 1000.0 - crop_start_s) * 100)
    re = max(re, rs + 500); re = min(re, 3999)
    t_rs = rs / 100.0 - 15.0
    t_re = re / 100.0 - 15.0

    time_ax = np.arange(4000) / 100.0 - 15.0
    ax = axes[si]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ch in enumerate(range(3, 13)):
        seg_min  = crop[rs:re, ch].min()
        pre_mean = crop[500:1400, ch].mean()
        lbl = f"ch{ch}(loc={locs_rev[ch]:.0f}cm) pre={pre_mean:.0f} relax_min={seg_min:.0f}"
        ax.plot(time_ax, crop[:, ch], color=colors[i], linewidth=1.2, label=lbl)

    ax.axvspan(t_rs, t_re, alpha=0.2, color="yellow", label="LES relax window")
    ax.axhline(les_p_xml, color="red", linestyle="--", linewidth=1.5,
               label=f"XML LES={les_p_xml:.1f} mmHg")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlim(-5, 15)
    ax.set_title(f"sw{si+2} (XML LES pressure={les_p_xml:.1f} mmHg)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time from swallow onset (s)")
plt.tight_layout()
plt.savefig("preprocessing/les_bottom_check.png", dpi=130, bbox_inches="tight")
print("Saved: preprocessing/les_bottom_check.png")
plt.close()
