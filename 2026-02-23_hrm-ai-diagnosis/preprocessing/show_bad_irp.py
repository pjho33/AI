"""
IRP > 15 mmHg인 환자들의 LES 채널 선택 결과 시각화
"""
import sys, numpy as np, json, os
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET
from pathlib import Path

SR = 100; PRE_SEC = 15.0

with open("data/processed/hrm_metrics_computed.json") as f:
    results = json.load(f)

# IRP > 15 환자 추출
bad = [r for r in results if r.get("IRP4s_mean") is not None and r["IRP4s_mean"] > 15]
print(f"IRP > 15 환자: {len(bad)}명")
for r in bad:
    print(f"  [{r['patient_id']}] IRP={r['IRP4s_mean']:.1f}, les_chs={r.get('les_chs')}")

DATA_DIR = Path("data/normal subjects")

def get_mva_path(pid):
    if pid.startswith("A"):
        num = int(pid[1:])
        for f in DATA_DIR.glob("*.mva"):
            if f.name.split(".")[0].isdigit() and int(f.name.split(".")[0]) == num:
                return str(f)
    else:
        num = int(pid[1:])
        for f in DATA_DIR.glob("*.mva"):
            parts = f.name.split(";")
            if len(parts) >= 2 and "." not in parts[0] and int(parts[0]) == num:
                return str(f)
    return None

def get_proc_dir(pid):
    if pid.startswith("A"):
        return f"data/processed/{pid[1:]}"
    return f"data/processed/{pid}"

n = len(bad)
fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
if n == 1:
    axes = [axes]
fig.suptitle("IRP > 15 mmHg 환자 — 선택된 LES 채널 확인\n(노란 음영=이완구간, 빨간선=XML LES pressure)", fontsize=11)

for ax, r in zip(axes, bad):
    pid = r["patient_id"]
    les_chs = r.get("les_chs", [])
    mva_path = get_mva_path(pid)
    proc_dir = get_proc_dir(pid)

    if not mva_path or not os.path.exists(proc_dir):
        ax.set_title(f"{pid} — file not found")
        continue

    blocks  = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    probe_root = ET.fromstring(xml_map["probe"])
    analysis   = ET.fromstring(xml_map["analysis"])
    sw_eps = [ep for ep in analysis.find("episodes") if ep.tag == "swallow"]

    sensors  = sorted(probe_root.find("pressure").findall("sensor"),
                      key=lambda s: int(s.findtext("index")))
    locs_rev = [float(s.findtext("location") or 0) for s in sensors][::-1]

    with open(proc_dir + "/meta.json") as f:
        meta = json.load(f)

    # sw03 사용
    si = min(1, len(meta["swallows"]) - 1)
    sw_meta = meta["swallows"][si]
    crop = np.load(proc_dir + "/swallows/" + sw_meta["file"])
    ep   = sw_eps[si]
    sw_begin_ms = float(ep.findtext("beginTime") or 0)
    les_elem    = ep.find("les")
    les_p_xml   = float(les_elem.findtext("pressure") or 0)
    les_bt = float(les_elem.find("begin").findtext("time") or sw_begin_ms)
    les_et = float(les_elem.find("end").findtext("time") or sw_begin_ms + 10000)
    crop_start_s = sw_begin_ms / 1000.0 - PRE_SEC
    rs = int((les_bt / 1000.0 - crop_start_s) * SR)
    re = int((les_et / 1000.0 - crop_start_s) * SR)
    re = max(re, rs + 500); re = min(re, len(crop) - 1)
    time_ax = np.arange(4000) / 100.0 - 15.0

    # 선택된 LES 채널 + 주변 채널 표시
    show_chs = sorted(set(les_chs + [max(0, c) for c in range(
        min(les_chs) - 3, max(les_chs) + 4)] if les_chs else range(5, 15)))
    show_chs = [c for c in show_chs if 0 <= c < 36][:10]

    colors = plt.cm.tab10(np.linspace(0, 1, len(show_chs)))
    for i, ch in enumerate(show_chs):
        lw = 2.5 if ch in les_chs else 1.0
        ls = "-" if ch in les_chs else "--"
        seg_min = crop[rs:re, ch].min() if re > rs else np.nan
        lbl = f"ch{ch}(loc={locs_rev[ch]:.0f}cm, min={seg_min:.0f})"
        if ch in les_chs:
            lbl += " ← SELECTED"
        ax.plot(time_ax, crop[:, ch], color=colors[i], linewidth=lw,
                linestyle=ls, label=lbl)

    ax.axvspan(les_bt/1000 - sw_begin_ms/1000, les_et/1000 - sw_begin_ms/1000,
               alpha=0.2, color="yellow", label="LES relax window")
    ax.axhline(les_p_xml, color="red", linestyle="--", linewidth=2,
               label=f"XML LES={les_p_xml:.1f} mmHg")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlim(-3, 15)
    ax.set_title(f"[{pid}] IRP={r['IRP4s_mean']:.1f} mmHg — selected ch={les_chs}, XML LES={les_p_xml:.1f}")
    ax.set_ylabel("Pressure (mmHg)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time from swallow onset (s)")
plt.tight_layout()
plt.savefig("preprocessing/bad_irp_channels.png", dpi=130, bbox_inches="tight")
print("Saved: preprocessing/bad_irp_channels.png")
plt.close()
