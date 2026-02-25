"""
A01 vs B01: LES 채널 선택 과정 디버그
"""
import sys, numpy as np, json
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

SR = 100; PRE_SEC = 15.0

def debug_les(label, mva_path, proc_dir):
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

    # XML LES pressure 수집
    xml_les_p = []
    for ep in sw_eps:
        les = ep.find("les")
        if les is not None:
            p = les.findtext("pressure")
            if p: xml_les_p.append(float(p))
    xml_mean = float(np.median(xml_les_p))
    print(f"\n{label}: XML LES pressure median = {xml_mean:.2f} mmHg")

    # 처음 5개 swallow 이완 구간 최저값
    sw_files = sorted(__import__("pathlib").Path(proc_dir + "/swallows").glob("sw*.npy"))[:5]
    ch_relax_mins = np.full((len(sw_files), 36), np.nan)
    ch_pre_means  = None

    for fi, fp in enumerate(sw_files):
        if fi >= len(sw_eps): break
        crop = np.load(fp)
        if ch_pre_means is None:
            ch_pre_means = np.array([crop[500:1400, ch].mean() for ch in range(36)])

        ep = sw_eps[fi]
        sw_begin_ms = float(ep.findtext("beginTime") or 0)
        les = ep.find("les")
        if les is None: continue
        b = les.find("begin"); e = les.find("end")
        if b is None or e is None: continue
        les_bt = float(b.findtext("time") or sw_begin_ms)
        les_et = float(e.findtext("time") or sw_begin_ms + 10000)
        crop_start_s = sw_begin_ms / 1000.0 - PRE_SEC
        rs = int((les_bt / 1000.0 - crop_start_s) * SR)
        re = int((les_et / 1000.0 - crop_start_s) * SR)
        re = max(re, rs + 500); re = min(re, len(crop) - 1)
        for ch in range(36):
            seg = crop[rs:re, ch]
            if len(seg) > 0:
                ch_relax_mins[fi, ch] = float(seg.min())

    ch_mean_mins = np.nanmean(ch_relax_mins, axis=0)
    candidates = [ch for ch in range(36) if ch_pre_means[ch] > 8.0]
    diffs = [abs(ch_mean_mins[ch] - xml_mean) if not np.isnan(ch_mean_mins[ch]) else 999
             for ch in candidates]
    best_ch = candidates[int(np.argmin(diffs))]

    print(f"  후보 채널 수: {len(candidates)}")
    print(f"  선택된 LES 채널: ch{best_ch} (loc={locs_rev[best_ch]:.0f}cm)")
    print(f"  ch{best_ch} 안정기 압력: {ch_pre_means[best_ch]:.1f} mmHg")
    print(f"  ch{best_ch} 이완 최저값: {ch_mean_mins[best_ch]:.1f} mmHg")
    print(f"  상위 5개 후보:")
    sorted_cands = sorted(candidates, key=lambda c: abs(ch_mean_mins[c] - xml_mean)
                          if not np.isnan(ch_mean_mins[c]) else 999)
    for ch in sorted_cands[:5]:
        print(f"    ch{ch:2d} (loc={locs_rev[ch]:5.0f}cm) "
              f"pre={ch_pre_means[ch]:6.1f} relax_min={ch_mean_mins[ch]:6.1f} "
              f"diff={abs(ch_mean_mins[ch]-xml_mean):.1f}")

    # 시각화: 선택된 채널 ±3 시계열
    crop = np.load(sw_files[1])
    ep   = sw_eps[1]
    sw_begin_ms = float(ep.findtext("beginTime") or 0)
    les  = ep.find("les")
    les_bt = float(les.find("begin").findtext("time"))
    les_et = float(les.find("end").findtext("time"))
    les_p_xml = float(les.findtext("pressure"))
    crop_start_s = sw_begin_ms / 1000.0 - PRE_SEC
    rs = int((les_bt / 1000.0 - crop_start_s) * SR)
    re = int((les_et / 1000.0 - crop_start_s) * SR)
    re = max(re, rs + 500); re = min(re, len(crop) - 1)
    time_ax = np.arange(4000) / 100.0 - 15.0

    fig, ax = plt.subplots(figsize=(13, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for i, ch in enumerate(sorted_cands[:5]):
        ax.plot(time_ax, crop[:, ch], color=colors[i], linewidth=1.2,
                label=f"ch{ch}(loc={locs_rev[ch]:.0f}cm, pre={ch_pre_means[ch]:.0f}, min={ch_mean_mins[ch]:.0f})")
    ax.axvspan(les_bt/1000 - sw_begin_ms/1000, les_et/1000 - sw_begin_ms/1000,
               alpha=0.2, color="yellow", label="LES relax window")
    ax.axhline(les_p_xml, color="red", linestyle="--", linewidth=2,
               label=f"XML LES={les_p_xml:.1f} mmHg")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlim(-3, 15); ax.set_title(f"{label} — Top 5 LES candidates (sw03)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pressure (mmHg)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"preprocessing/les_sel_{label}.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: les_sel_{label}.png")
    return best_ch

debug_les("A01",
          "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva",
          "data/processed/01")
debug_les("B01",
          "data/normal subjects/1;13-04-2012.mva",
          "data/processed/B01")
