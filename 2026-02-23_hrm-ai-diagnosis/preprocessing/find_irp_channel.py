"""
이완 구간에서 실제로 XML LES pressure에 가장 가까운 채널 찾기
A그룹과 B그룹 비교
"""
import sys, numpy as np, json, os
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

def find_irp_channel(label, mva_path, proc_dir, show_plot=True):
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    analysis = ET.fromstring(xml_map["analysis"])
    sw_eps = [ep for ep in analysis.find("episodes") if ep.tag == "swallow"]

    with open(os.path.join(proc_dir, "meta.json")) as f:
        meta = json.load(f)

    # 모든 swallow에 대해 이완 구간 내 채널별 최저값 수집
    ch_irp_all = np.zeros((len(meta["swallows"]), 36))

    for fi, sw_meta in enumerate(meta["swallows"]):
        crop_path = os.path.join(proc_dir, "swallows", sw_meta["file"])
        if not os.path.exists(crop_path):
            continue
        crop = np.load(crop_path)

        meta_begin = sw_meta["begin_ms"]
        # XML episode 매칭
        ep = None
        for e in sw_eps:
            if abs(float(e.findtext("beginTime")) - meta_begin) < 100:
                ep = e
                break
        if ep is None:
            continue

        les = ep.find("les")
        if les is None:
            continue

        sw_begin_ms = float(ep.findtext("beginTime"))
        les_bt = les.find("begin")
        les_et = les.find("end")
        if les_bt is None or les_et is None:
            continue

        les_begin_ms = float(les_bt.findtext("time"))
        les_end_ms   = float(les_et.findtext("time"))
        crop_start_s = sw_begin_ms / 1000.0 - 15.0
        rs = int((les_begin_ms / 1000.0 - crop_start_s) * 100)
        re = int((les_end_ms   / 1000.0 - crop_start_s) * 100)
        re = max(re, rs + 1000)
        re = min(re, len(crop) - 1)

        for ch in range(36):
            seg = crop[rs:re, ch]
            if len(seg) >= 400:
                ch_irp_all[fi, ch] = float(np.sort(seg)[:400].mean())
            elif len(seg) > 0:
                ch_irp_all[fi, ch] = float(seg.mean())

    # XML LES pressure 수집
    xml_irp_vals = []
    for sw_meta in meta["swallows"]:
        meta_begin = sw_meta["begin_ms"]
        for e in sw_eps:
            if abs(float(e.findtext("beginTime")) - meta_begin) < 100:
                les = e.find("les")
                if les is not None:
                    p = les.findtext("pressure")
                    if p:
                        xml_irp_vals.append(float(p))
                break

    xml_irp_mean = np.mean(xml_irp_vals) if xml_irp_vals else 0
    ch_irp_mean  = ch_irp_all.mean(axis=0)  # (36,)

    # XML IRP에 가장 가까운 채널
    best_ch = int(np.argmin(np.abs(ch_irp_mean - xml_irp_mean)))

    print(f"\n{label}:")
    print(f"  XML LES pressure mean: {xml_irp_mean:.2f} mmHg")
    print(f"  Best channel: ch{best_ch}, computed IRP mean: {ch_irp_mean[best_ch]:.2f} mmHg")
    print(f"  Top 5 closest channels:")
    diffs = np.abs(ch_irp_mean - xml_irp_mean)
    for ch in np.argsort(diffs)[:5]:
        print(f"    ch{ch:2d}: computed={ch_irp_mean[ch]:.2f}, diff={diffs[ch]:.2f}")

    if show_plot:
        # 시각화: sw03 이완 구간 전체 채널
        sw_meta = meta["swallows"][1]
        crop = np.load(os.path.join(proc_dir, "swallows", sw_meta["file"]))
        meta_begin = sw_meta["begin_ms"]
        ep = None
        for e in sw_eps:
            if abs(float(e.findtext("beginTime")) - meta_begin) < 100:
                ep = e; break

        les = ep.find("les")
        sw_begin_ms  = float(ep.findtext("beginTime"))
        les_begin_ms = float(les.find("begin").findtext("time"))
        les_end_ms   = float(les.find("end").findtext("time"))
        les_p_xml    = float(les.findtext("pressure"))
        crop_start_s = sw_begin_ms / 1000.0 - 15.0
        rs = int((les_begin_ms / 1000.0 - crop_start_s) * 100)
        re = int((les_end_ms   / 1000.0 - crop_start_s) * 100)
        re = max(re, rs + 1000); re = min(re, len(crop) - 1)
        time_ax = np.arange(4000) / 100.0 - 15.0

        fig, ax = plt.subplots(figsize=(14, 5))
        # 상위 5개 채널 표시
        colors = ["red", "orange", "green", "blue", "purple"]
        for ci, ch in enumerate(np.argsort(diffs)[:5]):
            ax.plot(time_ax, crop[:, ch], color=colors[ci], linewidth=1.2, alpha=0.8,
                    label=f"ch{ch} (IRP={ch_irp_mean[ch]:.1f})")
        ax.axvspan(les_begin_ms/1000 - sw_begin_ms/1000,
                   les_end_ms/1000   - sw_begin_ms/1000,
                   alpha=0.15, color="yellow", label="LES relax window")
        ax.axhline(les_p_xml, color="black", linestyle="--", linewidth=2,
                   label=f"XML LES={les_p_xml:.1f} mmHg")
        ax.axvline(0, color="green", linestyle="--")
        ax.set_xlim(-3, 15)
        ax.set_title(f"{label} — Top 5 IRP candidate channels (sw03)")
        ax.set_xlabel("Time from swallow onset (s)")
        ax.set_ylabel("Pressure (mmHg)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"preprocessing/irp_channels_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: irp_channels_{label}.png")

    return best_ch, xml_irp_mean, ch_irp_mean[best_ch]


ch_A, xml_A, comp_A = find_irp_channel(
    "A01",
    "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva",
    "data/processed/01"
)
ch_B, xml_B, comp_B = find_irp_channel(
    "B01",
    "data/normal subjects/1;13-04-2012.mva",
    "data/processed/B01"
)

print(f"\n=== 결론 ===")
print(f"A01: best IRP channel = ch{ch_A}, XML={xml_A:.1f}, computed={comp_A:.1f}")
print(f"B01: best IRP channel = ch{ch_B}, XML={xml_B:.1f}, computed={comp_B:.1f}")
