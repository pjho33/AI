"""UES 탐지 토폴로지 시각화 — 환자 ID를 인자로 받음"""
import sys, numpy as np, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pid = sys.argv[1]  # e.g. "B01" or "01"
SR = 100; onset = 1500; WIN = 6

def find_ao(crop):
    T2, T3, T4 = 40, 60, 90
    UW = 10; DW = int(10 * SR)
    seg = crop[:, 0:5]; events = []; idx = 0
    while idx <= len(seg) - UW:
        w = seg[idx:idx + UW]
        a2 = np.where(w[:, 2] > T2)[0]
        a3 = np.where(w[:, 3] > T3)[0]
        a4 = np.where(w[:, 4] > T4)[0]
        if len(a2) > 0 and len(a3) > 0 and len(a4) > 0 and a2[0] <= a3[0] <= a4[0]:
            events.append(idx + a2[0]); idx = events[0] + DW
        else:
            idx += 1
    return (None, False) if not events else (events[0], len(events) >= 2)

fps = sorted(glob.glob("data/processed/%s/swallows/sw*.npy" % pid))
if not fps:
    print("No swallows found for %s" % pid); sys.exit(1)

results = []
for fp in fps:
    c = np.load(fp); ao, dbl = find_ao(c)
    results.append((fp, ao, dbl))

detected = [(fp, ao, dbl) for fp, ao, dbl in results if ao is not None]
fallback = [fp for fp, ao, dbl in results if ao is None]
print("%s: 전체 %d / 탐지 %d / double %d / fallback %d" % (
    pid, len(fps), len(detected),
    sum(1 for _, _, d in detected if d), len(fallback)))

PER_PAGE = 7
for page_start in range(0, len(detected), PER_PAGE):
    batch = detected[page_start:page_start + PER_PAGE]
    nb = len(batch)
    fig, axes = plt.subplots(nb, 2, figsize=(22, 5 * nb))
    if nb == 1: axes = [axes]
    fig.suptitle("%s  탐지 %d/%d  하늘=actual_onset  흰점=XML  ⚠=double(IRP제외)" % (
        pid, len(detected), len(fps)), fontsize=11)
    for row, (fp, ao, is_dbl) in enumerate(batch):
        crop = np.load(fp); swname = fp.split("/")[-1].replace(".npy", "")
        status = "⚠double" if is_dbl else "✓"
        xml_t = (onset - ao) / SR
        s = max(0, ao - WIN * SR); e = min(len(crop), ao + WIN * SR)
        seg = crop[s:e, :]; t = (np.arange(len(seg)) - (ao - s)) / SR
        ax0 = axes[row][0]
        ax0.imshow(seg.T, aspect="auto", origin="upper",
                   extent=[t[0], t[-1], 35.5, -0.5], vmin=0, vmax=80, cmap="jet")
        ax0.axvline(0, color="cyan", lw=2.5)
        ax0.axvline(xml_t, color="white", ls="--", lw=1.5)
        ax0.axhline(1.5, color="yellow", ls="--", lw=1)
        ax0.axhline(4.5, color="yellow", ls="--", lw=1)
        ax0.set_yticks([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 35])
        ax0.set_yticklabels(["ch0","ch2","ch4","ch8","ch12","ch16","ch20","ch24","ch28","ch32","ch35"], fontsize=7)
        ax0.set_title("%s %s  XML=%+.1fs" % (swname, status, xml_t), fontsize=10,
                      color="orange" if is_dbl else "cyan")
        ax1 = axes[row][1]
        ax1.plot(t, seg[:, 2], "#4daf4a", lw=2.5, label="ch2(>40)")
        ax1.plot(t, seg[:, 3], "#377eb8", lw=2.5, label="ch3(>60)")
        ax1.plot(t, seg[:, 4], "#984ea3", lw=2.5, label="ch4(>90)")
        ax1.axhline(40, color="#4daf4a", ls=":", lw=1.2)
        ax1.axhline(60, color="#377eb8", ls=":", lw=1.2)
        ax1.axhline(90, color="#984ea3", ls=":", lw=1.2)
        ax1.axvline(0, color="cyan", lw=2.5, label="actual_onset")
        ax1.axvline(xml_t, color="white", ls="--", lw=1.5, label="XML")
        ax1.set_xlim(-WIN, WIN); ax1.set_ylim(-5, 350)
        ax1.legend(fontsize=8, ncol=3, loc="upper right")
        ax1.grid(alpha=0.3); ax1.set_ylabel("mmHg"); ax1.set_facecolor("#111")
    plt.tight_layout(h_pad=0.4)
    fname = "preprocessing/%s_p%d.png" % (pid, page_start // PER_PAGE + 1)
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print("saved: %s" % fname)
    plt.close()
