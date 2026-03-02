"""LES 채널 선택 + IRP window 시각화 — 환자 ID를 인자로 받음
Usage: python3 _plot_les.py A01
출력: /tmp/hrm_les/{pid}_p{n}.png
- 왼쪽: 전체 채널 토폴로지 (36채널)
  - 노란 수평선: best_les_ch 위치
  - 초록 수평선: dist_chs 범위
  - 하늘 수직선: actual_onset
  - 빨간 수직선: actual_onset + 10s (IRP window 끝)
- 오른쪽: LES 채널 압력 trace
  - 노란 영역: IRP window (actual_onset ~ +10s)
  - 수평 점선: IRP4s 값
"""
import sys, numpy as np, glob, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pid = sys.argv[1]
SR = 100

EXCLUDE = {"A22"}
if pid in EXCLUDE:
    print("%s: EXCLUDED" % pid); sys.exit(0)

# ── UES 탐지 (v1.2 dynamic) ─────────────────────────────────────
def find_actual_onset(crop):
    UW = 10; DW = int(10 * SR)
    pre = crop[:500, :]
    rest = np.median(pre[:, 1:5], axis=0)
    mad  = np.median(np.abs(pre[:, 1:5] - rest), axis=0)
    c0_idx = int(np.argmax(rest)); c0 = c0_idx + 1
    if crop[:, c0].max() < rest[c0_idx] + 15:
        return None, False
    ca = max(1, c0-1); cb = c0; cc = min(4, c0+1)
    def thr(ch): return rest[ch-1] + max(15, 2*mad[ch-1])
    ia = np.flatnonzero(crop[:, ca] > thr(ca))
    ib = np.flatnonzero(crop[:, cb] > thr(cb))
    ic = np.flatnonzero(crop[:, cc] > thr(cc))
    if not (len(ia) and len(ib) and len(ic)): return None, False
    ONSET = int(15 * SR)  # crop 내 swallow onset 고정 위치
    SEARCH_LO = int(8 * SR)   # onset - 7s
    SEARCH_HI = int(22 * SR)  # onset + 7s (탐색 상한)
    events = []; start = SEARCH_LO
    while True:
        idxa = np.searchsorted(ia, start)
        if idxa >= len(ia): break
        ja = ia[idxa]
        if ja >= SEARCH_HI: break  # 탐색 범위 초과
        if ja >= len(crop) - UW: break
        idxb = np.searchsorted(ib, ja)
        if idxb >= len(ib) or ib[idxb] >= ja + UW: start = ja+1; continue
        jb = ib[idxb]
        idxc = np.searchsorted(ic, jb)
        if idxc >= len(ic) or ic[idxc] >= ja + UW: start = ja+1; continue
        events.append(ja); start = ja + DW
        if np.searchsorted(ia, start) >= len(ia): break
    if not events: return None, False
    return events[0], len(events) >= 2

# ── LES 채널 자동 선택 (HRM_DEFINITIONS.md 기준) ────────────────
def select_les_ch(crops):
    onset = int(15 * SR)
    pre_s  = max(0, onset - int(10 * SR))
    pre_e  = max(0, onset - int(3  * SR))
    post_s = min(len(crops[0]), onset + int(10 * SR))
    post_e = min(len(crops[0]), onset + int(15 * SR))

    ch_pre  = np.nanmean([c[pre_s:pre_e,   :].mean(axis=0) for c in crops], axis=0)
    ch_post = np.nanmean([c[post_s:post_e, :].mean(axis=0) for c in crops], axis=0)
    ch_rest = (ch_pre + ch_post) / 2.0
    p_gastric = float(ch_pre[35])

    # ── 체부 수축파 distal end: first_hit 순차 진행 마지막 채널 ──
    # 먼저 전체 범위 osc_amp로 crus 채널 파악 → first_hit 탐색에서 제외
    sw_s = onset; sw_e = onset + int(10 * SR)

    MIN_DELAY = int(1.0 * SR)  # 수축파는 UES 이후 최소 1s 뒤에 LES 도달
    def _first_hit(c, ch):
        base = float(c[pre_s:pre_e, ch].mean())
        hits = np.flatnonzero(c[sw_s:sw_e, ch] > base + 10.0)
        # 동시 발화(crus/외부압박): onset 후 1s 이내 → 제외
        valid = [h for h in hits if h >= MIN_DELAY]
        return (sw_s + int(valid[0])) if valid else None

    # 탐색 범위: ch25~34 (skeletal/smooth 전환 이후 평활근 구간)
    # crus 필터 없이 fh 시간 순서만으로 distal 결정
    SEARCH_START = 25
    fh_mean = {}
    for ch in range(SEARCH_START, 35):
        times = [_first_hit(c, ch) for c in crops]
        valid = [t for t in times if t is not None]
        fh_mean[ch] = float(np.mean(valid)) if len(valid) >= 1 else None

    # 단조 진행 마지막 채널 (tolerance 0.5s, 연속 2회 역전/None 시 종료)
    TOLERANCE = 0.5 * SR
    distal_end_ch = SEARCH_START
    prev_t = fh_mean.get(SEARCH_START)
    miss = 0
    for ch in range(SEARCH_START + 1, 35):
        t = fh_mean.get(ch)
        if t is None:
            miss += 1
            if miss >= 2: break
            continue
        if prev_t is None or t >= prev_t - TOLERANCE:
            distal_end_ch = ch; prev_t = t; miss = 0
        else:
            miss += 1
            if miss >= 2: break
    if distal_end_ch <= SEARCH_START:
        distal_end_ch = 29  # fallback

    # ── LES 탐색 범위: distal+1 ~ distal+3, 최소 ch28 ────────────
    les_lo = max(distal_end_ch + 1, 28)
    les_hi = min(les_lo + 4, 34)  # ch33까지만 (ch34 gastric 경계 제외)
    if les_lo >= les_hi:
        les_lo = 28; les_hi = 34

    # ── LES 후보: distal+1 ~ distal+3 (최소 ch28) ────────────────
    les_cands = [ch for ch in range(les_lo, les_hi)]

    # ── 각 후보 채널의 IRP 계산 (actual_onset ~ +10s, 최저 400샘플) ──
    # actual_onset은 crop별로 구하고, 채널별 IRP 중앙값 비교
    ch_irp = {}
    for ch in les_cands:
        irps = []
        for c in crops:
            ao, dbl = find_actual_onset(c)
            if ao is None or dbl: continue
            irp_end = ao + int(10 * SR)
            if irp_end > len(c): continue
            # zero-pad 여부: IRP 범위 내 전체 채널 합이 0인 샘플 있으면 제외
            if np.any(c[ao:irp_end, :].sum(axis=1) == 0): continue
            irps.append(float(np.sort(c[ao:irp_end, ch])[:400].mean()))
        ch_irp[ch] = float(np.median(irps)) if irps else float('inf')

    # IRP 최솟값 채널 = LES (resting < 5mmHg인 채널 제외 - gastric 신호)
    valid_irp = {ch: v for ch, v in ch_irp.items()
                 if v < float('inf') and ch_pre[ch] >= 5.0}
    if valid_irp:
        best = min(valid_irp, key=valid_irp.get)
        best_irp = valid_irp[best]
    elif ch_irp:
        best = min(ch_irp, key=ch_irp.get)
        best_irp = ch_irp[best]
    else:
        best = les_lo; best_irp = float('nan')

    relax_drop = ch_pre[best] - ch_rest[best]
    dist_start = max(distal_end_ch - 5, 0)
    dist_chs   = list(range(dist_start, distal_end_ch + 1))
    return best, dist_chs, distal_end_ch, les_lo, les_hi, relax_drop, ch_pre[best], p_gastric

# ── 데이터 로드 ──────────────────────────────────────────────────
fps = sorted(glob.glob("data/processed/%s/swallows/sw*.npy" % pid))
if not fps:
    print("No swallows found for %s" % pid); sys.exit(1)

# LES 채널 결정: 수축파가 가장 깊이 내려간 swallow 상위 3개 기준
# → 같은 환자에서 체부 길이가 가장 잘 드러난 삼킴 = LES 위치 참조에 최적
all_crops = [(fp, np.load(fp, mmap_mode='r')) for fp in fps]
onset_s = int(15 * SR)
pre_base = np.nanmean(
    [np.asarray(c)[max(0,onset_s-int(10*SR)):max(0,onset_s-int(3*SR)), :].mean(0)
     for _, c in all_crops], axis=0)
p_gastric_quick = float(pre_base[35])
pthr_quick = max(30.0, p_gastric_quick + 15.0)

def _distal_end(c_arr):
    """수축파가 도달한 가장 낮은 채널 번호 (외부 압박 제외)"""
    seg = c_arr[onset_s:onset_s+int(10*SR), :]
    pk = seg.max(axis=0)
    chs = [ch for ch in range(20, 36)
           if pk[ch] > pthr_quick and pk[ch] > pre_base[ch] + 15.0]
    return max(chs) if chs else 20

distal_scores = [_distal_end(np.asarray(c)) for _, c in all_crops]
# double swallow 제외: pre 구간 UES 두 번 발화 여부 체크
_ao_list = [find_actual_onset(np.asarray(c)) for _, c in all_crops]
_valid_idx = [i for i, (ao, dbl) in enumerate(_ao_list) if ao is not None and not dbl]
if len(_valid_idx) >= 3:
    top_idx = sorted(_valid_idx, key=lambda i: distal_scores[i], reverse=True)[:3]
else:
    top_idx = sorted(range(len(all_crops)), key=lambda i: distal_scores[i], reverse=True)[:3]
crops_ref = [np.asarray(all_crops[i][1]) for i in top_idx]
best_les_ch, dist_chs, distal_end_ch, les_lo, les_hi, relax_drop, les_rest, p_gastric = select_les_ch(crops_ref)

print("%s: LES=ch%d  relax_drop=%.1f  resting=%.1f  gastric=%.1f  dist_chs=%s" % (
    pid, best_les_ch, relax_drop, les_rest, p_gastric, dist_chs))

# ── 각 swallow 플롯 ──────────────────────────────────────────────
WIN_L = 4   # onset 왼쪽 4초
WIN_R = 14  # onset 오른쪽 14초 (LES 이완+체부수축 전체 확인)
PER_PAGE = 6

results = []
for fp in fps:
    c = np.load(fp, mmap_mode='r')
    ao, dbl = find_actual_onset(c)
    results.append((fp, c, ao, dbl))

detected = [(fp, c, ao, dbl) for fp, c, ao, dbl in results if ao is not None]
fallback  = [fp for fp, c, ao, dbl in results if ao is None]
irp_valid = [(fp, c, ao, dbl) for fp, c, ao, dbl in detected
             if not dbl and ao + int(10*SR) <= len(c)]
print("전체 %d / 탐지 %d / double %d / IRP유효 %d / fallback %d" % (
    len(fps), len(detected),
    sum(1 for *_, d in detected if d),
    len(irp_valid), len(fallback)))

os.makedirs("/tmp/hrm_les", exist_ok=True)

for page_start in range(0, len(detected), PER_PAGE):
    batch = detected[page_start:page_start + PER_PAGE]
    nb = len(batch)
    fig, axes = plt.subplots(nb, 2, figsize=(22, 5 * nb))
    if nb == 1: axes = [axes]
    fig.suptitle("%s  LES=ch%d  dist=%s  gastric=%.1f  ⚠=double  ✗=IRP불가" % (
        pid, best_les_ch, dist_chs, p_gastric), fontsize=11)

    for row, (fp, crop, ao, is_dbl) in enumerate(batch):
        crop = np.asarray(crop)
        swname = fp.split("/")[-1].replace(".npy", "")
        irp_end_chk = ao + int(10*SR)
        has_zpad = (irp_end_chk <= len(crop)) and np.any(crop[ao:irp_end_chk, :].sum(axis=1) == 0)
        irp_ok = (not is_dbl) and (irp_end_chk <= len(crop)) and not has_zpad

        # IRP 계산
        irp_val = np.nan
        if irp_ok:
            les_trace = crop[:, best_les_ch]
            irp_win = les_trace[ao: ao + int(10*SR)]
            irp_val = float(np.sort(irp_win)[:400].mean())

        # 표시 구간: onset 기준 left=4s, right=14s
        s = max(0, ao - WIN_L*SR); e = min(len(crop), ao + WIN_R*SR)
        seg = crop[s:e, :]; t = (np.arange(len(seg)) - (ao - s)) / SR
        ao_rel = 0.0
        irp_end_rel = 10.0

        # ── 왼쪽: 토폴로지 ───────────────────────────────────────
        ax0 = axes[row][0]
        ax0.imshow(seg.T, aspect="auto", origin="upper", interpolation="nearest",
                   extent=[t[0], t[-1], 35.5, -0.5], vmin=0, vmax=80, cmap="jet")
        ax0.axvline(ao_rel, color="cyan", lw=2.0)
        if irp_ok:
            ax0.axvline(irp_end_rel, color="red", lw=1.5, ls="--")
            ax0.axvspan(ao_rel, irp_end_rel, alpha=0.10, color="yellow")
        ax0.axhline(best_les_ch, color="yellow", lw=2.0, ls="-", alpha=0.9)
        if dist_chs:
            ax0.axhspan(min(dist_chs)-0.5, max(dist_chs)+0.5, alpha=0.15, color="lime")
        ax0.axhline(distal_end_ch, color="white", lw=1.0, ls=":", alpha=0.6)
        ax0.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, 29, 30, 31, 32, 33, 35])
        ax0.set_yticklabels(["ch0","ch4","ch8","ch12","ch16","ch20","ch24",
                             "28","29","30","31","32","33","ch35"], fontsize=7)
        status = "dbl" if is_dbl else ("IRP=%.1f" % irp_val if irp_ok else "no_IRP")
        ax0.set_title("%s %s  LES=ch%d" % (swname, status, best_les_ch),
                      fontsize=10, color="orange" if is_dbl else ("lime" if irp_ok else "red"))
        ax0.set_xlabel("t (s, onset=0)"); ax0.set_ylabel("ch")
        ax0.set_xlim(-WIN_L, WIN_R)

        # ── 오른쪽: ch28~34 전체 + dist_chs trace ───────────────
        ax1 = axes[row][1]
        les_colors = {28:"#e41a1c", 29:"#ff7f00", 30:"#ffff33",
                      31:"#4daf4a", 32:"#377eb8", 33:"#984ea3"}
        for ch in range(28, 34):
            lw = 2.8 if ch == best_les_ch else 1.2
            alpha = 1.0 if ch == best_les_ch else 0.55
            ax1.plot(t, seg[:, ch], color=les_colors[ch], lw=lw,
                     alpha=alpha, label="ch%d%s" % (ch, "*" if ch==best_les_ch else ""))
        ax1.axvline(ao_rel, color="cyan", lw=2.0)
        if irp_ok:
            ax1.axvspan(ao_rel, irp_end_rel, alpha=0.10, color="yellow")
            ax1.axhline(irp_val, color="white", lw=1.5, ls="--", label="IRP=%.1f" % irp_val)
        ax1.axhline(p_gastric, color="gray", lw=1.0, ls=":", label="gastric=%.1f" % p_gastric)
        ax1.axhline(15, color="magenta", lw=1.0, ls="--", alpha=0.7, label="cutoff=15")
        ax1.set_xlim(-WIN_L, WIN_R); ax1.set_ylim(-5, 120)
        ax1.legend(fontsize=7, ncol=4, loc="upper right")
        ax1.grid(alpha=0.3); ax1.set_ylabel("mmHg"); ax1.set_facecolor("#111")

    plt.tight_layout(h_pad=0.4)
    fname = "/tmp/hrm_les/%s_p%d.png" % (pid, page_start // PER_PAGE + 1)
    plt.savefig(fname, dpi=90)
    print("saved: %s" % fname)
    plt.close()
