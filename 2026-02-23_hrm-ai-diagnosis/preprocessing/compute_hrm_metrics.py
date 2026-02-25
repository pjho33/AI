"""
HRM 주요 지표 직접 계산 (Chicago Classification v4.0 기준)
=============================================================
raw 압력 데이터 (4000, 36) @ 100Hz, pre=15s, post=25s 기준

계산 지표:
  - IRP4s  : swallow 후 10초 중 최저 4초 LES 압력 평균
  - LESP   : swallow 전 안정기 LES 압력 (pre 5~15s 평균)
  - DCI    : (peak - 20) × duration × length (원위부 식도 채널)
  - CFV    : 수축파 전파 속도 (cm/s)
  - IBP    : 볼루스 내압 (LES 바로 위 채널 peak)
  - UESP   : UES 안정압 (pre 5~15s 평균)
  - UES_relax: 삼킴 시 UES 최저압

채널 매핑 (probe XML location 기반):
  - LES 채널: location 0 ~ -5 cm (ch31~35, 역순 매핑 후)
  - 원위부 식도: location -5 ~ -20 cm
  - UES 채널: location -28 ~ -35 cm
"""

import json
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from parse_mva import extract_zlib_blocks, find_xml_blocks
    from parse_mva_binary import parse_pressure
    from crop_swallow import crop_swallows, estimate_block0_start
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from parse_mva import extract_zlib_blocks, find_xml_blocks
    from parse_mva_binary import parse_pressure
    from crop_swallow import crop_swallows, estimate_block0_start

SR = 100  # Hz
PRE_SEC = 15.0


def get_channel_locations(mva_path):
    """probe XML에서 채널별 위치(cm) 반환. 역순 매핑 적용."""
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    probe_root = ET.fromstring(xml_map['probe'])
    sensors = sorted(probe_root.find('pressure').findall('sensor'),
                     key=lambda s: int(s.findtext('index')))
    # decoded ch(35-tc) = txt ch(tc) → 역순
    locs = [float(s.findtext('location') or 0) for s in sensors]
    locs_reversed = locs[::-1]  # txt 채널 순서
    return np.array(locs_reversed)  # (36,) cm, 0=LES 근처, 음수=위쪽


def compute_irp4s(les_pressure_trace, relax_start=0, window_s=10.0, irp_s=4.0):
    """
    IRP4s: LES 이완 시작(relax_start) 후 window_s 초 중
           가장 낮은 irp_s 초(비연속 가능) 평균.
    les_pressure_trace: 1D array (n_t,) LES 채널 압력
    relax_start: LES 이완 시작 sample index (XML les.begin.time 기준)
    """
    n_window = int(window_s * SR)
    n_irp    = int(irp_s * SR)
    seg = les_pressure_trace[relax_start: relax_start + n_window]
    if len(seg) < n_irp:
        return np.nan
    # 가장 낮은 n_irp 샘플의 평균 (비연속)
    sorted_vals = np.sort(seg)
    return float(sorted_vals[:n_irp].mean())


def compute_dci(pressure_crop, ch_locs, post_start, les_loc,
                threshold=20.0, dist_chs=None):
    """
    DCI = integral of P(x,t) dx dt  (P > 20 mmHg 구간만, 미만은 0)
    단위: mmHg · s · cm
    """
    onset = post_start
    end   = min(pressure_crop.shape[0], onset + int(30 * SR))
    dt    = 1.0 / SR  # seconds per sample

    distal_chs = dist_chs if dist_chs else [
        i for i, loc in enumerate(ch_locs) if les_loc - 20 <= loc <= les_loc - 5
    ]
    if not distal_chs:
        return 0.0

    dci = 0.0
    for i, ch in enumerate(distal_chs):
        seg = pressure_crop[onset:end, ch]
        # 20 mmHg 미만은 0으로 처리
        p_above = np.where(seg > threshold, seg, 0.0)
        if p_above.sum() == 0:
            continue
        # 공간 간격 (cm): 인접 채널과의 거리
        if i + 1 < len(distal_chs):
            dx = abs(ch_locs[distal_chs[i]] - ch_locs[distal_chs[i + 1]])
        else:
            dx = 1.0
        # 시간 적분 후 공간 곱
        dci += float(p_above.sum()) * dt * dx

    return round(dci, 1)


def compute_cfv(pressure_crop, ch_locs, post_start, les_loc=-31.0, dist_chs=None):
    """
    CFV: 원위부 식도 채널에서 수축파 peak 시간 → 선형 회귀로 속도 계산
    """
    onset = post_start
    end   = min(pressure_crop.shape[0], onset + int(15 * SR))

    distal_chs = dist_chs if dist_chs else [
        i for i, loc in enumerate(ch_locs) if les_loc - 20 <= loc <= les_loc - 5
    ]
    if len(distal_chs) < 3:
        return np.nan

    peak_times = []
    locs_used  = []
    for ch in distal_chs:
        seg = pressure_crop[onset:end, ch]
        if seg.max() < 20:
            continue
        t_peak = float(np.argmax(seg)) / SR
        peak_times.append(t_peak)
        locs_used.append(ch_locs[ch])

    if len(peak_times) < 3:
        return np.nan

    # 선형 회귀: location(cm) vs time(s) → slope = cm/s
    locs_arr  = np.array(locs_used)
    times_arr = np.array(peak_times)
    try:
        slope, _ = np.polyfit(times_arr, locs_arr, 1)
        return round(abs(slope), 2)
    except Exception:
        return np.nan


def compute_ibp(pressure_crop, ch_locs, post_start, les_loc):
    """IBP: LES 바로 위 채널(les_loc-1 ~ les_loc-5cm)의 peak 압력."""
    onset = post_start
    end   = min(pressure_crop.shape[0], onset + int(10 * SR))
    ibp_chs = [i for i, loc in enumerate(ch_locs)
               if les_loc - 5 <= loc <= les_loc - 1]
    if not ibp_chs:
        return np.nan
    vals = [pressure_crop[onset:end, ch].max() for ch in ibp_chs]
    return round(float(np.mean(vals)), 1)


def compute_resting(pressure_crop, ch_indices, pre_start=500, pre_end=1400):
    """안정기 압력 (pre 5~14s = sample 500~1400)."""
    seg = pressure_crop[pre_start:pre_end, :][:, ch_indices]
    return float(seg.mean())


def classify_chicago(irp, dci_vals, cfv_vals):
    """
    Chicago Classification v4.0 간략 분류.
    irp: float (mmHg)
    dci_vals: list of float per swallow
    cfv_vals: list of float per swallow
    """
    n = len(dci_vals)
    if n == 0:
        return "Unknown"

    pct_failed = sum(1 for d in dci_vals if d < 100) / n * 100
    pct_ineffective = sum(1 for d in dci_vals if d < 450) / n * 100
    pct_hyper = sum(1 for d in dci_vals if d > 8000) / n * 100
    pct_premature = sum(1 for v in cfv_vals if v is not None
                        and not np.isnan(v) and v > 9) / max(1, sum(
        1 for v in cfv_vals if v is not None and not np.isnan(v))) * 100

    if irp > 15:
        if pct_failed == 100:
            return "Achalasia Type I"
        elif pct_hyper >= 20:
            return "Achalasia Type III"
        else:
            return "EGJ Outflow Obstruction"
    else:
        if pct_failed == 100:
            return "Absent Contractility"
        elif pct_premature >= 20:
            return "Distal Esophageal Spasm"
        elif pct_hyper >= 20:
            return "Hypercontractile (Jackhammer)"
        elif pct_ineffective > 50:
            return "Ineffective Esophageal Motility"
        else:
            return "Normal"


def compute_patient_metrics(mva_path, processed_dir):
    """환자 1명의 전체 지표 계산."""
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)

    # 환자 정보
    info = ET.fromstring(xml_map['info'])
    age    = info.findtext('dob') or ''
    gender = info.findtext('gender') or ''
    height = info.findtext('height') or ''
    proc   = info.findtext('procedure') or ''

    # 채널 위치
    ch_locs = get_channel_locations(mva_path)

    # analysis XML에서 swallow별 LES relaxation 시간 추출
    analysis = ET.fromstring(xml_map['analysis'])
    sw_episodes = [ep for ep in analysis.find('episodes') if ep.tag == 'swallow']

    # LES 채널 자동 탐색:
    # 1) XML les.pressure (이완 최저압) 수집
    # 2) 각 swallow 이완 구간에서 채널별 최저값 계산
    # 3) XML LES pressure에 가장 가까운 채널 = LES
    xml_les_pressures = []
    for ep in sw_episodes:
        les_elem = ep.find('les')
        if les_elem is not None:
            p = les_elem.findtext('pressure')
            if p: xml_les_pressures.append(float(p))
    xml_les_p_mean = float(np.median(xml_les_pressures)) if xml_les_pressures else 5.0

    # 처음 몇 개 swallow crop으로 채널별 이완 구간 최저값 계산
    sw_dir_tmp = Path(processed_dir) / 'swallows'
    sw_files_tmp = sorted(sw_dir_tmp.glob('sw*.npy'))[:min(5, len(sw_episodes))]
    ch_relax_mins = np.full((len(sw_files_tmp), 36), np.nan)

    for fi, fp_tmp in enumerate(sw_files_tmp):
        if fi >= len(sw_episodes):
            break
        crop_tmp = np.load(fp_tmp)
        ep = sw_episodes[fi]
        sw_begin_ms = float(ep.findtext('beginTime') or 0)
        les_elem = ep.find('les')
        if les_elem is None:
            continue
        b_elem = les_elem.find('begin')
        e_elem = les_elem.find('end')
        if b_elem is None or e_elem is None:
            continue
        les_bt = float(b_elem.findtext('time') or sw_begin_ms)
        les_et = float(e_elem.findtext('time') or sw_begin_ms + 10000)
        crop_start_s = sw_begin_ms / 1000.0 - PRE_SEC
        rs = int((les_bt / 1000.0 - crop_start_s) * SR)
        re = int((les_et / 1000.0 - crop_start_s) * SR)
        re = max(re, rs + 500); re = min(re, len(crop_tmp) - 1)
        for ch in range(36):
            seg = crop_tmp[rs:re, ch]
            if len(seg) > 0:
                ch_relax_mins[fi, ch] = float(seg.min())

    onset = int(PRE_SEC * SR)  # sample 1500 (swallow begin)

    # LES 채널 자동 탐색:
    # IRP window(swallow begin -5s~+3s)에서 최저 400 samples 평균이
    # XML les.pressure(이완 최저압)에 가장 가까운 채널 = LES
    irp_w_start = onset - int(5 * SR)   # sample 1000
    irp_w_end   = onset + int(3 * SR)   # sample 1800

    if sw_files_tmp:
        # 처음 5개 swallow에서 채널별 IRP4s 계산
        ch_irp4s_all = np.full((len(sw_files_tmp), 36), np.nan)
        for fi, fp_tmp in enumerate(sw_files_tmp):
            if fi >= len(sw_episodes): break
            c = np.load(fp_tmp)
            for ch in range(36):
                seg = c[irp_w_start:irp_w_end, ch]
                if len(seg) >= 400:
                    ch_irp4s_all[fi, ch] = float(np.sort(seg)[:400].mean())
        ch_irp4s_mean = np.nanmean(ch_irp4s_all, axis=0)  # (36,)
    else:
        ch_irp4s_mean = np.zeros(36)

    # XML LES pressure median
    xml_les_p_mean = float(np.median(xml_les_pressures)) if xml_les_pressures else 5.0

    # 채널 방향: ch0=pharynx(위), ch35=stomach(아래)
    # LES = ch28~34 (stomach 쪽), UES = ch0~4 (pharynx 쪽)
    # LES 채널 선택: ch28~34 중 삼킴 후(t=0~5s) after-contraction 최대 채널
    if sw_files_tmp:
        ch_pre_mean = np.nanmean(
            np.stack([np.load(fp)[500:1400, :].mean(axis=0) for fp in sw_files_tmp]),
            axis=0)
        ch_post_mean = np.nanmean(
            np.stack([np.load(fp)[onset:onset + int(5 * SR), :].mean(axis=0)
                      for fp in sw_files_tmp]),
            axis=0)
    else:
        ch_pre_mean = np.zeros(36)
        ch_post_mean = np.zeros(36)

    p_gastric = float(ch_pre_mean[35])  # ch35 = stomach baseline (QC용)

    # LES 채널 탐지:
    # 1) 수축파 distal end = t=0~10s 구간에서 peak > 30 mmHg인 채널 중 가장 높은 ch 번호
    # 2) LES = distal end 채널 바로 위(pharynx 방향) 1~4번째 채널
    if sw_files_tmp:
        ch_peak = np.nanmean(
            np.stack([np.load(fp)[onset:onset + int(10 * SR), :].max(axis=0)
                      for fp in sw_files_tmp]),
            axis=0)
    else:
        ch_peak = np.zeros(36)

    # 수축파 distal end: ch20~35 중 peak가 위내압(ch35)+15 mmHg 이상인 가장 높은 ch
    # ch20 미만은 상부 식도이므로 LES 탐지에 사용하지 않음
    peristalsis_threshold = max(30.0, p_gastric + 15.0)
    peristalsis_chs = [ch for ch in range(20, 36) if ch_peak[ch] > peristalsis_threshold]
    if peristalsis_chs:
        distal_end_ch = max(peristalsis_chs)
    else:
        distal_end_ch = 32  # fallback: LES 기본 위치

    # LES 채널 선택 로직:
    # 조건1 (주): ch20~33 중 resting(삼킴 전후 안정 구간) 평균이 가장 높은 채널
    #   - pre:  onset-10s ~ onset-3s  (이전 수축파 영향 제거)
    #   - post: onset+10s ~ onset+15s (수축파 완전 통과 후)
    # 조건2 (보조 tiebreak): 호흡성 oscillation 파워 (0.2~0.5Hz)
    #   - oscillation이 없는 환자도 있으므로 점수 가산에만 사용

    len_crop = 4000
    if sw_files_tmp:
        crops_tmp  = [np.load(fp) for fp in sw_files_tmp]
        len_crop   = len(crops_tmp[0])
    else:
        crops_tmp  = []

    pre_s  = max(0, onset - int(10 * SR))
    pre_e  = max(0, onset - int(3  * SR))
    post_s = min(len_crop, onset + int(10 * SR))
    post_e = min(len_crop, onset + int(15 * SR))

    if crops_tmp and pre_e > pre_s and post_e > post_s:
        ch_pre_mean  = np.nanmean([c[pre_s:pre_e,   :].mean(axis=0) for c in crops_tmp], axis=0)
        ch_post_mean = np.nanmean([c[post_s:post_e, :].mean(axis=0) for c in crops_tmp], axis=0)
    else:
        ch_pre_mean = ch_post_mean = np.zeros(36)

    ch_resting = (ch_pre_mean + ch_post_mean) / 2.0

    # 호흡성 oscillation 점수: pre 구간에서 0.2~0.5Hz 대역 FFT 파워
    osc_score = np.zeros(36)
    if crops_tmp and pre_e > pre_s:
        pre_len = pre_e - pre_s
        freqs   = np.fft.rfftfreq(pre_len, d=1.0/SR)
        band    = (freqs >= 0.2) & (freqs <= 0.5)
        for ch in range(20, 34):
            seg = np.nanmean([c[pre_s:pre_e, ch] for c in crops_tmp], axis=0)
            seg = seg - seg.mean()
            pw  = np.abs(np.fft.rfft(seg)) ** 2
            osc_score[ch] = pw[band].sum() if band.any() else 0.0
        # 정규화 (0~1)
        osc_max = osc_score[20:34].max()
        if osc_max > 0:
            osc_score = osc_score / osc_max

    # 조건3: 수축파 끝(distal_end_ch) 이후 채널만 탐색
    les_lo = max(distal_end_ch - 1, 20)
    les_hi = 34  # ch33 이하
    if les_lo >= les_hi:
        les_lo = 28  # fallback

    # ── actual_onset 탐지 (HRM_DEFINITIONS.md Section 2 기준) ──────
    # 탐색 구간: onset-5s ~ onset+20s
    # 조건: 0.1s(10샘플) 창 내에서
    #   ch2, ch3, ch4 각각 한 번 이상 40mmHg 이상 AND
    #   첫 도달 순서: ch2[0] <= ch3[0] <= ch4[0] (ch2→ch3→ch4 하강 순서)
    # 연달아 삼킴: 5초(500샘플) 이내 두 번째 이벤트 탐지 → is_double=True → IRP 제외
    # 반환: (actual_onset_sample, is_double)
    def _find_actual_onset(crop):
        # XML onset 무시 — crop 전체에서 UES 수축 탐지
        # 채널별 threshold: ch2>40, ch3>60, ch4>90 (UES resting pressure 반영)
        # 순서 조건: ch2→ch3→ch4 순서로 각 threshold 도달
        # actual_onset = 조건 만족 창 내 ch2가 40mmHg 처음 넘는 샘플
        # double = 첫 이벤트 후 10초 이내 두 번째 UES 이벤트 → IRP 제외
        T2, T3, T4 = 40, 60, 90
        UES_WIN = 10
        DOUBLE_WIN = int(10 * SR)
        seg = crop[:, 0:5]
        events = []
        idx = 0
        while idx <= len(seg) - UES_WIN:
            win = seg[idx:idx + UES_WIN, :]
            above2 = np.where(win[:, 2] > T2)[0]
            above3 = np.where(win[:, 3] > T3)[0]
            above4 = np.where(win[:, 4] > T4)[0]
            if (len(above2) > 0 and len(above3) > 0 and len(above4) > 0 and
                    above2[0] <= above3[0] <= above4[0]):
                ao = idx + above2[0]  # ch2가 T2 처음 넘는 샘플
                events.append(ao)
                # 첫 이벤트 후 10s 뒤부터 두 번째 탐색
                idx = events[0] + DOUBLE_WIN
            else:
                idx += 1
        if len(events) == 0:
            return None, False  # 탐지 실패
        is_double = len(events) >= 2
        return events[0], is_double

    actual_onsets_tmp = [_find_actual_onset(c) for c in crops_tmp] if crops_tmp else []
    # ao_sample, is_double 분리 (None=탐지실패)
    ao_samples_tmp  = [r[0] for r in actual_onsets_tmp]
    ao_doubles_tmp  = [r[1] for r in actual_onsets_tmp]
    valid_onsets = [ao for ao in ao_samples_tmp if ao is not None]

    # 이완 구간: actual_onset-1s ~ actual_onset+3s (actual_onset 기준)
    if valid_onsets and crops_tmp:
        relax_segs = []
        for c, ao, is_dbl in zip(crops_tmp, ao_samples_tmp, ao_doubles_tmp):
            if ao is None or is_dbl:
                continue  # 탐지실패 또는 연달아 삼킴 제외
            r_s = max(0, ao - int(1 * SR))
            r_e = min(len(c), ao + int(3 * SR))
            if r_e > r_s:
                relax_segs.append(c[r_s:r_e, :].mean(axis=0))
        ch_relax_mean = np.nanmean(relax_segs, axis=0) if relax_segs else np.zeros(36)
    else:
        relax_s = onset - int(1 * SR)
        relax_e = onset + int(3 * SR)
        ch_relax_mean = (np.nanmean([c[relax_s:relax_e, :].mean(axis=0)
                                     for c in crops_tmp], axis=0)
                         if crops_tmp else np.zeros(36))

    # LES 후보 조건:
    #   (A) resting > gastric + 5 mmHg  → 진짜 괄약근 고압대
    #   (B) relax_drop > 0              → 삼킴 시 실제로 이완됨 (대동맥 압박 채널 제외)
    # 두 조건 모두 만족하는 채널 중 drop이 가장 큰 채널 = LES
    relax_drop = ch_resting - ch_relax_mean   # 클수록 이완이 잘 됨

    # LES 후보: resting > gastric+5 AND 이완폭 > 2 mmHg (대동맥 압박 채널 제외)
    candidates = [ch for ch in range(les_lo, les_hi)
                  if ch_resting[ch] > p_gastric + 5 and relax_drop[ch] > 2]

    if candidates:
        les_score = np.zeros(36)
        for ch in candidates:
            les_score[ch] = relax_drop[ch] + 0.2 * osc_score[ch]
        best_les_ch = candidates[int(np.argmax([les_score[ch] for ch in candidates]))]
    else:
        # fallback1: drop > 0 이완 있는 채널 중 resting 가장 높은 것
        fb1 = [ch for ch in range(les_lo, les_hi) if relax_drop[ch] > 0]
        if fb1:
            best_les_ch = fb1[int(np.argmax([ch_resting[ch] for ch in fb1]))]
        else:
            # fallback2: 범위 내 resting 가장 높은 채널 (정상 환자 대비)
            best_les_ch = les_lo + int(np.argmax(ch_resting[les_lo:les_hi]))
    les_candidate_weak = len(candidates) == 0

    les_chs  = [best_les_ch]
    les_loc  = ch_locs[best_les_ch]
    # 원위부 식도: LES에서 pharynx 방향 (낮은 ch 번호), 약 5cm
    dist_end   = max(best_les_ch - 2, 0)
    dist_start = max(best_les_ch - 8, 0)
    dist_chs   = list(range(dist_start, dist_end))
    # UES: ch0~4 (pharynx 쪽)
    ues_chs  = list(range(0, 5))

    # swallow 크롭 로드
    sw_dir = Path(processed_dir) / 'swallows'
    sw_files = sorted(sw_dir.glob('sw*.npy'))

    irp_list, dci_list, cfv_list = [], [], []
    lesp_list, uesp_list, ibp_list = [], [], []
    ues_relax_list = []
    sw_level_records = []  # swallow-level: {sw_idx, irp, dci, cfv, lesp, qc_pass}

    # block0 시작 시간 (crop 시간 기준 계산용)
    meta_path = Path(processed_dir) / 'meta.json'
    block0_start_s = PRE_SEC  # 기본값
    if meta_path.exists():
        with open(meta_path) as f:
            import json as _json
            meta = _json.load(f)
            block0_start_s = meta.get('block0_start_s', PRE_SEC)

    n_total, n_qc_fail = 0, 0

    for fi, fp in enumerate(sw_files):
        crop = np.load(fp)  # (4000, 36)
        n_total += 1

        # ── Swallow QC: 수축파 존재 여부 확인 ──────────────────
        # ch8~16에서 crop 전체 구간 peak > 30mmHg 필요
        qc_seg_start = max(0, onset - int(5 * SR))
        qc_seg_end   = min(len(crop), onset + int(25 * SR))
        body_seg = crop[qc_seg_start:qc_seg_end, 8:17]
        if body_seg.max() < 30:
            n_qc_fail += 1
            continue  # 수축파 없는 swallow 제외

        # actual_onset: 인두(ch0~4) 수축이 3개 이상 채널에서 동시에 급증하는 시점
        # 탐색: onset -5s ~ +20s (XML onset이 실제 삼킴보다 많이 앞서는 경우 대비)
        sw_s = max(0, onset - int(5 * SR))
        sw_e = min(len(crop), onset + int(20 * SR))

        seg_5ch = crop[sw_s:sw_e, 0:5]  # (n_samples, 5)

        # ── UES 탐지: _find_actual_onset() 함수로 일원화 ──────────────
        actual_onset, is_double_swallow = _find_actual_onset(crop)

        # IRP4s: actual_onset 탐지 성공 + 연달아 삼킴 아닌 경우에만 계산
        pharynx_detected = (actual_onset is not None) and (not is_double_swallow)
        irp = np.nan
        if les_chs and pharynx_detected:
            irp_start = max(0, actual_onset)
            irp_end   = actual_onset + int(10 * SR)
            # IRP window가 crop 내에 충분히 있어야 함 (최소 1000 samples = 10s)
            if irp_end <= len(crop):
                les_trace  = crop[:, les_chs].mean(axis=1)
                irp_window = les_trace[irp_start:irp_end]
                irp = float(np.sort(irp_window)[:400].mean())
                irp = max(0.0, irp)
        if not np.isnan(irp):
            irp_list.append(irp)

        # LESP: 안정기 LES 압력
        if les_chs:
            lesp = compute_resting(crop, les_chs)
            lesp_list.append(lesp)

        # UESP: 안정기 UES 압력
        if ues_chs:
            uesp = compute_resting(crop, ues_chs)
            uesp_list.append(uesp)
            # UES 이완: swallow 후 3초 최저압
            ues_seg = crop[onset: onset + int(3 * SR), :][:, ues_chs].mean(axis=1)
            ues_relax_list.append(float(ues_seg.min()))

        # DCI
        dci = compute_dci(crop, ch_locs, onset, les_loc, dist_chs=dist_chs)
        dci_list.append(dci)

        # CFV
        cfv = compute_cfv(crop, ch_locs, onset, les_loc, dist_chs=dist_chs)
        cfv_list.append(cfv)

        # IBP
        ibp = compute_ibp(crop, ch_locs, onset, les_loc)
        ibp_list.append(ibp)

        # swallow-level 기록 저장
        sw_level_records.append({
            'sw_file':      fp.name,
            'actual_onset_t': round((actual_onset - onset) / SR, 2) if actual_onset is not None else None,
            'irp':  round(irp, 2) if not np.isnan(irp) else None,
            'dci':  round(dci, 2) if dci is not None and not np.isnan(dci) else None,
            'cfv':  round(cfv, 2) if cfv is not None and not np.isnan(cfv) else None,
            'lesp': round(lesp, 2) if les_chs and not np.isnan(lesp) else None,
        })

    def safe_mean(lst):
        vals = [v for v in lst if v is not None and not np.isnan(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def safe_median(lst):
        vals = [v for v in lst if v is not None and not np.isnan(v)]
        return round(float(np.median(vals)), 2) if vals else None

    def safe_std(lst):
        vals = [v for v in lst if v is not None and not np.isnan(v)]
        return round(float(np.std(vals)), 2) if vals else None

    irp_mean   = safe_mean(irp_list)
    irp_median = safe_median(irp_list)  # Chicago CC v3.0: median IRP 사용
    dci_mean   = safe_mean(dci_list)
    cfv_mean   = safe_mean(cfv_list)
    lesp_mean  = safe_mean(lesp_list)
    uesp_mean  = safe_mean(uesp_list)
    ibp_mean   = safe_mean(ibp_list)
    ues_relax  = safe_mean(ues_relax_list)

    # Chicago 분류: median IRP 기준 (CC v3.0)
    chicago = classify_chicago(
        irp_median if irp_median is not None else 0,
        dci_list,
        cfv_list
    )

    # 무효 연동 비율
    pct_ineffective = round(
        sum(1 for d in dci_list if d < 450) / len(dci_list) * 100, 1
    ) if dci_list else None

    # QC 플래그
    lesp_delta = round(lesp_mean - p_gastric, 2) if lesp_mean is not None else None
    qc_warnings = [f"swallows: {n_total - n_qc_fail}/{n_total} valid (QC pass)"]
    if n_qc_fail > 0:
        qc_warnings.append(f"{n_qc_fail} swallows excluded (no peristalsis)")
    if best_les_ch == 0:
        qc_warnings.append("les_ch==0 (stomach)")
    if lesp_mean is not None and lesp_mean < 0:
        qc_warnings.append("LESP<0")
    if les_candidate_weak:
        qc_warnings.append("fallback_used")
    if lesp_mean is not None and not (5 <= lesp_mean <= 60):
        qc_warnings.append("LESP_out_of_range")

    return {
        'age':            age,
        'gender':         gender,
        'height':         height,
        'procedure':      proc,
        'n_swallows':     len(sw_files),
        'IRP4s_mean':     irp_mean,
        'IRP4s_median':   irp_median,
        'IRP4s_std':      safe_std(irp_list),
        'LESP_mean':      lesp_mean,
        'LESP_std':       safe_std(lesp_list),
        'LESP_delta':     lesp_delta,
        'p_gastric':      round(p_gastric, 2),
        'DCI_mean':       dci_mean,
        'DCI_std':        safe_std(dci_list),
        'CFV_mean':       cfv_mean,
        'CFV_std':        safe_std(cfv_list),
        'IBP_mean':       ibp_mean,
        'UESP_mean':      uesp_mean,
        'UES_relax_mean': ues_relax,
        'pct_ineffective': pct_ineffective,
        'chicago':        chicago,
        'les_chs':        les_chs,
        'ues_chs':        ues_chs,
        'les_candidate_weak': les_candidate_weak,
        'qc_warnings':    qc_warnings,
        'swallow_records': sw_level_records,
    }


def run_all():
    from glob import glob

    PROCESSED = Path('data/processed')
    DATA_DIR  = Path('data/normal subjects')

    # A그룹 (01~31, 09 제외)
    a_files = {
        f"A{int(Path(f).name.split('.')[0]):02d}": f
        for f in glob(str(DATA_DIR / '*.mva'))
        if '.' in Path(f).name.split(';')[0]
        and Path(f).name.split('.')[0].isdigit()
    }
    # B그룹 (번호;날짜.mva)
    b_files = {
        f"B{int(Path(f).name.split(';')[0]):02d}": f
        for f in glob(str(DATA_DIR / '*.mva'))
        if ';' in Path(f).name
        and '.' not in Path(f).name.split(';')[0]
    }

    all_files = {**a_files, **b_files}
    print(f"Total: {len(all_files)} patients (A={len(a_files)}, B={len(b_files)})")

    # 수동 제외 목록 (카테터 꺾임 등 artifact)
    EXCLUDED = {'A22'}

    results = []
    for pid, mva_path in sorted(all_files.items()):
        if pid in EXCLUDED:
            print(f"[{pid}] EXCLUDED (catheter kink artifact)")
            results.append({'patient_id': pid, 'group': pid[0], 'excluded': True, 'qc_flag': 'catheter_kink'})
            continue
        # processed 폴더
        if pid.startswith('A'):
            num = pid[1:]
            proc_dir = PROCESSED / num
        else:
            proc_dir = PROCESSED / pid

        if not (proc_dir / 'swallows').exists():
            print(f"[{pid}] No swallows dir, skip")
            continue

        print(f"[{pid}] {Path(mva_path).name[:45]}...", end=' ')
        try:
            m = compute_patient_metrics(mva_path, proc_dir)
            m['patient_id'] = pid
            m['group']      = pid[0]
            results.append(m)
            print(f"IRP(median)={m['IRP4s_median']}, IRP(mean)={m['IRP4s_mean']}, "
                  f"DCI={m['DCI_mean']}, CFV={m['CFV_mean']}, Chicago={m['chicago']}")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    # JSON 저장
    out = PROCESSED / 'hrm_metrics_computed.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} patients → {out}")
    return results


if __name__ == '__main__':
    run_all()
