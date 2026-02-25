"""
Swallow 구간 자동 크롭
=====================================
- mva analysis XML의 beginTime/endTime으로 swallow 구간 추출
- 블록 0 시작 시간 = startTimeOffset 기반 자동 계산
- 각 swallow를 고정 길이(pad_sec)로 크롭

사용법:
    from crop_swallow import crop_swallows
    swallow_list = crop_swallows(mva_path, pressure, sample_rate)
    # swallow_list[i]: (n_t_crop, 36) ndarray
"""

import numpy as np
from xml.etree import ElementTree as ET

try:
    from parse_mva import extract_zlib_blocks, find_xml_blocks
    from parse_mva_binary import parse_pressure, get_swallow_times, get_recording_offset_ms
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from parse_mva import extract_zlib_blocks, find_xml_blocks
    from parse_mva_binary import parse_pressure, get_swallow_times, get_recording_offset_ms

SAMPLE_RATE = 100  # Hz


def estimate_block0_start(mva_path: str, pressure: np.ndarray) -> float:
    """
    블록 0의 전체 녹화 기준 시작 시간(s) 추정.

    방법: analysis XML의 swallow beginTime들과 블록 0 압력 데이터의
    LES 고압 채널(ch4/ch5) 패턴을 비교해서 시간 오프셋 추정.

    간단한 방법: startTimeOffset과 블록 0 길이로 역산.
    블록 0 = 전체 녹화의 마지막 구간 (startTimeOffset 이후)
    """
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)

    # examination XML에서 startTimeOffset
    exam = ET.fromstring(xml_map['examination'])
    start_offset_ms = 0.0
    for elem in exam.iter('startTimeOffset'):
        start_offset_ms = float(elem.text)
        break

    # 블록 0 총 길이
    n_t = pressure.shape[0]
    block0_duration_s = n_t / SAMPLE_RATE  # 1352.5s

    # 전체 녹화 길이 추정: analysis XML의 마지막 swallow endTime
    analysis = ET.fromstring(xml_map['analysis'])
    episodes = [ep for ep in analysis.find('episodes') if ep.tag == 'swallow']
    if episodes:
        last_end_ms = max(float(ep.findtext('endTime') or 0) for ep in episodes)
        total_duration_s = last_end_ms / 1000 + 60  # 여유 60s
    else:
        total_duration_s = start_offset_ms / 1000 + block0_duration_s

    # 블록 0 시작 = 전체 녹화 끝 - 블록 0 길이
    block0_start_s = total_duration_s - block0_duration_s

    return block0_start_s


def crop_swallows(mva_path: str,
                  pressure: np.ndarray,
                  sample_rate: int = SAMPLE_RATE,
                  pre_sec: float = 5.0,
                  post_sec: float = 35.0,
                  block0_start_s: float = None):
    """
    mva 파일에서 swallow 구간을 자동 크롭.

    Parameters
    ----------
    mva_path : str
    pressure : np.ndarray (n_t, 36)
    sample_rate : int
    pre_sec : float  swallow beginTime 이전 여유 시간
    post_sec : float swallow beginTime 이후 포함 시간
    block0_start_s : float  블록 0 시작 절대 시간(s), None이면 자동 추정

    Returns
    -------
    crops : list of dict
        [{'swallow_idx': int,
          'begin_ms': float,
          'end_ms': float,
          'pressure': np.ndarray (n_crop, 36)}, ...]
    """
    swallows = get_swallow_times(mva_path)
    n_t = pressure.shape[0]

    if block0_start_s is None:
        block0_start_s = estimate_block0_start(mva_path, pressure)

    crop_len = int((pre_sec + post_sec) * sample_rate)
    crops = []

    for sw in swallows:
        bt_s = sw['begin_ms'] / 1000.0
        et_s = sw['end_ms'] / 1000.0

        # 블록 0 내 sample index
        bt_sample = int((bt_s - block0_start_s) * sample_rate)
        et_sample = int((et_s - block0_start_s) * sample_rate)

        crop_start = bt_sample - int(pre_sec * sample_rate)
        crop_end = crop_start + crop_len

        # 범위 체크
        if crop_start < 0 or crop_end > n_t:
            continue

        seg = pressure[crop_start:crop_end, :]
        crops.append({
            'swallow_idx': sw['index'],
            'begin_ms': sw['begin_ms'],
            'end_ms': sw['end_ms'],
            'begin_sample': crop_start,
            'end_sample': crop_end,
            'pressure': seg,
        })

    return crops


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    fp = sys.argv[1] if len(sys.argv) > 1 else \
        "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"

    print(f"Parsing: {fp}")
    pressure, sr = parse_pressure(fp)
    print(f"Pressure: {pressure.shape}, {sr} Hz")

    block0_start = estimate_block0_start(fp, pressure)
    print(f"Block 0 start: {block0_start:.2f}s")

    crops = crop_swallows(fp, pressure, sr, pre_sec=15.0, post_sec=25.0,
                          block0_start_s=block0_start)
    print(f"\nCropped swallows: {len(crops)}")
    for c in crops:
        sw_dur = (c['end_ms'] - c['begin_ms']) / 1000
        print(f"  sw{c['swallow_idx']+1}: {c['begin_ms']/1000:.1f}s "
              f"(dur={sw_dur:.1f}s), crop shape={c['pressure'].shape}")

    # 첫 4개 swallow 시각화
    if crops:
        n_show = min(4, len(crops))
        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
        if n_show == 1:
            axes = [axes]
        for i, (ax, c) in enumerate(zip(axes, crops[:n_show])):
            im = ax.imshow(c['pressure'].T, aspect='auto', origin='upper',
                           vmin=0, vmax=200, cmap='jet',
                           extent=[0, c['pressure'].shape[0] / sr, 0.5, 36.5])
            ax.axvline(x=15.0, color='white', linestyle='--', alpha=0.7, label='swallow')
            ax.set_title(f"sw{c['swallow_idx']+1} ({c['begin_ms']/1000:.0f}s)")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channel')
            plt.colorbar(im, ax=ax, label='mmHg')
        plt.suptitle('Swallow Crops - Patient 1', fontsize=12)
        plt.tight_layout()
        plt.savefig('preprocessing/swallow_crops.png', dpi=120, bbox_inches='tight')
        print("\nSaved: preprocessing/swallow_crops.png")
        plt.close()
