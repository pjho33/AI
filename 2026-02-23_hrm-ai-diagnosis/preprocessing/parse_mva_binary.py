"""
HRM .mva 파일 바이너리 파싱 모듈
=====================================
인코딩 방식:
  - 블록 0: 압력 36채널 (delta encoding, channel-first layout)
  - 블록 1: 임피던스 18채널 (delta encoding, channel-first layout)

Delta encoding 규칙:
  - 0x80 바이트: escape → 다음 2바이트 = uint16 절대값 (little-endian)
  - 기타 바이트: int8 delta (부호 있는 차분값)

채널 레이아웃:
  - channel-first: ch0의 n_t samples, ch1의 n_t samples, ...
  - 채널 순서: 역순 (decoded ch35 = 압력 ch0, decoded ch34 = 압력 ch1, ...)

캘리브레이션:
  - probe XML의 sensors_cal[ch] 룩업 테이블로 raw signal → mmHg 변환

샘플링 레이트: 100 Hz (0.01초 간격)
"""

import struct
import numpy as np
from xml.etree import ElementTree as ET

try:
    from parse_mva import extract_zlib_blocks, find_xml_blocks
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from parse_mva import extract_zlib_blocks, find_xml_blocks

SAMPLE_RATE = 100  # Hz


def _delta_decode(data: bytes) -> np.ndarray:
    """
    Delta encoding 디코딩.
    0x80 = escape (다음 2바이트 uint16 절대값)
    기타 = int8 delta
    """
    values = []
    current = 0
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0x80:
            if i + 2 >= n:
                break
            current = struct.unpack_from('<H', data, i + 1)[0]
            i += 3
        else:
            delta = b if b < 128 else b - 256
            current = max(0, min(65535, current + delta))
            i += 1
        values.append(current)
    return np.array(values, dtype=np.float32)


def _load_calibration(probe_xml: str):
    """probe XML에서 압력/임피던스 캘리브레이션 커브 로드."""
    root = ET.fromstring(probe_xml)

    pressure_cal = []
    for sensor in sorted(root.find('pressure').findall('sensor'),
                         key=lambda s: int(s.findtext('index'))):
        scale_elem = sensor.find('scale')
        points = []
        if scale_elem is not None:
            for pt in scale_elem.findall('point'):
                sig = pt.findtext('signal')
                pres = pt.findtext('pressure')
                if sig is not None and pres is not None:
                    points.append((float(sig), float(pres)))
        pressure_cal.append(points)

    impedance_cal = []
    imp_elem = root.find('impedance')
    if imp_elem is not None:
        for sensor in sorted(imp_elem.findall('sensor'),
                             key=lambda s: int(s.findtext('index'))):
            scale_elem = sensor.find('scale')
            points = []
            if scale_elem is not None:
                for pt in scale_elem.findall('point'):
                    sig = pt.findtext('signal')
                    pres = pt.findtext('pressure')
                    if sig is not None and pres is not None:
                        points.append((float(sig), float(pres)))
            impedance_cal.append(points)

    return pressure_cal, impedance_cal


def _apply_calibration(raw_vals: np.ndarray, cal_points: list) -> np.ndarray:
    """캘리브레이션 커브 적용: raw signal → mmHg."""
    s_arr = np.array([p[0] for p in cal_points])
    p_arr = np.array([p[1] for p in cal_points])
    return np.interp(raw_vals, s_arr, p_arr)


def parse_pressure(mva_path: str):
    """
    .mva 파일에서 압력 시계열 추출.

    Parameters
    ----------
    mva_path : str

    Returns
    -------
    pressure : np.ndarray (n_t, 36), float32, mmHg
    sample_rate : int, 100 Hz
    """
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)

    pressure_cal, _ = _load_calibration(xml_map['probe'])

    binary_blocks = [(pos, dec) for pos, dec in blocks if not dec[:1] == b'<']
    if not binary_blocks:
        raise ValueError("No binary blocks found in mva file")

    dec0 = binary_blocks[0][1]
    raw0 = bytes(dec0) if not isinstance(dec0, (bytes, bytearray)) else dec0

    all_vals = _delta_decode(raw0)
    n_ch = 36
    n_t = len(all_vals) // n_ch
    mat = all_vals[:n_t * n_ch].reshape(n_ch, n_t)  # (36, n_t) channel-first

    # 역순 매핑: decoded ch(35-tc) → pressure ch(tc)
    mat_reordered = mat[::-1, :]  # (36, n_t)

    # 캘리브레이션 적용
    pressure = np.zeros((n_t, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        pressure[:, ch] = _apply_calibration(mat_reordered[ch], pressure_cal[ch])

    return pressure, SAMPLE_RATE


def parse_impedance(mva_path: str):
    """
    .mva 파일에서 임피던스 시계열 추출.

    Parameters
    ----------
    mva_path : str

    Returns
    -------
    impedance : np.ndarray (n_t, 18), float32, kΩ
    sample_rate : int, 100 Hz
    """
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    _, impedance_cal = _load_calibration(xml_map['probe'])

    binary_blocks = [(pos, dec) for pos, dec in blocks if not dec[:1] == b'<']
    if len(binary_blocks) < 2:
        raise ValueError("No impedance binary block found")

    dec1 = binary_blocks[1][1]
    raw1 = bytes(dec1) if not isinstance(dec1, (bytes, bytearray)) else dec1

    all_vals = _delta_decode(raw1)
    n_ch = 18
    n_t = len(all_vals) // n_ch
    mat = all_vals[:n_t * n_ch].reshape(n_ch, n_t)

    # 역순 매핑
    mat_reordered = mat[::-1, :]

    if impedance_cal:
        impedance = np.zeros((n_t, n_ch), dtype=np.float32)
        for ch in range(n_ch):
            impedance[:, ch] = _apply_calibration(mat_reordered[ch], impedance_cal[ch])
    else:
        # 캘리브레이션 없으면 raw 바이트값 그대로 (×0.1 kΩ 추정)
        impedance = (mat_reordered.T * 0.1).astype(np.float32)

    return impedance, SAMPLE_RATE


def get_swallow_times(mva_path: str):
    """
    analysis XML에서 swallow 구간 시간 추출.

    Returns
    -------
    swallows : list of dict
        [{'begin_ms': float, 'end_ms': float, 'index': int}, ...]
    """
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    analysis = ET.fromstring(xml_map['analysis'])

    swallows = []
    for i, ep in enumerate(analysis.find('episodes')):
        if ep.tag != 'swallow':
            continue
        swallows.append({
            'index': i,
            'begin_ms': float(ep.findtext('beginTime') or 0),
            'end_ms': float(ep.findtext('endTime') or 0),
        })
    return swallows


def get_recording_offset_ms(mva_path: str) -> float:
    """examination XML에서 startTimeOffset(ms) 반환."""
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    exam = ET.fromstring(xml_map['examination'])
    for elem in exam.iter('startTimeOffset'):
        return float(elem.text)
    return 0.0


if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else \
        "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"

    print(f"Parsing: {fp}")
    pressure, sr = parse_pressure(fp)
    print(f"Pressure : {pressure.shape}, {sr} Hz, "
          f"mean={pressure.mean():.1f} mmHg, "
          f"range={pressure.min():.1f}~{pressure.max():.1f}")

    swallows = get_swallow_times(fp)
    offset_ms = get_recording_offset_ms(fp)
    print(f"startTimeOffset: {offset_ms/1000:.1f}s")
    print(f"Swallows ({len(swallows)}):")
    for sw in swallows[:5]:
        print(f"  sw{sw['index']+1}: {sw['begin_ms']/1000:.1f}~{sw['end_ms']/1000:.1f}s")
