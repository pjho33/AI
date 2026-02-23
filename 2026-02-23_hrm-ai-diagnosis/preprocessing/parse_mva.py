"""
ManoView .mva 파일 파서
- 구조: 바이너리 컨테이너 (zlib 압축 XML 블록들)
- 추출 내용:
    1. 환자 정보 (patient info)
    2. 임상 지표 (LES 위치/압력, UES 위치/압력)
    3. 각 swallow별 분석 결과 (IRP 관련, DCI 관련)
    4. 원시 압력 데이터 (ASCII txt와 동일)
"""

import zlib
import re
import json
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET


# ─────────────────────────────────────────
# 1. zlib 블록 추출
# ─────────────────────────────────────────
def extract_zlib_blocks(filepath):
    """
    .mva 파일에서 모든 zlib 압축 블록을 찾아 압축 해제
    Returns: list of (position, decompressed_bytes)
    """
    data = open(filepath, 'rb').read()
    blocks = []
    for i in range(len(data) - 1):
        if data[i] == 0x78 and data[i+1] in (0x9c, 0xda, 0x01, 0x5e):
            try:
                dec = zlib.decompress(data[i:])
                blocks.append((i, dec))
            except Exception:
                pass
    return blocks


def find_xml_blocks(blocks):
    """
    압축 해제된 블록 중 XML 텍스트 블록만 분류
    같은 태그가 여러 개면 가장 큰 블록을 사용
    Returns: dict of tag → xml_string
    """
    xml_map = {}
    for pos, dec in blocks:
        try:
            text = dec.decode('utf-8', errors='replace').strip()
            if text.startswith('<'):
                m = re.match(r'<(\w+)', text)
                if m:
                    tag = m.group(1)
                    # 같은 태그면 더 큰 블록으로 교체
                    if tag not in xml_map or len(text) > len(xml_map[tag]):
                        xml_map[tag] = text
        except Exception:
            pass
    return xml_map


# ─────────────────────────────────────────
# 2. 환자 정보 파싱
# ─────────────────────────────────────────
def parse_patient_info(xml_map):
    """<info> 블록에서 환자 정보 추출"""
    if 'info' not in xml_map:
        return {}
    try:
        root = ET.fromstring(xml_map['info'])
        pt = root.find('patient')
        proc = root.find('procedure')
        info = {}
        if pt is not None:
            info['patient_id']  = _text(pt, 'id')
            info['gender']      = _text(pt, 'gender')
            info['height']      = _text(pt, 'height')
            info['dob']         = _text(pt, 'dob')
        if proc is not None:
            info['date']        = _text(proc, 'date')
            info['procedure']   = _text(proc, 'procedure')
            info['physician']   = _text(proc, 'physician')
        return info
    except Exception as e:
        return {'parse_error': str(e)}


# ─────────────────────────────────────────
# 3. 분석 결과 파싱 (swallow별 임상 지표)
# ─────────────────────────────────────────
def parse_analysis(xml_map):
    """
    <analysis> 블록에서 각 swallow의 임상 지표 추출
    - LES 위치, LES 압력 (IRP 대리 지표)
    - UES 위치, UES 압력
    - 각 채널 peak 압력 (DCI 계산용)
    - beginTime, endTime (타임스탬프, ms 단위)
    """
    if 'analysis' not in xml_map:
        return [], {}

    try:
        root     = ET.fromstring(xml_map['analysis'])
        episodes = root.find('episodes')
        if episodes is None:
            return [], {}

        swallows = []
        landmark = {}

        for ep in episodes:   # ET Element은 직접 iterate 가능
            tag = ep.tag      # 'swallow' or 'landmark'

            begin_t = ep.findtext('beginTime')
            end_t   = ep.findtext('endTime')
            entry = {
                'type':      tag,
                'beginTime': float(begin_t) if begin_t else None,
                'endTime':   float(end_t)   if end_t   else None,
                'status':    ep.findtext('status'),
            }

            # LES 정보
            les = ep.find('les')
            if les is not None:
                entry['les_location_cm']   = _ftext(les, 'location')
                entry['les_pressure_mmhg'] = _ftext(les, 'pressure')
                b = les.find('begin')
                e = les.find('end')
                entry['les_begin_ms'] = _ftext(b, 'time') if b is not None else None
                entry['les_end_ms']   = _ftext(e, 'time') if e is not None else None

            # UES 정보
            ues = ep.find('ues')
            if ues is not None:
                entry['ues_location_cm']   = _ftext(ues, 'location')
                entry['ues_pressure_mmhg'] = _ftext(ues, 'pressure')
                pk = ues.find('peak')
                if pk is not None:
                    entry['ues_peak_pressure'] = _ftext(pk, 'pressure')
                    entry['ues_peak_time_ms']  = _ftext(pk, 'time')

            # 식도 체부 채널별 peak 압력 (DCI 계산용)
            les_ch = ep.find('lesChannels')
            if les_ch is not None:
                ch_peaks = []
                ch_bases = []
                for ch in les_ch.findall('channel'):
                    pk = ch.find('peak')
                    bl = ch.find('baseline')
                    ch_peaks.append(_ftext(pk, 'pressure') if pk is not None else 0.0)
                    ch_bases.append(float(bl.text) if bl is not None and bl.text else 0.0)
                entry['channel_peak_pressures'] = ch_peaks
                entry['channel_baselines']       = ch_bases
                dur_s = (entry['endTime'] - entry['beginTime']) / 1000.0 \
                        if entry['endTime'] and entry['beginTime'] else 0
                dci = sum(max(0, (p or 0) - b) for p, b in zip(ch_peaks, ch_bases)) * dur_s
                entry['dci_approx'] = round(dci, 2)

            # PIP, gastric
            pip = ep.find('pip')
            if pip is not None:
                entry['pip_location_cm'] = _ftext(pip, 'location')
            gast = ep.find('gastric')
            if gast is not None:
                entry['gastric_location_cm'] = _ftext(gast, 'location')

            if tag == 'landmark':
                landmark = entry
            else:
                swallows.append(entry)

        return swallows, landmark

    except Exception as e:
        return [], {'parse_error': str(e)}


# ─────────────────────────────────────────
# 4. AVA 지표 파싱 (IRP 등)
# ─────────────────────────────────────────
def parse_ava(xml_map):
    """<ava> 블록: lesDist, lesProx, UES, PIP 위치"""
    if 'ava' not in xml_map:
        return {}
    try:
        root = ET.fromstring(xml_map['ava'])
        return {
            'les_dist_cm': _float(root, 'lesDist'),
            'les_prox_cm': _float(root, 'lesProx'),
            'ues_cm':      _float(root, 'ues'),
            'pip_cm':      _float(root, 'pip'),
        }
    except Exception as e:
        return {'parse_error': str(e)}


# ─────────────────────────────────────────
# 5. 통합 파서
# ─────────────────────────────────────────
def parse_mva(filepath):
    """
    .mva 파일 전체 파싱
    Returns:
        result : dict
            - patient_info
            - ava_metrics
            - landmark
            - swallows (list, 각 swallow별 임상 지표)
            - summary (전체 통계)
    """
    blocks  = extract_zlib_blocks(filepath)
    xml_map = find_xml_blocks(blocks)

    patient_info = parse_patient_info(xml_map)
    swallows, landmark = parse_analysis(xml_map)
    ava = parse_ava(xml_map)

    # 전체 요약 통계
    valid_sw = [s for s in swallows if s.get('les_pressure_mmhg') is not None]
    summary = {
        'n_swallows':          len(swallows),
        'n_valid_swallows':    len(valid_sw),
        'les_pressure_mean':   round(np.mean([s['les_pressure_mmhg'] for s in valid_sw]), 2) if valid_sw else None,
        'les_pressure_std':    round(np.std( [s['les_pressure_mmhg'] for s in valid_sw]), 2) if valid_sw else None,
        'ues_pressure_mean':   round(np.mean([s['ues_pressure_mmhg'] for s in valid_sw if s.get('ues_pressure_mmhg')]), 2) if valid_sw else None,
        'dci_mean':            round(np.mean([s['dci_approx'] for s in valid_sw if s.get('dci_approx') is not None]), 2) if valid_sw else None,
        'xml_blocks_found':    list(xml_map.keys()),
    }

    return {
        'filename':     Path(filepath).name,
        'patient_info': patient_info,
        'ava_metrics':  ava,
        'landmark':     landmark,
        'swallows':     swallows,
        'summary':      summary,
    }


# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────
def _text(elem, tag):
    if elem is None:
        return None
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else None

def _float(elem, tag):
    if elem is None:
        return None
    child = elem.find(tag)
    if child is not None and child.text:
        try:
            return float(child.text.strip())
        except ValueError:
            return None
    return None

def _ftext(elem, tag):
    """elem.findtext(tag)를 float으로 반환"""
    if elem is None:
        return None
    val = elem.findtext(tag)
    if val:
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent.parent / "data"
    mva_files = sorted(data_dir.glob("*.mva"))

    if not mva_files:
        print("No .mva files found in data/")
        sys.exit(1)

    for fpath in mva_files:
        print(f"\n{'='*60}")
        print(f"File: {fpath.name}")
        print(f"{'='*60}")

        result = parse_mva(fpath)

        print(f"\n[환자 정보]")
        for k, v in result['patient_info'].items():
            print(f"  {k:20s}: {v}")

        print(f"\n[AVA 지표]")
        for k, v in result['ava_metrics'].items():
            print(f"  {k:20s}: {v}")

        print(f"\n[요약]")
        for k, v in result['summary'].items():
            print(f"  {k:25s}: {v}")

        print(f"\n[Swallow별 임상 지표]")
        print(f"  {'#':>3}  {'begin(ms)':>10}  {'LES_loc':>8}  {'LES_p':>7}  {'UES_p':>7}  {'DCI_approx':>10}")
        print(f"  {'-'*60}")
        for i, sw in enumerate(result['swallows']):
            print(f"  {i+1:>3}  "
                  f"{sw.get('beginTime', 0)/1000:>9.1f}s  "
                  f"{sw.get('les_location_cm', 0) or 0:>8.2f}  "
                  f"{sw.get('les_pressure_mmhg', 0) or 0:>7.2f}  "
                  f"{sw.get('ues_pressure_mmhg', 0) or 0:>7.2f}  "
                  f"{sw.get('dci_approx', 0) or 0:>10.1f}")

        # JSON 저장
        out_path = data_dir / (fpath.stem + "_parsed.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved: {out_path}")
