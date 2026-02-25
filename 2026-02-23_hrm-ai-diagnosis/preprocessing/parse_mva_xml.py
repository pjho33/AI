"""
MVA XML 임상지표 추출
======================
swallow별:
  - beginTime, endTime, duration
  - LES: location, resting pressure, relaxation begin/end
  - UES: location, peak pressure, relaxation
  - lesChannels: 각 채널별 peak pressure, baseline, begin/end time
  - DCI (Distal Contractile Integral): lesChannels peak × duration × length
  - IRP (Integrated Relaxation Pressure): LES relaxation 구간 평균압
  - CFV (Contractile Front Velocity)
  - IBP (Intrabolus Pressure)
"""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from parse_mva import extract_zlib_blocks, find_xml_blocks
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from parse_mva import extract_zlib_blocks, find_xml_blocks


def _f(elem, tag, default=None):
    """안전한 float 추출."""
    v = elem.findtext(tag)
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def extract_swallow_metrics(mva_path: str) -> list:
    """
    모든 swallow의 임상지표 추출.

    Returns
    -------
    list of dict, 각 swallow당 하나
    """
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    analysis = ET.fromstring(xml_map['analysis'])
    episodes = analysis.find('episodes')

    swallows = []
    sw_num = 0
    for ep in episodes:
        if ep.tag != 'swallow':
            continue
        sw_num += 1

        begin_ms = _f(ep, 'beginTime', 0)
        end_ms   = _f(ep, 'endTime', 0)
        duration_s = (end_ms - begin_ms) / 1000.0

        # LES
        les = ep.find('les')
        les_location  = _f(les, 'location') if les is not None else None
        les_pressure  = _f(les, 'pressure') if les is not None else None
        les_begin_ms  = _f(les.find('begin'), 'time') if les is not None and les.find('begin') is not None else None
        les_end_ms    = _f(les.find('end'),   'time') if les is not None and les.find('end')   is not None else None

        # UES
        ues = ep.find('ues')
        ues_location  = _f(ues, 'location') if ues is not None else None
        ues_peak_p    = _f(ues.find('peak'), 'pressure') if ues is not None and ues.find('peak') is not None else None
        ues_begin_ms  = _f(ues.find('begin'), 'time') if ues is not None and ues.find('begin') is not None else None
        ues_end_ms    = _f(ues.find('end'),   'time') if ues is not None and ues.find('end')   is not None else None

        # lesChannels → DCI 계산
        # DCI = sum over channels: peak_pressure × contraction_duration × segment_length
        les_channels = []
        dci = 0.0
        les_ch_elem = ep.find('lesChannels')
        if les_ch_elem is not None:
            chs = les_ch_elem.findall('channel')
            for i, ch in enumerate(chs):
                peak_p   = _f(ch.find('peak'),  'pressure', 0)
                base_p   = _f(ch, 'baseline', 0)
                ch_begin = _f(ch.find('begin'), 'time', 0)
                ch_end   = _f(ch.find('end'),   'time', 0)
                ch_dur_s = (ch_end - ch_begin) / 1000.0 if ch_end and ch_begin else 0
                location = _f(ch, 'location', 0)

                # 인접 채널 간 거리 (cm)
                if i + 1 < len(chs):
                    next_loc = _f(chs[i+1], 'location', location - 1)
                    seg_len = abs(location - next_loc)
                else:
                    seg_len = 1.0

                ch_dci = (peak_p - 20) * ch_dur_s * seg_len  # 20 mmHg threshold
                dci += max(0, ch_dci)

                les_channels.append({
                    'index':    int(_f(ch, 'index', i)),
                    'location': location,
                    'peak_p':   peak_p,
                    'baseline': base_p,
                    'begin_ms': ch_begin,
                    'end_ms':   ch_end,
                    'dur_s':    ch_dur_s,
                })

        # uesChannels
        ues_channels = []
        ues_ch_elem = ep.find('uesChannels')
        if ues_ch_elem is not None:
            for ch in ues_ch_elem.findall('channel'):
                ues_channels.append({
                    'index':    int(_f(ch, 'index', 0)),
                    'location': _f(ch, 'location'),
                    'peak_p':   _f(ch.find('peak'), 'pressure'),
                    'baseline': _f(ch, 'baseline'),
                })

        # IRP (4s integrated relaxation pressure)
        # LES relaxation 구간(les_begin ~ les_end)의 최저 4초 평균
        # XML에 직접 없으므로 les_pressure를 proxy로 사용
        irp_proxy = les_pressure  # 실제 IRP는 raw 데이터 필요

        # CFV (Contractile Front Velocity)
        cfv = ep.find('cfv')
        cfv_prox_loc  = _f(cfv.find('prox'), 'location') if cfv is not None and cfv.find('prox') is not None else None
        cfv_dist_loc  = _f(cfv.find('dist'), 'location') if cfv is not None and cfv.find('dist') is not None else None
        cfv_prox_time = _f(cfv.find('prox'), 'time') if cfv is not None and cfv.find('prox') is not None else None
        cfv_dist_time = _f(cfv.find('dist'), 'time') if cfv is not None and cfv.find('dist') is not None else None
        if all(v is not None for v in [cfv_prox_loc, cfv_dist_loc, cfv_prox_time, cfv_dist_time]):
            dist_cm = abs(cfv_prox_loc - cfv_dist_loc)
            time_s  = abs(cfv_prox_time - cfv_dist_time) / 1000.0
            cfv_val = dist_cm / time_s if time_s > 0 else None
        else:
            cfv_val = None

        # IBP (Intrabolus Pressure)
        ibp = ep.find('ibp')
        ibp_val = _f(ibp, 'pressure') if ibp is not None else None

        # Integrated Volume
        iv = ep.find('integratedVolume')
        iv_pressure = _f(iv, 'pressure') if iv is not None else None
        iv_volume   = _f(iv, 'volume')   if iv is not None else None

        swallows.append({
            'swallow_num':  sw_num,
            'begin_ms':     begin_ms,
            'end_ms':       end_ms,
            'duration_s':   duration_s,
            'les_location': les_location,
            'les_pressure': les_pressure,
            'les_begin_ms': les_begin_ms,
            'les_end_ms':   les_end_ms,
            'ues_location': ues_location,
            'ues_peak_p':   ues_peak_p,
            'ues_begin_ms': ues_begin_ms,
            'ues_end_ms':   ues_end_ms,
            'dci':          round(dci, 2),
            'irp_proxy':    irp_proxy,
            'cfv':          round(cfv_val, 2) if cfv_val else None,
            'ibp':          ibp_val,
            'iv_pressure':  iv_pressure,
            'iv_volume':    iv_volume,
            'n_les_ch':     len(les_channels),
            'les_channels': les_channels,
            'ues_channels': ues_channels,
        })

    return swallows


def extract_patient_info(mva_path: str) -> dict:
    """info + examination XML에서 환자 기본 정보 추출."""
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)

    info = ET.fromstring(xml_map['info'])
    exam = ET.fromstring(xml_map['examination'])

    patient = {
        'name':           info.findtext('name') or '',
        'dob':            info.findtext('dob') or '',
        'gender':         info.findtext('gender') or '',
        'study_id':       info.findtext('studyId') or '',
        'start_offset_ms': float(exam.findtext('startTimeOffset') or 0),
    }
    return patient


def batch_extract(data_dir: str, out_path: str):
    """30명 전체 임상지표 추출 후 JSON 저장."""
    from glob import glob
    import traceback

    mva_files = sorted(glob(str(Path(data_dir) / '*.mva')))
    all_patients = []

    for mva_path in mva_files:
        fname = Path(mva_path).name
        pid_str = fname.split('.')[0]
        try:
            pid = int(pid_str)
        except ValueError:
            continue

        print(f"[{pid:02d}] {fname[:50]}...", end=' ')
        try:
            patient = extract_patient_info(mva_path)
            swallows = extract_swallow_metrics(mva_path)
            all_patients.append({
                'patient_id': pid,
                'file': fname,
                'info': patient,
                'n_swallows': len(swallows),
                'swallows': swallows,
            })
            print(f"→ {len(swallows)} swallows")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_patients, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_patients)} patients → {out_path}")
    return all_patients


if __name__ == '__main__':
    DATA_DIR = "data/normal subjects"
    OUT_PATH = "data/processed/clinical_metrics.json"
    Path("data/processed").mkdir(exist_ok=True)
    batch_extract(DATA_DIR, OUT_PATH)
