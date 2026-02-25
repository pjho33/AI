"""
60명 전체 mva 파일 일괄 처리 파이프라인
=========================================
출력:
  data/processed/{patient_id}/
    pressure.npy       (n_t, 36) float32
    swallows/
      sw{i:02d}.npy    (4000, 36) float32  (pre=15s, post=25s)
    meta.json          환자 정보 + swallow 시간 목록
"""

import os, sys, json, traceback
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent))
from parse_mva_binary import parse_pressure, get_swallow_times, get_recording_offset_ms
from crop_swallow import crop_swallows, estimate_block0_start
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

DATA_DIR = Path("data/normal subjects")
OUT_DIR  = Path("data/processed")
PRE_SEC  = 15.0
POST_SEC = 25.0


def get_patient_id(fname: str) -> str:
    """
    파일명에서 환자 ID 추출.
    - '1.박영미...' → 'A01'  (기존 이름 있는 파일)
    - '1;13-04-2012.mva' → 'B01'  (새 번호만 있는 파일)
    """
    if ';' in fname:
        # 새 형식: '번호;날짜.mva'
        num = fname.split(';')[0]
        try:
            return f"B{int(num):02d}"
        except ValueError:
            return f"B_{num}"
    else:
        # 기존 형식: '번호.이름...'
        num = fname.split('.')[0]
        try:
            return f"A{int(num):02d}"
        except ValueError:
            return num


def parse_patient_info(mva_path: str) -> dict:
    """info XML에서 환자 기본 정보 추출."""
    blocks = extract_zlib_blocks(mva_path)
    xml_map = find_xml_blocks(blocks)
    info = ET.fromstring(xml_map['info'])
    return {
        'name':     info.findtext('name') or '',
        'dob':      info.findtext('dob') or '',
        'gender':   info.findtext('gender') or '',
        'study_id': info.findtext('studyId') or '',
    }


def process_one(mva_path: str, out_dir: Path, pre_sec=PRE_SEC, post_sec=POST_SEC):
    """단일 mva 파일 처리."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sw_dir = out_dir / 'swallows'
    sw_dir.mkdir(exist_ok=True)

    # 압력 데이터 추출
    pressure, sr = parse_pressure(mva_path)
    np.save(out_dir / 'pressure.npy', pressure)

    # 블록 0 시작 시간 추정
    block0_start = estimate_block0_start(mva_path, pressure)

    # swallow 크롭
    crops = crop_swallows(mva_path, pressure, sr,
                          pre_sec=pre_sec, post_sec=post_sec,
                          block0_start_s=block0_start)

    crop_meta = []
    for c in crops:
        sw_idx = c['swallow_idx']
        np.save(sw_dir / f"sw{sw_idx+1:02d}.npy", c['pressure'])
        crop_meta.append({
            'swallow_idx': sw_idx,
            'begin_ms':    c['begin_ms'],
            'end_ms':      c['end_ms'],
            'file':        f"sw{sw_idx+1:02d}.npy",
        })

    # 환자 정보
    try:
        patient_info = parse_patient_info(mva_path)
    except Exception:
        patient_info = {}

    meta = {
        'mva_path':      str(mva_path),
        'patient':       patient_info,
        'pressure_shape': list(pressure.shape),
        'sample_rate':   sr,
        'block0_start_s': block0_start,
        'n_swallows':    len(crops),
        'swallows':      crop_meta,
    }
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(crops)


def run_batch(only_new=False):
    mva_files = sorted(glob(str(DATA_DIR / '*.mva')))
    if only_new:
        mva_files = [f for f in mva_files if ';' in Path(f).name]
    print(f"Found {len(mva_files)} mva files")

    results = []
    for mva_path in mva_files:
        fname = Path(mva_path).name
        pid = get_patient_id(fname)
        out_dir = OUT_DIR / pid
        print(f"\n[{pid}] {fname[:50]}...")

        try:
            n_sw = process_one(mva_path, out_dir)
            print(f"  → {n_sw} swallows saved to {out_dir}")
            results.append({'id': pid, 'file': fname, 'n_swallows': n_sw, 'ok': True})
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({'id': pid, 'file': fname, 'error': str(e), 'ok': False})

    # 요약
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    ok = [r for r in results if r['ok']]
    fail = [r for r in results if not r['ok']]
    print(f"Success: {len(ok)}/{len(results)}")
    for r in ok:
        print(f"  [{r['id']}] {r['n_swallows']} swallows")
    if fail:
        print(f"Failed: {len(fail)}")
        for r in fail:
            print(f"  [{r['id']}] {r['error']}")

    # 전체 요약 저장
    with open(OUT_DIR / 'batch_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_DIR}/batch_summary.json")


if __name__ == '__main__':
    run_batch()
