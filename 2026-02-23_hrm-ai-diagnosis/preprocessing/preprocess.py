"""
HRM 전처리 파이프라인
- 파일 1개 = 삼킴 1회 = 샘플 1개
- 첫 행(더미 0.0) 제거, 절대 타임스탬프 기반 겹치는 구간 제거
1. ASCII 파싱 + 중복 제거
2. Min-Max Normalization (-10~300 mmHg)
3. Tensor 변환 (1, 36, TARGET_TIME) — 고정 길이 crop/pad
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json


# ─────────────────────────────────────────
# 1. 파싱
# ─────────────────────────────────────────
def parse_hrm_ascii(filepath):
    """
    파일 1개 = 삼킴 1회 전체 기록
    - 첫 행(더미 0.0) 제거
    Returns:
        time      : (N,) 절대 타임스탬프 (초)
        pressure  : (N, 36) float32  [mmHg]
        impedance : (N, 18) float32
    """
    data = np.loadtxt(filepath, dtype=np.float32)
    # 첫 행은 더미(col0=0.0) → 제거
    data = data[1:]
    time      = data[:, 0]
    pressure  = data[:, 1:37]
    impedance = data[:, 37:55]
    return time, pressure, impedance


def remove_overlap(time, pressure, impedance, prev_t_end):
    """
    이전 파일의 끝 시간(prev_t_end) 이후 행만 유지
    """
    if prev_t_end is None:
        return time, pressure, impedance
    mask = time > prev_t_end
    return time[mask], pressure[mask], impedance[mask]


# ─────────────────────────────────────────
# 2. Normalization
# ─────────────────────────────────────────
PRESSURE_MIN = -10.0
PRESSURE_MAX = 300.0

def normalize_pressure(pressure):
    """Min-Max Scaling to [0, 1] based on fixed clinical range"""
    p = np.clip(pressure, PRESSURE_MIN, PRESSURE_MAX)
    return (p - PRESSURE_MIN) / (PRESSURE_MAX - PRESSURE_MIN)


# ─────────────────────────────────────────
# 3. Tensor 변환
# ─────────────────────────────────────────
TARGET_TIME = 6000   # 60초 @ 100Hz (파일 전체, 6001행 기준)
N_SENSORS   = 36

def to_tensor(pressure_norm, target_time=TARGET_TIME):
    """
    (T, 36) → (1, 36, target_time) 텐서
    - T < target_time: zero-padding (오른쪽)
    - T > target_time: 앞부분 crop
    """
    T = pressure_norm.shape[0]
    if T >= target_time:
        arr = pressure_norm[:target_time, :]
    else:
        pad = target_time - T
        arr = np.pad(pressure_norm, ((0, pad), (0, 0)), mode='constant')

    # (target_time, 36) → (1, 36, target_time)
    tensor = arr.T[np.newaxis, :, :]
    return tensor.astype(np.float32)


# ─────────────────────────────────────────
# 4. 전체 파이프라인
# ─────────────────────────────────────────
def process_file(filepath, prev_t_end=None):
    """
    단일 파일 전처리 → 텐서 1개 반환
    prev_t_end: 이전 파일의 마지막 타임스탬프 (겹침 제거용)

    Returns:
        tensor    : np.ndarray (1, 36, TARGET_TIME)
        t_end     : float (이 파일의 마지막 타임스탬프)
        meta      : dict
    """
    time, pressure, impedance = parse_hrm_ascii(filepath)
    time, pressure, impedance = remove_overlap(time, pressure, impedance, prev_t_end)

    pressure_norm = normalize_pressure(pressure)
    tensor = to_tensor(pressure_norm)

    meta = {
        "filename":       Path(filepath).name,
        "t_start":        float(time[0]),
        "t_end":          float(time[-1]),
        "timepoints":     len(pressure),
        "duration_s":     float(time[-1] - time[0]),
        "pressure_min":   float(pressure.min()),
        "pressure_max":   float(pressure.max()),
        "tensor_shape":   list(tensor.shape),
        "overlap_removed_s": float(prev_t_end) if prev_t_end else 0.0,
    }
    return tensor, float(time[-1]), meta


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    npy_dir  = Path(__file__).parent.parent / "data" / "processed"
    npy_dir.mkdir(parents=True, exist_ok=True)

    # 파일을 절대 타임스탬프 시작 시간 기준으로 정렬
    files = sorted(data_dir.glob("*.txt"))
    files_with_tstart = []
    for fpath in files:
        data = np.loadtxt(fpath, dtype=np.float32)
        t_start = data[1, 0]   # 첫 행 더미 제외
        files_with_tstart.append((t_start, fpath))
    files_with_tstart.sort(key=lambda x: x[0])

    print("파일 시간순 정렬:")
    for ts, fp in files_with_tstart:
        data = np.loadtxt(fp, dtype=np.float32)
        te = data[-1, 0]
        print(f"  {fp.name}: {ts:.2f}s ~ {te:.2f}s")

    print()

    all_tensors = []
    all_meta    = []
    prev_t_end  = None

    for ts, fpath in files_with_tstart:
        tensor, t_end, meta = process_file(fpath, prev_t_end)
        all_tensors.append(tensor)
        all_meta.append(meta)

        overlap = max(0, prev_t_end - ts) if prev_t_end else 0
        print(f"{meta['filename']:20s} | "
              f"{meta['t_start']:.2f}~{meta['t_end']:.2f}s | "
              f"rows: {meta['timepoints']:4d} | "
              f"overlap removed: {overlap:.2f}s")

        prev_t_end = t_end

    # 전체 텐서 저장: (N, 1, 36, TARGET_TIME)
    arr = np.stack(all_tensors, axis=0)
    npy_path = npy_dir / "normal_tensors.npy"
    np.save(npy_path, arr)

    meta_path = npy_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nTotal samples : {len(all_tensors)}")
    print(f"Tensor shape  : {arr.shape}  (N, 1, sensors, time)")
    print(f"Saved to      : {npy_path}")
