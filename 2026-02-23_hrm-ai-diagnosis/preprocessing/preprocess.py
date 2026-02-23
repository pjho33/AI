"""
HRM 전처리 파이프라인
1. ASCII 파싱
2. Wet Swallow 구간 자동 감지 & Cropping
3. Min-Max Normalization (-10~300 mmHg)
4. 4D Tensor 변환 (Batch, 1, Sensors, Time)
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
    Returns:
        pressure  : (N, 36) float32  [mmHg]
        impedance : (N, 18) float32
    """
    data = np.loadtxt(filepath, dtype=np.float32)
    pressure  = data[:, 1:37]
    impedance = data[:, 37:55]
    return pressure, impedance


# ─────────────────────────────────────────
# 2. Wet Swallow 자동 감지
# ─────────────────────────────────────────
def detect_swallows(pressure,
                    mid_sensors=(10, 25),   # 식도 체부 센서 범위 (0-indexed)
                    threshold=20.0,          # mmHg, 연하 감지 임계값
                    min_gap_s=3.0,           # 연하 간 최소 간격 (초)
                    pre_s=2.0,               # 연하 전 여유 (초)
                    post_s=8.0,              # 연하 후 여유 (초)
                    fs=100):
    """
    식도 체부 센서의 압력 피크를 기반으로 wet swallow 구간 감지.

    Returns:
        swallows : list of (start_idx, end_idx)
    """
    # 식도 체부 센서 평균 압력
    mid_pressure = pressure[:, mid_sensors[0]:mid_sensors[1]].max(axis=1)

    # 간단한 smoothing (0.1초 이동평균)
    kernel = int(0.1 * fs)
    smoothed = np.convolve(mid_pressure, np.ones(kernel)/kernel, mode='same')

    # threshold 초과 구간 감지
    above = smoothed > threshold
    min_gap = int(min_gap_s * fs)
    pre  = int(pre_s  * fs)
    post = int(post_s * fs)

    # 연속 구간 찾기
    swallows = []
    in_swallow = False
    start = 0
    last_end = -min_gap

    for i in range(len(above)):
        if above[i] and not in_swallow:
            in_swallow = True
            start = i
        elif not above[i] and in_swallow:
            in_swallow = False
            end = i
            # 최소 간격 필터
            if start - last_end >= min_gap:
                s = max(0, start - pre)
                e = min(len(pressure), end + post)
                swallows.append((s, e))
                last_end = end

    return swallows


# ─────────────────────────────────────────
# 3. Normalization
# ─────────────────────────────────────────
PRESSURE_MIN = -10.0
PRESSURE_MAX = 300.0

def normalize_pressure(pressure):
    """Min-Max Scaling to [0, 1] based on fixed clinical range"""
    p = np.clip(pressure, PRESSURE_MIN, PRESSURE_MAX)
    return (p - PRESSURE_MIN) / (PRESSURE_MAX - PRESSURE_MIN)


# ─────────────────────────────────────────
# 4. Tensor 변환
# ─────────────────────────────────────────
TARGET_TIME = 1000   # 10초 @ 100Hz (패딩/크롭 기준)
N_SENSORS   = 36

def to_tensor(pressure_norm, target_time=TARGET_TIME):
    """
    (T, 36) → (1, 36, target_time) 텐서
    - T < target_time: zero-padding
    - T > target_time: center crop
    """
    T = pressure_norm.shape[0]
    if T >= target_time:
        start = (T - target_time) // 2
        arr = pressure_norm[start:start + target_time, :]
    else:
        pad = target_time - T
        pad_left  = pad // 2
        pad_right = pad - pad_left
        arr = np.pad(pressure_norm, ((pad_left, pad_right), (0, 0)), mode='constant')

    # (target_time, 36) → (1, 36, target_time)
    tensor = arr.T[np.newaxis, :, :]   # (1, 36, target_time)
    return tensor.astype(np.float32)


# ─────────────────────────────────────────
# 5. 전체 파이프라인
# ─────────────────────────────────────────
def process_file(filepath, visualize=False, out_dir=None):
    """
    단일 파일 전처리 → swallow별 텐서 리스트 반환

    Returns:
        tensors : list of np.ndarray (1, 36, TARGET_TIME)
        meta    : dict (filename, n_swallows, swallow_durations)
    """
    pressure, impedance = parse_hrm_ascii(filepath)
    swallows = detect_swallows(pressure)

    tensors = []
    durations = []

    for i, (s, e) in enumerate(swallows):
        seg = pressure[s:e, :]
        seg_norm = normalize_pressure(seg)
        tensor = to_tensor(seg_norm)
        tensors.append(tensor)
        durations.append((e - s) / 100.0)

        if visualize and out_dir:
            _plot_swallow(seg, i, filepath, out_dir, s, e)

    meta = {
        "filename":         Path(filepath).name,
        "total_timepoints": len(pressure),
        "n_swallows":       len(swallows),
        "swallow_durations_s": durations,
        "tensor_shape":     list(tensors[0].shape) if tensors else None,
    }
    return tensors, meta


def _plot_swallow(seg, idx, filepath, out_dir, s, e):
    """개별 swallow 시각화"""
    t = np.arange(len(seg)) * 0.01
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(seg.T, aspect='auto', origin='upper',
                   extent=[t[0], t[-1], 36.5, 0.5],
                   cmap='jet', vmin=-10, vmax=150, interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Pressure (mmHg)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor')
    ax.set_title(f"{Path(filepath).stem} — Swallow #{idx+1} "
                 f"(idx {s}~{e}, {(e-s)/100:.1f}s)")
    plt.tight_layout()
    out = Path(out_dir) / f"{Path(filepath).stem}_swallow{idx+1:02d}.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir  = Path(__file__).parent.parent / "data" / "figures" / "swallows"
    npy_dir  = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for fpath in sorted(data_dir.glob("*.txt")):
        tensors, meta = process_file(fpath, visualize=True, out_dir=out_dir)
        all_meta.append(meta)

        # 텐서 저장 (파일당 하나의 npy: shape = (n_swallows, 1, 36, TARGET_TIME))
        if tensors:
            arr = np.stack(tensors, axis=0)   # (N, 1, 36, 1000)
            npy_path = npy_dir / (fpath.stem + "_tensors.npy")
            np.save(npy_path, arr)

        print(f"{meta['filename']:20s} | "
              f"swallows: {meta['n_swallows']:2d} | "
              f"durations: {[f'{d:.1f}s' for d in meta['swallow_durations_s']]}")

    # 메타 저장
    meta_path = npy_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nTensors saved to : {npy_dir}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Swallow figures  : {out_dir}")
