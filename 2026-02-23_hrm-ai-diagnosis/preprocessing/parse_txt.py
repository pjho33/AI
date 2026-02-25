"""
HRM txt 파일 파서
ManoView에서 export한 ASCII txt 파일을 파싱합니다.

파일 구조:
  행 0: TIME:\t1\t2\t...\t36\t\t1\t2\t...\t18  (헤더)
  행 1~N: 시간(s)\t압력1\t...\t압력36\t\t임피던스1\t...\t임피던스18
  마지막 2행: Annotations:\n  시간\t설명

반환:
  time_s     : (N,) float64, 절대 시간 (s)
  pressure   : (N, 36) float32, 압력 (mmHg)
  impedance  : (N, 18) float32, 임피던스 (kΩ)
  annotations: list of (time_s, text)
"""

import numpy as np
from pathlib import Path


def parse_hrm_txt(filepath):
    """
    ManoView export txt 파일 파싱.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    time_s : np.ndarray (N,)
    pressure : np.ndarray (N, 36)
    impedance : np.ndarray (N, 18)
    annotations : list of (float, str)
    """
    filepath = Path(filepath)
    rows = []
    annotations = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    in_annotations = False
    for line in lines[1:]:  # 헤더 행 건너뜀
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('Annotations:'):
            in_annotations = True
            continue
        if in_annotations:
            parts = stripped.split('\t')
            if len(parts) >= 2:
                try:
                    t = float(parts[0].strip())
                    text = parts[1].strip()
                    annotations.append((t, text))
                except ValueError:
                    pass
            continue
        # 데이터 행
        vals = [float(x) for x in stripped.split()]
        if len(vals) >= 37:
            rows.append(vals)

    if not rows:
        raise ValueError(f"No data rows found in {filepath}")

    data = np.array(rows, dtype=np.float64)

    time_s   = data[:, 0].astype(np.float64)
    pressure = data[:, 1:37].astype(np.float32)

    # 임피던스: 컬럼 38~55 (컬럼 37은 빈 구분자)
    if data.shape[1] >= 56:
        impedance = data[:, 38:56].astype(np.float32)
    elif data.shape[1] >= 55:
        impedance = data[:, 37:55].astype(np.float32)
    else:
        impedance = np.full((len(rows), 18), np.nan, dtype=np.float32)

    return time_s, pressure, impedance, annotations


def txt_summary(filepath):
    """파일 요약 출력."""
    time_s, pressure, impedance, annotations = parse_hrm_txt(filepath)
    dt = np.diff(time_s[:10]).mean() if len(time_s) > 1 else 0.01
    print(f"File   : {Path(filepath).name}")
    print(f"Samples: {len(time_s)}  ({len(time_s)*dt:.1f}s @ {1/dt:.0f}Hz)")
    print(f"TIME   : {time_s[0]:.2f} ~ {time_s[-1]:.2f} s")
    print(f"Pressure : {pressure.min():.1f} ~ {pressure.max():.1f} mmHg  "
          f"(mean={pressure.mean():.1f})")
    print(f"Impedance: {np.nanmin(impedance):.3f} ~ {np.nanmax(impedance):.3f} kΩ")
    print(f"Annotations: {annotations}")
    return time_s, pressure, impedance, annotations


if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/1.txt"
    txt_summary(fp)
