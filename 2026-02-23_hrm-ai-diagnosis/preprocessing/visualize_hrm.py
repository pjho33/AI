"""
HRM ASCII 데이터 파싱 및 압력 지형도(Esophageal Pressure Topography) 시각화
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys


def parse_hrm_ascii(filepath):
    """
    HRM ASCII 파일 파싱
    Returns:
        time      : (N,) 타임스탬프 배열 (초)
        pressure  : (N, 36) 압력 배열 (mmHg)
        impedance : (N, 18) 임피던스 배열
    """
    data = np.loadtxt(filepath)
    time      = data[:, 0]
    pressure  = data[:, 1:37]   # col 2~37 (0-indexed: 1~36)
    impedance = data[:, 37:55]  # col 38~55 (0-indexed: 37~54)
    return time, pressure, impedance


def plot_pressure_topography(time, pressure, title="Esophageal Pressure Topography",
                              save_path=None, vmin=-10, vmax=150):
    """
    압력 지형도 (Clouse plot) 시각화
    x축: 상대 시간 (0부터 시작, 초)
    y축: 센서 번호 (1=UES, 36=LES 방향)
    색상: 압력 (mmHg)
    """
    # 상대 시간 (0-based, 0.01s 간격)
    n = len(time)
    t_rel = np.arange(n) * 0.01  # 0, 0.01, 0.02, ...
    duration = t_rel[-1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    # --- 상단: 압력 지형도 ---
    ax = axes[0]
    img = pressure.T  # (36, N)

    im = ax.imshow(
        img,
        aspect='auto',
        origin='upper',          # sensor 1(UES)이 위쪽
        extent=[0, duration, 36.5, 0.5],
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear'
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Pressure (mmHg)', fontsize=11)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Sensor (1=proximal → 36=distal)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 센서 위치 레이블 (대략적 해부학적 위치)
    ax.axhline(y=3,  color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=30, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(1.0, 2,  'UES', color='white', fontsize=9, va='center')
    ax.text(1.0, 31, 'LES', color='white', fontsize=9, va='center')

    # --- 하단: 압력 시계열 (특정 센서) ---
    ax2 = axes[1]
    for sensor_idx, label, color in [(2,  'UES (S3)',  'blue'),
                                      (17, 'Mid (S18)', 'green'),
                                      (32, 'LES (S33)', 'red')]:
        ax2.plot(t_rel, pressure[:, sensor_idx], label=label,
                 color=color, lw=1.0, alpha=0.8)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Pressure (mmHg)', fontsize=11)
    ax2.set_title('Representative Sensor Traces', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.axhline(y=0, color='gray', linestyle=':', lw=0.8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, duration)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def print_summary(filepath, time, pressure, impedance):
    """데이터 기본 통계 출력"""
    print(f"\n{'='*50}")
    print(f"File: {Path(filepath).name}")
    print(f"{'='*50}")
    dt = 0.01  # 100 Hz, 0.01s interval (confirmed)
    print(f"Duration     : {time[-1] - time[0]:.2f} s ({len(time)} time points)")
    print(f"Sampling rate: 100 Hz (0.01s interval, fixed)")
    print(f"Pressure     : shape={pressure.shape}, "
          f"min={pressure.min():.1f}, max={pressure.max():.1f} mmHg")
    print(f"Impedance    : shape={impedance.shape}, "
          f"min={impedance.min():.3f}, max={impedance.max():.3f}")
    print(f"Pressure mean per sensor (first 5): "
          f"{pressure.mean(axis=0)[:5].round(2)}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir  = Path(__file__).parent.parent / "data" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        print("No .txt files found in data/raw/")
        sys.exit(1)

    for fpath in files:
        time, pressure, impedance = parse_hrm_ascii(fpath)
        print_summary(fpath, time, pressure, impedance)

        save_path = out_dir / (fpath.stem + "_topography.png")
        plot_pressure_topography(
            time, pressure,
            title=f"EPT - {fpath.stem}",
            save_path=save_path
        )

    print(f"\nAll figures saved to: {out_dir}")
