"""
압력 지형도(pressure topography) 시각화
- mva delta decode 결과 vs txt ground truth 비교
"""
import sys, struct, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, "preprocessing")
from parse_mva import extract_zlib_blocks, find_xml_blocks
from xml.etree import ElementTree as ET

# txt 로드
rows = []
with open("data/1.txt") as f:
    for line in f.readlines()[1:]:
        line = line.strip()
        if not line or line.startswith("Annotations"):
            break
        vals = [float(x) for x in line.split()]
        if len(vals) >= 37:
            rows.append(vals)
data_txt = np.array(rows)
pres_txt = data_txt[:, 1:37]   # (6000, 36)
time_txt = data_txt[:, 0]

# mva 파싱
fpath = "data/normal subjects/1.박영미 03-14-2012;PJH1Study 1.mva"
blocks = extract_zlib_blocks(fpath)
xml_map = find_xml_blocks(blocks)
probe_root = ET.fromstring(xml_map['probe'])
sensors_cal = []
for sensor in sorted(probe_root.find('pressure').findall('sensor'),
                     key=lambda s: int(s.findtext('index'))):
    points = [(float(pt.findtext('signal')), float(pt.findtext('pressure')))
              for pt in sensor.find('scale').findall('point')]
    sensors_cal.append(points)

binary_blocks = [(pos, dec) for pos, dec in blocks if not dec[:1] == b'<']
dec0 = binary_blocks[0][1]
raw0 = np.frombuffer(dec0, dtype=np.uint8)

def delta_decode_1ch(data):
    values = []; current = 0; i = 0
    while i < len(data):
        b = int(data[i])
        if b == 0x80:
            if i + 2 >= len(data): break
            val = struct.unpack_from('<H', data, i + 1)[0]
            current = val; i += 3
        else:
            delta = b if b < 128 else b - 256
            current = max(0, min(65535, current + delta))
            i += 1
        values.append(current)
    return np.array(values, dtype=np.float32)

print("Delta decoding...")
all_vals = delta_decode_1ch(raw0)
n_t = 135248
mat = all_vals[:n_t * 36].reshape(36, n_t)

pres_all = np.zeros((36, n_t), dtype=np.float32)
for ch in range(36):
    s = np.array([p[0] for p in sensors_cal[ch]])
    p = np.array([p[1] for p in sensors_cal[ch]])
    pres_all[ch] = np.interp(mat[ch], s, p)

# 역순 매핑 + 시간 오프셋 적용
best_start = 109600
window = 6000
seg_mva = pres_all[::-1, best_start:best_start + window]  # (36, 6000) 역순
seg_mva = seg_mva.T  # (6000, 36)

time_rel = np.arange(window) * 0.01  # 0~60s 상대 시간

# ============================================================
# 시각화
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

vmin, vmax = 0, 200

# 1. txt pressure topography
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(pres_txt.T, aspect='auto', origin='upper',
                  extent=[time_txt[0]-time_txt[0], time_txt[-1]-time_txt[0], 0.5, 36.5],
                  vmin=vmin, vmax=vmax, cmap='jet')
ax1.set_title('txt (Ground Truth)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Channel')
plt.colorbar(im1, ax=ax1, label='mmHg')

# 2. mva delta decode pressure topography
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(seg_mva.T, aspect='auto', origin='upper',
                  extent=[0, 60, 0.5, 36.5],
                  vmin=vmin, vmax=vmax, cmap='jet')
ax2.set_title('mva (delta decode, reversed ch)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Channel')
plt.colorbar(im2, ax=ax2, label='mmHg')

# 3. 채널별 시계열 비교 (ch4, ch5 - LES 영역)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_rel, pres_txt[:, 4], 'b-', label='txt ch4 (115mmHg)', alpha=0.8)
ax3.plot(time_rel, seg_mva[:, 4], 'r--', label='mva ch4', alpha=0.8)
ax3.set_title('Channel 4 (LES high pressure)', fontsize=11)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Pressure (mmHg)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 채널별 시계열 비교 (ch0 - 식도 상부)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_rel, pres_txt[:, 0], 'b-', label='txt ch0 (11mmHg)', alpha=0.8)
ax4.plot(time_rel, seg_mva[:, 0], 'r--', label='mva ch0', alpha=0.8)
ax4.set_title('Channel 0 (esophageal body)', fontsize=11)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Pressure (mmHg)')
ax4.legend()
ax4.grid(True, alpha=0.3)

fig.suptitle('HRM Pressure Topography: txt vs mva\n(Patient 1, 박영미)', fontsize=13)
plt.savefig('preprocessing/pressure_topology_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: preprocessing/pressure_topology_comparison.png")
plt.close()

# 상관계수 요약
ch_corrs = [float(np.corrcoef(pres_txt[:, tc], seg_mva[:, tc])[0, 1]) for tc in range(36)]
print(f"\nMean corr (36ch): {np.mean(ch_corrs):.3f}")
print(f"Corr > 0.8: {sum(c > 0.8 for c in ch_corrs)}/36 channels")
print(f"Per-ch: {[round(c,2) for c in ch_corrs]}")
