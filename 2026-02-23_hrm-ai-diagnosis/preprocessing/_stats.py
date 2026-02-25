import numpy as np, glob, sys
sys.path.insert(0, '.')
SR = 100

def find_ao(crop):
    T2, T3, T4 = 40, 60, 90
    UW = 10
    DW = int(10 * SR)
    seg = crop[:, 0:5]
    events = []
    idx = 0
    while idx <= len(seg) - UW:
        w = seg[idx:idx + UW]
        a2 = np.where(w[:, 2] > T2)[0]
        a3 = np.where(w[:, 3] > T3)[0]
        a4 = np.where(w[:, 4] > T4)[0]
        if len(a2) > 0 and len(a3) > 0 and len(a4) > 0 and a2[0] <= a3[0] <= a4[0]:
            events.append(idx + a2[0])
            idx = events[0] + DW
        else:
            idx += 1
    return (None, False) if not events else (events[0], len(events) >= 2)

import os

# A군 + B군 patient ID 목록 자동 수집
pids = []
for d in sorted(os.listdir("data/processed")):
    if os.path.isdir("data/processed/%s/swallows" % d):
        pids.append(d)

print("환자    전체  탐지  double  fallback  IRP유효")
for pid in pids:
    fps = sorted(glob.glob("data/processed/%s/swallows/sw*.npy" % pid))
    if not fps:
        continue
    n = len(fps); nd = 0; nb = 0; ni = 0
    for fp in fps:
        c = np.load(fp)
        ao, dbl = find_ao(c)
        if ao is not None:
            nd += 1
            if dbl:
                nb += 1
            elif ao + 10 * SR <= len(c):
                ni += 1
    label = ("A" + pid) if pid.isdigit() else pid
    print("%-7s  %3d   %3d   %3d     %3d       %3d" % (label, n, nd, nb, n - nd, ni))
