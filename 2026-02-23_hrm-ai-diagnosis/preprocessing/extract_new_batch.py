"""
새 30명 임상지표 추출 + 60명 합친 clinical_metrics_all.json 생성
"""
import sys, json
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent))
from parse_mva_xml import extract_swallow_metrics, extract_patient_info

DATA_DIR = Path("data/normal subjects")
OUT_DIR  = Path("data/processed")

# ── 새 30명 파일 (번호;날짜.mva, 이름 없음) ──────────────────────
new_files = sorted([
    f for f in glob(str(DATA_DIR / "*.mva"))
    if ";" in Path(f).name
    and "." not in Path(f).name.split(";")[0]   # 번호만 있는 것
])
print(f"New files: {len(new_files)}")

new_patients = []
for mva_path in new_files:
    fname = Path(mva_path).name
    num   = fname.split(";")[0]
    pid   = f"B{int(num):02d}"
    print(f"[{pid}] {fname[:50]}...", end=" ")
    try:
        swallows = extract_swallow_metrics(mva_path)
        # 이름 없는 파일은 info XML이 다를 수 있으므로 안전하게 처리
        try:
            info = extract_patient_info(mva_path)
        except Exception:
            info = {"name": "", "dob": "", "gender": "", "study_id": pid}
        new_patients.append({
            "patient_id": pid,
            "file": fname,
            "group": "B",
            "info": info,
            "n_swallows": len(swallows),
            "swallows": swallows,
        })
        print(f"→ {len(swallows)} swallows")
    except Exception as e:
        print(f"ERROR: {e}")

# ── 기존 30명 로드 ────────────────────────────────────────────────
with open(OUT_DIR / "clinical_metrics.json", encoding="utf-8") as f:
    old_patients = json.load(f)

# group 태그 추가
for pt in old_patients:
    pt["group"] = "A"
    pt["patient_id"] = f"A{int(pt['patient_id']):02d}"

# ── 합치기 ────────────────────────────────────────────────────────
all_patients = old_patients + new_patients
print(f"\nTotal: {len(all_patients)} patients")

out_path = OUT_DIR / "clinical_metrics_all.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_patients, f, ensure_ascii=False, indent=2)
print(f"Saved → {out_path}")
