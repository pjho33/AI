# Run PQQ+Ca Transplant - Quick Guide

---

## ✅ 자동 실행 (Drug-MD 환경)

### 한 줄 명령어

```bash
cd /home/pjho3/projects/AI/2026-02-21_pqq-sorbitol-md/scripts
bash run_chimerax_transplant.sh
```

---

## 📋 스크립트가 하는 일

1. **Drug-MD conda 환경 활성화**
2. **ChimeraX 확인**
3. **필요 파일 체크**:
   - receptor_af.pdb ✓
   - 1WPQ.pdb ✓
   - chimerax_transplant.cxc ✓
4. **ChimeraX 실행** (nogui 모드):
   - receptor_af.pdb와 1WPQ.pdb 정렬
   - PQQ + Ca 추출
   - 이식 및 저장
5. **결과 검증**:
   - receptor_PQQ_Ca.pdb 생성 확인
   - PQQ, Ca 원자 수 확인

---

## 📊 예상 출력

```
============================================================
PQQ+Ca Transplant - Automated Script
============================================================

[1/5] Activating Drug-MD conda environment...
  ✓ ChimeraX found: /home/pjho3/miniconda3/envs/Drug-MD/bin/chimerax

[2/5] Checking required files...
  ✓ receptor_af.pdb found
  ✓ 1WPQ.pdb found
  ✓ chimerax_transplant.cxc found

[3/5] Running ChimeraX transplant...
  This will:
    - Align receptor_af.pdb with 1WPQ.pdb
    - Extract PQQ and Ca from 1WPQ
    - Transplant to receptor_af
    - Save as receptor_PQQ_Ca.pdb

[4/5] Verifying output...
  ✓ receptor_PQQ_Ca.pdb created
  PQQ atoms found: 24
  Ca atoms found: 1

[5/5] Summary
============================================================

Output file: receptor_PQQ_Ca.pdb
  Size: 450K
  Total atoms: 5678
  PQQ atoms: 24
  Ca atoms: 1

✓ Transplant successful!

Next steps:
  1. Visual check (optional):
     chimerax receptor_PQQ_Ca.pdb

  2. Proceed to Phase 2 (Parameterization):
     cd ../scripts
     # Generate CGenFF parameters for PQQ and sorbitol

============================================================
```

---

## ✅ 성공 확인

**파일 생성됨**:
```bash
ls -lh structures/receptor_PQQ_Ca.pdb
```

**PQQ, Ca 확인**:
```bash
grep "HETATM" structures/receptor_PQQ_Ca.pdb | grep -E "PQQ|CA"
```

---

## 🔍 시각적 확인 (선택사항)

```bash
conda activate Drug-MD
cd structures
chimerax receptor_PQQ_Ca.pdb
```

**확인 사항**:
- PQQ가 binding pocket 중앙에 있는지
- Ca²⁺가 PQQ 근처에 있는지
- 충돌 없는지

---

## ⚠️ 문제 해결

### ChimeraX not found
```bash
conda activate Drug-MD
conda install -c conda-forge chimerax
```

### 출력 파일 없음
```bash
# 로그 확인
cat structures/chimerax_transplant.log

# 수동 실행
cd structures
chimerax --nogui --script ../scripts/chimerax_transplant.cxc
```

### PQQ/Ca 없음
- 1WPQ.pdb 재다운로드
- 수동으로 ChimeraX GUI에서 확인

---

## 🚀 다음 단계

**Phase 1 완료** ✓
- receptor_af.pdb (AlphaFold)
- receptor_PQQ_Ca.pdb (PQQ+Ca 이식)

**Phase 2 시작**: Parameterization
- PQQ CGenFF 파라미터
- L-sorbitol 파라미터
- D-sorbitol 파라미터
- PQQ position restraints
