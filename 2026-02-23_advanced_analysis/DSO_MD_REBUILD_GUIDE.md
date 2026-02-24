# System 3 (DSO) MD 재빌드 가이드라인

> 작성일: 2026-02-23  
> 배경: 기존 DSO MD에서 DSO 리간드가 결합 포켓에서 완전 이탈 확인 (PQQ C5 ↔ DSO O2 최소거리 19.0 Å)  
> 목적: 처음부터 올바른 파이프라인으로 재빌드

---

## 핵심 원칙

**모든 분자를 처음부터 제자리에 놓고 시작**

---

## Phase 1: 파라미터화 (이미 완료 ✓)

- PQQ itp (GAFF2, -3 전하) ✓
- DSO itp (GAFF2, 중성) ✓
- `DSO_GMX.gro`, `DSO_GMX.itp` ✓

---

## Phase 2: 도킹 → 좌표 변환

```bash
# 1. DSO 도킹 (이미 완료 ✓)
# DSO_docked.pdbqt, affinity -5.15 kcal/mol

# 2. mode 1 추출
# DSO_pose1.pdbqt 생성

# 3. 좌표 변환 (Vina → GROMACS 좌표계)
# receptor.pdbqt CA ↔ protein.gro CA shift 계산
# DSO_repositioned.gro 생성
```

---

## Phase 3: Complex 구성

```bash
# 순서 엄수!
# protein.gro (pdb2gmx, 단백질만)
# + PQQ_GMX.gro (원점 기준)
# + DSO_repositioned.gro (shift 적용된 것)
# → complex.gro 병합
```

---

## Phase 4: Topology 구성

```
topol.top 구조:

#include "charmm36-jul2022.ff/forcefield.itp"
#include "ligand_atomtypes.itp"       ← PQQ+DSO 통합 atomtypes
#include "charmm36-jul2022.ff/tip3p.itp"
#include "PQQ_GMX_no_atomtypes.itp"
#include "DSO_GMX_no_atomtypes.itp"   ← DSO 추가
#include "charmm36-jul2022.ff/ions.itp"

[ molecules ]
Protein_chain_A   1
PQQ               1
DSO               1   ← gro 순서와 일치해야 함!
SOL               ?
NA                ?
CL                ?
```

---

## Phase 5: 시뮬레이션 준비

```bash
# 1. 박스 설정
gmx editconf -f complex.gro -o complex_box.gro -c -d 1.0 -bt cubic

# 2. 솔베이션
gmx solvate -cp complex_box.gro -cs spc216.gro -o solvated.gro -p topol.top

# 3. 이온화
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
echo "SOL" | gmx genion -s ions.tpr -o ionized.gro -p topol.top \
            -pname NA -nname CL -neutral -conc 0.15

# 4. 인덱스 그룹 생성 ← System 2에서 빠뜨렸던 것
echo "q" | gmx make_ndx -f ionized.gro 2>&1 | grep -E "^\s+[0-9]+"
# UNL(PQQ), DSO 번호 확인 후:
printf "1 | [PQQ번호] | [DSO번호]\nname XX Protein_PQQ_DSO\n[SOL번호] | [Ion번호]\nname XX SOL_ions\nq\n" | \
gmx make_ndx -f ionized.gro -o index.ndx

# 5. posre.itp 생성
echo "1" | gmx genrestr -f ionized.gro -n index.ndx -o posre.itp -fc 1000 1000 1000
```

---

## Phase 6: EM → NVT → NPT → MD

```bash
# EM
gmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 4
# 확인: Fmax < 1000 ✓

# NVT (index.ndx 필수!)
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr
gmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp 4

# NPT
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr
gmx mdrun -v -deffnm npt -ntmpi 1 -ntomp 4

# Production MD
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md.tpr
gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 4
```

---

## 체크리스트

| 단계 | 확인 사항 |
|------|-----------|
| `complex.gro` | gro 순서 = topology molecules 순서 |
| `atomtypes` | PQQ + DSO 중복 타입 병합 확인 |
| `index.ndx` | grompp 전에 반드시 생성 |
| `posre.itp` | NVT grompp 전에 반드시 생성 |
| 박스 크기 | ~12 nm (21 nm 아님) |
| 이온화 | NA + CL 둘 다 추가 (`-conc 0.15`) |

---

## 비고

- DSO는 LSO와 파라미터 동일, 좌표만 다름 → System 2보다 빠르게 완료 가능
- 기존 DSO MD 실패 원인 추정: DSO_repositioned.gro 좌표 오류 또는 complex.gro 병합 순서 문제
- 재빌드 결과물 저장 위치: `gromacs_dso_v2/`
