# HRM AI 진단 — 진행 현황

> 마지막 업데이트: 2026-02-26

---

## 현재 단계: Phase 1 — 전처리 파이프라인 구축 중

---

## ✅ 완료된 작업

### 데이터 파싱 & 전처리
- MVA 바이너리 파일 파싱 완료 (`parse_mva_binary.py`)
- Swallow crop 추출 완료 (`crop_swallow.py`) → `data/processed/XX/swallows/sw*.npy`
- A군(01~31), B군(B01~) 데이터 처리 완료

### HRM 지표 계산 (`compute_hrm_metrics.py`)
- **IRP**: median IRP (Chicago Classification v3.0 기준) — ✅ 완료
- **DCI, CFV, LESP, IBP**: 계산 로직 구현 완료
- **LES 채널 자동 선택**: resting > gastric+5, relax_drop > 2mmHg, 점수 기반 선택

### UES 탐지 (`_find_actual_onset`)
- **확정 알고리즘 (2026-02-25)**:
  - XML onset 완전 무시 — crop 전체 탐색
  - ch2 > 40, ch3 > 60, ch4 > 90 mmHg (채널별 threshold)
  - 첫 도달 순서: ch2 ≤ ch3 ≤ ch4 (생리적 하강 순서)
  - actual_onset = ch2가 40mmHg 처음 넘는 샘플 = LES 이완 시작점
  - 탐지 실패 → None → IRP 제외
  - 첫 이벤트 후 10초 이내 두 번째 UES → is_double=True → IRP 제외
- 코드 수정 완료, `HRM_DEFINITIONS.md` 업데이트 완료

---

## 🔄 진행 중

| ID | 내용 | 상태 |
|----|------|------|
| `ues_thresh` | UES 탐지 로직 B군 검증 | **미완** — IDE 문제로 중단 |
| `double_swallow` | 연달아 삼킴 제외 로직 검증 | **미완** — 코드 완료, 시각화 검증 필요 |

---

## ⏳ 대기 중

| ID | 내용 |
|----|------|
| `les_search` | B군 이상값 환자 LES 오선택 확인 및 수정 |
| Phase 2 | EfficientNet 모델 개발 (데이터 파이프라인 완성 후) |

---

## 다음 세션 시작 시 할 일

1. B01부터 UES 탐지 토폴로지 검증:
   ```bash
   cd /home/pjho3tr/projects/AI/2026-02-23_hrm-ai-diagnosis
   /home/pjho3tr/miniforge3/envs/drug-md/bin/python3 preprocessing/_plot_ues.py B01
   ```
2. B군 전체 통계 확인 (`preprocessing/_stats.py` — B군 경로 추가 필요)
3. LES 채널 검증 (B군 이상값 환자)

---

## 핵심 파일 경로

| 파일 | 역할 |
|------|------|
| `preprocessing/compute_hrm_metrics.py` | 핵심 지표 계산 |
| `HRM_DEFINITIONS.md` | 알고리즘 정의 (수정 전 반드시 확인) |
| `preprocessing/_plot_ues.py` | UES 탐지 토폴로지 시각화 (환자 ID 인자) |
| `preprocessing/_stats.py` | 전체 통계 확인 (A군만, B군 추가 필요) |
| `data/processed/XX/swallows/` | swallow crop 데이터 |

---

## 알고리즘 요약 (HRM_DEFINITIONS.md 전체 기준)

| 지표 | 기준 |
|------|------|
| actual_onset | crop 전체 탐색, ch2>40/ch3>60/ch4>90, 순서조건, ch2 처음 40 넘는 샘플 |
| double swallow | 첫 이벤트 후 10초 이내 두 번째 UES → IRP 제외 |
| IRP window | actual_onset ~ actual_onset+10s, 최저 400샘플 평균 |
| IRP 진단 | median IRP, cutoff 15mmHg (CC v3.0) |
| LES 선택 | relax_drop + 0.2×osc_score 최대 채널 |
