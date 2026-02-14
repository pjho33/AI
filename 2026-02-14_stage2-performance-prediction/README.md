# Stage 2: Performance Prediction (성능 예측)

**날짜**: 2026-02-14  
**목표**: 실험 성능 예측 시스템 구축

## 개요

Stage 1에서 "반응이 가능한가?"를 예측했다면, Stage 2는 "얼마나 잘 일어나는가?"를 예측합니다.

### 예측 대상
- **수율 (Yield)**: 기질 → 생성물 전환율
- **kcat**: 효소 회전수 (turnover number)
- **Km**: 미카엘리스 상수 (기질 친화도)
- **반응 시간**: 90% 완료까지 소요 시간
- **선택성**: 원하는 생성물 비율

## Stage 1과의 연결

```python
# Stage 1 출력
feasibility = {
    "P_feasible": 0.914,
    "reaction_centers": [1, 2],
    "enzyme_compatible": True,
    "cofactor": "NAD+"
}

# Stage 2 입력
performance = predict_performance(
    reaction=feasibility,
    conditions={
        "substrate_conc": 10,  # mM
        "enzyme_conc": 0.1,    # μM
        "cofactor_conc": 1.0,  # mM
        "pH": 7.4,
        "temperature": 37
    }
)

# Stage 2 출력
{
    "yield": {"mean": 0.85, "std": 0.05},
    "kcat": {"mean": 120, "std": 15},
    "Km": {"mean": 0.6, "std": 0.1},
    "time_to_90pct": {"mean": 1.5, "std": 0.2}
}
```

## 주요 도전 과제

### 1. 데이터 결측 (70-90%)
BRENDA, 문헌 데이터는 대부분 결측값 포함
- **해결**: Masked loss, 전이 학습

### 2. 조건 의존성
같은 반응도 조건에 따라 10배 이상 차이
- **해결**: 조건 정규화, 조건별 모델

### 3. 불확실성 정량화
측정값 vs 추정값 구분 필요
- **해결**: 불확실성 전파, 신뢰 구간

## 프로젝트 구조

```
2026-02-14_stage2-performance-prediction/
├── src/
│   ├── data_loaders/
│   │   ├── brenda_loader.py          # BRENDA API
│   │   └── literature_loader.py      # 문헌 데이터
│   ├── models/
│   │   ├── stage2_predictor.py       # 메인 예측기
│   │   ├── masked_loss.py            # 결측 대응
│   │   └── uncertainty.py            # 불확실성
│   └── utils/
│       ├── transfer_learning.py      # 전이 학습
│       └── evaluation.py             # 평가
├── data/
│   ├── brenda_kinetics.json          # kcat, Km
│   └── assay_records.json            # 실험 데이터
├── docs/
│   └── STAGE2_DESIGN.md              # 설계 문서
└── notebooks/
    └── 01_brenda_exploration.ipynb   # 데이터 탐색
```

## 시작하기

```bash
cd /home/pjho3/projects/AI/2026-02-14_stage2-performance-prediction

# 가상환경 (Stage 1과 공유 가능)
source ../2026-02-13_reaction-center-prediction-baseline/venv/bin/activate

# 추가 패키지
pip install scipy uncertainties
```

## 다음 단계

1. BRENDA 데이터 로더 구현
2. Stage 2 예측기 기본 구조
3. Masked loss 구현
4. 불확실성 정량화
5. 평가 및 검증
