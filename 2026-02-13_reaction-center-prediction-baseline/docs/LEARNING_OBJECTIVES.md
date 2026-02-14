# AI 학습 목표 상세 정의

## 전체 학습 목표

AI가 화학반응과 효소에 대해 배워야 할 핵심 지식

---

## 1. 반응 패턴 인식 (Reaction Pattern Recognition)

### 목표
분자 구조를 보고 가능한 반응 유형을 예측

### 학습 내용

#### Level 1: 기본 작용기 인식
```
입력: SMILES 구조
출력: 존재하는 작용기 목록

예시:
OCC(O)C(O)C(O)C(O)CO
→ 1차 알코올 2개 (C1, C6)
→ 2차 알코올 4개 (C2, C3, C4, C5)
```

#### Level 2: 반응 가능성 판단
```
작용기 → 가능한 반응

1차 알코올:
  ✓ 산화 → 알데히드
  ✓ 에스테르화
  ✗ 케톤화 (불가능)

2차 알코올:
  ✓ 산화 → 케톤
  ✓ 에스테르화
  ✓ 탈수 → 알켄
```

#### Level 3: 반응 우선순위
```
같은 분자에 여러 반응 가능 위치가 있을 때:

D-sorbitol (OCC(O)C(O)C(O)C(O)CO):
  C2 산화: 우선순위 ⭐⭐⭐ (가장 접근 가능)
  C3 산화: 우선순위 ⭐⭐
  C4 산화: 우선순위 ⭐⭐
  C5 산화: 우선순위 ⭐⭐
  
이유:
- 입체 장애
- 전자적 환경
- 효소 특이성
```

### 평가 지표
- **Top-1 accuracy**: 가장 반응 가능성 높은 위치 맞추기
- **Top-5 accuracy**: 상위 5개 후보에 정답 포함

---

## 2. 구조 변화 예측 (Structural Change Prediction)

### 목표
반응이 일어나면 분자의 어느 부분이 어떻게 바뀌는지 예측

### 학습 내용

#### Level 1: 원자 수준 변화
```
반응 전: [CH2:1][OH:2]
반응 후: [CH:1]=[O:2]

변화:
- Atom 1 (C): H 개수 2 → 1
- Atom 2 (O): 결합 차수 1 → 2
- Atom 2 (O): H 개수 1 → 0
```

#### Level 2: 결합 변화
```
산화 반응:
C-OH (단일결합) → C=O (이중결합)

이성질화:
C1-C2-C3 → C1=C2-C3 (위치 이동)
```

#### Level 3: 혼성화 변화
```
sp3 → sp2: 알코올 → 카보닐
sp2 → sp3: 카보닐 → 알코올
sp3 → sp3: 이성질화 (혼성 유지)
```

### 학습 데이터 형식
```python
{
    "atom_changes": [
        {
            "atom_idx": 2,
            "before": {
                "hybridization": "sp3",
                "bonds": {"O": 1, "C": 2, "H": 1},
                "formal_charge": 0
            },
            "after": {
                "hybridization": "sp2",
                "bonds": {"O": 1, "C": 2},
                "formal_charge": 0
            }
        }
    ],
    "bond_changes": [
        {
            "atoms": [2, 7],
            "before": "SINGLE",
            "after": "DOUBLE"
        }
    ]
}
```

### 평가 지표
- **Atom-level accuracy**: 바뀐 원자 정확히 예측
- **Bond-level accuracy**: 결합 변화 정확히 예측
- **Transformation type accuracy**: 변환 유형 분류 정확도

---

## 3. 효소 요구사항 학습 (Enzyme Requirements)

### 목표
특정 반응에 어떤 효소가 필수적인지, 어떤 효소가 선택적인지 학습

### 학습 내용

#### Level 1: EC 클래스 매핑
```
반응 유형 → EC 클래스

산화반응 → EC 1 (Oxidoreductase)
  - EC 1.1.1.x: NAD(P)+ 의존성
  - EC 1.1.3.x: O2 의존성
  - EC 1.1.99.x: 기타 전자수용체

이성질화 → EC 5 (Isomerase)
  - EC 5.3.1.x: 케토-엔올 이성질화
  - EC 5.3.3.x: 탄소 골격 이성질화
```

#### Level 2: 보조인자 요구사항
```
효소 → 필수 보조인자

EC 1.1.1.14 (L-iditol 2-dehydrogenase):
  필수: NAD+
  선택: Mg2+ (활성 증가)
  
EC 5.3.1.5 (glucose-6-phosphate isomerase):
  필수: 없음
  선택: Mg2+ (안정성 증가)
```

#### Level 3: 효소 특이성
```
반응 + 기질 → 적합한 효소

L-sorbitol 산화:
  ✓ EC 1.1.1.14 (L-iditol 2-dehydrogenase) - 특이적
  △ EC 1.1.1.21 (aldose reductase) - 역반응 선호
  ✗ EC 1.1.1.1 (alcohol dehydrogenase) - 특이성 낮음

D-glucose 이성질화:
  ✓ EC 5.3.1.5 (glucose-6-phosphate isomerase)
  ✗ EC 5.3.1.9 (다른 isomerase) - 기질 특이성 다름
```

### 학습 데이터 형식
```python
{
    "reaction_id": "sorbitol_oxidation",
    "substrate": "L-sorbitol",
    "product": "L-sorbose",
    "reaction_type": "oxidation",
    
    "enzyme_requirements": {
        "essential_ec_class": "1.1.1",
        "essential_cofactors": ["NAD+"],
        "optional_cofactors": ["Mg2+"],
        
        "suitable_enzymes": [
            {
                "ec": "1.1.1.14",
                "name": "L-iditol 2-dehydrogenase",
                "specificity": "high",
                "kcat": 120,  # s^-1
                "km": 0.5,    # mM
                "preference_score": 0.95
            },
            {
                "ec": "1.1.1.21",
                "name": "aldose reductase",
                "specificity": "medium",
                "kcat": 30,
                "km": 2.0,
                "preference_score": 0.45
            }
        ],
        
        "unsuitable_enzymes": [
            {
                "ec": "1.1.1.1",
                "reason": "낮은 기질 특이성"
            }
        ]
    }
}
```

### 평가 지표
- **EC class accuracy**: 올바른 EC 클래스 예측
- **Cofactor prediction**: 필수 보조인자 예측
- **Enzyme ranking**: 적합한 효소 순위 매기기

---

## 4. 효소 선택 최적화 (Enzyme Selection Optimization)

### 목표
같은 반응을 촉매할 수 있는 여러 효소 중 가장 적합한 것 선택

### 학습 내용

#### Level 1: 활성도 기반 순위
```
반응: L-sorbitol → L-sorbose

효소 후보:
1. EC 1.1.1.14 - 활성도 95/100 ⭐⭐⭐⭐⭐
2. EC 1.1.1.15 - 활성도 60/100 ⭐⭐⭐
3. EC 1.1.1.21 - 활성도 30/100 ⭐
```

#### Level 2: 다중 기준 최적화
```
평가 기준:
- 기질 특이성 (Km 값)
- 촉매 효율 (kcat/Km)
- 안정성 (온도, pH 범위)
- 생산 용이성 (발현 수율)
- 비용

가중치 적용:
효소 점수 = 0.4×특이성 + 0.3×효율 + 0.2×안정성 + 0.1×비용
```

#### Level 3: 조건별 최적 효소
```
조건에 따른 효소 선택:

고온 반응 (50°C):
  1순위: Thermostable variant
  2순위: Wild-type with stabilizer

저온 반응 (4°C):
  1순위: Cold-adapted enzyme
  2순위: Mesophilic enzyme

대량 생산:
  1순위: E. coli 고발현 효소
  2순위: Yeast 발현 효소
```

### 학습 데이터 형식
```python
{
    "reaction": "sorbitol_to_sorbose",
    
    "enzyme_comparison": [
        {
            "enzyme_id": "EC_1.1.1.14",
            "kinetics": {
                "kcat": 120,      # s^-1
                "km": 0.5,        # mM
                "kcat_km": 240    # M^-1 s^-1
            },
            "conditions": {
                "optimal_temp": 37,
                "temp_range": [25, 45],
                "optimal_ph": 7.5,
                "ph_range": [6.5, 8.5]
            },
            "production": {
                "host": "E. coli",
                "yield": "high",
                "cost": "low"
            },
            "overall_score": 0.92
        },
        {
            "enzyme_id": "EC_1.1.1.15",
            "kinetics": {
                "kcat": 45,
                "km": 2.0,
                "kcat_km": 22.5
            },
            "conditions": {
                "optimal_temp": 30,
                "temp_range": [20, 40],
                "optimal_ph": 7.0,
                "ph_range": [6.0, 8.0]
            },
            "production": {
                "host": "E. coli",
                "yield": "medium",
                "cost": "medium"
            },
            "overall_score": 0.65
        }
    ],
    
    "recommendation": {
        "best_enzyme": "EC_1.1.1.14",
        "reason": "최고 촉매 효율 + 넓은 조건 범위",
        "alternatives": ["EC_1.1.1.15"],
        "avoid": ["EC_1.1.1.21"]
    }
}
```

### 평가 지표
- **Ranking accuracy**: 효소 순위 정확도 (Kendall's tau)
- **Top-3 recommendation**: 상위 3개 추천 정확도
- **Condition-specific accuracy**: 조건별 최적 효소 선택

---

## 통합 학습 파이프라인

### Stage 1: 반응 인식
```
입력: 기질 분자
↓
반응 가능 위치 예측
↓
반응 유형 분류
```

### Stage 2: 변화 예측
```
반응 위치 + 반응 유형
↓
구조 변화 예측
↓
생성물 구조 검증
```

### Stage 3: 효소 매칭
```
반응 유형 + 기질 구조
↓
적합한 EC 클래스 예측
↓
효소 후보 목록 생성
```

### Stage 4: 효소 최적화
```
효소 후보 목록 + 반응 조건
↓
다중 기준 평가
↓
최적 효소 Top-5 추천
```

## 성공 기준

### Phase 1 (현재)
- ✓ 반응 위치 예측: Top-5 accuracy ≥70%
- ⧗ EC 클래스 예측: accuracy ≥80%

### Phase 2 (1개월)
- 구조 변화 예측: atom-level accuracy ≥85%
- 효소 추천: Top-3 accuracy ≥75%

### Phase 3 (3개월)
- 통합 시스템: end-to-end accuracy ≥80%
- 실험 검증: 추천 효소의 실제 성공률 ≥70%

## 다음 단계

1. **데이터 수집**: Rhea, BRENDA, UniProt에서 효소-반응 데이터
2. **라벨링**: 효소 활성도, 특이성 데이터 정리
3. **학습**: 단계별 모델 훈련
4. **검증**: 실험 데이터로 검증
