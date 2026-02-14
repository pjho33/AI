# 2026-02-13: Reaction Center Prediction Baseline

## 오늘의 목표

**반응 중심 예측 시스템의 기초 구축**

### 달성한 것

1. ✓ 프로젝트 구조 설계
2. ✓ Rule-based predictor 구현
3. ✓ Reaction center extractor 구현
4. ✓ Evaluation framework 구축
5. ✓ Demo notebook 작성

## 학습 목표 정의

### AI가 배워야 할 것

#### 1. 반응 패턴 인식
```
입력: 분자 구조 (SMILES)
학습: 이 구조에서 어떤 반응이 가능한가?

예시:
- 1차 알코올 (R-CH2-OH) → 산화 가능 → 알데히드 (R-CHO)
- 2차 알코올 (R-CH(OH)-R') → 산화 가능 → 케톤 (R-CO-R')
- 케톤 ⇌ 알도스 → 이성질화 가능
```

#### 2. 구조 변화 예측
```
질문: 반응이 일어나면 어느 원자가 바뀌는가?

학습 내용:
- C2 위치의 -OH가 =O로 변함
- 혼성화 변화: sp3 → sp2
- 수소 개수 변화
- 결합 차수 변화
```

#### 3. 효소 요구사항 학습
```
반응 유형 → 필수 효소 클래스

산화반응:
- EC 1.1.1.x (NAD+ 의존성 dehydrogenase) 필수
- 보조인자: NAD+ 또는 NADP+

이성질화:
- EC 5.3.x (isomerase) 필수
- 보조인자: 대부분 불필요
```

#### 4. 효소 선택 최적화
```
같은 반응이라도 효소마다 차이:

L-sorbitol → L-sorbose:
- EC 1.1.1.14 (L-iditol 2-dehydrogenase) ⭐ 최적
- EC 1.1.1.15 (다른 dehydrogenase) - 덜 효율적
- EC 1.1.99.x (다른 산화효소) - 부적합

AI 학습 목표: 가장 효율적인 효소 상위 5개 추천
```

## 구현된 시스템

### 1. Rule-Based Predictor
**위치**: `src/rules/rule_based_predictor.py`

**기능**:
- 화학 변환 규칙 5개 구현
- SMARTS 패턴 매칭
- 신뢰도 점수 계산

**사용법**:
```python
from rules.rule_based_predictor import RuleBasedPredictor

predictor = RuleBasedPredictor()
predictions = predictor.predict_reaction_centers(
    smiles="OCC(O)C(O)C(O)C(O)CO",  # sorbitol
    reaction_type="oxidation"
)
```

### 2. Reaction Center Extractor
**위치**: `src/data_extraction/reaction_center_extractor.py`

**기능**:
- Atom-mapped SMILES 파싱
- 반응 중심 자동 추출
- 변환 유형 분류

**학습 데이터 생성**:
```python
from data_extraction.reaction_center_extractor import ReactionCenterExtractor

extractor = ReactionCenterExtractor()
center = extractor.extract_from_mapped_smiles(
    "[CH2:1][OH:2]>>[CH:1]=[O:2]"
)
# → center.atom_indices = [0, 1]
# → center.change_type = "alcohol_to_carbonyl"
```

### 3. Evaluation Framework
**위치**: `src/evaluation/evaluator.py`

**메트릭**:
- Top-1, Top-3, Top-5 accuracy
- 반응 유형별 성능
- 실용적 영향 분석

## 데이터 구조

### 학습 데이터 스키마

```python
{
    "reaction_id": "RHEA:12345",
    "substrate": {
        "smiles": "OCC(O)C(O)C(O)C(O)CO",
        "name": "D-sorbitol"
    },
    "product": {
        "smiles": "OCC(=O)C(O)C(O)C(O)CO",
        "name": "D-fructose"
    },
    "reaction_info": {
        "type": "oxidation",
        "reaction_center_atoms": [2],  # C2 위치
        "change_description": "secondary_alcohol_to_ketone",
        "cofactor": "NAD+"
    },
    "enzyme_info": {
        "ec_number": "1.1.1.14",
        "name": "L-iditol 2-dehydrogenase",
        "specificity": "high",  # 이 반응에 대한 특이성
        "activity": 95  # 상대적 활성도 (0-100)
    },
    "alternative_enzymes": [
        {
            "ec_number": "1.1.1.15",
            "activity": 60,
            "note": "덜 효율적"
        }
    ]
}
```

## 다음 단계

### 즉시 가능
1. Demo notebook 실행 (`notebooks/01_pilot_demo.ipynb`)
2. Rule-based predictor 테스트
3. 샘플 반응으로 검증

### 1주일 내
1. Rhea 데이터베이스에서 실제 반응 추출
2. 100+ 반응으로 테스트 세트 구축
3. Rule-based baseline 성능 측정

### 1개월 내
1. 효소-반응 관계 데이터 수집
2. 효소 활성도 데이터 통합
3. 효소 추천 시스템 프로토타입

## 파일 구조

```
2026-02-13_reaction-center-prediction-baseline/
├── README.md                           # 이 파일
├── src/
│   ├── data_extraction/
│   │   ├── rhea_parser.py             # Rhea DB 파서
│   │   └── reaction_center_extractor.py
│   ├── rules/
│   │   └── rule_based_predictor.py    # 규칙 기반 예측
│   └── evaluation/
│       └── evaluator.py               # 평가 시스템
├── notebooks/
│   └── 01_pilot_demo.ipynb            # 데모
├── docs/
│   ├── README.md                      # 프로젝트 개요
│   ├── RECOMMENDATION.md              # 추천 접근법
│   ├── IMPLEMENTATION_GUIDE.md        # 실행 가이드
│   └── project_structure.md           # 상세 설계
└── data/
    ├── raw/                           # 원본 데이터
    └── processed/                     # 처리된 데이터
```

## 실행 방법

```bash
# 환경 설정
cd /home/pjho3/projects/AI/2026-02-13_reaction-center-prediction-baseline
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

# 테스트
python src/rules/rule_based_predictor.py

# Jupyter 실행
jupyter notebook notebooks/01_pilot_demo.ipynb
```

## 핵심 인사이트

### 학습 전략
1. **간단한 반응부터**: Polyol 산화/이성질화
2. **명확한 패턴**: 원자 보존 반응
3. **점진적 확장**: 복잡한 반응으로

### 효소 학습 전략
1. **필수 vs 선택**: 어떤 효소가 꼭 필요한가?
2. **효율성 순위**: 같은 반응에 여러 효소 중 최선은?
3. **특이성 학습**: 왜 이 효소가 이 반응에 좋은가?

## 참고 문서

- `docs/README.md` - 전체 프로젝트 개요
- `docs/RECOMMENDATION.md` - 추천 접근 방식 (한국어)
- `docs/IMPLEMENTATION_GUIDE.md` - 상세 실행 가이드
- `docs/project_structure.md` - 기술 설계 문서
