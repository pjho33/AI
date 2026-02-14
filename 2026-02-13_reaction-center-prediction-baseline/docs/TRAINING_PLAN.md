# 간단한 화학반응으로 AI 학습 시작하기

## 학습 전략

### 왜 간단한 반응부터?

1. **명확한 패턴**: 복잡한 변수 없이 핵심 패턴 학습
2. **검증 가능**: 결과를 쉽게 확인하고 이해 가능
3. **점진적 확장**: 학습한 패턴을 복잡한 반응에 적용

---

## Phase 1: 초간단 반응 5개로 시작

### 선택한 반응들

#### 1. 에탄올 → 아세트알데히드 (가장 기본)
```
반응: CH3-CH2-OH → CH3-CHO
효소: EC 1.1.1.1 (alcohol dehydrogenase)
보조인자: NAD+

학습 포인트:
- 1차 알코올이 알데히드로 산화
- C2 위치의 -OH가 =O로 변화
- NAD+ 필수
```

#### 2. 2-프로판올 → 아세톤
```
반응: CH3-CH(OH)-CH3 → CH3-CO-CH3
효소: EC 1.1.1.1 (alcohol dehydrogenase)
보조인자: NAD+

학습 포인트:
- 2차 알코올이 케톤으로 산화
- 중앙 탄소의 -OH가 =O로
- 1차와 2차 알코올 차이 학습
```

#### 3. 글리세롤 → 디하이드록시아세톤
```
반응: HO-CH2-CH(OH)-CH2-OH → HO-CH2-CO-CH2-OH
효소: EC 1.1.1.6 (glycerol dehydrogenase)
보조인자: NAD+

학습 포인트:
- 여러 -OH 중 C2만 산화
- 위치 선택성 학습
- 대칭 분자에서의 반응 중심
```

#### 4. D-소르비톨 → D-프럭토스
```
반응: OCC(O)C(O)C(O)C(O)CO → OCC(=O)C(O)C(O)C(O)CO
효소: EC 1.1.1.14 (L-iditol 2-dehydrogenase)
보조인자: NAD+

학습 포인트:
- 6개 탄소 중 C2만 산화
- 입체화학 중요성
- 효소 특이성
```

#### 5. D-글루코스 → D-프럭토스 (이성질화)
```
반응: C1 알데히드 ⇌ C2 케톤
효소: EC 5.3.1.5 (glucose-6-phosphate isomerase)
보조인자: 없음

학습 포인트:
- 산화 없이 구조만 변화
- 보조인자 불필요
- 이성질화 vs 산화 구분
```

---

## AI가 배우는 것

### Level 1: 패턴 인식 (1-2일)

**입력**: 분자 구조
**출력**: 반응 가능 위치

```python
# AI가 학습하는 패턴
패턴1: "R-CH2-OH" → 1차 알코올 → 산화 가능
패턴2: "R-CH(OH)-R'" → 2차 알코올 → 산화 가능
패턴3: "R-CO-R'" → 케톤 → 환원 가능
패턴4: "R-CHO" → 알데히드 → 환원 가능
```

**학습 결과**:
- 5개 반응에서 공통 패턴 추출
- 새로운 알코올 분자 보면 → 어디가 산화될지 예측 가능

### Level 2: 반응 조건 학습 (3-5일)

**입력**: 분자 + 반응 유형
**출력**: 필요한 효소 + 보조인자

```python
# AI가 학습하는 규칙
규칙1: 알코올 산화 → EC 1.1.1.x + NAD+ 필요
규칙2: 이성질화 → EC 5.3.x + 보조인자 불필요
규칙3: 1차 알코올 → 알데히드
규칙4: 2차 알코올 → 케톤
```

**학습 결과**:
- 반응 유형만 보고 필요한 효소 클래스 예측
- 보조인자 요구사항 예측

### Level 3: 효소 선택 학습 (1주)

**입력**: 반응 + 기질 구조
**출력**: 최적 효소 순위

```python
# AI가 학습하는 효소-기질 관계
에탄올 산화:
  - EC 1.1.1.1 (일반 alcohol dehydrogenase) ⭐⭐⭐⭐⭐
  - EC 1.1.1.71 (특정 dehydrogenase) ⭐⭐

소르비톨 산화:
  - EC 1.1.1.14 (L-iditol 2-dehydrogenase) ⭐⭐⭐⭐⭐
  - EC 1.1.1.1 (일반 alcohol dehydrogenase) ⭐⭐
```

**학습 결과**:
- 기질 구조에 따른 효소 특이성 이해
- 일반 효소 vs 특이적 효소 구분

### Level 4: 예측 능력 (2주)

**입력**: 새로운 분자 (학습 안 한 것)
**출력**: 반응 예측 + 효소 추천

```python
# 테스트: 자일리톨 (학습 안 함)
입력: OCC(O)C(O)C(O)CO

AI 예측:
1. C2 산화 가능 (확률 85%)
2. C3 산화 가능 (확률 75%)
3. 필요 효소: EC 1.1.1.x
4. 보조인자: NAD+
5. 추천 효소: EC 1.1.1.9 (xylitol dehydrogenase)
```

**학습 결과**:
- 학습한 패턴을 새 분자에 적용
- 일반화 능력 획득

---

## 실제 학습 데이터 형식

### 훈련 데이터 예시

```python
training_data = [
    {
        "id": "reaction_001",
        "name": "ethanol_oxidation",
        "substrate": {
            "smiles": "CCO",
            "name": "ethanol",
            "structure": "CH3-CH2-OH"
        },
        "product": {
            "smiles": "CC=O",
            "name": "acetaldehyde",
            "structure": "CH3-CHO"
        },
        "reaction_center": {
            "atoms": [1],  # C2 위치
            "change": "alcohol_to_aldehyde",
            "bond_change": "C-OH → C=O"
        },
        "enzyme": {
            "ec": "1.1.1.1",
            "name": "alcohol dehydrogenase",
            "specificity": "broad",
            "activity": 90
        },
        "cofactor": "NAD+",
        "reaction_type": "oxidation"
    },
    
    {
        "id": "reaction_002",
        "name": "isopropanol_oxidation",
        "substrate": {
            "smiles": "CC(C)O",
            "name": "2-propanol",
            "structure": "CH3-CH(OH)-CH3"
        },
        "product": {
            "smiles": "CC(C)=O",
            "name": "acetone",
            "structure": "CH3-CO-CH3"
        },
        "reaction_center": {
            "atoms": [1],  # 중앙 C
            "change": "alcohol_to_ketone",
            "bond_change": "C-OH → C=O"
        },
        "enzyme": {
            "ec": "1.1.1.1",
            "name": "alcohol dehydrogenase",
            "specificity": "broad",
            "activity": 85
        },
        "cofactor": "NAD+",
        "reaction_type": "oxidation"
    },
    
    # ... 나머지 3개 반응
]
```

---

## 학습 후 얻을 수 있는 것

### 1. 반응 예측 시스템
```
새 분자 입력 → AI가 자동으로:
- 반응 가능한 위치 찾기
- 반응 유형 분류
- 생성물 예측
```

### 2. 효소 추천 시스템
```
반응 + 기질 입력 → AI가 자동으로:
- 적합한 효소 Top-5 추천
- 필요한 보조인자 제시
- 예상 효율 점수
```

### 3. 지식 베이스
```
AI가 학습한 지식:
- 20+ 화학 패턴
- 10+ 효소-기질 관계
- 반응 조건 규칙
```

### 4. 확장 가능한 기반
```
5개 반응 학습 → 50개로 확장 → 500개로 확장
같은 방법론으로 계속 성장
```

---

## 구체적 실행 계획

### Day 1: 데이터 준비
```bash
1. 5개 반응의 SMILES 수집
2. 효소 정보 정리
3. training_data.json 생성
```

### Day 2: 학습 실행
```python
1. 데이터 로딩
2. Rule-based predictor로 패턴 추출
3. 성능 측정 (Top-1, Top-5 accuracy)
```

### Day 3: 검증 및 분석
```python
1. 새로운 분자로 테스트
2. 예측 정확도 확인
3. 학습한 패턴 시각화
```

### Day 4: 문서화
```markdown
1. 학습 결과 정리
2. AI가 배운 것 요약
3. 다음 단계 계획
```

---

## 성공 기준

### 최소 목표 (MVP)
- ✓ 5개 반응 100% 정확히 예측
- ✓ 새 알코올 분자에서 산화 위치 70% 정확도
- ✓ 효소 클래스 80% 정확히 예측

### 이상적 목표
- ✓ 새 분자에서 90% 정확도
- ✓ 효소 Top-3 추천 85% 정확도
- ✓ 학습한 패턴을 다른 반응에 적용 가능

---

## 다음 확장

### Phase 2: 10개 더 추가
- 다른 polyol 산화
- 다른 이성질화
- 환원 반응

### Phase 3: 복잡한 반응
- 여러 단계 반응
- 입체선택적 반응
- C-C 결합 형성

---

## 시작하기

바로 시작하려면:

```bash
cd 2026-02-13_reaction-center-prediction-baseline
python src/create_simple_training_data.py
python src/train_simple_model.py
```

다음 파일 생성 필요:
1. `data/simple_reactions.json` - 5개 반응 데이터
2. `src/create_simple_training_data.py` - 데이터 생성 스크립트
3. `src/train_simple_model.py` - 학습 스크립트
4. `notebooks/02_simple_learning.ipynb` - 학습 과정 시각화

만들어줄까?
