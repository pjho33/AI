# 빠른 시작 가이드

## 간단한 화학반응으로 AI 학습 시작하기

### 1분 요약

**목표**: 5개 간단한 반응으로 AI가 화학반응 패턴을 배우게 함

**학습 내용**:
- 어떤 분자 구조가 어떻게 반응하는가
- 어느 위치가 반응 중심인가
- 어떤 효소가 필요한가

**결과**: 새로운 분자를 보고 반응 예측 + 효소 추천 가능

---

## 즉시 실행

```bash
cd /home/pjho3/projects/AI/2026-02-13_reaction-center-prediction-baseline

# 환경 설정 (처음 한 번만)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 학습 실행
python src/train_simple_model.py
```

**실행 시간**: 약 10초  
**출력**: 학습 과정 + 예측 결과 + 성능 평가

---

## 학습 데이터

### 5개 간단한 반응

1. **에탄올 → 아세트알데히드** (가장 기본)
   - 1차 알코올 산화
   - 효소: EC 1.1.1.1

2. **2-프로판올 → 아세톤**
   - 2차 알코올 산화
   - 효소: EC 1.1.1.1

3. **글리세롤 → 디하이드록시아세톤**
   - 여러 OH 중 C2만 선택적 산화
   - 효소: EC 1.1.1.6

4. **소르비톨 → 프럭토스**
   - 6탄당 알코올 산화
   - 효소: EC 1.1.1.14

5. **글루코스 → 프럭토스** (이성질화)
   - 산화 없이 구조만 변화
   - 효소: EC 5.3.1.5

데이터 위치: `data/simple_reactions.json`

---

## AI가 배우는 것

### Level 1: 패턴 인식
```
R-CH2-OH → 1차 알코올 → 알데히드로 산화
R-CH(OH)-R' → 2차 알코올 → 케톤으로 산화
```

### Level 2: 효소 요구사항
```
산화반응 → EC 1.1.1.x + NAD+ 필수
이성질화 → EC 5.3.x + 보조인자 불필요
```

### Level 3: 위치 선택성
```
여러 OH기 중 어디가 반응하는가?
→ 입체화학, 효소 특이성에 따라 결정
```

---

## 학습 후 얻는 것

### 1. 반응 예측 능력
새 분자 입력 → AI가 자동으로:
- ✓ 반응 가능한 위치 찾기
- ✓ 반응 유형 분류
- ✓ 생성물 예측

### 2. 효소 추천 시스템
반응 + 기질 입력 → AI가 자동으로:
- ✓ 적합한 효소 Top-5 추천
- ✓ 필요한 보조인자 제시
- ✓ 예상 효율 점수

### 3. 지식 베이스
- 20+ 화학 패턴
- 10+ 효소-기질 관계
- 반응 조건 규칙

---

## 예상 성능

### 학습 데이터 (5개 반응)
- Top-1 Accuracy: 60-80%
- Top-5 Accuracy: 90-100%

### 새로운 분자 (일반화)
- Top-1 Accuracy: 40-60%
- Top-5 Accuracy: 70-85%

**실용성**: Top-5 정확도 70%면 효소 스크리닝 비용 90% 절감 가능

---

## 출력 예시

```
[rxn_001] ethanol_to_acetaldehyde
반응: ethanol → acetaldehyde
효소: alcohol dehydrogenase (EC 1.1.1.1)
보조인자: NAD+
반응 중심: C2 (primary alcohol carbon)

학습 포인트:
  • 1차 알코올은 알데히드로 산화된다
  • NAD+가 필수 보조인자
  • C-OH 결합이 C=O로 변한다
```

---

## 다음 단계

### 즉시 (오늘)
1. ✓ 5개 반응 학습 실행
2. ✓ 결과 확인
3. ✓ 새 분자로 예측 테스트

### 1주일 내
1. 10개 더 추가 (총 15개)
2. 다양한 polyol 반응
3. 성능 향상 확인

### 1개월 내
1. 50개 반응으로 확장
2. ML 모델 도입
3. 효소 활성도 예측 추가

---

## 문제 해결

### 오류: 모듈을 찾을 수 없음
```bash
# 경로 확인
cd /home/pjho3/projects/AI/2026-02-13_reaction-center-prediction-baseline
pwd

# 가상환경 활성화 확인
which python
# 출력: .../venv/bin/python 이어야 함
```

### 오류: RDKit 설치 실패
```bash
# conda 사용 권장
conda install -c conda-forge rdkit
```

### 데이터 파일 없음
```bash
# 파일 확인
ls data/simple_reactions.json

# 없으면 프로젝트 재다운로드
```

---

## 더 알아보기

- `docs/TRAINING_PLAN.md` - 상세 학습 계획
- `docs/LEARNING_OBJECTIVES.md` - AI 학습 목표
- `data/simple_reactions.json` - 학습 데이터
- `src/train_simple_model.py` - 학습 코드

---

## 요약

**5개 간단한 반응 → AI 학습 → 새 분자 예측 가능**

```bash
python src/train_simple_model.py
```

이 한 줄로 시작!
