# Chemical Reaction Prediction & Enzyme Selection AI

## 프로젝트 개요

화학반응의 가능성을 예측하고 적절한 효소를 선택하는 AI 시스템

### 학습 목표

AI가 학습할 내용:
1. **반응 패턴 인식**: 가장 간단한 화학반응부터 시작해서 어떤 구조가 어떻게 반응하는지
2. **구조 변화 예측**: 반응이 일어나면 분자의 어느 부분에 변화가 생기는지
3. **효소 요구사항**: 어떤 효소가 필수적이고, 어떤 효소가 반응을 더 잘 촉매하는지
4. **효소-반응 관계**: 특정 반응에 가장 적합한 효소 선택

## 프로젝트 구조 (날짜별 정리)

```
AI/
├── 2026-02-13_reaction-center-prediction-baseline/
│   ├── src/                    # 코드
│   ├── data/                   # 데이터
│   ├── notebooks/              # 분석 노트북
│   └── docs/                   # 문서
├── 2026-02-14_[다음작업]/
└── ...
```

### 날짜별 작업 내용

#### 2026-02-13: Reaction Center Prediction Baseline
- 반응 중심 예측 시스템 기초 구축
- Rule-based predictor 구현
- Evaluation framework 구축
- Polyol/sugar 산화환원 + 이성질화 타겟팅

## 핵심 철학

> "모든 화학반응을 예측하려 하지 않는다.  
> AI가 잘 배울 수 있는 반응부터 정복하고,  
> 그 틀을 관심 반응으로 확장한다."

## 빠른 시작

각 날짜 폴더의 README를 참조하세요.

```bash
cd 2026-02-13_reaction-center-prediction-baseline
cat docs/README.md
```
