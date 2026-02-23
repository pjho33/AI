# HRM AI 진단 보조 시스템 연구계획

> 고해상도 내압검사(HRM) 데이터 기반 식도 운동 질환 자동 진단 AI 개발

---

## 1. 연구 배경 및 필요성

### 문제 정의
- **고해상도 내압검사(HRM)**: 식도 운동 질환의 표준 진단법
  - 식도 전체 압력 변화를 시공간적 패턴으로 시각화
  - 시카고 분류법(Chicago Classification v4.0) 기반 진단
- **현재 한계**:
  - 전문가 수동 분석에 전적으로 의존 → 시간 부담 큼
  - 랜드마크 설정 및 압력 패턴 해석에 주관적 판단 개입
  - 숙련된 전문가 간에도 진단 불일치 발생
  - 임상 활용도 및 접근성 저해

### 기존 AI 연구 현황
- CNN(EfficientNet, Inception V3), LSTM 등 적용 → 높은 정확도 달성
- LLM 보조 도구 활용 시도
- **한계**: 진단 결과만 제시, 의료진이 신뢰할 수 있는 근거 미제공

### 본 연구의 방향
- 대규모 HRM 데이터 학습 → 식도 운동 질환 자동 진단 알고리즘 개발
- 설명 가능한 AI(XAI) 적용 → 판단 근거 명확히 제시
- 진단 정확도·일관성 향상 + 의료진 업무 효율 극대화

---

## 2. 연구 목표

| 목표 | 내용 |
|------|------|
| **다기관 데이터** | 5개 대학병원 대규모 데이터 수집 → 일반화 가능한 모델 구축 |
| **Raw Data 학습** | 이미지 아닌 ASCII/Text 원본 수치 데이터 직접 학습 → 데이터 손실 원천 차단 |
| **앙상블 모델** | EfficientNet + XGBoost 결합 → 패턴 + 수치 모두 고려 |
| **Dual XAI** | Grad-CAM(시각적) + SHAP(임상적) 이중 설명 시스템 |
| **아형 탐색** | VAE 비지도 학습 → 기존 분류로 구분 안 되는 새로운 Sub-phenotype 발굴 |

---

## 3. 연구 내용 및 방법

### 3-1. 데이터 수집

- **연구 설계**: 다기관 후향적 코호트 연구 (Multicenter Retrospective Cohort Study)
- **데이터 소스**: 5개 대학병원 소화기내과 (2010~2026년)
  - 강북삼성병원, 삼성서울병원, 이대목동병원, 강남세브란스병원, 은평성모병원
- **핵심 차별점**: 이미지 캡처 방식 지양 → **ASCII(Text) Raw Numerical Data 추출**
  - Time(시간) × Channel(센서 위치) 좌표의 정확한 압력값(mmHg) 확보
- **Ground Truth**: 소화기내과 전문의가 Chicago Classification v4.0으로 판독한 결과

### 3-2. 데이터 전처리

```
ASCII Raw Data
    ↓ Python 파싱
2D 행렬 (Time × Sensor)
    ↓ Min-Max Scaling (min=-10, max=300 mmHg, 병원 간 장비 차이 보정)
    ↓ Wet Swallow 구간 Cropping
4D Tensor: (Batch, 1, Sensors, Time)
```

- 정규화 범위: -10 ~ 300 mmHg (HRM 의미 있는 압력 범위 기준 고정)
- 채널 `1`: 압력 정보 단일 채널

### 3-3. AI 모델 아키텍처 (3단계 계층적 구조)

#### Step 1. 특징 추출 (Feature Extraction)
- **모델**: EfficientNet-B0
  - Inception V3 대비 적은 파라미터로 SOTA 성능
  - 입력 채널: RGB(3) → Grayscale(1) 수정
  - 식도 연동, IRP 구간 등 시공간적 패턴 학습

#### Step 2. 최종 진단 + 설명 가능성 (Diagnosis & XAI)
- **앙상블 분류기**:
  - EfficientNet Feature map + IRP·DCI 등 임상 지표 결합
  - XGBoost 또는 LightGBM 최종 분류
  - 시카고 분류의 계층적 진단 논리 모방

- **Dual XAI System**:
  - **Grad-CAM** (시각적 설명): EfficientNet이 판단 근거로 삼은 식도 부위(LES, UES, 체부)를 압력 지형도 위에 히트맵으로 시각화
  - **SHAP** (임상적 설명): XGBoost에서 어떤 임상 변수(예: IRP > 15mmHg)가 진단에 기여했는지 수치 제시

#### Step 3. 미세 아형 탐색 (Unsupervised Phenotyping)
- **모델**: Variational Autoencoder (VAE)
- **분석 흐름**:
  ```
  HRM Raw Data → VAE → 저차원 잠재 공간(Latent Space) z
      ↓ t-SNE / UMAP 시각화
  Sub-cluster 식별 → 임상 증상·치료 반응과 연관 분석
  ```
- **목표**: '정상' 또는 '비특이적 운동장애'로 분류되던 환자 중 새로운 하위 그룹 발굴
  - 특정 약물 반응 또는 예후와 연결 시 → Novel Phenotype 정의

### 3-4. 통계 분석 및 평가

| 항목 | 내용 |
|------|------|
| 데이터 분할 | Train : Validation : Test = 8 : 1 : 1 |
| 평가 지표 | Accuracy, Sensitivity, Specificity, F1-score, AUROC |
| 검증 방법 | 5-fold Cross Validation |
| 개발 환경 | Linux (Ubuntu), Python 3.9, PyTorch, NVIDIA RTX 3090 |

---

## 4. 기대 효과 및 활용 방안

### 4-1. 본 연구의 강점

| 강점 | 설명 |
|------|------|
| Multicenter Raw Data | 이미지 아닌 원본 수치 데이터 → 기술적 신뢰도 향상 |
| EfficientNet + XGBoost | SOTA 딥러닝 + Tabular 데이터 최강자 결합 |
| Dual XAI | "어디를 봤어?(Grad-CAM)" + "뭐가 중요했어?(SHAP)" 동시 제공 |
| VAE Phenotyping | 단순 진단을 넘어 미진단 영역의 새로운 연구적 발견 가능 |

### 4-2. 연구사적 기여도

- **데이터 분석 패러다임 전환**: Raw Data(ASCII) 직접 학습으로 생체 신호 분석의 새로운 표준 방법론 제시
- **설명 가능한 AI 실현**: Dual XAI로 '블랙박스' 문제 해결 → 의료진 신뢰 확보
- **새로운 질환 분류 체계 제시**: VAE 기반 Sub-phenotype 발굴 → 학문적 지평 확장

### 4-3. 활용 방안

- **임상 의사결정 지원(CDSS)**: 전문의 부족 기관·수련의에게 전문가 수준 판독 보조, Inter-observer variability 최소화
- **정밀 의료(Precision Medicine)**: 아형별 약물 반응·시술 예후 추적 → 데이터 기반 최적 치료 전략
- **기술 확장성**: 항문 내압검사(HR-ARM) 등 유사 시공간 생체 신호 분석으로 즉시 확장 가능

---

## 프로젝트 구조 (예정)

```
2026-02-23_hrm-ai-diagnosis/
├── README.md                   # 이 파일 (연구계획)
├── data/
│   ├── raw/                    # ASCII 원본 데이터 (gitignore)
│   ├── processed/              # 전처리된 텐서
│   └── splits/                 # train/val/test 분할 인덱스
├── preprocessing/
│   ├── parse_ascii.py          # ASCII → 2D 행렬 파싱
│   ├── normalize.py            # Min-Max Scaling (-10~300 mmHg)
│   └── crop_swallow.py         # Wet Swallow 구간 Cropping
├── models/
│   ├── efficientnet.py         # EfficientNet-B0 (채널 수정)
│   ├── ensemble.py             # EfficientNet + XGBoost 앙상블
│   └── vae.py                  # VAE (Sub-phenotype 탐색)
├── xai/
│   ├── gradcam.py              # Grad-CAM 시각화
│   └── shap_analysis.py        # SHAP Value 분석
├── evaluation/
│   └── metrics.py              # Accuracy, AUROC, F1 등
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_training.ipynb
    └── 03_xai_visualization.ipynb
```
