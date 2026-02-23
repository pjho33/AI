# HRM AI 진단 연구 실행 계획

> 예상 총 기간: 18~24개월

---

## 전체 로드맵

```
Phase 1 (1~3개월)   : 데이터 수집 & 전처리 파이프라인 구축
Phase 2 (3~6개월)   : EfficientNet 기반 특징 추출 모델 개발
Phase 3 (6~10개월)  : 앙상블 + Dual XAI 시스템 구축
Phase 4 (10~14개월) : VAE 기반 Sub-phenotype 탐색
Phase 5 (14~18개월) : 다기관 검증 & 논문 작성
```

---

## Phase 1. 데이터 수집 & 전처리 파이프라인 (1~3개월)

### 1-1. 데이터 수집 및 IRB 승인
- [ ] 5개 기관 IRB 신청 (강북삼성, 삼성서울, 이대목동, 강남세브란스, 은평성모)
- [ ] 각 기관 HRM 소프트웨어에서 ASCII Export 방법 확인 및 표준화
- [ ] 데이터 수집 기준 통일: 2010~2026년, 성인, 표준 wet swallow 프로토콜
- [ ] 진단 레이블 기준 확정: Chicago Classification v4.0 (전문의 2인 독립 판독)

### 1-2. 데이터 현황 파악 (목표 샘플 수)
| 진단 | 예상 샘플 수 | 비고 |
|------|------------|------|
| 정상 (Normal) | ~1,000 | 가장 많음 |
| 아칼라지아 Type I/II/III | ~300 each | 핵심 타겟 |
| EGJ Outflow Obstruction | ~200 | |
| 비특이적 운동장애 | ~500 | VAE 탐색 대상 |
| 기타 | ~200 | |
| **합계** | **~3,000+** | 다기관 합산 |

> ⚠️ 클래스 불균형 대비: Oversampling(SMOTE) 또는 Class-weighted Loss 적용 예정

### 1-3. 전처리 파이프라인 구현
```python
# 목표 파이프라인
ASCII file
  → parse_ascii.py        # 헤더 제거, Time×Sensor 행렬 추출
  → normalize.py          # Min-Max Scaling (min=-10, max=300 mmHg)
  → crop_swallow.py       # Wet Swallow 구간 자동 감지 & Cropping
  → tensor_builder.py     # (Batch, 1, Sensors, Time) 4D Tensor 변환
  → quality_check.py      # 불량 데이터 필터링 (센서 탈락, 아티팩트)
```

**핵심 구현 포인트:**
- 병원별 센서 수 차이 처리 (보통 32~36채널) → Zero-padding 또는 Interpolation
- Wet Swallow 자동 감지: UES 압력 패턴 기반 threshold 알고리즘
- 1인당 여러 swallow → 각 swallow를 독립 샘플로 처리 vs 평균화 전략 결정 필요

### 1-4. 탐색적 데이터 분석 (EDA)
- [ ] 기관별 압력 분포 비교 (장비 차이 확인)
- [ ] 진단군별 평균 압력 지형도 시각화
- [ ] IRP, DCI 등 임상 지표 분포 확인
- [ ] 클래스 불균형 정도 파악

**산출물**: `data/processed/` 텐서 데이터셋, EDA 노트북

---

## Phase 2. EfficientNet 특징 추출 모델 (3~6개월)

### 2-1. 모델 구조 설계
```python
# EfficientNet-B0 수정 사항
- 입력: (Batch, 1, Sensors, Time)  # Grayscale 1채널
- 첫 번째 Conv 레이어: in_channels=3 → 1 로 수정
- 출력: Feature vector (1280-dim) → 진단 클래스 수로 FC 연결
```

**실험 비교 대상:**
| 모델 | 비고 |
|------|------|
| EfficientNet-B0 | 기본 (파라미터 적음) |
| EfficientNet-B2 | 성능 상향 시 |
| ResNet-50 | 비교 베이스라인 |
| Inception V3 | 선행 연구 재현 |

### 2-2. 학습 전략
- **Loss**: Cross-Entropy + Label Smoothing (0.1)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-2
- **Scheduler**: CosineAnnealingLR
- **Augmentation**: 시간축 Jitter, 압력값 Gaussian Noise 추가
- **Batch size**: 32 (RTX 3090 기준)
- **Early stopping**: Validation loss 기준 patience=10

### 2-3. 평가
- 5-fold Cross Validation
- 지표: Accuracy, Sensitivity, Specificity, F1, AUROC
- **목표**: AUROC > 0.90 (아칼라지아 분류 기준)

**산출물**: 학습된 EfficientNet 가중치, 성능 리포트

---

## Phase 3. 앙상블 + Dual XAI 시스템 (6~10개월)

### 3-1. 앙상블 구성
```
EfficientNet Feature (1280-dim)
        +
임상 지표 (IRP, DCI, CFV, IBP 등 ~10개)
        ↓
  XGBoost / LightGBM
        ↓
  최종 진단 (Chicago Classification)
```

**임상 지표 추출 자동화:**
- IRP (Integrated Relaxation Pressure): LES 구간 자동 감지 후 계산
- DCI (Distal Contractile Integral): 원위부 식도 수축 적분값
- CFV (Contractile Front Velocity): 수축 전파 속도

### 3-2. Grad-CAM 구현
```python
# 목표 시각화
- EfficientNet의 마지막 Conv layer에서 Gradient 추출
- 압력 지형도(Time × Sensor) 위에 히트맵 오버레이
- 해부학적 랜드마크(UES, LES, 식도 체부) 위치 표시
- 출력: 환자별 판단 근거 이미지 자동 생성
```

### 3-3. SHAP 분석
```python
# 목표
- XGBoost 모델에 SHAP TreeExplainer 적용
- 변수 중요도 순위: IRP, DCI, Feature_xxx 등
- 환자별 개인화된 SHAP waterfall plot 생성
- 전체 코호트 SHAP summary plot (beeswarm)
```

### 3-4. 통합 리포트 자동 생성
- 환자 1명당 PDF 리포트: 압력 지형도 + Grad-CAM + SHAP + 최종 진단
- 임상 현장 즉시 활용 가능한 형태

**산출물**: 앙상블 모델, XAI 시각화 모듈, 자동 리포트 생성기

---

## Phase 4. VAE 기반 Sub-phenotype 탐색 (10~14개월)

### 4-1. VAE 아키텍처
```python
# Convolutional VAE
Encoder: Conv2D layers → μ, σ (잠재 벡터 z, dim=64)
Decoder: ConvTranspose2D layers → 원본 복원
Loss: Reconstruction Loss (MSE) + KL Divergence
```

### 4-2. 잠재 공간 분석
```
전체 환자 데이터 → VAE Encoder → z (64-dim)
        ↓
  t-SNE / UMAP (2D/3D 시각화)
        ↓
  클러스터링 (K-means, HDBSCAN)
        ↓
  각 클러스터의 임상 특성 분석
```

### 4-3. Sub-phenotype 검증 기준
- [ ] 클러스터 간 임상 지표(IRP, DCI) 통계적 유의미한 차이 확인
- [ ] 치료 반응(약물, 시술) 차이 분석
- [ ] 증상 점수(Eckardt score 등)와 상관관계
- [ ] 독립 코호트에서 재현성 확인

**핵심 타겟 환자군**: 
- 시카고 분류 "비특이적 운동장애(IEM)" 환자
- 증상은 심하나 검사 정상인 환자
- 아칼라지아 경계선 환자

**산출물**: VAE 모델, Sub-phenotype 분류 기준, 임상 의미 분석 리포트

---

## Phase 5. 다기관 검증 & 논문 작성 (14~18개월)

### 5-1. 외부 검증 (External Validation)
- 학습에 사용하지 않은 1개 기관 데이터로 독립 검증
- 성능 지표 재확인 (AUROC, F1)
- 기관 간 일반화 성능 비교

### 5-2. 임상 유용성 평가
- **Inter-observer variability 감소 효과**: AI 보조 전후 전문의 판독 일치율 비교
- **판독 시간 단축**: AI 보조 전후 소요 시간 측정
- **수련의 vs 전문의**: AI 보조 시 수련의 성능 향상 정도

### 5-3. 논문 전략

| 논문 | 저널 타겟 | 내용 |
|------|----------|------|
| Paper 1 | Gut / GIE (IF ~20) | 전체 시스템: EfficientNet+XGBoost+Dual XAI |
| Paper 2 | Neurogastroenterology | VAE Sub-phenotype 발굴 |
| Paper 3 | 국내 학회지 | 한국인 HRM 데이터 특성 분석 |

---

## 우선순위 & 즉시 시작 가능한 작업

### 지금 당장 할 수 있는 것 (데이터 없이)
1. **전처리 파이프라인 코드 작성** (샘플 ASCII 파일 1~2개로 테스트)
2. **EfficientNet-B0 모델 코드 작성** (랜덤 텐서로 forward pass 확인)
3. **Grad-CAM, SHAP 모듈 구현** (더미 데이터로 시각화 테스트)
4. **VAE 아키텍처 구현** (MNIST 등으로 사전 검증)

### 데이터 수집과 병행
5. **IRB 신청 서류 준비**
6. **각 기관 HRM 소프트웨어 ASCII Export 형식 파악** (기관마다 다를 수 있음)

---

## 기술 스택

| 분야 | 도구 |
|------|------|
| 딥러닝 | PyTorch, timm (EfficientNet) |
| 앙상블 | XGBoost, LightGBM, scikit-learn |
| XAI | pytorch-grad-cam, shap |
| 차원 축소 | umap-learn, scikit-learn (t-SNE) |
| 클러스터링 | hdbscan, scikit-learn |
| 시각화 | matplotlib, seaborn, plotly |
| 데이터 | pandas, numpy, scipy |
| 실험 관리 | MLflow 또는 Weights & Biases |
| 환경 | Python 3.9, CUDA 11.x, RTX 3090 |

---

## 리스크 & 대응 방안

| 리스크 | 대응 |
|--------|------|
| 데이터 수 부족 (특히 희귀 진단) | Transfer Learning + Data Augmentation |
| 기관 간 장비 차이 (센서 수, 압력 범위) | Domain Adaptation 또는 Normalization 강화 |
| 클래스 불균형 | Focal Loss, SMOTE, Class-weighted sampling |
| IRB 지연 | 1개 기관 데이터로 먼저 파이프라인 검증 |
| 모델 과적합 | Dropout, Weight Decay, 5-fold CV |
