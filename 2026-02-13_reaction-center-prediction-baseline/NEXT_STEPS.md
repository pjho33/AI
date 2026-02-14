# 다음 단계 (내일 시작할 것)

## Stage 2: Performance Prediction (성능 예측)

### 🎯 목표
실험 성능 예측 시스템 구축
- 수율 (yield) 예측
- 효소 동역학 (kcat, Km) 예측
- 반응 시간 예측
- 선택성 예측

---

## 📋 작업 순서

### 1. BRENDA 데이터 수집
**목표**: kcat, Km 값 수집

**작업**:
```python
# src/data_loaders/brenda_loader.py 생성
- BRENDA API 연동
- EC 번호별 kcat, Km 수집
- 결측치 표시
```

**예상 결과**:
- 50-100개 효소 동역학 데이터
- 결측률: 70% (예상)

### 2. Stage 2 예측기 구현
**목표**: 성능 예측 모델

**작업**:
```python
# src/stage2_performance_predictor.py 생성

class Stage2PerformancePredictor:
    def predict_performance(
        self,
        reaction,  # Stage 1 출력
        assay_conditions
    ) -> PerformancePrediction:
        # 수율 예측
        # kcat, Km 예측
        # 시간 예측
        pass
```

**출력 형식**:
```python
{
    "yield": {
        "mean": 0.85,
        "std": 0.05,
        "source": "estimated"
    },
    "kcat": {
        "mean": 120,
        "std": 15,
        "source": "measured"
    },
    "Km": {
        "mean": 0.6,
        "std": 0.1,
        "source": "transferred"
    }
}
```

### 3. Masked Loss 구현
**목표**: 결측 데이터 대응

**작업**:
```python
def masked_mse_loss(predicted, actual, mask):
    """
    mask: 1 = 측정됨, 0 = 결측
    """
    error = (predicted - actual) ** 2
    masked_error = error * mask
    return masked_error.sum() / mask.sum()
```

**적용**:
- 결측치 무시
- 측정값만 학습
- 불확실성 증가

### 4. 불확실성 정량화
**목표**: 예측 신뢰도 계산

**작업**:
```python
# 불확실성 소스
- 측정 불확실성
- 모델 불확실성
- 결측치 불확실성

# 전파
total_uncertainty = sqrt(
    measurement_var +
    model_var +
    missing_data_var
)
```

### 5. 평가 및 검증
**목표**: Stage 2 성능 평가

**메트릭**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (결정계수)
- 불확실성 보정 정확도

---

## 🔧 필요한 도구

### 데이터 소스
1. **BRENDA** (효소 동역학)
   - URL: https://www.brenda-enzymes.org/
   - 데이터: kcat, Km, Ki
   - API: SOAP 또는 웹 스크래핑

2. **문헌** (수율, 선택성)
   - PubChem
   - 논문 데이터베이스
   - 수동 큐레이션

### 라이브러리
```bash
pip install scipy  # 통계 함수
pip install uncertainties  # 불확실성 전파
```

---

## 📊 예상 데이터 구조

### Assay Record (실험 데이터)
```python
{
    "assay_id": "assay_001",
    "reaction_id": "rxn_sorbitol_001",
    
    "conditions": {
        "pH": 7.4,
        "temperature": 37,
        "substrate_conc": 10,  # mM
        "enzyme_conc": 0.1,    # μM
        "cofactor_conc": 1.0   # mM
    },
    
    "measurements": {
        "kcat": {
            "value": 150,
            "unit": "s-1",
            "source": "brenda",
            "measured": true,
            "confidence": 0.9
        },
        "Km": {
            "value": 0.5,
            "unit": "mM",
            "source": "brenda",
            "measured": true,
            "confidence": 0.9
        },
        "yield": {
            "value": 0.92,
            "source": "literature",
            "measured": true,
            "confidence": 0.7
        },
        "time_to_90pct": {
            "value": null,  # 결측
            "measured": false,
            "estimated": 1.5,
            "confidence": 0.3
        }
    },
    
    "data_completeness": 0.6  # 60% 측정됨
}
```

---

## 🎓 학습 전략

### 1. 전이 학습
```python
# 유사 반응에서 학습
similar_reactions = find_similar(
    substrate_structure,
    enzyme_family
)

estimated_kcat = transfer_from_similar(
    similar_reactions,
    similarity_weights
)
```

### 2. 불확실성 기반 가중치
```python
# 신뢰도 높은 데이터에 더 큰 가중치
loss = sum(
    weight[i] * error[i]
    for i in range(n)
)

weight[i] = confidence[i] / sum(confidence)
```

### 3. 앙상블
```python
# 여러 예측 결합
predictions = [
    model1.predict(),
    model2.predict(),
    transfer_learning.predict()
]

final = weighted_average(predictions, confidences)
```

---

## 📈 성공 기준

### 최소 목표
- MAE < 30% (kcat, Km)
- 수율 예측 MAE < 15%
- 불확실성 보정 정확도 > 70%

### 이상적 목표
- MAE < 20% (kcat, Km)
- 수율 예측 MAE < 10%
- 불확실성 보정 정확도 > 85%

---

## ⚠️ 예상 문제 및 해결책

### 문제 1: 데이터 결측 70%
**해결**:
- Masked loss 사용
- 전이 학습
- 불확실성 명시

### 문제 2: BRENDA API 제한
**해결**:
- 캐싱 사용
- 속도 제한 준수
- 대안: 웹 스크래핑

### 문제 3: 조건 의존성
**해결**:
- 조건별 모델 학습
- 조건 정규화
- 외삽 주의

---

## 📝 체크리스트

### Day 1 (내일)
- [ ] BRENDA 데이터 로더 구현
- [ ] 10-20개 효소 데이터 수집
- [ ] 데이터 구조 검증

### Day 2
- [ ] Stage 2 예측기 기본 구조
- [ ] Masked loss 구현
- [ ] 간단한 예측 테스트

### Day 3
- [ ] 불확실성 정량화
- [ ] 전이 학습 구현
- [ ] 평가 메트릭

### Day 4-5
- [ ] 성능 최적화
- [ ] 문서화
- [ ] Stage 2 완료

---

## 🚀 시작 명령어

```bash
# 프로젝트 디렉토리로 이동
cd /home/pjho3/projects/AI/2026-02-13_reaction-center-prediction-baseline

# 가상환경 활성화
source venv/bin/activate

# 새 파일 생성
touch src/data_loaders/brenda_loader.py
touch src/stage2_performance_predictor.py

# 작업 시작!
```

---

## 📚 참고 자료

### BRENDA
- 웹사이트: https://www.brenda-enzymes.org/
- API 문서: https://www.brenda-enzymes.org/brenda_download.php
- 데이터 형식: XML, JSON

### 불확실성 전파
- uncertainties 라이브러리
- 베이지안 추론
- 몬테카를로 시뮬레이션

### 전이 학습
- 분자 유사도 (Tanimoto)
- 효소 패밀리 유사도
- 반응 패턴 유사도

---

## 💡 팁

1. **작은 것부터 시작**
   - 5-10개 반응으로 프로토타입
   - 검증 후 확장

2. **불확실성 항상 명시**
   - 측정값 vs 추정값 구분
   - 신뢰도 표시

3. **결측치 전략**
   - 무시하지 말고 명시
   - 불확실성 증가

4. **검증 중요**
   - 실험 데이터와 비교
   - 물리적 타당성 확인

---

**준비 완료! Stage 2 시작하자! 🚀**
