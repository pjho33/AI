# 반응 성공 평가 프레임워크 (3-Stage Architecture)

## 핵심 설계 원칙

### 1. 단계적 분리 (Staged Evaluation)
- **Stage 1**: Chemistry Feasibility (화학적 가능성)
- **Stage 2**: Performance (실험 성능)
- **Stage 3**: Process/Economics (공정/경제성)

### 2. 스펙 기반 평가 (Spec-driven)
- 임계값을 하드코딩하지 않음
- 외부 스펙으로 입력받음
- 도메인/목표에 따라 유연하게 조정

### 3. 확률적 점수 (Probabilistic Scoring)
- 경직된 AND 게이트 대신
- 가중합 점수 + 최소 게이트 1-2개
- Trade-off 허용

### 4. 데이터 현실성 (Data Realism)
- 측정 가능한 것만 학습 목표로
- 결측치 많은 데이터 대응
- 추정치로 라벨 만들지 않음

---

## Stage 1: Chemistry Feasibility

### 목표
**"이 반응이 화학적으로 일어날 수 있는가?"**

### 입력
```python
{
    "substrate": {
        "smiles": "OCC(O)C(O)C(O)C(O)CO",
        "structure_features": {...}
    },
    "reaction_type": "oxidation",
    "enzyme_family": "EC 1.1.1",
    "cofactor": "NAD+",
    "conditions": {
        "pH": 7.4,
        "temperature": 37,
        "solvent": "water"
    }
}
```

### 출력
```python
{
    "P_feasible": 0.92,  # 가능성 확률
    "predicted_center": [2],  # 반응 중심
    "confidence": 0.85,
    "limiting_factors": ["none"]
}
```

### 평가 기준 (학습 가능)
- ✓ 기질-효소 적합성
- ✓ 반응 중심 예측
- ✓ 작용기 인식
- ✓ 입체화학 호환성
- ✓ 보조인자 요구사항

### 데이터 소스
- Rhea, KEGG (풍부)
- UniProt (효소 정보)
- 문헌 (반응 존재 여부)

---

## Stage 2: Performance (Experimental)

### 목표
**"이 반응이 실험실에서 얼마나 잘 작동하는가?"**

### 입력
```python
{
    "reaction": {...},  # Stage 1 출력
    "assay_conditions": {
        "substrate_conc": 10,  # mM
        "enzyme_conc": 0.1,    # μM
        "cofactor_conc": 1.0,  # mM
        "time": 2,             # hours
        "volume": 1            # mL
    }
}
```

### 출력 (확률 분포)
```python
{
    "yield": {
        "mean": 0.85,
        "std": 0.05,
        "confidence_interval": [0.75, 0.92]
    },
    "selectivity": {
        "mean": 0.92,
        "std": 0.03,
        "confidence_interval": [0.86, 0.96]
    },
    "activity": {
        "kcat": {"mean": 120, "std": 15},
        "Km": {"mean": 0.6, "std": 0.1},
        "kcat_Km": {"mean": 200, "std": 30}
    },
    "time_to_completion": {
        "mean": 1.5,  # hours
        "std": 0.2
    }
}
```

### 평가 기준 (부분적으로 학습 가능)

#### 측정 가능 (학습 목표로 적합)
- ✓ kcat, Km (BRENDA에 일부 존재)
- ✓ 수율 (문헌에 일부)
- ✓ 선택성 (문헌에 일부)

#### 측정 어려움 (추정 필요)
- △ 시간-농도 프로파일
- △ 부산물 분포
- △ 재현성 (실험 반복 필요)

### 데이터 소스
- BRENDA (kcat, Km 일부)
- 문헌 (수율, 선택성 일부)
- **문제**: 결측치 70-90%

### 결측 데이터 전략
```python
# 1. 유사 반응에서 전이 학습
similar_reactions = find_similar(substrate, enzyme)
estimated_kcat = transfer_learning(similar_reactions)

# 2. 불확실성 명시
output = {
    "kcat": {
        "value": 120,
        "source": "measured",  # or "estimated" or "unknown"
        "confidence": 0.9
    }
}

# 3. 결측치 허용 loss
loss = masked_mse(predicted, actual, mask=is_available)
```

---

## Stage 3: Process & Economics

### 목표
**"이 반응이 산업적으로 성공할 수 있는가?"**

### 입력
```python
{
    "reaction": {...},
    "performance": {...},  # Stage 2 출력
    "process_spec": {
        "scale": 1000,  # L
        "target_productivity": 10,  # g/L/h
        "max_cost_per_kg": 200,
        "required_purity": 0.98
    }
}
```

### 출력
```python
{
    "P_scalable": 0.75,
    "estimated_COGS": {
        "mean": 145,
        "breakdown": {
            "enzyme": 80,
            "cofactor": 40,
            "purification": 25
        }
    },
    "P_profitable": 0.82,
    "bottlenecks": ["oxygen_transfer", "heat_removal"]
}
```

### 평가 기준 (**별도 모델**)

이것들은 **화학 모델이 아니라 공정 모델**:
- 스케일업 수율 변화
- 혼합/전열/물질전달
- 원가 구성 (원료/장치/downstream/폐수)
- ROI, NPV
- 재현성 (배치간 변동)

### 데이터 소스
- 공정 데이터베이스 (매우 희소)
- 파일럿/산업 데이터 (기밀)
- **현실**: 대부분 추정/시뮬레이션

### 접근 방법
```python
# 화학 모델과 분리
class ProcessModel:
    def predict_scalability(self, lab_performance, scale_params):
        # 공정공학 모델
        # - 혼합 시간
        # - 산소 전달 계수
        # - 열 제거 용량
        pass
    
    def estimate_economics(self, process_params, market_data):
        # 경제성 모델
        # - 원가 계산
        # - 시장 가격
        # - ROI
        pass
```

---

## 통합 평가: 스펙 기반 성공 확률

### 스펙 정의
```python
spec = {
    # Stage 1 게이트 (필수)
    "min_feasibility": 0.7,
    
    # Stage 2 목표 (가중합)
    "target_yield": 0.7,
    "target_selectivity": 0.85,
    "max_time_hours": 6,
    "min_kcat": 30,
    
    # Stage 3 목표
    "max_COGS": 200,
    "min_margin": 100,
    
    # 가중치
    "weights": {
        "yield": 0.3,
        "selectivity": 0.25,
        "speed": 0.2,
        "cost": 0.15,
        "scalability": 0.1
    }
}
```

### 점수 계산
```python
def calculate_success_score(prediction, spec):
    """스펙 기반 성공 점수"""
    
    # 1. 필수 게이트 (하나라도 실패하면 0점)
    if prediction["P_feasible"] < spec["min_feasibility"]:
        return 0.0, "failed_feasibility_gate"
    
    if prediction["selectivity"]["mean"] < 0.7:  # 치명적 부산물 방지
        return 0.0, "failed_selectivity_gate"
    
    # 2. 성능 점수 (가중합, trade-off 허용)
    scores = {}
    
    # 수율 점수
    yield_score = sigmoid_score(
        prediction["yield"]["mean"],
        target=spec["target_yield"],
        steepness=10
    )
    scores["yield"] = yield_score
    
    # 선택성 점수
    selectivity_score = sigmoid_score(
        prediction["selectivity"]["mean"],
        target=spec["target_selectivity"],
        steepness=10
    )
    scores["selectivity"] = selectivity_score
    
    # 속도 점수
    time_score = 1.0 - sigmoid_score(
        prediction["time_to_completion"]["mean"],
        target=spec["max_time_hours"],
        steepness=0.5
    )
    scores["speed"] = time_score
    
    # 비용 점수 (Stage 3)
    if "estimated_COGS" in prediction:
        cost_score = 1.0 - sigmoid_score(
            prediction["estimated_COGS"]["mean"],
            target=spec["max_COGS"],
            steepness=0.01
        )
        scores["cost"] = cost_score
    
    # 3. 가중 합산
    total_score = sum(
        scores[key] * spec["weights"][key]
        for key in scores
    )
    
    # 4. 불확실성 페널티
    uncertainty_penalty = calculate_uncertainty_penalty(prediction)
    final_score = total_score * (1 - uncertainty_penalty)
    
    return final_score, scores


def sigmoid_score(value, target, steepness):
    """부드러운 점수 함수 (경직된 threshold 대신)"""
    import numpy as np
    return 1 / (1 + np.exp(-steepness * (value - target)))
```

### 출력 예시
```python
{
    "overall_score": 0.87,
    "grade": "A",
    "P_success": 0.85,  # 스펙 충족 확률
    
    "breakdown": {
        "feasibility": 0.92,
        "yield": 0.90,
        "selectivity": 0.95,
        "speed": 0.85,
        "cost": 0.80
    },
    
    "gates_passed": ["feasibility", "selectivity"],
    "recommendation": "proceed_to_pilot",
    
    "uncertainty": {
        "total": 0.15,
        "sources": {
            "yield_variance": 0.05,
            "kcat_estimated": 0.08,
            "cost_projected": 0.02
        }
    },
    
    "improvement_suggestions": [
        "optimize_pH_to_8.0_for_higher_yield",
        "increase_cofactor_conc_to_reduce_time"
    ]
}
```

---

## 데이터 스키마 (3-Layer)

### Layer 1: Reaction Record (화학 정보)
```python
{
    "reaction_id": "rxn_sorbitol_001",
    "substrate": {
        "smiles": "OCC(O)C(O)C(O)C(O)CO",
        "smiles_stereo": "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
        "name": "D-sorbitol",
        "mw": 182.17
    },
    "product": {
        "smiles": "OCC(=O)C(O)C(O)C(O)CO",
        "name": "D-fructose"
    },
    "enzyme": {
        "ec": "1.1.1.14",
        "name": "L-iditol 2-dehydrogenase",
        "uniprot": "P12345"
    },
    "cofactor": "NAD+",
    "reaction_center": [2],
    "reaction_type": "oxidation",
    
    # Stage 1 데이터 (대부분 존재)
    "feasibility": {
        "source": "rhea",
        "validated": true,
        "confidence": 0.95
    }
}
```

### Layer 2: Assay Record (실험 데이터)
```python
{
    "assay_id": "assay_001",
    "reaction_id": "rxn_sorbitol_001",
    
    "conditions": {
        "pH": 7.4,
        "temperature": 37,
        "substrate_conc": 10,
        "enzyme_conc": 0.1,
        "cofactor_conc": 1.0
    },
    
    # 측정값 (결측 가능)
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
            "confidence": 0.7,
            "reference": "DOI:xxx"
        },
        "selectivity": {
            "value": 0.95,
            "source": "literature",
            "measured": true,
            "confidence": 0.6
        },
        "time_to_90pct": {
            "value": null,  # 결측
            "source": null,
            "measured": false,
            "estimated": 1.5,
            "confidence": 0.3
        }
    },
    
    # 결측 표시
    "data_completeness": 0.6,  # 60% 측정됨
    "missing_fields": ["time_profile", "byproducts"]
}
```

### Layer 3: Process Record (공정 데이터)
```python
{
    "process_id": "proc_001",
    "assay_id": "assay_001",
    
    "scale": {
        "lab": {"volume": 0.001, "yield": 0.92},
        "pilot": {"volume": 10, "yield": 0.88},
        "industrial": {"volume": 1000, "yield": null}  # 예측 필요
    },
    
    "economics": {
        "enzyme_cost": 80,
        "cofactor_cost": 40,
        "purification_cost": 25,
        "total_COGS": 145,
        "source": "estimated",  # not measured
        "confidence": 0.4
    },
    
    "scalability": {
        "mixing_adequate": true,
        "heat_removal_adequate": true,
        "oxygen_transfer_adequate": false,  # 병목
        "estimated_scale_yield": 0.85,
        "confidence": 0.3
    },
    
    # 대부분 결측/추정
    "data_completeness": 0.2
}
```

---

## 학습 전략 (결측 데이터 대응)

### 1. 단계별 학습
```python
# Stage 1: 데이터 풍부 → 직접 학습
model_stage1 = train_feasibility_model(
    data=rhea_reactions,  # 10,000+ 반응
    labels=reaction_centers
)

# Stage 2: 데이터 희소 → 전이학습 + 불확실성
model_stage2 = train_performance_model(
    data=brenda_assays,  # 1,000+ assays (결측 많음)
    pretrained=model_stage1,
    loss=masked_loss,  # 결측치 무시
    uncertainty=True  # 불확실성 출력
)

# Stage 3: 데이터 극히 희소 → 공정 모델 (별도)
model_stage3 = ProcessEngineeringModel(
    physics_based=True,  # 물리 기반
    data_driven=False    # 데이터 부족
)
```

### 2. Masked Loss (결측 대응)
```python
def masked_mse_loss(predicted, actual, mask):
    """결측치가 있는 데이터용 loss"""
    # mask: 1 = 측정됨, 0 = 결측
    error = (predicted - actual) ** 2
    masked_error = error * mask
    return masked_error.sum() / mask.sum()

# 사용
loss = masked_mse_loss(
    predicted_yield,
    actual_yield,
    mask=is_measured  # [1, 1, 0, 1, 0, ...]
)
```

### 3. 불확실성 전파
```python
# Stage 1 불확실성
P_feasible = 0.92 ± 0.05

# Stage 2에서 전파
if P_feasible < 0.7:
    # 가능성 낮으면 성능 예측 불확실성 증가
    yield_uncertainty *= 2.0

# 최종 점수에 반영
final_score = base_score * (1 - total_uncertainty)
```

---

## 실용 예시

### 입력: 새로운 반응 평가
```python
spec = {
    "min_feasibility": 0.7,
    "target_yield": 0.75,
    "target_selectivity": 0.85,
    "max_time_hours": 4,
    "max_COGS": 180,
    "weights": {
        "yield": 0.3,
        "selectivity": 0.25,
        "speed": 0.2,
        "cost": 0.15,
        "scalability": 0.1
    }
}

result = evaluate_reaction(
    substrate="OCC(O)C(O)C(O)CO",  # xylitol
    reaction_type="oxidation",
    enzyme="EC 1.1.1.9",
    conditions={"pH": 7.4, "temp": 37},
    spec=spec
)
```

### 출력
```python
{
    "overall_score": 0.82,
    "P_success": 0.78,
    "grade": "B+",
    
    "stage1": {
        "P_feasible": 0.88,
        "predicted_center": [2],
        "confidence": 0.85
    },
    
    "stage2": {
        "yield": {"mean": 0.80, "std": 0.06, "source": "estimated"},
        "selectivity": {"mean": 0.90, "std": 0.04, "source": "estimated"},
        "kcat": {"mean": 95, "std": 20, "source": "transferred"},
        "time": {"mean": 2.5, "std": 0.5}
    },
    
    "stage3": {
        "COGS": {"mean": 160, "std": 25, "source": "estimated"},
        "P_scalable": 0.70
    },
    
    "gates_passed": true,
    "recommendation": "proceed_with_caution",
    "uncertainty_level": "medium",
    
    "suggestions": [
        "validate_yield_experimentally",
        "optimize_pH_to_8.0",
        "consider_cofactor_regeneration"
    ]
}
```

---

## 요약

### 핵심 변경점

1. **3-Stage 분리**
   - Stage 1: 화학 (학습 가능, 데이터 풍부)
   - Stage 2: 성능 (부분 학습, 데이터 희소)
   - Stage 3: 공정 (별도 모델, 데이터 극히 희소)

2. **스펙 기반**
   - 임계값 하드코딩 ❌
   - 외부 스펙 입력 ✓
   - 도메인별 유연성 ✓

3. **확률적 점수**
   - 경직된 AND ❌
   - 가중합 + 최소 게이트 ✓
   - Trade-off 허용 ✓

4. **데이터 현실성**
   - 결측치 무시 ❌
   - Masked loss ✓
   - 불확실성 명시 ✓

### 다음 단계

1. Stage 1 모델 구현 (가능성 예측)
2. Stage 2 모델 구현 (성능 예측, 결측 대응)
3. 스펙 기반 평가 시스템
4. 불확실성 정량화
