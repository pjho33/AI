# 반응 성공의 정의

## 현재 시스템의 문제

### 현재 가정
```python
"반응이 일어날 수 있다" = "성공"
```

**하지만 실제로는?**
- 반응이 일어나도 **수율 1%**면 실패
- 반응이 빨라도 **부반응**이 많으면 실패
- 효소가 작동해도 **비용**이 너무 높으면 실패

---

## 반응 성공의 다층적 정의

### Level 1: 열역학적 가능성
```
ΔG < 0 → 열역학적으로 가능

하지만:
- ΔG = -1 kJ/mol: 가능하지만 평형이 50:50
- ΔG = -20 kJ/mol: 거의 완전히 생성물로
```

**판단 기준**:
```python
if ΔG < -10 kJ/mol:
    thermodynamically_favorable = True
else:
    thermodynamically_favorable = False
```

### Level 2: 동역학적 실현 가능성
```
활성화 에너지 (Ea)가 낮아야 함

Ea < 50 kJ/mol: 상온에서 빠름
Ea = 50-100 kJ/mol: 효소 필요
Ea > 100 kJ/mol: 효소 있어도 느림
```

**판단 기준**:
```python
if Ea < 80 kJ/mol:  # 효소 촉매
    kinetically_feasible = True
else:
    kinetically_feasible = False
```

### Level 3: 수율 (Yield)
```
실제로 얼마나 생성물이 만들어지는가?

수율 > 90%: 우수
수율 70-90%: 양호
수율 50-70%: 보통
수율 < 50%: 불량
```

**영향 요인**:
- 평형 상수 (Keq)
- 부반응
- 생성물 분해
- 역반응

**판단 기준**:
```python
if yield > 70%:
    acceptable_yield = True
```

### Level 4: 선택성 (Selectivity)
```
원하는 생성물만 만들어지는가?

선택성 > 95%: 우수
선택성 80-95%: 양호
선택성 < 80%: 정제 필요
```

**예시**:
```
D-sorbitol 산화:
- C2 산화 (원하는 것): 90%
- C3 산화 (부반응): 8%
- C4 산화 (부반응): 2%

→ 선택성 90%
```

**판단 기준**:
```python
if selectivity > 80%:
    acceptable_selectivity = True
```

### Level 5: 반응 속도
```
얼마나 빨리 완료되는가?

t½ < 1시간: 매우 빠름
t½ = 1-6시간: 빠름
t½ = 6-24시간: 보통
t½ > 24시간: 느림
```

**효소 반응**:
```
kcat > 100 s⁻¹: 빠른 효소
kcat = 10-100 s⁻¹: 보통
kcat < 10 s⁻¹: 느린 효소
```

**판단 기준**:
```python
if kcat > 50 s⁻¹:
    fast_enough = True
```

### Level 6: 경제성
```
비용 대비 효과

효소 비용 + 보조인자 비용 + 정제 비용 < 생성물 가치
```

**예시**:
```
L-sorbitol → L-sorbose:
- 효소 비용: $100/kg
- NAD+ 재생: $50/kg
- 정제: $30/kg
- 총 비용: $180/kg

L-sorbose 가격: $500/kg
→ 이익: $320/kg
→ 경제적으로 성공
```

### Level 7: 재현성
```
같은 조건에서 항상 같은 결과?

표준편차 < 5%: 우수
표준편차 5-10%: 양호
표준편차 > 10%: 불안정
```

### Level 8: 확장성 (Scale-up)
```
실험실 규모 → 산업 규모

실험실 (1 mL): 수율 95%
파일럿 (1 L): 수율 90%
산업 (1000 L): 수율 ???

→ 확장 가능성 검증 필요
```

---

## 통합 성공 기준

### 최소 성공 기준 (Minimum Viable)
```python
reaction_success = (
    thermodynamically_favorable and  # ΔG < -10
    kinetically_feasible and         # Ea < 80
    yield > 50% and                  # 수율 50% 이상
    selectivity > 70%                # 선택성 70% 이상
)
```

### 실용 성공 기준 (Practical)
```python
practical_success = (
    ΔG < -15 kJ/mol and
    Ea < 60 kJ/mol and
    yield > 70% and
    selectivity > 85% and
    kcat > 50 s⁻¹ and
    economically_viable
)
```

### 우수 성공 기준 (Excellent)
```python
excellent_success = (
    ΔG < -20 kJ/mol and
    Ea < 50 kJ/mol and
    yield > 90% and
    selectivity > 95% and
    kcat > 100 s⁻¹ and
    cost_per_kg < market_price * 0.3 and
    reproducible and
    scalable
)
```

---

## 실제 데이터 예시

### 성공 사례: D-sorbitol → D-fructose

```python
{
    "thermodynamics": {
        "ΔG": -18.5,  # kJ/mol
        "Keq": 2500,
        "status": "favorable"
    },
    "kinetics": {
        "Ea": 45,  # kJ/mol (효소 촉매)
        "kcat": 150,  # s⁻¹
        "Km": 0.5,  # mM
        "status": "fast"
    },
    "yield": {
        "theoretical": 100,
        "actual": 92,
        "status": "excellent"
    },
    "selectivity": {
        "C2_oxidation": 95,  # 원하는 반응
        "C3_oxidation": 3,
        "C4_oxidation": 2,
        "total_selectivity": 95,
        "status": "excellent"
    },
    "economics": {
        "enzyme_cost": 80,  # $/kg
        "cofactor_cost": 40,
        "purification": 25,
        "total_cost": 145,
        "product_value": 450,
        "profit_margin": 305,
        "status": "profitable"
    },
    "overall_success": "EXCELLENT"
}
```

### 실패 사례: 저온에서 에탄올 산화

```python
{
    "thermodynamics": {
        "ΔG": -22,  # 여전히 favorable
        "status": "favorable"
    },
    "kinetics": {
        "Ea": 65,  # 높아짐 (저온)
        "kcat": 13,  # 매우 느림
        "status": "too_slow"  # ← 실패 원인
    },
    "yield": {
        "theoretical": 100,
        "actual": 15,  # 24시간 후
        "status": "poor"
    },
    "selectivity": {
        "desired": 90,
        "status": "good"  # 선택성은 괜찮음
    },
    "economics": {
        "time_cost": "너무 오래 걸림",
        "status": "not_viable"
    },
    "overall_success": "FAIL",
    "reason": "kinetically_too_slow"
}
```

### 부분 성공: 고염 조건

```python
{
    "thermodynamics": {
        "ΔG": -18,
        "status": "favorable"
    },
    "kinetics": {
        "kcat": 90,  # 감소했지만 괜찮음
        "Km": 1.2,  # 증가 (결합 약화)
        "status": "acceptable"
    },
    "yield": {
        "actual": 75,
        "status": "acceptable"
    },
    "selectivity": {
        "desired": 88,
        "status": "good"
    },
    "overall_success": "PARTIAL",
    "recommendation": "이온강도 낮추면 개선 가능"
}
```

---

## AI가 배워야 할 것

### 현재 (Phase 1-2)
```
"이 반응이 일어날 수 있나?" → Yes/No
```

### 필요 (Phase 3+)
```
"이 반응이 성공할까?" → 
  - 열역학: Yes (ΔG = -18)
  - 동역학: Yes (kcat = 150)
  - 수율: 92% (우수)
  - 선택성: 95% (우수)
  - 경제성: 이익 $305/kg
  → 종합: EXCELLENT SUCCESS
```

---

## 데이터 스키마 확장

### 완전한 반응 성공 평가

```python
{
    "reaction_id": "sorbitol_to_fructose",
    
    # 기본 정보
    "substrate": "D-sorbitol",
    "product": "D-fructose",
    "enzyme": "EC 1.1.1.14",
    
    # 성공 평가
    "success_evaluation": {
        
        # 1. 열역학
        "thermodynamics": {
            "delta_G": -18.5,
            "delta_G_unit": "kJ/mol",
            "Keq": 2500,
            "favorable": true,
            "score": 95
        },
        
        # 2. 동역학
        "kinetics": {
            "Ea": 45,
            "Ea_unit": "kJ/mol",
            "kcat": 150,
            "kcat_unit": "s-1",
            "Km": 0.5,
            "Km_unit": "mM",
            "kcat_Km": 300,
            "half_life": 0.5,
            "half_life_unit": "hours",
            "fast_enough": true,
            "score": 90
        },
        
        # 3. 수율
        "yield": {
            "theoretical": 100,
            "actual": 92,
            "unit": "percent",
            "acceptable": true,
            "score": 92
        },
        
        # 4. 선택성
        "selectivity": {
            "desired_product": 95,
            "byproduct_1": 3,
            "byproduct_2": 2,
            "regioselectivity": 95,
            "stereoselectivity": 99,
            "acceptable": true,
            "score": 95
        },
        
        # 5. 경제성
        "economics": {
            "enzyme_cost": 80,
            "cofactor_cost": 40,
            "purification_cost": 25,
            "total_cost": 145,
            "product_value": 450,
            "profit": 305,
            "roi": 210,
            "roi_unit": "percent",
            "viable": true,
            "score": 85
        },
        
        # 6. 재현성
        "reproducibility": {
            "n_experiments": 10,
            "mean_yield": 92,
            "std_dev": 2.1,
            "cv": 2.3,
            "cv_unit": "percent",
            "reproducible": true,
            "score": 95
        },
        
        # 7. 확장성
        "scalability": {
            "lab_scale": 95,
            "pilot_scale": 90,
            "industrial_scale": 85,
            "scalable": true,
            "score": 90
        },
        
        # 종합 평가
        "overall": {
            "total_score": 91.7,
            "grade": "A",
            "success_level": "EXCELLENT",
            "recommendation": "산업화 추천",
            "limiting_factors": [],
            "improvement_potential": "minimal"
        }
    }
}
```

---

## 실용적 구현

### Phase 3: 성공 예측 시스템

```python
class ReactionSuccessPredictor:
    
    def predict_success(self, reaction, conditions):
        """반응 성공 예측"""
        
        # 1. 열역학 평가
        thermo_score = self.evaluate_thermodynamics(reaction)
        
        # 2. 동역학 평가
        kinetic_score = self.evaluate_kinetics(reaction, conditions)
        
        # 3. 수율 예측
        yield_score = self.predict_yield(reaction, conditions)
        
        # 4. 선택성 예측
        selectivity_score = self.predict_selectivity(reaction)
        
        # 5. 경제성 평가
        economic_score = self.evaluate_economics(reaction)
        
        # 종합 점수
        total_score = (
            thermo_score * 0.15 +
            kinetic_score * 0.25 +
            yield_score * 0.25 +
            selectivity_score * 0.20 +
            economic_score * 0.15
        )
        
        # 성공 등급
        if total_score > 90:
            return "EXCELLENT"
        elif total_score > 75:
            return "GOOD"
        elif total_score > 60:
            return "ACCEPTABLE"
        else:
            return "POOR"
```

---

## 결론

**반응 성공 ≠ 반응 가능**

**진짜 성공**:
1. ✓ 열역학적으로 favorable
2. ✓ 동역학적으로 빠름
3. ✓ 수율 높음 (>70%)
4. ✓ 선택성 좋음 (>85%)
5. ✓ 경제적으로 viable
6. ✓ 재현 가능
7. ✓ 확장 가능

**현재 시스템**: 1번만 봄
**필요한 시스템**: 1-7번 모두 평가

다음 단계에서 이걸 구현할까?
