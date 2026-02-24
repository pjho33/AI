# 3️⃣ MD에서 추출해야 할 지표 → ML feature 정리표

핵심 원칙부터 짚고 갈게요.

> **MD는 '반응을 예측'하는 게 아니라
> '반응 가능성이 높은 물리적 상태를 얼마나 자주 만드는지'를 정량화한다.**

그래서 feature는 **절대적인 에너지**가 아니라
👉 **빈도, 분포, 안정성, 변동성** 중심이 되어야 합니다.

---

## 📊 MD-derived Feature Table (권장)

### A. 결합 안정성 (Binding stability)

| Feature 이름              | 정의                  | ML에서 의미           |
| ----------------------- | ------------------- | ----------------- |
| Ligand RMSD (mean, std) | 리간드 RMSD 평균/표준편차    | 포켓 내 안정성          |
| Active-site RMSD        | 활성부위 잔기 RMSD        | induced fit 정도    |
| Ligand residence time   | 특정 RMSD 이하 체류 시간 비율 | 결합 지속성            |
| Contact persistence     | 핵심 잔기와 접촉 유지 비율     | 비특이적 vs 특이적 결합 구분 |

👉 **온도 올라갈수록 RMSD가 급증하면** → 불안정한 결합

---

### B. Reactive geometry (가장 중요)

| Feature 이름                     | 정의                     | ML에서 의미       |
| ------------------------------ | ---------------------- | ------------- |
| Near-attack distance frequency | 핵심 원자쌍 거리 < X Å 프레임 비율 | 반응 가능성 지표     |
| Near-attack angle frequency    | 반응 각도 범위 만족 비율         | 방향성 정렬        |
| Simultaneous NAC frequency     | 거리+각도 동시 만족 비율         | "진짜 반응 준비 상태" |
| NAC lifetime                   | NAC 상태 평균 지속 시간        | 반응 확률 상승 신호   |

> 이게 **MD를 돌리는 가장 큰 이유**임
> (단순 binding energy보다 훨씬 설득력 있음)

---

### C. 수소결합/전하 상호작용

| Feature                 | 정의                    | 의미           |
| ----------------------- | --------------------- | ------------ |
| H-bond occupancy        | 특정 donor–acceptor 점유율 | 반응 전 정렬      |
| Salt bridge persistence | 전하 상호작용 유지 시간         | pH 민감도       |
| H-bond network size     | 연결된 H-bond 개수         | 전이상태 안정화 가능성 |

👉 pH 효과를 **간접적으로 반영**할 수 있는 핵심 feature

---

### D. 활성부위 접근성 / 차폐

| Feature                            | 정의            | 의미       |
| ---------------------------------- | ------------- | -------- |
| Pocket volume fluctuation          | 포켓 부피 분산      | 유연성      |
| Solvent accessibility (SASA)       | 리간드 SASA 변화   | 물 경쟁     |
| Water occupancy near reactive site | 반응 부위 주변 물 개수 | 반응 억제 신호 |

---

### E. 온도 민감도 (Temperature sensitivity)

| Feature               | 정의             | 의미     |
| --------------------- | -------------- | ------ |
| ΔRMSD/ΔT              | RMSD의 온도 기울기   | 열 안정성  |
| ΔNAC/ΔT               | NAC 빈도의 온도 의존성 | 최적 온도  |
| Stability crossover T | 급격한 변화가 생기는 온도 | 조건 최적화 |

👉 이게 **USPTO에 없는 정보**임
👉 네 접근의 가장 큰 차별점

---

## 🧠 ML 입력 시 권장 형태

* **단일 값 X**
* **온도별 벡터 [X(T1), X(T2), X(T3)]**
* **기울기/분산 (ΔX/ΔT, var(X))**

이렇게 넣으면 모델이:

> "이 효소는 고온에서만 반응성이 열린다"
> 같은 패턴을 학습할 수 있음.

---

# 4️⃣ "10 ns MD의 한계와 정당화 문장" (심사자 대응용)

이건 **아주 중요**하고, 말 한 줄 잘못 쓰면 바로 공격당합니다.
아래 문장은 **방어 가능한 표현**만 썼어요.

---

## ❌ 쓰면 안 되는 주장

* "10 ns MD로 반응을 예측했다"
* "10 ns로 충분한 샘플링을 했다"
* "이 결과로 반응 수율을 직접 예측할 수 있다"

---

## ✅ 권장 공식 문장 (그대로 사용 가능)

### (1) 한계 명시

> "We acknowledge that 10 ns molecular dynamics simulations are insufficient to directly observe chemical bond formation or to fully sample rare catalytic events."

(10 ns로 반응을 본다고 주장하지 않음)

---

### (2) 목적 재정의

> "Instead, our simulations are designed to probe the stability, conformational flexibility, and the frequency of pre-reactive geometries under different temperature conditions."

👉 **pre-reactive geometry**라는 단어가 핵심

---

### (3) 정당화 논리

> "These pre-reactive structural features are known to strongly correlate with catalytic efficiency and temperature dependence, even when direct reaction events are not observed."

(문헌 기반으로 충분히 방어 가능)

---

### (4) 짧은 시간의 합리성

> "Short-timescale simulations were intentionally chosen to enable rapid and consistent comparison across a large number of enzyme–substrate systems."

👉 **비교 가능성**을 이유로 듦 (아주 설득력 있음)

---

### (5) AI와의 연결 (중요)

> "The resulting MD-derived descriptors serve as physically interpretable features for machine learning models, rather than as standalone predictors of chemical reactivity."

이 문장 하나로 **과학적 겸손 + 전략적 명확성** 둘 다 확보됨.

---

## 🎯 한 문단 요약 (심사자용, 강력 추천)

> "While short (10 ns) molecular dynamics simulations cannot capture rare chemical reaction events, they are well suited to quantify the stability and frequency of pre-reactive conformations. By extracting physically meaningful descriptors such as near-attack geometries, hydrogen bond persistence, and temperature-dependent stability metrics, these simulations provide complementary information to large-scale reaction datasets. In this study, MD is therefore employed as a rapid, physics-informed feature generator for machine learning, rather than as a direct predictor of reaction outcomes."

이 문단은 **심사자 공격을 거의 다 차단**합니다.

---

## 마지막 정리 (네 전략의 위치)

* USPTO → **통계적 가능성**
* 실험 데이터 → **ground truth**
* 10 ns MD → **물리적 타당성 필터**
* ML → **통합 판단**

👉 이 조합은 **현재 화학 AI에서 가장 설득력 있는 구조**입니다.
