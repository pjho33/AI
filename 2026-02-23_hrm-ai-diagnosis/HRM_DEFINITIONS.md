# HRM 분석 정의서 (UES / LES / IRP / DCI)

> **최종 확정 기준** — `compute_hrm_metrics.py` 수정 전 반드시 확인
>
> | 버전 | 날짜 | 변경 내용 |
> |------|------|-----------|
> | v1.0 | 2026-02-25 | UES 탐지 초기 확정 (고정 ch2/3/4, 고정 threshold) |
> | v1.1 | 2026-03-02 | searchsorted 벡터화, start=10s, ch0_onset 추가, dt QC |
> | v1.2 | 2026-03-02 | Dynamic UES 중심 채널 자동선택 + 상대 threshold (실험 중) |
>
> **현재 compute_hrm_metrics.py 적용 버전: v1.1**

---

## 1. 채널 방향 (절대 기준)

| 채널 | 위치 |
|------|------|
| ch0 | Pharynx (위쪽, 상부) |
| ch1~4 | UES 고압대 |
| ch5~19 | 근위부 식도 체부 |
| ch20~27 | 원위부 식도 체부 (LES 위쪽) |
| ch28~33 | LES 고압대 후보 범위 |
| ch34~35 | Stomach (위내압 기준) |

- `imshow` 시각화: `origin="upper"` → ch0이 위에, ch35가 아래
- **ch34~35는 절대 LES로 선택하지 않음**

---

## 2. UES 탐지 (actual_onset) ← 수정 시 반드시 이 정의 확인

### 생리학적 배경
- 삼킴파는 **ch2 → ch3 → ch4** 순서로 순차적으로 내려옴
- **UES 수축 시작 = LES 이완 시작** → ch2가 40mmHg를 처음 넘는 시점 = `actual_onset`
- ch4는 UES 고압대의 핵심 채널로 resting pressure가 높아 threshold를 높게 설정

### ✅ 확정 알고리즘 (2026-03-02 업데이트)
```
1. 탐색 구간: sample 1000(10초) 이후부터 탐색
   - pre=15s 구간 앞부분의 이전 삼킴 잔류 신호 오탐 방지
   - XML onset은 지시 시점일 뿐, 실제 삼킴 시점과 다를 수 있음

2. searchsorted 기반 탐색 (완전 벡터화):
   각 채널의 threshold 돌파 샘플 인덱스를 미리 계산 후:
   - ch2 > 40mmHg 첫 샘플(j2) → 같은 0.1s 창 내에
   - ch3 > 60mmHg 가 j2 이후 존재(j3) → 같은 창 내에
   - ch4 > 90mmHg 가 j3 이후 존재
   → 조건 만족: actual_onset(ch2_onset) = j2

3. ch0_onset (시각화 참고용, IRP 계산에는 미사용):
   - j2 기준 3s~1s 이전 구간에서 baseline = median
   - thr = baseline + max(10, 3×MAD)
   - j2 이전 1.5s 내 thr 첫 돌파 + 이후 0.2s 중 30% 이상 유지 → ch0_onset
   - A군 전체 dt(ch0→ch2) 분포: median=0.45s, p5=0.0s, p95=0.92s
   - QC 기준: 0.0 ≤ dt < 1.2s → ch0 신뢰, 범위 밖 → IRP에 ch2_onset 사용

4. 조건 만족 없으면 → 탐지 실패 → IRP 계산 제외

5. actual_onset + 10s > crop 끝 → IRP 계산 불가 → 제외

6. 연달아 삼킴 제외:
   - 첫 이벤트 탐지 후 10s 이내 두 번째 UES 이벤트 → is_double=True → IRP 계산 제외
```

### ⚠️ 주의
- ch2, ch3, ch4 = `crop[:, 2]`, `crop[:, 3]`, `crop[:, 4]` (0-indexed)
- **fallback 또는 연달아 삼킴(is_double) swallow는 IRP 계산 제외**, DCI/CFV/LESP는 포함
- IRP window: `actual_onset ~ actual_onset + 10s` (LES 채널 기준)

---

## 3. LES 채널 선택

### 생리학적 기준
LES = **괄약근** 이므로:
1. 삼킴 전후(resting)에 지속적으로 높은 압력 유지
2. 삼킴 시 실제로 이완됨 (pressure drop 발생)
3. 수축파(peristalsis)가 끝난 이후에 위치
4. 호흡성 oscillation 있을 수 있음 (보조 조건)

### ⚠️ 대동맥 압박 채널 주의
- 식도 체부 중간에 resting 압력이 높은 구간 존재 → 대동맥에 의한 외부 압박
- **특징**: resting은 높지만 삼킴 시 이완이 없음 (relax_drop ≈ 0)
- → `relax_drop > 2 mmHg` 조건으로 자동 제외

### 알고리즘
```
1. 수축파 끝 채널(distal_end_ch) 결정:
   - ch20~35 중 peak > max(30, gastric+15) mmHg인 가장 높은 ch
   - fallback: ch32

2. 탐색 범위:
   les_lo = max(distal_end_ch - 1, 20)
   les_hi = 34  (ch33 이하, stomach 제외)

3. 각 채널별 값 계산 (여러 swallow 평균):
   - ch_resting = (pre_mean + post_mean) / 2
     - pre:  onset-10s ~ onset-3s  (이전 수축파 영향 제거)
     - post: onset+10s ~ onset+15s (수축파 완전 통과 후)
   - ch_relax_mean = onset-1s ~ onset+3s 평균
   - relax_drop = ch_resting - ch_relax_mean

4. LES 후보 조건 (AND):
   (A) ch_resting > gastric + 5 mmHg  → 진짜 괄약근 고압대
   (B) relax_drop > 2 mmHg            → 삼킴 시 실제 이완됨

5. 후보 중 최종 선택:
   les_score = relax_drop + 0.2 * osc_score (호흡 oscillation 보조)
   → les_score 최대 채널 = best_les_ch

6. Fallback (후보 없을 때):
   - fb1: relax_drop > 0 인 채널 중 resting 최대
   - fb2: les_lo ~ les_hi 중 resting 최대
   - les_candidate_weak = True (QC 플래그)
```

---

## 4. IRP 계산

### 정의 (Chicago Classification v3.0/v4.0)
- **IRP (Integrated Relaxation Pressure)**: actual_onset부터 10초 window 내 가장 낮은 4초(400 samples) 평균
- **단위**: mmHg
- **기준**: `median IRP` (mean 아님) — outlier 무시

### 알고리즘
```
irp_window = les_trace[actual_onset : actual_onset + 10s]
irp = mean(sorted(irp_window)[:400])  # 가장 낮은 400 samples
```

### 핵심 규칙
- **pharyngeal contraction이 탐지된 swallow에만 계산**
- fallback swallow(actual_onset = XML onset) → IRP = NaN → 제외
- 환자 IRP = `median(valid_irp_list)`

### Cutoff (CC v3.0 기준)
| 값 | 의미 |
|----|------|
| IRP > 15 mmHg | EGJ relaxation impaired |
| IRP ≤ 15 mmHg | EGJ relaxation normal |

---

## 5. DCI 계산

### 정의
- **DCI (Distal Contractile Integral)**: 원위부 식도 수축 강도
- **단위**: mmHg·s·cm
- 압력 > 20 mmHg 구간만 적분

### 채널 범위
```
dist_chs = [best_les_ch - 8 : best_les_ch - 2]  (LES 위쪽 6채널)
```

### Cutoff
| 값 | 분류 |
|----|------|
| DCI < 450 | Ineffective swallow |
| DCI > 8000 | Hypercontractile (Jackhammer) |

---

## 6. Chicago Classification 진단 기준

```
if median_IRP > 15:
    if 100% failed (DCI < 100):  → Achalasia Type I
    elif ≥20% premature (CFV > 9): → Achalasia Type III (with spasm)
    else:                          → EGJ Outflow Obstruction

else (IRP ≤ 15):
    if 100% failed:                → Absent Contractility
    elif ≥20% premature CFV > 9:   → Distal Esophageal Spasm
    elif ≥20% DCI > 8000:          → Hypercontractile (Jackhammer)
    elif >50% DCI < 450:           → Ineffective Esophageal Motility (IEM)
    else:                          → Normal
```

---

## 7. Swallow QC 기준

| 조건 | 기준 |
|------|------|
| 식도 체부 수축파 | ch8~16에서 onset-5s~+8s 내 peak > 30 mmHg |
| 진짜 삼킴 (UES 탐지) | 0.1s 창 내 ch2, ch3, ch4 각각 40 mmHg 이상 |
| IRP 포함 조건 | 두 조건 모두 만족 (actual_onset ≠ XML onset) |

---

## 8. 데이터 규격

| 항목 | 값 |
|------|----|
| 샘플링 레이트 | 100 Hz |
| Crop 길이 | 4000 samples = 40초 |
| Pre-swallow | 15초 (onset = sample 1500) |
| Post-swallow | 25초 |
| 채널 수 | 36 (ch0~35) |
| 처리 환자 수 | 60명 (A01~A31, B01~B30) |

---

## 9. 분석 제외 환자 목록

| 환자 ID | 제외 사유 | 확인일 |
|---------|-----------|--------|
| A22 | 카테터 꺾임 — UES 고압대(ch3/ch4)가 삼킴 시 오히려 하락하는 역전 패턴. 시각화 및 데이터 모두 확인. 분석 불가. | 2026-03-02 |

> 코드 적용: `EXCLUDE = {"A22"}` — `_plot_ues.py`, `_stats.py`, `batch_pipeline.py` 동일 적용

---

*최종 업데이트: 2026-02-25 (UES 탐지 기준 확정)*

---

## [v1.2] UES 탐지 업그레이드 — Dynamic 중심 채널 + 상대 threshold (실험 중)

> ⚠️ **현재 _plot_ues.py에만 적용 중 (검증 단계). compute_hrm_metrics.py는 v1.1 유지.**
> v1.2 검증 완료 후 compute_hrm_metrics.py에 반영 예정.

### 배경 및 동기

A군 분석 결과, UES 고압대의 중심 채널이 환자/검사마다 다름이 확인됨:
- ch4 중심: 422/629 swallows (67%) — 대부분
- ch2 중심: 178/629 (28%) — A08, A20, A26, A30 등
- ch1 중심:  21/629 (3%)  — A04 전체
- A15: ch2(9개)/ch4(12개) swallow별로 혼재

고정 채널(ch2/3/4) + 고정 threshold(40/60/90)는 카테터가 1채널만 이동해도 탐지 실패.

### v1.2 알고리즘

```
Step 0) pre 구간 정의
  - crop[:500, :] (처음 5초) → 각 채널 resting baseline & MAD 계산

Step 1) UES 중심 채널(c0) 자동 선택
  - ch1~ch4 중 pre resting(median) 가장 높은 채널 = c0
  - 생리적 근거: UES 고압대는 resting pressure가 높음

Step 2) QC — UES 신호 존재 확인
  - crop 전체에서 c0 채널 peak < resting[c0] + 15mmHg → 탐지 실패
  - (A17처럼 카테터가 UES 밖에 있는 경우 자동 제외)

Step 3) 탐색 채널 3개 설정
  - ca = max(1, c0-1), cb = c0, cc = min(4, c0+1)
  - 위→아래 순서 (생리적 하강 방향)

Step 4) 채널별 adaptive threshold
  - thr(ch) = resting[ch] + max(15, 2 × MAD[ch])

Step 5) 순차 상승 탐지 (searchsorted 벡터화)
  - start = sample 1000 (10초) 이후부터 탐색
  - 0.1s(10샘플) 창 내에서 ca → cb → cc 순서로 thr 돌파 확인
  - 조건 만족 시 actual_onset = ca 첫 돌파 샘플

Step 6) double swallow 처리 (v1.1과 동일)
  - 첫 이벤트 후 10s 이내 두 번째 이벤트 → is_double=True → IRP 제외
```

### v1.1 → v1.2 변경 요약

| 항목 | v1.1 | v1.2 |
|------|------|------|
| 탐색 채널 | 고정 ch2, ch3, ch4 | 자동 선택 [c0-1, c0, c0+1] |
| threshold | 고정 40/60/90 mmHg | adaptive: resting + max(15, 2×MAD) |
| UES QC | 없음 | peak-baseline < 15mmHg 시 실패 |
| ch0_onset | 시각화 참고용 유지 | 동일 |

### 롤백 방법 (v1.1 복원)
`_plot_ues.py`의 `find_ao_fast()` 함수를 아래로 교체:
```python
T2, T3, T4 = 40, 60, 90; UW = 10; DW = int(10*SR)
i2 = np.flatnonzero(crop[:,2]>T2)
i3 = np.flatnonzero(crop[:,3]>T3)
i4 = np.flatnonzero(crop[:,4]>T4)
# ... searchsorted 탐색 (start=int(10*SR))
```
