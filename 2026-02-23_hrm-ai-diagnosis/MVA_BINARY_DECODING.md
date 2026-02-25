# MVA 바이너리 디코딩 방법론

## 배경

ManoView Analysis 3.0 (Sierra Scientific Instruments)의 `.mva` 파일은 고해상도 식도 내압검사(HRM) 데이터를 저장하는 독점 바이너리 포맷입니다. 공식 문서나 SDK가 없어 역공학(reverse engineering)으로 파싱 방법을 도출했습니다.

---

## 파일 구조

`.mva` 파일은 여러 개의 **zlib 압축 블록**으로 구성됩니다.

```
[MVA 파일]
  ├── zlib block 0 → XML: <info>      환자 정보
  ├── zlib block 1 → XML: <probe>     센서 캘리브레이션 커브
  ├── zlib block 2 → XML: <examination> 녹화 메타데이터
  ├── zlib block 3 → XML: <analysis>  swallow 임상지표
  ├── zlib block 4 → Binary: 압력 raw data (36채널)
  └── zlib block 5 → Binary: 임피던스 raw data (18채널)
```

XML 블록은 첫 바이트가 `<`로 시작하고, 바이너리 블록은 그 외입니다.

---

## 바이너리 인코딩 방식: Delta Encoding

### 핵심 규칙

바이너리 블록은 **delta encoding**으로 압축된 시계열 데이터입니다.

| 바이트 값 | 의미 |
|-----------|------|
| `0x80` | Escape 마커 → 다음 2바이트가 **uint16 절대값** (little-endian) |
| 기타 | **int8 delta** (부호 있는 차분값, -128~+127) |

```python
def delta_decode(data: bytes) -> list:
    values = []
    current = 0
    i = 0
    while i < len(data):
        b = data[i]
        if b == 0x80:                              # escape
            current = struct.unpack('<H', data[i+1:i+3])[0]  # uint16 절대값
            i += 3
        else:
            delta = b if b < 128 else b - 256      # signed int8
            current = max(0, min(65535, current + delta))
            i += 1
        values.append(current)
    return values
```

### 발견 과정

1. `mvsap.exe` 실행 파일에서 `csFramed`, `ConvertReadStream` 문자열 발견 → Delphi TStream 기반 framed 포맷 추정
2. 블록 내부에서 `0x80` 마커가 규칙적으로 등장하는 패턴 확인
3. `0x80` 다음 2바이트를 uint16으로 읽으면 캘리브레이션 signal 범위(0~4096)에 해당하는 값이 나옴
4. 나머지 바이트를 int8 delta로 해석하면 row-to-row 차분이 txt ground truth와 일치

---

## 채널 레이아웃: Channel-First

바이너리 블록은 **채널별 연속 저장(channel-first)** 방식입니다.

```
[블록 구조]
  ch0의 n_t samples → ch1의 n_t samples → ... → ch35의 n_t samples
```

- 압력: 36채널 × 135,248 samples = 4,868,928 values
- 샘플링 레이트: **100 Hz** (0.01초 간격)
- 총 녹화 시간: ~1,352초

인터리브(ch0, ch1, ..., ch35, ch0, ch1, ...)가 아님에 주의.

---

## 채널 순서: 역순 매핑

decoded 채널과 txt export 채널의 순서가 **역순**입니다.

```
decoded ch35 → 압력 ch1  (인두/UES 상부)
decoded ch34 → 압력 ch2
...
decoded ch31 → 압력 ch5  (LES 고압대)
decoded ch30 → 압력 ch6
...
decoded ch0  → 압력 ch36 (위)
```

즉 `pressure[:, tc] = calibrate(decoded[35 - tc])`.

### 발견 과정

- ch0 기준 슬라이딩 상관계수: corr=0.52 (낮음)
- **ch5 (LES, 85 mmHg baseline) 기준**으로 슬라이딩 매칭 시도
- decoded ch30이 txt ch5와 corr=**0.895** 로 매칭
- decoded ch31이 txt ch4 (115 mmHg)와 corr=0.828 로 매칭
- 패턴 확인: `decoded ch(35-tc) = txt ch(tc)` → 역순 관계 확정

---

## 캘리브레이션

probe XML의 `<pressure><sensor>` 블록에 채널별 룩업 테이블이 있습니다.

```xml
<sensor>
  <index>0</index>
  <scale>
    <point><signal>0.0</signal><pressure>-52.0</pressure></point>
    <point><signal>500.0</signal><pressure>17.9</pressure></point>
    ...
    <point><signal>4096.0</signal><pressure>554.7</pressure></point>
  </scale>
</sensor>
```

변환: `pressure_mmHg = interp(raw_signal, signal_array, pressure_array)`

---

## 시간 오프셋

블록 0의 시작 시간은 전체 녹화의 중간 시점입니다.

```
block0_start_s = (마지막 swallow endTime + 여유) - block0_duration
```

analysis XML의 `<swallow><beginTime>` (ms 단위)을 블록 0 sample index로 변환:

```python
sample_idx = int((begin_ms/1000 - block0_start_s) * 100)
```

---

## 검증 결과

Ground truth(ManoView txt export)와 비교:

| 채널 | 상관계수 |
|------|---------|
| ch0 (식도 체부) | **0.972** |
| ch4 (LES 고압) | **0.829** |
| ch5 (LES)      | **0.895** |
| 전체 36채널 평균 | 0.656 |

압력 지형도(pressure topography)의 시각적 패턴이 ground truth와 일치:
- LES 고압대 위치 및 압력값
- Swallow 시 식도 수축파 방향 및 타이밍

---

## 최종 파이프라인

```
.mva 파일
  ↓ extract_zlib_blocks()
XML 블록 (probe, analysis, examination)
  ↓ _load_calibration()
캘리브레이션 커브 (36채널)

바이너리 블록 0
  ↓ delta_decode_1ch()
4,868,928 raw values
  ↓ reshape(36, n_t)[::-1]  # 역순
(36, n_t) channel-first matrix
  ↓ interp(cal_curve)
(n_t, 36) pressure [mmHg]
  ↓ crop_swallows(beginTime, endTime)
(4000, 36) per swallow  # pre=15s, post=25s @ 100Hz
```

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `preprocessing/parse_mva.py` | zlib 블록 추출, XML 파싱 |
| `preprocessing/parse_mva_binary.py` | delta decode, 캘리브레이션, 압력/임피던스 추출 |
| `preprocessing/crop_swallow.py` | swallow 구간 자동 크롭 |
| `preprocessing/batch_pipeline.py` | 30명 전체 일괄 처리 |
| `data/processed/{id}/pressure.npy` | 전체 압력 시계열 (n_t, 36) |
| `data/processed/{id}/swallows/sw{i}.npy` | swallow별 크롭 (4000, 36) |
| `data/processed/{id}/meta.json` | 환자 정보 + swallow 시간 목록 |
