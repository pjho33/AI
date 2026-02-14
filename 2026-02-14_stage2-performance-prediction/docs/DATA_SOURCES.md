# 화학 반응 데이터 소스

## GNN 학습을 위한 대규모 데이터베이스

---

## 🌟 추천 데이터 소스 (우선순위 순)

### 1. **USPTO (미국 특허청 반응 데이터)** ⭐⭐⭐
**가장 추천!**

**규모**:
- **1,000,000+** 반응
- 1976-2016년 특허 데이터
- SMILES 형식

**내용**:
- 기질 → 생성물
- 반응 조건 (일부)
- 수율 (일부)

**다운로드**:
```bash
# USPTO-50K (50,000개 반응)
wget https://github.com/rxn4chemistry/OpenNMT-py/raw/master/data/USPTO_50K.zip

# USPTO-MIT (1M+ 반응)
wget https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
```

**형식**:
```
reactant1.reactant2>reagent>product
CCO.CC(=O)O>>CCOC(=O)C  # 에스터화
```

**장점**:
- ✅ 대규모 (1M+)
- ✅ 실제 반응
- ✅ 무료
- ✅ 전처리 도구 많음

**단점**:
- ⚠️ 수율 데이터 부족
- ⚠️ 조건 정보 불완전

---

### 2. **Rhea (생화학 반응)** ⭐⭐⭐
**우리가 이미 사용 중!**

**규모**:
- **12,000+** 생화학 반응
- EC 번호 매핑
- ChEBI 구조

**내용**:
- 효소 반응
- 기질 → 생성물
- EC 번호
- 조건 (일부)

**다운로드**:
```bash
# TSV 형식
curl "https://www.rhea-db.org/rhea/?query=&columns=rhea-id,equation,ec&format=tsv&limit=10000" > rhea_all.tsv

# RDF 형식 (전체)
wget ftp://ftp.expasy.org/databases/rhea/rdf/rhea.rdf.gz
```

**장점**:
- ✅ 고품질
- ✅ EC 번호 매핑
- ✅ 생화학 특화
- ✅ 무료

**단점**:
- ⚠️ 규모 작음 (12K)
- ⚠️ 동역학 데이터 별도 (BRENDA)

---

### 3. **BRENDA (효소 동역학)** ⭐⭐⭐
**우리가 이미 사용 중!**

**규모**:
- **83,000+** 효소
- **7,000+** EC 번호
- kcat, Km 데이터

**내용**:
- kcat, Km, Ki
- pH, 온도 의존성
- 기질 특이성

**다운로드**:
```python
# SOAP API (복잡)
from suds.client import Client
client = Client('https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl')

# 또는 웹 스크래핑
# 또는 라이센스 구매 (학술용 무료)
```

**장점**:
- ✅ 동역학 데이터
- ✅ 조건 정보 풍부
- ✅ 고품질

**단점**:
- ⚠️ API 복잡
- ⚠️ 결측 많음 (70%)
- ⚠️ 반응식 없음 (효소만)

---

### 4. **Reaxys (상업용)** ⭐⭐
**유료지만 최고 품질**

**규모**:
- **50,000,000+** 반응
- 1771년부터 현재까지
- 문헌 기반

**내용**:
- 반응 조건 상세
- 수율
- 반응 시간
- 문헌 출처

**접근**:
- 대학 라이센스 (보통 있음)
- API 제공
- 웹 인터페이스

**장점**:
- ✅ 최대 규모
- ✅ 최고 품질
- ✅ 조건 상세
- ✅ 수율 데이터

**단점**:
- ❌ 유료
- ❌ API 제한적

---

### 5. **ORD (Open Reaction Database)** ⭐⭐⭐
**새로운 오픈 소스!**

**규모**:
- **1,000,000+** 반응 (목표)
- 현재 **100,000+**
- 계속 증가 중

**내용**:
- 반응 조건 상세
- 수율
- 실험 절차
- 구조화된 데이터

**다운로드**:
```bash
# GitHub
git clone https://github.com/open-reaction-database/ord-data.git

# Python API
pip install ord-schema
from ord_schema import message_helpers
```

**형식**:
```python
# Protocol Buffers
reaction = Reaction()
reaction.inputs['substrate'].components.add(
    smiles='CCO',
    amount=Amount(mass=Mass(value=10, units='GRAM'))
)
```

**장점**:
- ✅ 오픈 소스
- ✅ 구조화
- ✅ 조건 상세
- ✅ 계속 성장

**단점**:
- ⚠️ 아직 규모 작음
- ⚠️ 형식 복잡

---

### 6. **ChEMBL (생물활성)** ⭐⭐
**약물 발견 데이터**

**규모**:
- **2,000,000+** 화합물
- **15,000,000+** 활성 데이터
- 표적-화합물 상호작용

**내용**:
- IC50, EC50
- 결합 친화도
- 약물 표적

**다운로드**:
```bash
# SQLite 데이터베이스
wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_31_sqlite.tar.gz

# Python API
pip install chembl_webresource_client
from chembl_webresource_client.new_client import new_client
```

**장점**:
- ✅ 대규모
- ✅ 고품질
- ✅ 무료

**단점**:
- ⚠️ 반응 데이터 아님 (활성 데이터)
- ⚠️ 효소 반응 적음

---

### 7. **PubChem BioAssay** ⭐⭐
**생물학적 검정 데이터**

**규모**:
- **1,000,000+** 검정
- **200,000,000+** 측정값

**내용**:
- 효소 활성
- IC50, EC50
- 조건

**다운로드**:
```bash
# FTP
wget ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/

# API
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1234/JSON"
```

**장점**:
- ✅ 대규모
- ✅ 무료
- ✅ API 좋음

**단점**:
- ⚠️ 반응 데이터 아님
- ⚠️ 효소 동역학 적음

---

## 📊 데이터 소스 비교

| 데이터베이스 | 규모 | 반응식 | 조건 | 수율 | kcat/Km | 무료 | 추천 |
|------------|------|--------|------|------|---------|------|------|
| **USPTO** | 1M+ | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | ⭐⭐⭐ |
| **Rhea** | 12K | ✅ | ⚠️ | ❌ | ❌ | ✅ | ⭐⭐⭐ |
| **BRENDA** | 83K | ❌ | ✅ | ❌ | ✅ | ✅ | ⭐⭐⭐ |
| **Reaxys** | 50M+ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⭐⭐ |
| **ORD** | 100K+ | ✅ | ✅ | ✅ | ❌ | ✅ | ⭐⭐⭐ |
| **ChEMBL** | 2M+ | ❌ | ⚠️ | ❌ | ⚠️ | ✅ | ⭐⭐ |
| **PubChem** | 1M+ | ❌ | ⚠️ | ❌ | ⚠️ | ✅ | ⭐⭐ |

---

## 🎯 우리 프로젝트 추천 조합

### Phase 1: 소규모 테스트 (지금)
```
Rhea (12K) + BRENDA (시뮬레이션)
→ GNN 구조 검증
→ 파이프라인 구축
```

### Phase 2: 중규모 학습 (1주일)
```
USPTO-50K (50K) + Rhea (12K)
→ 실제 GNN 학습
→ 성능 평가
```

### Phase 3: 대규모 학습 (1개월)
```
USPTO-MIT (1M+) + ORD (100K+) + Rhea (12K)
→ 고성능 모델
→ 실험 검증
```

---

## 💻 데이터 다운로드 스크립트

### USPTO-50K 다운로드
```python
import urllib.request
import zipfile

# 다운로드
url = "https://github.com/rxn4chemistry/OpenNMT-py/raw/master/data/USPTO_50K.zip"
urllib.request.urlretrieve(url, "USPTO_50K.zip")

# 압축 해제
with zipfile.ZipFile("USPTO_50K.zip", 'r') as zip_ref:
    zip_ref.extractall("data/USPTO_50K")

print("USPTO-50K 다운로드 완료!")
```

### Rhea 전체 다운로드
```python
import requests

# TSV 형식 (전체)
url = "https://www.rhea-db.org/rhea/"
params = {
    "query": "",
    "columns": "rhea-id,equation,ec,chebi-reactant,chebi-product",
    "format": "tsv",
    "limit": 20000
}

response = requests.get(url, params=params)

with open("data/rhea_all.tsv", 'w') as f:
    f.write(response.text)

print("Rhea 전체 다운로드 완료!")
```

### ORD 다운로드
```bash
# GitHub 클론
git clone https://github.com/open-reaction-database/ord-data.git data/ord-data

# Python으로 읽기
pip install ord-schema
```

---

## 🔧 데이터 전처리 필요사항

### 1. SMILES 정규화
```python
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

# "CCO" == "OCC" → "CCO" (정규화)
```

### 2. 반응 밸런싱
```python
def balance_reaction(reactants, products):
    # 원자 수 확인
    # 밸런스 안 맞으면 제거
    pass
```

### 3. 결측 데이터 처리
```python
# 수율 없으면 None
# kcat 없으면 전이 학습
# 조건 없으면 기본값
```

---

## 📈 예상 데이터 규모

### 최소 (테스트)
- **10,000개** 반응
- Rhea + BRENDA 시뮬레이션
- GNN 구조 검증

### 중간 (학습)
- **50,000개** 반응
- USPTO-50K + Rhea
- 실제 성능 평가

### 대규모 (프로덕션)
- **1,000,000개** 반응
- USPTO-MIT + ORD + Rhea
- 고성능 모델

---

## 🚀 다음 단계

1. **USPTO-50K 다운로드** (50,000개)
2. **데이터 전처리** (SMILES 정규화)
3. **GNN 구현** (PyTorch Geometric)
4. **학습 및 평가**
5. **규칙 기반 vs GNN 비교**

---

## 📚 참고 자료

### 논문
- "Molecular Graph Convolutions" (Kearnes et al., 2016)
- "Predicting Reaction Performance" (Schwaller et al., 2021)
- "USPTO Dataset Analysis" (Lowe, 2012)

### 코드
- RXN4Chemistry: https://github.com/rxn4chemistry
- ChemBERTa: https://github.com/seyonechithrananda/bert-loves-chemistry
- ORD: https://github.com/open-reaction-database

### 도구
- RDKit: 분자 처리
- PyTorch Geometric: GNN
- DeepChem: 화학 ML

---

**결론**: USPTO-50K로 시작하는 게 최선!
- 50,000개 반응
- 무료
- 전처리 도구 많음
- GNN 학습 충분

바로 다운로드하고 GNN 구현 시작할까?
