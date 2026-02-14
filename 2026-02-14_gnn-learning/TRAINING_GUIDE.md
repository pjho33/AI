# GNN 학습 가이드

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# GPU 지원 PyTorch 설치 (권장)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

# 기타 의존성
pip install rdkit pandas tqdm requests
```

### 2. 데이터 준비

```bash
# USPTO 데이터 다운로드 및 파싱
python src/data_processing/download_uspto_official.py

# 생성되는 파일:
# - data/uspto_official_1k.json (1,000개)
# - data/uspto_official_100k.json (100,000개)
# - data/uspto_official_1m.json (1,000,000개)
```

---

## 📚 모델별 학습 가이드

### GCN (Graph Convolutional Network)

**특징**: 가장 빠르고 간단한 모델

```bash
# 100K 학습
python src/train_gnn_100k.py

# 1M 학습 (GPU 권장)
python src/train_gnn_1m.py
```

**하이퍼파라미터**:
```python
hidden_dim = 256
num_layers = 3
dropout = 0.3
learning_rate = 0.001
batch_size = 128
epochs = 50
```

**예상 시간**:
- 100K: CPU ~45분, GPU ~5분
- 1M: CPU ~7시간, GPU ~1시간

---

### GAT (Graph Attention Network)

**특징**: Attention 메커니즘으로 해석 가능

```bash
# 100K 학습 (Full 50 epochs)
python src/train_gat_100k_full.py
```

**하이퍼파라미터**:
```python
hidden_dim = 256
num_heads = 4  # Attention heads
num_layers = 3
dropout = 0.3
learning_rate = 0.001
batch_size = 128
epochs = 50
```

**예상 시간**:
- 100K: CPU ~75분, GPU ~8분

---

### MPNN (Message Passing Neural Network)

**특징**: 가장 강력하지만 GPU 필수

```bash
# 100K 학습 (GPU 필수)
python src/train_mpnn_100k.py
```

**하이퍼파라미터**:
```python
hidden_dim = 256
edge_features = 4
num_layers = 3
dropout = 0.3
learning_rate = 0.001
batch_size = 128
epochs = 50
```

**예상 시간**:
- 100K: CPU ~22시간, GPU ~35분 ⚠️ GPU 필수!

---

## 🎭 앙상블 예측

```bash
# 앙상블 시스템 테스트
python src/ensemble_predictor.py
```

**사용 예시**:
```python
from ensemble_predictor import EnsemblePredictor

# 앙상블 생성
ensemble = EnsemblePredictor(device='cuda')

# 모델 로드
ensemble.load_models({
    'gcn': 'data/best_gnn_100k.pt',
    'gat': 'data/best_gat_100k_full.pt',
    'mpnn': 'data/best_mpnn_100k.pt'
})

# 예측
smiles = "CCO"  # Ethanol
prob, predictions = ensemble.predict(smiles, method='weighted')

print(f"Ensemble: {prob:.3f}")
print(f"GCN: {predictions['gcn']:.3f}")
print(f"GAT: {predictions['gat']:.3f}")
print(f"MPNN: {predictions['mpnn']:.3f}")
```

---

## ⚙️ GPU 설정

### GPU 확인

```bash
# NVIDIA GPU 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### GPU 메모리 관리

**RTX 3090 (24GB) 권장 설정**:
```python
# 100K 데이터
batch_size = 128  # ~10GB VRAM

# 1M 데이터
batch_size = 256  # ~15GB VRAM
```

**메모리 부족 시**:
```python
# 배치 크기 줄이기
batch_size = 64

# Gradient accumulation
accumulation_steps = 2
```

---

## 📊 학습 모니터링

### 로그 파일

모든 학습은 자동으로 로그 저장:
```
data/training_log_gnn_100k.txt
data/training_log_gat_100k_full.txt
data/training_log_mpnn_100k_gpu.txt
data/training_log_1m.txt
```

### 실시간 모니터링

```bash
# 학습 진행 상황 확인
tail -f data/training_log_mpnn_100k_gpu.txt

# GPU 사용률 모니터링
watch -n 1 nvidia-smi
```

### TensorBoard (선택)

```python
# 코드에 추가
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_name')

# 학습 중
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
```

```bash
# TensorBoard 실행
tensorboard --logdir=runs
```

---

## 🔧 문제 해결

### 1. CUDA Out of Memory

**증상**: `RuntimeError: CUDA out of memory`

**해결**:
```python
# 배치 크기 줄이기
batch_size = 64  # 또는 32

# num_workers 줄이기
num_workers = 2  # 또는 0

# 메모리 정리
torch.cuda.empty_cache()
```

### 2. SMILES 파싱 오류

**증상**: `SMILES Parse Error`

**원인**: Silicon, Tellurium 등 특수 원소

**해결**: 자동으로 스킵됨 (정상)
```
성공률: ~99.4%
실패: ~0.6% (주로 Si, Te 화합물)
```

### 3. 학습이 너무 느림

**CPU 사용 중**:
```bash
# GPU 버전 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**GPU 사용 중인데 느림**:
```python
# num_workers 조정
num_workers = 4  # CPU 코어 수에 맞게

# pin_memory 활성화
DataLoader(..., pin_memory=True)
```

### 4. 모델 로드 오류

**증상**: `RuntimeError: Error(s) in loading state_dict`

**원인**: 모델 구조 불일치

**해결**:
```python
# 정확한 파라미터로 모델 생성
model = ReactionGCN(
    node_features=22,
    hidden_dim=256,  # 학습 시와 동일
    output_dim=1
)
model.load_state_dict(torch.load('model.pt'))
```

---

## 📈 성능 최적화

### 1. 데이터 로딩 최적화

```python
# 멀티프로세싱
DataLoader(..., num_workers=4, pin_memory=True)

# 프리페칭
DataLoader(..., prefetch_factor=2)

# 영구 워커
DataLoader(..., persistent_workers=True)
```

### 2. 학습 최적화

```python
# Mixed precision training (GPU)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(batch)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 모델 최적화

```python
# 모델 컴파일 (PyTorch 2.0+)
model = torch.compile(model)

# JIT 컴파일
model = torch.jit.script(model)
```

---

## 🎯 권장 학습 전략

### 프로토타이핑
```
1. 1K 데이터로 빠른 테스트 (1분)
2. 10K로 검증 (5분)
3. 100K로 최종 확인 (45분)
```

### 프로덕션
```
1. 100K로 모델 개발 (1시간)
2. 1M으로 최종 학습 (3-4시간, GPU)
3. 앙상블로 안정성 확보
```

### 연구
```
1. GCN으로 베이스라인 (빠름)
2. GAT로 Attention 분석 (해석)
3. MPNN으로 최고 성능 (정확)
4. 앙상블로 종합 (안정)
```

---

## 📝 체크리스트

### 학습 전
- [ ] GPU 설치 확인 (`nvidia-smi`)
- [ ] PyTorch CUDA 확인 (`torch.cuda.is_available()`)
- [ ] 데이터 다운로드 완료
- [ ] 충분한 디스크 공간 (1M: ~5GB)
- [ ] 충분한 VRAM (MPNN: ~10GB)

### 학습 중
- [ ] 로그 파일 생성 확인
- [ ] GPU 사용률 모니터링
- [ ] Loss 감소 확인
- [ ] Accuracy 증가 확인
- [ ] 메모리 누수 없음

### 학습 후
- [ ] 모델 파일 저장 확인 (`data/best_*.pt`)
- [ ] Test accuracy 확인
- [ ] 추론 테스트
- [ ] 앙상블 통합
- [ ] 문서화

---

## 🚀 고급 기능

### 커스텀 모델

```python
class CustomGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 여기에 레이어 정의
        
    def forward(self, data):
        # 여기에 forward 로직
        return output
```

### 커스텀 손실 함수

```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # 여기에 손실 계산
        return loss
```

### 학습 재개

```python
# 체크포인트 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# 체크포인트 로드
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

**작성일**: 2026-02-14
**버전**: 1.0
**프레임워크**: PyTorch Geometric 2.7.0
