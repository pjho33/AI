"""
GNN 학습 파이프라인
실제 USPTO 데이터로 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('src/data_processing')
sys.path.append('src/models')

from smiles_to_graph import MoleculeGraphConverter
from reaction_gcn import ReactionGCN


def load_uspto_data(json_file: str, max_samples: int = None):
    """USPTO JSON 데이터 로드"""
    
    print(f"데이터 로드 중: {json_file}")
    
    with open(json_file, 'r') as f:
        reactions = json.load(f)
    
    if max_samples:
        reactions = reactions[:max_samples]
    
    print(f"✓ {len(reactions)}개 반응 로드")
    
    return reactions


def create_graph_dataset(reactions, converter):
    """
    반응 데이터를 그래프 데이터셋으로 변환
    
    간단한 버전: 반응물만 사용, 이진 분류 (성공/실패)
    """
    
    print("\n그래프 데이터셋 생성 중...")
    
    graphs = []
    labels = []
    
    for rxn in tqdm(reactions, desc="변환"):
        # 반응물 SMILES (첫 번째만 사용)
        reactant_smiles = rxn['reactants'].split('.')[0]
        
        # 그래프 변환
        graph = converter.smiles_to_graph(reactant_smiles)
        
        if graph is None:
            continue
        
        # 레이블: 생성물이 있으면 성공 (1), 없으면 실패 (0)
        # 실제로는 모두 성공이지만, 데모용으로 간단히 설정
        label = 1.0 if rxn['products'] else 0.0
        
        graphs.append(graph)
        labels.append(label)
    
    print(f"✓ {len(graphs)}개 그래프 생성")
    
    return graphs, labels


def train_epoch(model, loader, optimizer, criterion, device):
    """1 에폭 학습"""
    
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward
        output = model(batch)
        loss = criterion(output.squeeze(), batch.y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """평가"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            
            # 이진 분류
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    """메인 학습 루프"""
    
    print("="*70)
    print("GNN 학습 파이프라인")
    print("="*70)
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # 데이터 로드
    reactions = load_uspto_data('data/uspto_official_1k.json', max_samples=500)
    
    # 그래프 변환
    converter = MoleculeGraphConverter()
    graphs, labels = create_graph_dataset(reactions, converter)
    
    # 레이블 추가
    for graph, label in zip(graphs, labels):
        graph.y = torch.tensor([label], dtype=torch.float)
    
    # Train/Test 분할
    split_idx = int(len(graphs) * 0.8)
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:]
    
    print(f"\nTrain: {len(train_graphs)}개")
    print(f"Test: {len(test_graphs)}개")
    
    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # 모델
    model = ReactionGCN(
        node_features=22,
        hidden_dim=64,
        output_dim=1
    ).to(device)
    
    print(f"\n모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 옵티마이저 & 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 학습
    print("\n" + "="*70)
    print("학습 시작")
    print("="*70)
    
    num_epochs = 20
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Test Acc: {test_acc:.3f}")
        
        # 최고 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'data/best_gnn_model.pt')
    
    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)
    print(f"최고 Test Accuracy: {best_acc:.3f}")
    print(f"모델 저장: data/best_gnn_model.pt")
    
    # 예측 테스트
    print("\n" + "="*70)
    print("예측 테스트")
    print("="*70)
    
    model.eval()
    test_samples = test_graphs[:5]
    
    with torch.no_grad():
        for i, graph in enumerate(test_samples):
            graph = graph.to(device)
            output = model(graph)
            prob = torch.sigmoid(output).item()
            
            print(f"\n샘플 {i+1}:")
            print(f"  SMILES: {graph.smiles}")
            print(f"  예측 확률: {prob:.3f}")
            print(f"  실제 레이블: {graph.y.item():.0f}")


if __name__ == "__main__":
    main()
