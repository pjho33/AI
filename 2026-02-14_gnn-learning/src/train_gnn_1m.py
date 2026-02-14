"""
대규모 GNN 학습 - 1,000,000개 반응
초대규모 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time
import re

sys.path.append('src/data_processing')
sys.path.append('src/models')

from smiles_to_graph import MoleculeGraphConverter
from reaction_gcn import ReactionGCN


def load_and_prepare_data_1m(json_file: str):
    """1M 데이터 로드 및 준비"""
    
    print("="*70)
    print("초대규모 데이터 준비 (1,000,000개)")
    print("="*70)
    
    # 데이터 로드
    print(f"\n데이터 로드 중: {json_file}")
    print("⚠️  1M 데이터 로딩은 시간이 걸립니다...")
    
    with open(json_file, 'r') as f:
        reactions = json.load(f)
    
    print(f"✓ {len(reactions):,}개 반응 로드")
    
    # 그래프 변환
    print(f"\n그래프 변환 중 (10-15분 예상)...\n")
    converter = MoleculeGraphConverter()
    
    graphs = []
    failed = 0
    
    for i, rxn in enumerate(tqdm(reactions, desc="변환")):
        # 반응물 SMILES (첫 번째만)
        reactant_smiles = rxn['reactants'].split('.')[0]
        
        # Atom mapping 제거
        clean_smiles = re.sub(r'\[([A-Z][a-z]?):\d+\]', r'\1', reactant_smiles)
        clean_smiles = re.sub(r':(\d+)', '', clean_smiles)
        
        # 그래프 변환
        graph = converter.smiles_to_graph(clean_smiles)
        
        if graph is None:
            failed += 1
            continue
        
        # 레이블
        graph.y = torch.tensor([1.0], dtype=torch.float)
        graphs.append(graph)
        
        # 진행 상황 출력 (매 100K)
        if (i + 1) % 100000 == 0:
            print(f"\n진행: {i+1:,}개 처리, {len(graphs):,}개 성공, {failed:,}개 실패")
    
    print(f"\n✓ {len(graphs):,}개 그래프 생성 ({failed}개 실패, {failed/len(reactions)*100:.1f}%)")
    
    return graphs


def train_1m_model(graphs, epochs=30, batch_size=256):
    """1M 모델 학습"""
    
    print("\n" + "="*70)
    print("초대규모 모델 학습")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # Train/Val/Test 분할
    n = len(graphs)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size+val_size]
    test_graphs = graphs[train_size+val_size:]
    
    print(f"\nTrain: {len(train_graphs):,}개")
    print(f"Val: {len(val_graphs):,}개")
    print(f"Test: {len(test_graphs):,}개")
    
    # DataLoader (더 큰 배치)
    print("\nDataLoader 생성 중...")
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 더 큰 모델
    model = ReactionGCN(
        node_features=22,
        hidden_dim=512,  # 256 → 512
        output_dim=1,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델 파라미터: {total_params:,}")
    print(f"Hidden Dim: 512")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # 학습
    print("\n학습 시작...\n")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output.squeeze(), batch.y)
                val_loss += loss.item()
                val_batches += 1
                
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_loss /= val_batches
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/best_gnn_1m.pt')
            print(f"  → Best model saved!")
    
    # Test 평가
    print("\n" + "="*70)
    print("최종 평가")
    print("="*70)
    
    model.load_state_dict(torch.load('data/best_gnn_1m.pt'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            test_correct += (pred == batch.y).sum().item()
            test_total += batch.y.size(0)
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Samples: {test_total:,}")
    print(f"모델 저장: data/best_gnn_1m.pt")
    
    return model, test_acc


def main():
    """메인"""
    
    # 데이터 준비
    graphs = load_and_prepare_data_1m('data/uspto_official_1m.json')
    
    # 학습
    model, test_acc = train_1m_model(graphs, epochs=30, batch_size=256)
    
    print("\n" + "="*70)
    print("1M 학습 완료!")
    print("="*70)
    print(f"최종 Test Accuracy: {test_acc:.4f}")
    print(f"총 데이터: {len(graphs):,}개")


if __name__ == "__main__":
    main()
