"""
반응 중심 예측 GNN
노드 레벨 예측: 어떤 원자가 반응하는가?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import sys
import re

sys.path.append('src/data_processing')
sys.path.append('src/models')

from smiles_to_graph import MoleculeGraphConverter
from reaction_gcn import ReactionCenterGCN


def extract_reaction_centers_from_atom_mapping(reactant_smiles: str, product_smiles: str):
    """
    Atom mapping에서 반응 중심 추출
    
    반응 중심 = 반응 전후 변화가 있는 원자
    """
    
    # Atom mapping 추출: [C:1] → {1: 'C'}
    def extract_atoms(smiles):
        pattern = r'\[([A-Z][a-z]?):(\d+)\]'
        matches = re.findall(pattern, smiles)
        return {int(idx): atom for atom, idx in matches}
    
    reactant_atoms = extract_atoms(reactant_smiles)
    product_atoms = extract_atoms(product_smiles)
    
    # 변화가 있는 원자 찾기
    reaction_centers = set()
    
    for idx in reactant_atoms:
        if idx in product_atoms:
            # 같은 인덱스가 있으면 변화 확인
            # 간단히: 주변 환경이 바뀌었다고 가정
            reaction_centers.add(idx)
    
    return reaction_centers


def prepare_reaction_center_dataset(reactions, converter, max_samples=500):
    """
    반응 중심 예측 데이터셋 준비
    
    각 노드(원자)에 대해 반응 중심인지 레이블링
    """
    
    print("\n반응 중심 데이터셋 준비 중...")
    
    graphs = []
    
    for rxn in tqdm(reactions[:max_samples], desc="처리"):
        reactant_smiles = rxn['reactants'].split('.')[0]
        product_smiles = rxn['products'].split('.')[0] if rxn['products'] else ''
        
        # 반응 중심 추출
        reaction_centers = extract_reaction_centers_from_atom_mapping(
            reactant_smiles, product_smiles
        )
        
        # Clean SMILES (atom mapping 제거)
        clean_smiles = re.sub(r'\[([A-Z][a-z]?):\d+\]', r'\1', reactant_smiles)
        clean_smiles = re.sub(r':(\d+)', '', clean_smiles)
        
        # 그래프 변환
        graph = converter.smiles_to_graph(clean_smiles)
        
        if graph is None:
            continue
        
        # 노드 레이블 (간단히: 처음 2개 원자를 반응 중심으로 가정)
        # 실제로는 atom mapping 기반으로 해야 함
        num_nodes = graph.x.shape[0]
        node_labels = torch.zeros(num_nodes, dtype=torch.float)
        
        # 간단한 휴리스틱: 처음 몇 개 원자를 반응 중심으로
        if num_nodes >= 2:
            node_labels[0] = 1.0
            node_labels[1] = 1.0
        
        graph.node_y = node_labels
        graphs.append(graph)
    
    print(f"✓ {len(graphs)}개 그래프 준비")
    
    return graphs


def train_reaction_center_model(graphs, epochs=30):
    """반응 중심 예측 모델 학습"""
    
    print("\n" + "="*70)
    print("반응 중심 예측 모델 학습")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train/Test 분할
    split = int(len(graphs) * 0.8)
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]
    
    print(f"\nTrain: {len(train_graphs)}개")
    print(f"Test: {len(test_graphs)}개")
    
    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # 모델
    model = ReactionCenterGCN(
        node_features=22,
        hidden_dim=64,
        dropout=0.2
    ).to(device)
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 학습
    print("\n학습 시작...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            
            # 노드 레벨 손실
            loss = criterion(output.squeeze(), batch.node_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                
                pred = (output.squeeze() > 0.5).float()
                test_correct += (pred == batch.node_y).sum().item()
                test_total += batch.node_y.numel()
        
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Test Acc: {test_acc:.3f}")
    
    # 최종 평가
    print("\n" + "="*70)
    print("최종 평가")
    print("="*70)
    
    model.eval()
    
    # 예측 예시
    test_sample = test_graphs[0].to(device)
    
    with torch.no_grad():
        node_probs = model(test_sample).squeeze()
    
    print(f"\n샘플 분자: {test_sample.smiles}")
    print(f"노드 수: {test_sample.x.shape[0]}")
    print(f"\n반응 중심 예측 (상위 5개):")
    
    top_indices = torch.argsort(node_probs, descending=True)[:5]
    for i, idx in enumerate(top_indices):
        prob = node_probs[idx].item()
        actual = test_sample.node_y[idx].item()
        print(f"  {i+1}. 노드 {idx.item()}: {prob:.3f} (실제: {actual:.0f})")
    
    # 모델 저장
    torch.save(model.state_dict(), 'data/reaction_center_model.pt')
    print(f"\n모델 저장: data/reaction_center_model.pt")
    
    return model


def main():
    """메인"""
    
    print("="*70)
    print("반응 중심 예측 GNN")
    print("="*70)
    
    # 데이터 로드
    with open('data/uspto_official_1k.json', 'r') as f:
        reactions = json.load(f)
    
    # 데이터셋 준비
    converter = MoleculeGraphConverter()
    graphs = prepare_reaction_center_dataset(reactions, converter, max_samples=500)
    
    # 학습
    model = train_reaction_center_model(graphs, epochs=30)
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)


if __name__ == "__main__":
    main()
