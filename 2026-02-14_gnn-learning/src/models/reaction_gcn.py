"""
GNN 모델: 반응 예측
2-layer GCN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple


class ReactionGCN(nn.Module):
    """
    반응 예측을 위한 GCN 모델
    
    아키텍처:
    - 2-layer GCN
    - Global mean pooling
    - MLP predictor
    
    입력: 분자 그래프
    출력: 반응 가능성, kcat, Km 등
    """
    
    def __init__(
        self,
        node_features: int = 22,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Args:
            node_features: 노드 특징 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원 (1=가능성, 3=가능성+kcat+Km)
            dropout: 드롭아웃 비율
        """
        super(ReactionGCN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # MLP predictor
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data 객체
                - x: 노드 특징 [N, node_features]
                - edge_index: 엣지 인덱스 [2, E]
                - batch: 배치 인덱스 [N]
        
        Returns:
            예측값 [batch_size, output_dim]
        """
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling (그래프 → 벡터)
        x = global_mean_pool(x, batch)
        
        # MLP predictor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ReactionCenterGCN(nn.Module):
    """
    반응 중심 예측을 위한 GCN
    
    노드 레벨 예측:
    - 각 원자가 반응 중심인지 예측
    """
    
    def __init__(
        self,
        node_features: int = 22,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super(ReactionCenterGCN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Node-level predictor
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            노드별 반응 중심 확률 [N, 1]
        """
        
        x, edge_index = data.x, data.edge_index
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Node-level prediction
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x


def demo_gcn():
    """GCN 모델 데모"""
    
    print("="*70)
    print("GCN 모델 데모")
    print("="*70)
    
    # 모델 초기화
    model = ReactionGCN(
        node_features=22,
        hidden_dim=64,
        output_dim=1  # 반응 가능성
    )
    
    print(f"\n모델 구조:")
    print(model)
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n파라미터:")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능: {trainable_params:,}")
    
    # 테스트 데이터 생성
    print("\n" + "="*70)
    print("테스트 예측")
    print("="*70)
    
    from smiles_to_graph import MoleculeGraphConverter
    
    converter = MoleculeGraphConverter()
    
    # 테스트 분자들
    test_smiles = ["CCO", "OCC(O)CO", "c1ccccc1"]
    graphs = []
    
    for smiles in test_smiles:
        graph = converter.smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    
    # 배치 생성
    batch = Batch.from_data_list(graphs)
    
    # 예측
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
    
    print(f"\n예측 결과:")
    for i, (smiles, pred) in enumerate(zip(test_smiles, predictions)):
        print(f"  {smiles}: {pred.item():.4f}")
    
    # 반응 중심 예측 모델
    print("\n" + "="*70)
    print("반응 중심 예측 모델")
    print("="*70)
    
    rc_model = ReactionCenterGCN(
        node_features=22,
        hidden_dim=64
    )
    
    print(f"\n모델 구조:")
    print(rc_model)
    
    # 테스트
    graph = graphs[0]  # 에탄올
    
    rc_model.eval()
    with torch.no_grad():
        node_probs = rc_model(graph)
    
    print(f"\n에탄올 (CCO) 반응 중심 예측:")
    print(f"  노드 수: {graph.x.shape[0]}")
    print(f"  반응 중심 확률 (상위 3개):")
    
    top_indices = torch.argsort(node_probs.squeeze(), descending=True)[:3]
    for idx in top_indices:
        print(f"    노드 {idx.item()}: {node_probs[idx].item():.4f}")
    
    print("\n" + "="*70)
    print("다음 단계")
    print("="*70)
    print("1. 학습 데이터 준비 (USPTO)")
    print("2. 학습 루프 구현")
    print("3. 모델 학습")
    print("4. 평가 및 비교")


if __name__ == "__main__":
    demo_gcn()
