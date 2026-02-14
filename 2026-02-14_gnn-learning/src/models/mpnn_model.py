"""
MPNN (Message Passing Neural Network) 모델
가장 강력한 GNN 아키텍처
"""

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool
import torch.nn.functional as F


class ReactionMPNN(nn.Module):
    """
    MPNN 기반 반응 예측 모델
    
    Message Passing Neural Network:
    - Edge features 활용
    - 더 표현력 있는 메시지 전달
    - 최고 성능
    """
    
    def __init__(self, node_features=22, edge_features=4, hidden_dim=128, dropout=0.3):
        super(ReactionMPNN, self).__init__()
        
        # Edge network (메시지 함수)
        self.edge_network1 = nn.Sequential(
            nn.Linear(edge_features, node_features * hidden_dim),
            nn.ReLU()
        )
        
        self.edge_network2 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim),
            nn.ReLU()
        )
        
        self.edge_network3 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim),
            nn.ReLU()
        )
        
        # MPNN layers
        self.conv1 = NNConv(node_features, hidden_dim, self.edge_network1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_network2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = NNConv(hidden_dim, hidden_dim, self.edge_network3)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # MPNN layers with edge features
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Prediction
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ReactionCenterMPNN(nn.Module):
    """
    MPNN 기반 반응 중심 예측
    노드 레벨 예측
    """
    
    def __init__(self, node_features=22, edge_features=4, hidden_dim=128, dropout=0.2):
        super(ReactionCenterMPNN, self).__init__()
        
        # Edge networks
        self.edge_network1 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * node_features),
            nn.ReLU()
        )
        
        self.edge_network2 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim),
            nn.ReLU()
        )
        
        # MPNN layers
        self.conv1 = NNConv(node_features, hidden_dim, self.edge_network1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_network2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Node-level prediction
        self.fc = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # MPNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Node predictions
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x


def demo():
    """데모"""
    
    print("="*70)
    print("MPNN 모델 데모")
    print("="*70)
    
    from torch_geometric.data import Data, Batch
    
    # 샘플 그래프 (edge features 포함)
    x = torch.randn(10, 22)
    edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
    edge_attr = torch.randn(5, 4)  # Edge features (4 dims)
    
    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    batch = Batch.from_data_list([data1, data2])
    
    # MPNN 모델
    print("\n1. 반응 예측 MPNN")
    model = ReactionMPNN(node_features=22, edge_features=4, hidden_dim=128)
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   출력 shape: {output.shape}")
    print(f"   예측값: {torch.sigmoid(output).squeeze().tolist()}")
    
    # 반응 중심 MPNN
    print("\n2. 반응 중심 예측 MPNN")
    model2 = ReactionCenterMPNN(node_features=22, edge_features=4, hidden_dim=128)
    print(f"   파라미터: {sum(p.numel() for p in model2.parameters()):,}")
    
    model2.eval()
    with torch.no_grad():
        output2 = model2(data1)
    print(f"   출력 shape: {output2.shape}")
    print(f"   노드별 확률 (상위 5개):")
    probs = output2.squeeze()
    top_indices = torch.argsort(probs, descending=True)[:5]
    for i, idx in enumerate(top_indices):
        print(f"     {i+1}. 노드 {idx.item()}: {probs[idx].item():.3f}")


if __name__ == "__main__":
    demo()
