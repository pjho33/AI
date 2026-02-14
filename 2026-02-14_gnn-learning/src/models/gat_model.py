"""
GAT (Graph Attention Network) 모델
더 강력한 메시지 전달
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class ReactionGAT(nn.Module):
    """
    GAT 기반 반응 예측 모델
    
    GCN과 달리 attention mechanism 사용
    - 중요한 원자에 더 집중
    - 더 표현력 있는 학습
    """
    
    def __init__(self, node_features=22, hidden_dim=128, num_heads=4, dropout=0.3):
        super(ReactionGAT, self).__init__()
        
        # GAT layers
        self.conv1 = GATConv(
            node_features, 
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            dropout=dropout
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers with attention
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Prediction
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ReactionCenterGAT(nn.Module):
    """
    GAT 기반 반응 중심 예측
    노드 레벨 예측
    """
    
    def __init__(self, node_features=22, hidden_dim=128, num_heads=4, dropout=0.2):
        super(ReactionCenterGAT, self).__init__()
        
        self.conv1 = GATConv(
            node_features,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Node-level prediction
        self.fc = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GAT layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Node predictions
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x


def demo():
    """데모"""
    
    print("="*70)
    print("GAT 모델 데모")
    print("="*70)
    
    from torch_geometric.data import Data, Batch
    
    # 샘플 그래프
    x = torch.randn(10, 22)
    edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
    
    data1 = Data(x=x, edge_index=edge_index)
    data2 = Data(x=x, edge_index=edge_index)
    
    batch = Batch.from_data_list([data1, data2])
    
    # GAT 모델
    print("\n1. 반응 예측 GAT")
    model = ReactionGAT(node_features=22, hidden_dim=128, num_heads=4)
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   출력 shape: {output.shape}")
    
    # 반응 중심 GAT
    print("\n2. 반응 중심 예측 GAT")
    model2 = ReactionCenterGAT(node_features=22, hidden_dim=128, num_heads=4)
    print(f"   파라미터: {sum(p.numel() for p in model2.parameters()):,}")
    
    model2.eval()
    with torch.no_grad():
        output2 = model2(data1)
    print(f"   출력 shape: {output2.shape}")
    print(f"   노드별 확률: {output2.squeeze().tolist()}")


if __name__ == "__main__":
    demo()
