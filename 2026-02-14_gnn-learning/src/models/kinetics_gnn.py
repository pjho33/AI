"""
효소 동역학 예측 GNN
kcat, Km 값 예측
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class KineticsGNN(nn.Module):
    """
    효소 동역학 파라미터 예측 GNN
    
    출력:
        - kcat (turnover number, s^-1)
        - Km (Michaelis constant, mM)
    """
    
    def __init__(self, node_features=22, hidden_dim=128, dropout=0.3):
        super(KineticsGNN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Prediction heads
        self.dropout = nn.Dropout(dropout)
        
        # kcat 예측 (log scale)
        self.kcat_fc1 = nn.Linear(hidden_dim, 64)
        self.kcat_fc2 = nn.Linear(64, 1)
        
        # Km 예측 (log scale)
        self.km_fc1 = nn.Linear(hidden_dim, 64)
        self.km_fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
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
        
        # kcat 예측
        kcat = self.relu(self.kcat_fc1(x))
        kcat = self.kcat_fc2(kcat)
        
        # Km 예측
        km = self.relu(self.km_fc1(x))
        km = self.km_fc2(km)
        
        return kcat, km


def demo():
    """데모"""
    
    print("="*70)
    print("효소 동역학 예측 GNN 데모")
    print("="*70)
    
    from torch_geometric.data import Data, Batch
    
    # 샘플 그래프
    x = torch.randn(10, 22)  # 10 nodes, 22 features
    edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
    
    data1 = Data(x=x, edge_index=edge_index)
    data2 = Data(x=x, edge_index=edge_index)
    
    batch = Batch.from_data_list([data1, data2])
    
    # 모델
    model = KineticsGNN(node_features=22, hidden_dim=128)
    
    print(f"\n모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward
    model.eval()
    with torch.no_grad():
        kcat, km = model(batch)
    
    print(f"\n예측 결과:")
    print(f"  kcat (log): {kcat.squeeze().tolist()}")
    print(f"  Km (log): {km.squeeze().tolist()}")
    
    # 실제 값으로 변환
    kcat_real = torch.exp(kcat)
    km_real = torch.exp(km)
    
    print(f"\n실제 값:")
    print(f"  kcat (s^-1): {kcat_real.squeeze().tolist()}")
    print(f"  Km (mM): {km_real.squeeze().tolist()}")


if __name__ == "__main__":
    demo()
