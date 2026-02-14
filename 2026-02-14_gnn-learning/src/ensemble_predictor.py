"""
앙상블 예측 시스템
여러 GNN 모델을 결합하여 더 강력한 예측
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append('src/models')
sys.path.append('src/data_processing')

from reaction_gcn import ReactionGCN
from gat_model import ReactionGAT
from mpnn_model import ReactionMPNN
from smiles_to_graph import MoleculeGraphConverter


class EnsemblePredictor:
    """
    앙상블 예측기
    
    여러 모델의 예측을 결합:
    - GCN (빠르고 정확)
    - GAT (Attention 기반)
    - MPNN (가장 강력)
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.converter = MoleculeGraphConverter()
        self.models = {}
        
    def load_models(self, model_paths):
        """
        모델 로드
        
        Args:
            model_paths: dict of {model_name: model_path}
        """
        
        print("="*70)
        print("앙상블 모델 로드")
        print("="*70)
        
        # GCN 모델
        if 'gcn' in model_paths:
            print("\n1. GCN 모델 로드...")
            gcn = ReactionGCN(node_features=22, hidden_dim=256, output_dim=1)
            gcn.load_state_dict(torch.load(model_paths['gcn'], map_location=self.device))
            gcn.eval()
            self.models['gcn'] = gcn
            print(f"   ✓ {model_paths['gcn']}")
        
        # GAT 모델
        if 'gat' in model_paths:
            print("\n2. GAT 모델 로드...")
            gat = ReactionGAT(node_features=22, hidden_dim=256, num_heads=4)
            gat.load_state_dict(torch.load(model_paths['gat'], map_location=self.device))
            gat.eval()
            self.models['gat'] = gat
            print(f"   ✓ {model_paths['gat']}")
        
        # MPNN 모델
        if 'mpnn' in model_paths:
            print("\n3. MPNN 모델 로드...")
            mpnn = ReactionMPNN(node_features=22, edge_features=4, hidden_dim=256)
            mpnn.load_state_dict(torch.load(model_paths['mpnn'], map_location=self.device))
            mpnn.eval()
            self.models['mpnn'] = mpnn
            print(f"   ✓ {model_paths['mpnn']}")
        
        print(f"\n총 {len(self.models)}개 모델 로드 완료")
    
    def predict(self, smiles, method='average'):
        """
        앙상블 예측
        
        Args:
            smiles: SMILES 문자열
            method: 'average', 'weighted', 'voting'
        
        Returns:
            probability, individual_predictions
        """
        
        # SMILES → Graph
        graph = self.converter.smiles_to_graph(smiles)
        if graph is None:
            return None, None
        
        # 각 모델 예측
        predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                output = model(graph)
                prob = torch.sigmoid(output).item()
                predictions[name] = prob
        
        # 앙상블
        if method == 'average':
            # 평균
            ensemble_prob = sum(predictions.values()) / len(predictions)
        
        elif method == 'weighted':
            # 가중 평균 (MPNN > GAT > GCN)
            weights = {'gcn': 0.3, 'gat': 0.3, 'mpnn': 0.4}
            ensemble_prob = sum(predictions[k] * weights.get(k, 1.0) for k in predictions) / sum(weights.get(k, 1.0) for k in predictions)
        
        elif method == 'voting':
            # 투표 (0.5 기준)
            votes = sum(1 for p in predictions.values() if p > 0.5)
            ensemble_prob = votes / len(predictions)
        
        else:
            ensemble_prob = sum(predictions.values()) / len(predictions)
        
        return ensemble_prob, predictions
    
    def predict_batch(self, smiles_list, method='average'):
        """
        배치 예측
        
        Args:
            smiles_list: SMILES 문자열 리스트
            method: 앙상블 방법
        
        Returns:
            list of (ensemble_prob, individual_predictions)
        """
        
        results = []
        for smiles in smiles_list:
            ensemble_prob, predictions = self.predict(smiles, method)
            results.append((ensemble_prob, predictions))
        
        return results
    
    def get_model_info(self):
        """모델 정보"""
        
        info = {}
        for name, model in self.models.items():
            params = sum(p.numel() for p in model.parameters())
            info[name] = {
                'parameters': params,
                'type': type(model).__name__
            }
        
        return info


def demo():
    """데모"""
    
    print("="*70)
    print("앙상블 예측 시스템 데모")
    print("="*70)
    
    # 앙상블 생성
    ensemble = EnsemblePredictor()
    
    # 모델 로드 (사용 가능한 모델만)
    model_paths = {}
    
    if Path('data/best_gnn_100k.pt').exists():
        model_paths['gcn'] = 'data/best_gnn_100k.pt'
    
    if Path('data/best_gat_100k_full.pt').exists():
        model_paths['gat'] = 'data/best_gat_100k_full.pt'
    elif Path('data/best_gat_100k.pt').exists():
        model_paths['gat'] = 'data/best_gat_100k.pt'
    
    if Path('data/best_mpnn_100k.pt').exists():
        model_paths['mpnn'] = 'data/best_mpnn_100k.pt'
    
    if not model_paths:
        print("\n⚠️  학습된 모델이 없습니다.")
        print("먼저 모델을 학습해주세요.")
        return
    
    ensemble.load_models(model_paths)
    
    # 모델 정보
    print("\n" + "="*70)
    print("모델 정보")
    print("="*70)
    
    info = ensemble.get_model_info()
    for name, details in info.items():
        print(f"\n{name.upper()}:")
        print(f"  타입: {details['type']}")
        print(f"  파라미터: {details['parameters']:,}")
    
    # 예측 테스트
    print("\n" + "="*70)
    print("예측 테스트")
    print("="*70)
    
    test_smiles = [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene
        "CC(C)O",        # Isopropanol
    ]
    
    for smiles in test_smiles:
        print(f"\nSMILES: {smiles}")
        
        # 평균 앙상블
        ensemble_prob, predictions = ensemble.predict(smiles, method='average')
        
        if ensemble_prob is not None:
            print(f"  앙상블 (평균): {ensemble_prob:.3f}")
            print(f"  개별 예측:")
            for model_name, prob in predictions.items():
                print(f"    {model_name.upper()}: {prob:.3f}")
        else:
            print("  ⚠️  변환 실패")
    
    # 가중 평균
    print("\n" + "="*70)
    print("가중 평균 앙상블 (MPNN 40%, GAT 30%, GCN 30%)")
    print("="*70)
    
    smiles = "CCO"
    ensemble_prob, predictions = ensemble.predict(smiles, method='weighted')
    print(f"\nSMILES: {smiles}")
    print(f"가중 앙상블: {ensemble_prob:.3f}")
    
    print("\n" + "="*70)
    print("앙상블 시스템 준비 완료!")
    print("="*70)


if __name__ == "__main__":
    demo()
