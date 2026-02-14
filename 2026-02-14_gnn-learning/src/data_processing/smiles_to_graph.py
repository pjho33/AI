"""
SMILES를 PyTorch Geometric 그래프로 변환
RDKit 사용
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import Optional, List, Tuple


class MoleculeGraphConverter:
    """
    분자를 그래프로 변환
    
    Node features (원자):
    - 원자 번호
    - 하이브리드화 (sp, sp2, sp3)
    - 방향족성
    - 형식 전하
    - 수소 개수
    
    Edge features (결합):
    - 결합 차수 (단일, 이중, 삼중)
    - 방향족 결합
    - 링 포함 여부
    """
    
    # 원자 번호 매핑 (주요 원소)
    ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']
    
    # 하이브리드화 타입
    HYBRIDIZATION_LIST = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[Data]:
        """
        SMILES를 PyG Data 객체로 변환
        
        Args:
            smiles: SMILES 문자열
        
        Returns:
            PyG Data 객체 또는 None (변환 실패 시)
        """
        
        # SMILES → RDKit Mol
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # 수소 추가 (명시적)
        mol = Chem.AddHs(mol)
        
        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            features = MoleculeGraphConverter._get_atom_features(atom)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge index & features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 양방향 엣지
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Edge features
            bond_features = MoleculeGraphConverter._get_bond_features(bond)
            edge_features.append(bond_features)
            edge_features.append(bond_features)  # 양방향 동일
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # 단일 원자 (엣지 없음)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        
        # PyG Data 객체
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )
        
        return data
    
    @staticmethod
    def _get_atom_features(atom) -> List[float]:
        """원자 특징 추출"""
        
        features = []
        
        # 1. 원자 번호 (one-hot)
        atom_symbol = atom.GetSymbol()
        atom_one_hot = [0] * len(MoleculeGraphConverter.ATOM_LIST)
        if atom_symbol in MoleculeGraphConverter.ATOM_LIST:
            idx = MoleculeGraphConverter.ATOM_LIST.index(atom_symbol)
            atom_one_hot[idx] = 1
        features.extend(atom_one_hot)
        
        # 2. 하이브리드화 (one-hot)
        hybridization = atom.GetHybridization()
        hybrid_one_hot = [0] * len(MoleculeGraphConverter.HYBRIDIZATION_LIST)
        if hybridization in MoleculeGraphConverter.HYBRIDIZATION_LIST:
            idx = MoleculeGraphConverter.HYBRIDIZATION_LIST.index(hybridization)
            hybrid_one_hot[idx] = 1
        features.extend(hybrid_one_hot)
        
        # 3. 방향족성
        features.append(float(atom.GetIsAromatic()))
        
        # 4. 형식 전하
        features.append(float(atom.GetFormalCharge()))
        
        # 5. 수소 개수
        features.append(float(atom.GetTotalNumHs()))
        
        # 6. Degree (연결된 원자 수)
        features.append(float(atom.GetDegree()))
        
        return features
    
    @staticmethod
    def _get_bond_features(bond) -> List[float]:
        """결합 특징 추출"""
        
        features = []
        
        # 1. 결합 차수 (one-hot)
        bond_type = bond.GetBondType()
        bond_type_one_hot = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]
        features.extend(bond_type_one_hot)
        
        return features
    
    @staticmethod
    def batch_smiles_to_graphs(smiles_list: List[str]) -> List[Data]:
        """여러 SMILES를 그래프로 변환"""
        
        graphs = []
        failed = 0
        
        for smiles in smiles_list:
            graph = MoleculeGraphConverter.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
            else:
                failed += 1
        
        if failed > 0:
            print(f"⚠️ {failed}/{len(smiles_list)}개 변환 실패")
        
        return graphs


def demo_graph_conversion():
    """그래프 변환 데모"""
    
    print("="*70)
    print("SMILES → PyG Graph 변환 데모")
    print("="*70)
    
    # 테스트 분자들
    test_molecules = [
        ("에탄올", "CCO"),
        ("글리세롤", "OCC(O)CO"),
        ("벤젠", "c1ccccc1"),
        ("아세트산", "CC(=O)O"),
        ("글루코스", "OCC(O)C(O)C(O)C(O)C=O")
    ]
    
    converter = MoleculeGraphConverter()
    
    for name, smiles in test_molecules:
        print(f"\n[{name}] {smiles}")
        print("-"*70)
        
        graph = converter.smiles_to_graph(smiles)
        
        if graph is None:
            print("✗ 변환 실패")
            continue
        
        print(f"✓ 변환 성공")
        print(f"  노드 수: {graph.x.shape[0]}")
        print(f"  노드 특징 차원: {graph.x.shape[1]}")
        print(f"  엣지 수: {graph.edge_index.shape[1]}")
        print(f"  엣지 특징 차원: {graph.edge_attr.shape[1]}")
        
        # 첫 번째 노드 특징 출력
        print(f"\n  첫 번째 원자 특징 (일부):")
        print(f"    원자 타입 one-hot: {graph.x[0, :12].tolist()}")
        print(f"    방향족: {graph.x[0, 18]:.0f}")
        print(f"    형식 전하: {graph.x[0, 19]:.0f}")
    
    print("\n" + "="*70)
    print("그래프 구조 정보")
    print("="*70)
    
    print(f"\n노드 특징 차원: {graph.x.shape[1]}")
    print(f"  - 원자 타입 (one-hot): 12")
    print(f"  - 하이브리드화 (one-hot): 6")
    print(f"  - 방향족성: 1")
    print(f"  - 형식 전하: 1")
    print(f"  - 수소 개수: 1")
    print(f"  - Degree: 1")
    
    print(f"\n엣지 특징 차원: {graph.edge_attr.shape[1]}")
    print(f"  - 결합 타입 (one-hot): 4 (단일/이중/삼중/방향족)")
    
    print("\n" + "="*70)
    print("다음 단계")
    print("="*70)
    print("1. GNN 모델 구축 (GCN/GAT/MPNN)")
    print("2. USPTO 데이터 로드 및 변환")
    print("3. 반응 중심 예측 학습")


if __name__ == "__main__":
    demo_graph_conversion()
