"""
전이 학습 (Transfer Learning)
유사 반응에서 동역학 파라미터 전이
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimilarReaction:
    """유사 반응 정보"""
    ec_number: str
    substrate: str
    similarity_score: float
    kcat: Optional[float]
    Km: Optional[float]
    conditions: Dict


class TransferLearner:
    """
    전이 학습
    
    핵심 아이디어:
    - 유사한 반응의 동역학 파라미터를 전이
    - 분자 구조 유사도 기반
    - 효소 패밀리 유사도 기반
    """
    
    def __init__(self, kinetics_database: List[Dict]):
        """
        Args:
            kinetics_database: 동역학 데이터베이스
        """
        self.database = kinetics_database
    
    def find_similar_reactions(
        self,
        target_ec: str,
        target_substrate: str,
        top_k: int = 5
    ) -> List[SimilarReaction]:
        """
        유사 반응 찾기
        
        Args:
            target_ec: 타겟 EC 번호
            target_substrate: 타겟 기질 (SMILES)
            top_k: 상위 k개
        
        Returns:
            List[SimilarReaction]: 유사 반응 리스트
        """
        
        similar_reactions = []
        
        for entry in self.database:
            # 유사도 계산
            similarity = self._calculate_similarity(
                target_ec,
                target_substrate,
                entry["ec_number"],
                entry.get("substrate", "")
            )
            
            if similarity > 0.3:  # 최소 유사도
                similar = SimilarReaction(
                    ec_number=entry["ec_number"],
                    substrate=entry.get("substrate", ""),
                    similarity_score=similarity,
                    kcat=entry.get("kcat"),
                    Km=entry.get("Km"),
                    conditions=entry.get("conditions", {})
                )
                similar_reactions.append(similar)
        
        # 유사도 순으로 정렬
        similar_reactions.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similar_reactions[:top_k]
    
    def _calculate_similarity(
        self,
        ec1: str,
        substrate1: str,
        ec2: str,
        substrate2: str
    ) -> float:
        """
        반응 유사도 계산
        
        Args:
            ec1, substrate1: 반응 1
            ec2, substrate2: 반응 2
        
        Returns:
            유사도 (0-1)
        """
        
        # 1. EC 번호 유사도
        ec_similarity = self._ec_similarity(ec1, ec2)
        
        # 2. 기질 유사도 (간단한 버전)
        substrate_similarity = self._substrate_similarity(substrate1, substrate2)
        
        # 종합 유사도
        similarity = (
            ec_similarity * 0.6 +
            substrate_similarity * 0.4
        )
        
        return similarity
    
    def _ec_similarity(self, ec1: str, ec2: str) -> float:
        """EC 번호 유사도"""
        
        parts1 = ec1.split('.')
        parts2 = ec2.split('.')
        
        # 같은 레벨까지 일치하는 정도
        matches = 0
        for i in range(min(len(parts1), len(parts2))):
            if parts1[i] == parts2[i]:
                matches += 1
            else:
                break
        
        # 가중치: 상위 레벨이 더 중요
        weights = [0.4, 0.3, 0.2, 0.1]
        similarity = sum(weights[i] for i in range(matches))
        
        return similarity
    
    def _substrate_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        기질 유사도 (간단한 버전)
        
        실제로는 Tanimoto 계수 등 사용
        여기서는 간단히 구현
        """
        
        if not smiles1 or not smiles2:
            return 0.0
        
        # 간단한 문자열 유사도
        if smiles1 == smiles2:
            return 1.0
        
        # 공통 부분 문자열 비율
        common = sum(1 for c in smiles1 if c in smiles2)
        max_len = max(len(smiles1), len(smiles2))
        
        return common / max_len if max_len > 0 else 0.0
    
    def transfer_kinetics(
        self,
        similar_reactions: List[SimilarReaction],
        parameter: str = "kcat"
    ) -> Tuple[float, float, float]:
        """
        동역학 파라미터 전이
        
        Args:
            similar_reactions: 유사 반응 리스트
            parameter: "kcat" or "Km"
        
        Returns:
            (transferred_value, uncertainty, confidence)
        """
        
        # 측정값이 있는 반응만 사용
        valid_reactions = [
            r for r in similar_reactions
            if getattr(r, parameter) is not None
        ]
        
        if not valid_reactions:
            return None, None, 0.0
        
        # 유사도 가중 평균
        values = []
        weights = []
        
        for reaction in valid_reactions:
            value = getattr(reaction, parameter)
            weight = reaction.similarity_score
            
            values.append(value)
            weights.append(weight)
        
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 정규화
        
        # 가중 평균
        transferred_value = np.sum(values * weights)
        
        # 불확실성 (가중 표준편차)
        variance = np.sum(weights * (values - transferred_value) ** 2)
        uncertainty = np.sqrt(variance)
        
        # 신뢰도 (유사도와 데이터 수에 비례)
        avg_similarity = np.mean([r.similarity_score for r in valid_reactions])
        data_confidence = min(1.0, len(valid_reactions) / 5)
        confidence = avg_similarity * data_confidence
        
        return transferred_value, uncertainty, confidence
    
    def transfer_with_conditions(
        self,
        similar_reactions: List[SimilarReaction],
        target_conditions: Dict,
        parameter: str = "kcat"
    ) -> Tuple[float, float, float]:
        """
        조건을 고려한 전이 학습
        
        Args:
            similar_reactions: 유사 반응 리스트
            target_conditions: 타겟 조건
            parameter: "kcat" or "Km"
        
        Returns:
            (transferred_value, uncertainty, confidence)
        """
        
        # 조건 유사도 추가 고려
        adjusted_reactions = []
        
        for reaction in similar_reactions:
            # 조건 유사도 계산
            condition_similarity = self._condition_similarity(
                reaction.conditions,
                target_conditions
            )
            
            # 전체 유사도 재계산
            adjusted_similarity = (
                reaction.similarity_score * 0.7 +
                condition_similarity * 0.3
            )
            
            adjusted = SimilarReaction(
                ec_number=reaction.ec_number,
                substrate=reaction.substrate,
                similarity_score=adjusted_similarity,
                kcat=reaction.kcat,
                Km=reaction.Km,
                conditions=reaction.conditions
            )
            adjusted_reactions.append(adjusted)
        
        # 조정된 유사도로 전이
        return self.transfer_kinetics(adjusted_reactions, parameter)
    
    def _condition_similarity(
        self,
        conditions1: Dict,
        conditions2: Dict
    ) -> float:
        """조건 유사도"""
        
        pH1 = conditions1.get("pH", 7.4)
        pH2 = conditions2.get("pH", 7.4)
        temp1 = conditions1.get("temperature", 37)
        temp2 = conditions2.get("temperature", 37)
        
        # pH 유사도
        pH_diff = abs(pH1 - pH2)
        pH_sim = max(0, 1.0 - pH_diff / 3.0)
        
        # 온도 유사도
        temp_diff = abs(temp1 - temp2)
        temp_sim = max(0, 1.0 - temp_diff / 20.0)
        
        return (pH_sim + temp_sim) / 2


def demo_transfer_learning():
    """전이 학습 데모"""
    
    print("="*70)
    print("전이 학습 데모")
    print("="*70)
    
    # 시뮬레이션 데이터베이스
    database = [
        {
            "ec_number": "1.1.1.1",
            "substrate": "CCO",
            "kcat": 100,
            "Km": 0.5,
            "conditions": {"pH": 7.4, "temperature": 37}
        },
        {
            "ec_number": "1.1.1.1",
            "substrate": "CCCO",
            "kcat": 120,
            "Km": 0.6,
            "conditions": {"pH": 7.0, "temperature": 30}
        },
        {
            "ec_number": "1.1.1.6",
            "substrate": "OCC(O)CO",
            "kcat": 80,
            "Km": 0.8,
            "conditions": {"pH": 8.0, "temperature": 37}
        },
        {
            "ec_number": "1.1.1.14",
            "substrate": "OCC(O)C(O)CO",
            "kcat": 90,
            "Km": 0.7,
            "conditions": {"pH": 7.4, "temperature": 37}
        },
        {
            "ec_number": "5.3.1.5",
            "substrate": "glucose",
            "kcat": 50,
            "Km": 1.0,
            "conditions": {"pH": 7.5, "temperature": 37}
        }
    ]
    
    learner = TransferLearner(database)
    
    # 테스트: 새로운 알코올 산화
    print("\n[테스트] 새로운 알코올 산화")
    print("-"*70)
    print("타겟: EC 1.1.1.1, 기질: CCCCO (1-butanol)")
    print("조건: pH 7.4, 37°C")
    
    # 유사 반응 찾기
    similar = learner.find_similar_reactions(
        target_ec="1.1.1.1",
        target_substrate="CCCCO",
        top_k=3
    )
    
    print(f"\n유사 반응 {len(similar)}개 발견:")
    for i, rxn in enumerate(similar, 1):
        print(f"\n{i}. EC {rxn.ec_number}")
        print(f"   유사도: {rxn.similarity_score:.3f}")
        print(f"   kcat: {rxn.kcat} s⁻¹")
        print(f"   Km: {rxn.Km} mM")
    
    # kcat 전이
    print("\n" + "="*70)
    print("kcat 전이 학습")
    print("="*70)
    
    kcat, kcat_unc, kcat_conf = learner.transfer_kinetics(similar, "kcat")
    
    print(f"\n전이된 kcat: {kcat:.1f} ± {kcat_unc:.1f} s⁻¹")
    print(f"신뢰도: {kcat_conf:.3f}")
    
    # Km 전이
    Km, Km_unc, Km_conf = learner.transfer_kinetics(similar, "Km")
    
    print(f"\n전이된 Km: {Km:.2f} ± {Km_unc:.2f} mM")
    print(f"신뢰도: {Km_conf:.3f}")
    
    # 조건 고려 전이
    print("\n" + "="*70)
    print("조건 고려 전이 학습")
    print("="*70)
    
    target_conditions = {"pH": 7.0, "temperature": 30}
    
    kcat_adj, kcat_unc_adj, kcat_conf_adj = learner.transfer_with_conditions(
        similar,
        target_conditions,
        "kcat"
    )
    
    print(f"\n타겟 조건: pH {target_conditions['pH']}, {target_conditions['temperature']}°C")
    print(f"조정된 kcat: {kcat_adj:.1f} ± {kcat_unc_adj:.1f} s⁻¹")
    print(f"신뢰도: {kcat_conf_adj:.3f}")
    
    print("\n" + "="*70)
    print("전이 학습의 장점")
    print("="*70)
    print("1. 측정 데이터 없어도 예측 가능")
    print("2. 유사 반응의 지식 활용")
    print("3. 불확실성 정량화")
    print("4. 조건 의존성 고려")
    
    print("\n전이 학습 신뢰도:")
    print("  높은 유사도 (>0.8): 신뢰도 높음")
    print("  중간 유사도 (0.5-0.8): 신뢰도 중간")
    print("  낮은 유사도 (<0.5): 신뢰도 낮음")


if __name__ == "__main__":
    demo_transfer_learning()
