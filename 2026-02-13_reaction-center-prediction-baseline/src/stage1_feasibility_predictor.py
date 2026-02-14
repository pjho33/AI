"""
Stage 1: Chemistry Feasibility Predictor
화학적 가능성 예측 + 반응 중심 예측
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).parent))
from rules.rule_based_predictor import RuleBasedPredictor, ReactionPrediction


@dataclass
class FeasibilityPrediction:
    """Stage 1 예측 결과"""
    P_feasible: float  # 가능성 확률
    predicted_centers: List[int]  # 반응 중심 원자들
    confidence: float  # 예측 신뢰도
    enzyme_compatible: bool  # 효소 호환성
    cofactor_required: Optional[str]  # 필요 보조인자
    limiting_factors: List[str]  # 제한 요인들
    details: Dict  # 상세 정보


class Stage1FeasibilityPredictor:
    """
    Stage 1: 화학적 가능성 예측기
    
    목표:
    - 이 반응이 화학적으로 일어날 수 있는가?
    - 어디서 반응이 일어나는가?
    - 어떤 효소/보조인자가 필요한가?
    
    학습 가능: ✓ (데이터 풍부: Rhea, KEGG)
    """
    
    def __init__(self):
        self.base_predictor = RuleBasedPredictor()
        self.feasibility_rules = self._load_feasibility_rules()
        
    def _load_feasibility_rules(self) -> Dict:
        """가능성 평가 규칙"""
        return {
            "substrate_enzyme_compatibility": {
                # 기질-효소 호환성 규칙
                "alcohol_dehydrogenase": {
                    "required_groups": ["OH"],
                    "forbidden_groups": [],
                    "optimal_pH_range": (6.0, 9.0),
                    "cofactor": "NAD+"
                },
                "isomerase": {
                    "required_groups": ["C=O", "OH"],
                    "forbidden_groups": [],
                    "optimal_pH_range": (6.5, 8.5),
                    "cofactor": None
                }
            },
            "reaction_type_rules": {
                "oxidation": {
                    "required": ["oxidizable_group"],
                    "produces": ["carbonyl"],
                    "cofactor": "NAD+",
                    "base_feasibility": 0.85
                },
                "isomerization": {
                    "required": ["isomerizable_group"],
                    "produces": ["isomer"],
                    "cofactor": None,
                    "base_feasibility": 0.75
                }
            },
            "condition_constraints": {
                "pH": {
                    "min": 3.0,
                    "max": 11.0,
                    "optimal": (6.5, 8.5)
                },
                "temperature": {
                    "min": 0,
                    "max": 60,
                    "optimal": (25, 40)
                }
            }
        }
    
    def predict_feasibility(
        self,
        substrate_smiles: str,
        reaction_type: str,
        enzyme_ec: str,
        conditions: Optional[Dict] = None
    ) -> FeasibilityPrediction:
        """
        화학적 가능성 예측
        
        Args:
            substrate_smiles: 기질 SMILES
            reaction_type: 반응 유형 (oxidation, isomerization 등)
            enzyme_ec: 효소 EC 번호
            conditions: 반응 조건 (pH, 온도 등)
        
        Returns:
            FeasibilityPrediction: 가능성 예측 결과
        """
        
        if conditions is None:
            conditions = {"pH": 7.4, "temperature": 37}
        
        # 1. 기본 반응 중심 예측
        base_predictions = self.base_predictor.predict_reaction_centers(
            substrate_smiles, reaction_type
        )
        
        if not base_predictions:
            return FeasibilityPrediction(
                P_feasible=0.1,
                predicted_centers=[],
                confidence=0.9,
                enzyme_compatible=False,
                cofactor_required=None,
                limiting_factors=["no_reactive_site"],
                details={"reason": "No reactive sites found"}
            )
        
        # 2. 효소 호환성 평가
        enzyme_compat_score = self._evaluate_enzyme_compatibility(
            substrate_smiles, enzyme_ec, reaction_type
        )
        
        # 3. 조건 적합성 평가
        condition_score = self._evaluate_conditions(
            conditions, enzyme_ec
        )
        
        # 4. 보조인자 요구사항
        cofactor = self._determine_cofactor(reaction_type, enzyme_ec)
        cofactor_score = 1.0 if cofactor else 0.9  # 보조인자 없으면 약간 유리
        
        # 5. 입체화학 고려 (간단한 버전)
        stereo_score = self._evaluate_stereochemistry(
            substrate_smiles, enzyme_ec
        )
        
        # 6. 종합 가능성 점수
        base_feasibility = self.feasibility_rules["reaction_type_rules"].get(
            reaction_type, {}
        ).get("base_feasibility", 0.7)
        
        P_feasible = (
            base_feasibility * 0.4 +
            enzyme_compat_score * 0.25 +
            condition_score * 0.15 +
            cofactor_score * 0.1 +
            stereo_score * 0.1
        )
        
        # 7. 제한 요인 식별
        limiting_factors = self._identify_limiting_factors(
            enzyme_compat_score,
            condition_score,
            cofactor_score,
            stereo_score
        )
        
        # 8. 신뢰도 계산
        confidence = self._calculate_confidence(
            base_predictions,
            enzyme_compat_score,
            condition_score
        )
        
        # 9. 반응 중심 추출
        predicted_centers = base_predictions[0].atom_indices
        
        return FeasibilityPrediction(
            P_feasible=round(P_feasible, 3),
            predicted_centers=predicted_centers,
            confidence=round(confidence, 3),
            enzyme_compatible=enzyme_compat_score > 0.6,
            cofactor_required=cofactor,
            limiting_factors=limiting_factors,
            details={
                "base_feasibility": base_feasibility,
                "enzyme_compatibility": enzyme_compat_score,
                "condition_score": condition_score,
                "cofactor_score": cofactor_score,
                "stereo_score": stereo_score,
                "top_predictions": [
                    {
                        "centers": pred.atom_indices,
                        "rule": pred.rule_name,
                        "confidence": pred.confidence
                    }
                    for pred in base_predictions[:3]
                ]
            }
        )
    
    def _evaluate_enzyme_compatibility(
        self,
        smiles: str,
        enzyme_ec: str,
        reaction_type: str
    ) -> float:
        """효소-기질 호환성 평가"""
        
        # EC 번호 파싱
        ec_parts = enzyme_ec.split('.')
        ec_class = ec_parts[0] if ec_parts else "1"
        
        # 반응 유형별 효소 클래스 매칭
        expected_ec_class = {
            "oxidation": "1",
            "reduction": "1",
            "isomerization": "5",
            "transfer": "2"
        }.get(reaction_type, "1")
        
        if ec_class == expected_ec_class:
            base_score = 0.95
        else:
            # 매우 강화된 페널티: 잘못된 효소 클래스는 거의 불가능
            base_score = 0.08
        
        # 기질 구조 특징 (개선된 버전)
        has_alcohol = "O" in smiles and "C" in smiles
        has_carbonyl = "=O" in smiles or "C=O" in smiles
        
        # 반응 유형별 기질 적합성
        if reaction_type == "oxidation":
            if has_alcohol:
                base_score *= 1.05
            else:
                base_score *= 0.5  # 알코올 없으면 산화 어려움
        elif reaction_type == "isomerization":
            # 이성질화는 알코올 또는 카보닐 필요
            if has_alcohol or has_carbonyl:
                base_score *= 1.05
            else:
                base_score *= 0.3
        
        return min(1.0, base_score)
    
    def _evaluate_conditions(
        self,
        conditions: Dict,
        enzyme_ec: str
    ) -> float:
        """반응 조건 적합성 평가"""
        
        pH = conditions.get("pH", 7.4)
        temp = conditions.get("temperature", 37)
        
        # pH 평가 (매우 강화된 페널티)
        pH_constraints = self.feasibility_rules["condition_constraints"]["pH"]
        if pH < pH_constraints["min"] or pH > pH_constraints["max"]:
            pH_score = 0.02  # 극한 조건: 거의 불가능
        elif pH_constraints["optimal"][0] <= pH <= pH_constraints["optimal"][1]:
            pH_score = 1.0
        else:
            # 최적 범위에서 벗어난 정도에 따라 (매우 가파른 페널티)
            deviation = min(
                abs(pH - pH_constraints["optimal"][0]),
                abs(pH - pH_constraints["optimal"][1])
            )
            # pH 3.0: deviation=3.5 → score=0.05
            # pH 6.0: deviation=0.5 → score=0.70
            # pH 5.0: deviation=1.5 → score=0.30
            pH_score = max(0.05, 1.0 - deviation * 0.6)
        
        # 온도 평가 (매우 강화된 페널티)
        temp_constraints = self.feasibility_rules["condition_constraints"]["temperature"]
        if temp < temp_constraints["min"] or temp > temp_constraints["max"]:
            temp_score = 0.02  # 극한 온도: 거의 불가능
        elif temp_constraints["optimal"][0] <= temp <= temp_constraints["optimal"][1]:
            temp_score = 1.0
        else:
            deviation = min(
                abs(temp - temp_constraints["optimal"][0]),
                abs(temp - temp_constraints["optimal"][1])
            )
            # 매우 가파른 페널티
            # 80°C: deviation=40 → score=0.05
            # 4°C: deviation=21 → score=0.10
            temp_score = max(0.05, 1.0 - deviation * 0.06)
        
        return (pH_score * 0.6 + temp_score * 0.4)
    
    def _determine_cofactor(
        self,
        reaction_type: str,
        enzyme_ec: str
    ) -> Optional[str]:
        """필요한 보조인자 결정"""
        
        if reaction_type in ["oxidation", "reduction"]:
            return "NAD+"
        elif reaction_type == "isomerization":
            return None
        else:
            return None
    
    def _evaluate_stereochemistry(
        self,
        smiles: str,
        enzyme_ec: str
    ) -> float:
        """입체화학 호환성 평가 (간단한 버전)"""
        
        # 입체화학 정보가 있으면 더 정확
        if "@" in smiles:
            return 0.95  # 입체화학 명시됨
        else:
            return 0.75  # 입체화학 불명확
    
    def _identify_limiting_factors(
        self,
        enzyme_score: float,
        condition_score: float,
        cofactor_score: float,
        stereo_score: float
    ) -> List[str]:
        """제한 요인 식별"""
        
        factors = []
        threshold = 0.6
        
        if enzyme_score < threshold:
            factors.append("enzyme_compatibility")
        if condition_score < threshold:
            factors.append("suboptimal_conditions")
        if cofactor_score < threshold:
            factors.append("cofactor_requirement")
        if stereo_score < threshold:
            factors.append("stereochemistry_unclear")
        
        if not factors:
            factors.append("none")
        
        return factors
    
    def _calculate_confidence(
        self,
        base_predictions: List[ReactionPrediction],
        enzyme_score: float,
        condition_score: float
    ) -> float:
        """예측 신뢰도 계산"""
        
        if not base_predictions:
            return 0.1
        
        # 최상위 예측의 신뢰도
        top_confidence = base_predictions[0].confidence
        
        # 예측 일관성 (상위 예측들이 비슷한가?)
        if len(base_predictions) > 1:
            consistency = 1.0 - (
                base_predictions[0].confidence - 
                base_predictions[1].confidence
            )
        else:
            consistency = 1.0
        
        # 종합 신뢰도
        confidence = (
            top_confidence * 0.5 +
            enzyme_score * 0.25 +
            condition_score * 0.15 +
            consistency * 0.1
        )
        
        return min(1.0, confidence)
    
    def batch_predict(
        self,
        reactions: List[Dict]
    ) -> List[FeasibilityPrediction]:
        """배치 예측"""
        
        results = []
        for rxn in reactions:
            pred = self.predict_feasibility(
                substrate_smiles=rxn["substrate"]["smiles"],
                reaction_type=rxn.get("reaction_type", "oxidation"),
                enzyme_ec=rxn["enzyme"]["ec_number"],
                conditions=rxn.get("reaction_conditions")
            )
            results.append(pred)
        
        return results
    
    def evaluate_performance(
        self,
        predictions: List[FeasibilityPrediction],
        ground_truth: List[bool]
    ) -> Dict:
        """성능 평가"""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # 임계값별 성능
        thresholds = [0.5, 0.6, 0.7, 0.8]
        results = {}
        
        for threshold in thresholds:
            predicted_positive = [p.P_feasible >= threshold for p in predictions]
            
            tp = sum(p and g for p, g in zip(predicted_positive, ground_truth))
            fp = sum(p and not g for p, g in zip(predicted_positive, ground_truth))
            tn = sum(not p and not g for p, g in zip(predicted_positive, ground_truth))
            fn = sum(not p and g for p, g in zip(predicted_positive, ground_truth))
            
            accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"threshold_{threshold}"] = {
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3)
            }
        
        # 평균 신뢰도
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        
        results["average_confidence"] = round(avg_confidence, 3)
        results["average_P_feasible"] = round(
            sum(p.P_feasible for p in predictions) / len(predictions), 3
        )
        
        return results


def main():
    """Stage 1 데모"""
    
    print("="*70)
    print("Stage 1: Chemistry Feasibility Predictor")
    print("="*70)
    
    predictor = Stage1FeasibilityPredictor()
    
    # 테스트 케이스
    test_cases = [
        {
            "name": "에탄올 산화 (최적 조건)",
            "substrate": "CCO",
            "reaction_type": "oxidation",
            "enzyme": "1.1.1.1",
            "conditions": {"pH": 7.4, "temperature": 37},
            "expected": True
        },
        {
            "name": "에탄올 산화 (극한 pH)",
            "substrate": "CCO",
            "reaction_type": "oxidation",
            "enzyme": "1.1.1.1",
            "conditions": {"pH": 3.0, "temperature": 37},
            "expected": False
        },
        {
            "name": "글리세롤 산화",
            "substrate": "OCC(O)CO",
            "reaction_type": "oxidation",
            "enzyme": "1.1.1.6",
            "conditions": {"pH": 8.0, "temperature": 30},
            "expected": True
        },
        {
            "name": "소르비톨 산화",
            "substrate": "OCC(O)C(O)C(O)C(O)CO",
            "reaction_type": "oxidation",
            "enzyme": "1.1.1.14",
            "conditions": {"pH": 7.4, "temperature": 37},
            "expected": True
        },
        {
            "name": "잘못된 효소 (산화에 이성질화효소)",
            "substrate": "CCO",
            "reaction_type": "oxidation",
            "enzyme": "5.3.1.5",
            "conditions": {"pH": 7.4, "temperature": 37},
            "expected": False
        }
    ]
    
    print("\n" + "="*70)
    print("테스트 케이스 평가")
    print("="*70)
    
    predictions = []
    ground_truth = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['name']}")
        print(f"  기질: {case['substrate']}")
        print(f"  효소: EC {case['enzyme']}")
        print(f"  조건: pH {case['conditions']['pH']}, {case['conditions']['temperature']}°C")
        
        pred = predictor.predict_feasibility(
            substrate_smiles=case['substrate'],
            reaction_type=case['reaction_type'],
            enzyme_ec=case['enzyme'],
            conditions=case['conditions']
        )
        
        predictions.append(pred)
        ground_truth.append(case['expected'])
        
        print(f"\n  결과:")
        print(f"    P_feasible: {pred.P_feasible:.3f}")
        print(f"    신뢰도: {pred.confidence:.3f}")
        print(f"    반응 중심: {pred.predicted_centers}")
        print(f"    효소 호환: {pred.enzyme_compatible}")
        print(f"    보조인자: {pred.cofactor_required}")
        print(f"    제한 요인: {', '.join(pred.limiting_factors)}")
        
        expected_str = "가능" if case['expected'] else "불가능"
        predicted_str = "가능" if pred.P_feasible >= 0.7 else "불가능"
        match = "✓" if (pred.P_feasible >= 0.7) == case['expected'] else "✗"
        
        print(f"\n  예상: {expected_str} | 예측: {predicted_str} {match}")
    
    # 성능 평가
    print("\n" + "="*70)
    print("전체 성능 평가")
    print("="*70)
    
    performance = predictor.evaluate_performance(predictions, ground_truth)
    
    for threshold, metrics in performance.items():
        if threshold.startswith("threshold"):
            print(f"\n{threshold}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    print(f"\n평균 신뢰도: {performance['average_confidence']}")
    print(f"평균 P_feasible: {performance['average_P_feasible']}")
    
    print("\n" + "="*70)
    print("Stage 1 학습 내용")
    print("="*70)
    
    print("\n1. 화학적 가능성 평가:")
    print("   • 기질-효소 호환성")
    print("   • 반응 조건 적합성")
    print("   • 보조인자 요구사항")
    
    print("\n2. 반응 중심 예측:")
    print("   • 어디서 반응이 일어나는가")
    print("   • 여러 후보 중 가장 가능성 높은 위치")
    
    print("\n3. 제한 요인 식별:")
    print("   • 무엇이 반응을 방해하는가")
    print("   • 개선 방향 제시")
    
    print("\n4. 신뢰도 정량화:")
    print("   • 예측이 얼마나 확실한가")
    print("   • 불확실성 명시")


if __name__ == "__main__":
    main()
