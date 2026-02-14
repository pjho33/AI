"""
조건 인식 반응 예측기 (Condition-Aware Predictor)
pH, 온도, 용매 등 반응 조건을 고려한 예측
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).parent))
from rules.rule_based_predictor import RuleBasedPredictor, ReactionPrediction


@dataclass
class ReactionConditions:
    """반응 조건"""
    pH: float
    temperature: float
    solvent: str
    ionic_strength: float = 0.15
    cofactor_concentration: Optional[Dict[str, float]] = None
    metal_ions: Optional[Dict[str, float]] = None


@dataclass
class ConditionAwarePrediction(ReactionPrediction):
    """조건을 고려한 예측"""
    condition_score: float = 1.0
    activity_at_conditions: float = 100.0
    limiting_factor: Optional[str] = None


class ConditionAwarePredictor:
    """조건을 고려한 반응 예측기"""
    
    def __init__(self):
        self.base_predictor = RuleBasedPredictor()
        self.condition_effects = self._load_condition_effects()
    
    def _load_condition_effects(self) -> Dict:
        """조건 효과 규칙 로딩"""
        return {
            "pH_effects": {
                "EC_1.1.1": {
                    "optimal": 7.5,
                    "range": (6.0, 9.0),
                    "tolerance": 1.5
                },
                "EC_5.3.1": {
                    "optimal": 7.5,
                    "range": (6.5, 8.5),
                    "tolerance": 1.0
                }
            },
            "temperature_effects": {
                "optimal": 37,
                "Q10": 2.0,
                "denaturation_start": 50,
                "cold_penalty": 0.05
            },
            "solvent_effects": {
                "water": 1.0,
                "water:ethanol": 0.3,
                "organic": 0.1
            },
            "ionic_strength_effects": {
                "optimal": 0.15,
                "high_salt_penalty": 0.4
            }
        }
    
    def predict_with_conditions(
        self, 
        smiles: str, 
        reaction_type: str,
        conditions: ReactionConditions
    ) -> List[ConditionAwarePrediction]:
        """조건을 고려한 반응 예측"""
        
        base_predictions = self.base_predictor.predict_reaction_centers(
            smiles, reaction_type
        )
        
        condition_aware_predictions = []
        
        for pred in base_predictions:
            condition_score = self._calculate_condition_score(
                pred, conditions
            )
            
            activity = self._estimate_enzyme_activity(
                pred, conditions
            )
            
            limiting_factor = self._identify_limiting_factor(
                pred, conditions
            )
            
            adjusted_confidence = pred.confidence * condition_score
            
            ca_pred = ConditionAwarePrediction(
                atom_indices=pred.atom_indices,
                rule_name=pred.rule_name,
                ec_class=pred.ec_class,
                confidence=adjusted_confidence,
                cofactor=pred.cofactor,
                condition_score=condition_score,
                activity_at_conditions=activity,
                limiting_factor=limiting_factor
            )
            
            condition_aware_predictions.append(ca_pred)
        
        condition_aware_predictions.sort(
            key=lambda x: x.confidence, reverse=True
        )
        
        return condition_aware_predictions
    
    def _calculate_condition_score(
        self, 
        prediction: ReactionPrediction,
        conditions: ReactionConditions
    ) -> float:
        """조건 점수 계산"""
        
        score = 1.0
        
        score *= self._pH_score(prediction.ec_class, conditions.pH)
        score *= self._temperature_score(conditions.temperature)
        score *= self._solvent_score(conditions.solvent)
        score *= self._ionic_strength_score(conditions.ionic_strength)
        
        return score
    
    def _pH_score(self, ec_class: str, pH: float) -> float:
        """pH 효과 점수"""
        ec_main = ec_class.split('.')[0] if '.' in ec_class else ec_class
        ec_key = f"EC_{ec_main}"
        
        if ec_key not in self.condition_effects["pH_effects"]:
            ec_key = "EC_1.1.1"
        
        pH_params = self.condition_effects["pH_effects"][ec_key]
        optimal = pH_params["optimal"]
        tolerance = pH_params["tolerance"]
        
        deviation = abs(pH - optimal)
        
        if deviation == 0:
            return 1.0
        elif deviation <= tolerance:
            return 1.0 - (deviation / tolerance) * 0.3
        elif deviation <= tolerance * 2:
            return 0.7 - (deviation - tolerance) / tolerance * 0.5
        else:
            return 0.2
    
    def _temperature_score(self, temperature: float) -> float:
        """온도 효과 점수"""
        optimal = self.condition_effects["temperature_effects"]["optimal"]
        Q10 = self.condition_effects["temperature_effects"]["Q10"]
        
        if temperature == optimal:
            return 1.0
        elif temperature < optimal:
            delta = optimal - temperature
            penalty = self.condition_effects["temperature_effects"]["cold_penalty"]
            return max(0.1, 1.0 - delta * penalty)
        else:
            delta = temperature - optimal
            if temperature >= 50:
                return 0.2
            return max(0.5, 1.0 - delta * 0.03)
    
    def _solvent_score(self, solvent: str) -> float:
        """용매 효과 점수"""
        if solvent == "water":
            return 1.0
        elif "ethanol" in solvent.lower():
            return 0.3
        else:
            return 0.5
    
    def _ionic_strength_score(self, ionic_strength: float) -> float:
        """이온 강도 효과 점수"""
        optimal = self.condition_effects["ionic_strength_effects"]["optimal"]
        
        if abs(ionic_strength - optimal) < 0.05:
            return 1.0
        elif ionic_strength > 0.5:
            penalty = self.condition_effects["ionic_strength_effects"]["high_salt_penalty"]
            return 1.0 - penalty
        else:
            return 0.9
    
    def _estimate_enzyme_activity(
        self,
        prediction: ReactionPrediction,
        conditions: ReactionConditions
    ) -> float:
        """조건에서의 효소 활성 추정"""
        
        base_activity = 100.0
        
        pH_factor = self._pH_score(prediction.ec_class, conditions.pH)
        temp_factor = self._temperature_score(conditions.temperature)
        solvent_factor = self._solvent_score(conditions.solvent)
        ionic_factor = self._ionic_strength_score(conditions.ionic_strength)
        
        activity = base_activity * pH_factor * temp_factor * solvent_factor * ionic_factor
        
        return round(activity, 1)
    
    def _identify_limiting_factor(
        self,
        prediction: ReactionPrediction,
        conditions: ReactionConditions
    ) -> Optional[str]:
        """제한 인자 식별"""
        
        factors = {
            "pH": self._pH_score(prediction.ec_class, conditions.pH),
            "temperature": self._temperature_score(conditions.temperature),
            "solvent": self._solvent_score(conditions.solvent),
            "ionic_strength": self._ionic_strength_score(conditions.ionic_strength)
        }
        
        min_factor = min(factors.items(), key=lambda x: x[1])
        
        if min_factor[1] < 0.7:
            return min_factor[0]
        
        return None
    
    def compare_conditions(
        self,
        smiles: str,
        reaction_type: str,
        condition_list: List[ReactionConditions]
    ) -> Dict:
        """여러 조건 비교"""
        
        results = {}
        
        for i, conditions in enumerate(condition_list):
            predictions = self.predict_with_conditions(
                smiles, reaction_type, conditions
            )
            
            if predictions:
                best_pred = predictions[0]
                results[f"condition_{i+1}"] = {
                    "conditions": {
                        "pH": conditions.pH,
                        "temperature": conditions.temperature,
                        "solvent": conditions.solvent,
                        "ionic_strength": conditions.ionic_strength
                    },
                    "predicted_activity": best_pred.activity_at_conditions,
                    "confidence": best_pred.confidence,
                    "limiting_factor": best_pred.limiting_factor
                }
        
        best_condition = max(
            results.items(),
            key=lambda x: x[1]["predicted_activity"]
        )
        
        return {
            "all_conditions": results,
            "best_condition": best_condition[0],
            "best_activity": best_condition[1]["predicted_activity"]
        }


def main():
    """조건 인식 예측 데모"""
    
    print("="*70)
    print("조건 인식 반응 예측 시스템 (Phase 2)")
    print("="*70)
    
    predictor = ConditionAwarePredictor()
    
    test_molecule = "CCO"
    reaction_type = "oxidation"
    
    print(f"\n테스트 분자: 에탄올 (CCO)")
    print(f"반응 유형: {reaction_type}")
    
    conditions_list = [
        ("생리적 조건", ReactionConditions(
            pH=7.4, temperature=37, solvent="water", ionic_strength=0.15
        )),
        ("산성 조건", ReactionConditions(
            pH=5.0, temperature=37, solvent="water", ionic_strength=0.15
        )),
        ("염기성 조건", ReactionConditions(
            pH=9.0, temperature=37, solvent="water", ionic_strength=0.15
        )),
        ("저온 조건", ReactionConditions(
            pH=7.4, temperature=4, solvent="water", ionic_strength=0.15
        )),
        ("고온 조건", ReactionConditions(
            pH=7.4, temperature=50, solvent="water", ionic_strength=0.15
        )),
        ("유기용매", ReactionConditions(
            pH=7.0, temperature=25, solvent="water:ethanol", ionic_strength=0.15
        ))
    ]
    
    print("\n" + "="*70)
    print("조건별 예측 결과")
    print("="*70)
    
    for name, conditions in conditions_list:
        print(f"\n[{name}]")
        print(f"  pH: {conditions.pH}, 온도: {conditions.temperature}°C")
        print(f"  용매: {conditions.solvent}, 이온강도: {conditions.ionic_strength}M")
        
        predictions = predictor.predict_with_conditions(
            test_molecule, reaction_type, conditions
        )
        
        if predictions:
            pred = predictions[0]
            print(f"\n  예측 활성: {pred.activity_at_conditions:.1f}%")
            print(f"  신뢰도: {pred.confidence:.3f}")
            print(f"  조건 점수: {pred.condition_score:.3f}")
            if pred.limiting_factor:
                print(f"  제한 인자: {pred.limiting_factor}")
    
    print("\n" + "="*70)
    print("최적 조건 찾기")
    print("="*70)
    
    comparison = predictor.compare_conditions(
        test_molecule, 
        reaction_type,
        [cond for _, cond in conditions_list]
    )
    
    print(f"\n최적 조건: {comparison['best_condition']}")
    print(f"예상 활성: {comparison['best_activity']:.1f}%")
    
    print("\n" + "="*70)
    print("AI가 배운 것")
    print("="*70)
    
    print("\n1. pH 의존성:")
    print("   • pH 7.4 (생리적): 최적 활성")
    print("   • pH 5.0 (산성): 활성 감소")
    print("   • pH 9.0 (염기성): 활성 유지")
    
    print("\n2. 온도 의존성:")
    print("   • 37°C: 최적 온도")
    print("   • 4°C: 활성 크게 감소")
    print("   • 50°C: 효소 변성 시작")
    
    print("\n3. 용매 효과:")
    print("   • 물: 최적")
    print("   • 유기용매 혼합: 활성 감소")
    
    print("\n4. 조건 최적화:")
    print("   • 제한 인자 식별 가능")
    print("   • 최적 조건 추천 가능")
    print("   • 조건별 활성 예측 가능")


if __name__ == "__main__":
    main()
