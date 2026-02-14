"""
Stage 2: Performance Predictor
실험 성능 예측 (수율, kcat, Km, 시간)
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Stage 1 import
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "2026-02-13_reaction-center-prediction-baseline/src"))

from masked_loss import MaskedLoss


@dataclass
class PerformancePrediction:
    """성능 예측 결과"""
    
    # 예측값 (평균)
    yield_mean: float
    kcat_mean: Optional[float]
    Km_mean: Optional[float]
    time_to_90pct_mean: float
    
    # 불확실성 (표준편차)
    yield_std: float
    kcat_std: Optional[float]
    Km_std: Optional[float]
    time_to_90pct_std: float
    
    # 데이터 소스
    kcat_source: str  # "measured", "estimated", "transferred"
    Km_source: str
    
    # 신뢰도
    confidence: float
    
    # 제한 요인
    limiting_factors: List[str]


class Stage2PerformancePredictor:
    """
    Stage 2: 성능 예측기
    
    입력: Stage 1 출력 + 실험 조건
    출력: 수율, kcat, Km, 시간
    """
    
    def __init__(self, kinetics_db: Dict = None):
        """
        Args:
            kinetics_db: BRENDA 동역학 데이터베이스
        """
        self.kinetics_db = kinetics_db or {}
        
        # 기본 추정값 (EC 클래스별)
        self.default_kinetics = {
            "1": {"kcat": 100, "Km": 0.5},  # Oxidoreductases
            "2": {"kcat": 75, "Km": 0.8},   # Transferases
            "3": {"kcat": 50, "Km": 1.0},   # Hydrolases
            "4": {"kcat": 60, "Km": 0.7},   # Lyases
            "5": {"kcat": 50, "Km": 1.0},   # Isomerases
            "6": {"kcat": 40, "Km": 1.2},   # Ligases
        }
    
    def predict_performance(
        self,
        feasibility: Dict,  # Stage 1 출력
        assay_conditions: Dict
    ) -> PerformancePrediction:
        """
        성능 예측
        
        Args:
            feasibility: Stage 1 출력
                {
                    "P_feasible": 0.914,
                    "enzyme_compatible": True,
                    "cofactor": "NAD+",
                    "reaction_type": "oxidation",
                    "enzyme_ec": "1.1.1.1"
                }
            
            assay_conditions:
                {
                    "substrate_conc": 10,  # mM
                    "enzyme_conc": 0.1,    # μM
                    "cofactor_conc": 1.0,  # mM
                    "pH": 7.4,
                    "temperature": 37
                }
        
        Returns:
            PerformancePrediction
        """
        
        # 1. 효소 동역학 예측 (kcat, Km)
        kcat, kcat_std, kcat_source = self._predict_kcat(
            feasibility["enzyme_ec"],
            assay_conditions
        )
        
        Km, Km_std, Km_source = self._predict_Km(
            feasibility["enzyme_ec"],
            assay_conditions
        )
        
        # 2. 수율 예측
        yield_mean, yield_std = self._predict_yield(
            feasibility,
            kcat,
            Km,
            assay_conditions
        )
        
        # 3. 반응 시간 예측
        time_mean, time_std = self._predict_time(
            kcat,
            Km,
            assay_conditions
        )
        
        # 4. 신뢰도 계산
        confidence = self._calculate_confidence(
            feasibility["P_feasible"],
            kcat_source,
            Km_source
        )
        
        # 5. 제한 요인 식별
        limiting_factors = self._identify_limiting_factors(
            feasibility,
            kcat,
            Km,
            assay_conditions
        )
        
        return PerformancePrediction(
            yield_mean=yield_mean,
            kcat_mean=kcat,
            Km_mean=Km,
            time_to_90pct_mean=time_mean,
            yield_std=yield_std,
            kcat_std=kcat_std,
            Km_std=Km_std,
            time_to_90pct_std=time_std,
            kcat_source=kcat_source,
            Km_source=Km_source,
            confidence=confidence,
            limiting_factors=limiting_factors
        )
    
    def _predict_kcat(
        self,
        enzyme_ec: str,
        conditions: Dict
    ) -> tuple:
        """kcat 예측"""
        
        # 1. 데이터베이스 검색
        if enzyme_ec in self.kinetics_db:
            measured_kcat = self.kinetics_db[enzyme_ec].get("kcat")
            if measured_kcat is not None:
                # 조건 보정
                kcat = self._adjust_for_conditions(
                    measured_kcat,
                    conditions,
                    "kcat"
                )
                return kcat, kcat * 0.15, "measured"
        
        # 2. EC 클래스 기본값
        ec_class = enzyme_ec.split('.')[0]
        default_kcat = self.default_kinetics.get(ec_class, {}).get("kcat", 75)
        
        # 조건 보정
        kcat = self._adjust_for_conditions(default_kcat, conditions, "kcat")
        
        return kcat, kcat * 0.3, "estimated"
    
    def _predict_Km(
        self,
        enzyme_ec: str,
        conditions: Dict
    ) -> tuple:
        """Km 예측"""
        
        # 1. 데이터베이스 검색
        if enzyme_ec in self.kinetics_db:
            measured_Km = self.kinetics_db[enzyme_ec].get("Km")
            if measured_Km is not None:
                Km = self._adjust_for_conditions(
                    measured_Km,
                    conditions,
                    "Km"
                )
                return Km, Km * 0.15, "measured"
        
        # 2. EC 클래스 기본값
        ec_class = enzyme_ec.split('.')[0]
        default_Km = self.default_kinetics.get(ec_class, {}).get("Km", 0.8)
        
        Km = self._adjust_for_conditions(default_Km, conditions, "Km")
        
        return Km, Km * 0.3, "estimated"
    
    def _adjust_for_conditions(
        self,
        base_value: float,
        conditions: Dict,
        parameter: str
    ) -> float:
        """조건에 따른 파라미터 보정"""
        
        pH = conditions.get("pH", 7.4)
        temp = conditions.get("temperature", 37)
        
        # pH 효과 (최적 pH 7.4 기준)
        pH_factor = 1.0
        if abs(pH - 7.4) > 0.5:
            pH_factor = max(0.5, 1.0 - abs(pH - 7.4) * 0.1)
        
        # 온도 효과 (최적 온도 37°C 기준)
        temp_factor = 1.0
        if parameter == "kcat":
            # kcat는 온도에 비례 (Q10 = 2)
            temp_factor = 2 ** ((temp - 37) / 10)
            temp_factor = min(2.0, max(0.3, temp_factor))
        
        return base_value * pH_factor * temp_factor
    
    def _predict_yield(
        self,
        feasibility: Dict,
        kcat: float,
        Km: float,
        conditions: Dict
    ) -> tuple:
        """수율 예측"""
        
        # 기본 수율 (가능성에 비례)
        base_yield = feasibility["P_feasible"] * 0.9
        
        # 기질 농도 효과 (Michaelis-Menten)
        S = conditions.get("substrate_conc", 10)  # mM
        saturation = S / (S + Km)
        
        # 효소 농도 효과
        E = conditions.get("enzyme_conc", 0.1)  # μM
        enzyme_factor = min(1.0, E / 0.1)  # 기준 0.1 μM
        
        # 최종 수율
        yield_mean = base_yield * saturation * enzyme_factor
        yield_mean = min(0.95, max(0.1, yield_mean))
        
        # 불확실성
        yield_std = 0.05 + (1 - feasibility["P_feasible"]) * 0.1
        
        return yield_mean, yield_std
    
    def _predict_time(
        self,
        kcat: float,
        Km: float,
        conditions: Dict
    ) -> tuple:
        """반응 시간 예측 (90% 완료)"""
        
        S0 = conditions.get("substrate_conc", 10)  # mM
        E = conditions.get("enzyme_conc", 0.1)  # μM
        
        # Michaelis-Menten 속도
        v_max = kcat * E  # mM/s
        v0 = v_max * S0 / (S0 + Km)
        
        # 90% 완료 시간 (간단한 추정)
        # t ≈ -ln(0.1) / (v0 / S0)
        if v0 > 0:
            time_mean = 2.3 * S0 / v0 / 3600  # hours
            time_mean = max(0.1, min(24, time_mean))
        else:
            time_mean = 24.0
        
        time_std = time_mean * 0.2
        
        return time_mean, time_std
    
    def _calculate_confidence(
        self,
        P_feasible: float,
        kcat_source: str,
        Km_source: str
    ) -> float:
        """신뢰도 계산"""
        
        # 가능성 기여
        feasibility_conf = P_feasible
        
        # 데이터 소스 기여
        source_scores = {
            "measured": 0.9,
            "transferred": 0.6,
            "estimated": 0.3
        }
        
        kcat_conf = source_scores.get(kcat_source, 0.3)
        Km_conf = source_scores.get(Km_source, 0.3)
        
        # 종합 신뢰도
        confidence = (
            feasibility_conf * 0.4 +
            kcat_conf * 0.3 +
            Km_conf * 0.3
        )
        
        return confidence
    
    def _identify_limiting_factors(
        self,
        feasibility: Dict,
        kcat: float,
        Km: float,
        conditions: Dict
    ) -> List[str]:
        """제한 요인 식별"""
        
        factors = []
        
        # 1. 낮은 가능성
        if feasibility["P_feasible"] < 0.7:
            factors.append("low_feasibility")
        
        # 2. 낮은 kcat
        if kcat < 10:
            factors.append("low_kcat")
        
        # 3. 높은 Km (낮은 친화도)
        if Km > 5.0:
            factors.append("high_Km")
        
        # 4. 낮은 기질 농도
        S = conditions.get("substrate_conc", 10)
        if S < Km:
            factors.append("substrate_limited")
        
        # 5. 낮은 효소 농도
        E = conditions.get("enzyme_conc", 0.1)
        if E < 0.05:
            factors.append("enzyme_limited")
        
        if not factors:
            factors.append("none")
        
        return factors


def demo_stage2():
    """Stage 2 예측기 데모"""
    
    print("="*70)
    print("Stage 2: Performance Predictor 데모")
    print("="*70)
    
    # 예측기 초기화
    predictor = Stage2PerformancePredictor()
    
    # 테스트 케이스
    test_cases = [
        {
            "name": "에탄올 산화 (최적 조건)",
            "feasibility": {
                "P_feasible": 0.914,
                "enzyme_compatible": True,
                "cofactor": "NAD+",
                "reaction_type": "oxidation",
                "enzyme_ec": "1.1.1.1"
            },
            "conditions": {
                "substrate_conc": 10,
                "enzyme_conc": 0.1,
                "cofactor_conc": 1.0,
                "pH": 7.4,
                "temperature": 37
            }
        },
        {
            "name": "글리세롤 산화 (낮은 효소)",
            "feasibility": {
                "P_feasible": 0.913,
                "enzyme_compatible": True,
                "cofactor": "NAD+",
                "reaction_type": "oxidation",
                "enzyme_ec": "1.1.1.6"
            },
            "conditions": {
                "substrate_conc": 10,
                "enzyme_conc": 0.02,  # 낮음
                "cofactor_conc": 1.0,
                "pH": 7.4,
                "temperature": 37
            }
        },
        {
            "name": "이성질화 (낮은 온도)",
            "feasibility": {
                "P_feasible": 0.85,
                "enzyme_compatible": True,
                "cofactor": None,
                "reaction_type": "isomerization",
                "enzyme_ec": "5.3.1.5"
            },
            "conditions": {
                "substrate_conc": 10,
                "enzyme_conc": 0.1,
                "pH": 7.4,
                "temperature": 25  # 낮음
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['name']}")
        print("-"*70)
        
        pred = predictor.predict_performance(
            case["feasibility"],
            case["conditions"]
        )
        
        print(f"예측 결과:")
        print(f"  수율: {pred.yield_mean:.1%} ± {pred.yield_std:.1%}")
        
        if pred.kcat_mean:
            print(f"  kcat: {pred.kcat_mean:.1f} ± {pred.kcat_std:.1f} s⁻¹ ({pred.kcat_source})")
        
        if pred.Km_mean:
            print(f"  Km: {pred.Km_mean:.2f} ± {pred.Km_std:.2f} mM ({pred.Km_source})")
        
        print(f"  반응 시간 (90%): {pred.time_to_90pct_mean:.2f} ± {pred.time_to_90pct_std:.2f} hours")
        print(f"  신뢰도: {pred.confidence:.3f}")
        print(f"  제한 요인: {', '.join(pred.limiting_factors)}")
    
    print("\n" + "="*70)
    print("Stage 2 학습 내용")
    print("="*70)
    print("1. 효소 동역학 예측 (kcat, Km)")
    print("2. 수율 예측 (기질 농도, 효소 농도 고려)")
    print("3. 반응 시간 예측 (Michaelis-Menten)")
    print("4. 불확실성 정량화 (측정 vs 추정)")
    print("5. 제한 요인 식별 (개선 방향)")


if __name__ == "__main__":
    demo_stage2()
