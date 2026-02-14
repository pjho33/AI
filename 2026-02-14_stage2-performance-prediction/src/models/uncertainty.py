"""
불확실성 정량화 (Uncertainty Quantification)
예측의 신뢰도를 정량적으로 표현
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class UncertaintyComponents:
    """불확실성 구성 요소"""
    measurement_uncertainty: float  # 측정 불확실성
    model_uncertainty: float        # 모델 불확실성
    missing_data_uncertainty: float # 결측 데이터 불확실성
    total_uncertainty: float        # 전체 불확실성


class UncertaintyQuantifier:
    """
    불확실성 정량화
    
    3가지 불확실성 소스:
    1. 측정 불확실성: 실험 데이터의 변동성
    2. 모델 불확실성: 예측 모델의 한계
    3. 결측 데이터 불확실성: 추정값의 불확실성
    """
    
    @staticmethod
    def propagate_uncertainty(
        value: float,
        uncertainties: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> Tuple[float, float]:
        """
        불확실성 전파
        
        Args:
            value: 예측값
            uncertainties: {"measurement": 0.1, "model": 0.2, "missing": 0.3}
            weights: 각 불확실성의 가중치
        
        Returns:
            (mean, total_std)
        """
        
        if weights is None:
            weights = {k: 1.0 for k in uncertainties.keys()}
        
        # 분산의 합 (독립적 가정)
        total_variance = 0.0
        
        for source, uncertainty in uncertainties.items():
            weight = weights.get(source, 1.0)
            total_variance += (weight * uncertainty) ** 2
        
        total_std = np.sqrt(total_variance)
        
        return value, total_std
    
    @staticmethod
    def calculate_measurement_uncertainty(
        measurements: List[float],
        default_cv: float = 0.15
    ) -> float:
        """
        측정 불확실성 계산
        
        Args:
            measurements: 측정값 리스트
            default_cv: 기본 변동계수 (coefficient of variation)
        
        Returns:
            측정 불확실성 (표준편차)
        """
        
        if len(measurements) > 1:
            # 실제 측정값이 여러 개 있으면 표준편차 계산
            return np.std(measurements)
        elif len(measurements) == 1:
            # 측정값이 하나면 기본 CV 사용
            return measurements[0] * default_cv
        else:
            # 측정값 없음
            return 0.0
    
    @staticmethod
    def calculate_model_uncertainty(
        prediction: float,
        training_error: float = 0.2
    ) -> float:
        """
        모델 불확실성 계산
        
        Args:
            prediction: 예측값
            training_error: 학습 오차 (상대값)
        
        Returns:
            모델 불확실성
        """
        
        return prediction * training_error
    
    @staticmethod
    def calculate_missing_data_uncertainty(
        prediction: float,
        data_source: str,
        completeness: float = 0.5
    ) -> float:
        """
        결측 데이터 불확실성 계산
        
        Args:
            prediction: 예측값
            data_source: "measured", "transferred", "estimated"
            completeness: 데이터 완전성 (0-1)
        
        Returns:
            결측 데이터 불확실성
        """
        
        # 데이터 소스별 기본 불확실성
        base_uncertainty = {
            "measured": 0.1,      # 측정값: 낮은 불확실성
            "transferred": 0.3,   # 전이 학습: 중간 불확실성
            "estimated": 0.5      # 추정값: 높은 불확실성
        }
        
        base = base_uncertainty.get(data_source, 0.5)
        
        # 데이터 완전성에 반비례
        uncertainty = base * (1.0 + (1.0 - completeness))
        
        return prediction * uncertainty
    
    @staticmethod
    def quantify_total_uncertainty(
        prediction: float,
        measurements: List[float],
        data_source: str,
        completeness: float,
        training_error: float = 0.2
    ) -> UncertaintyComponents:
        """
        전체 불확실성 정량화
        
        Args:
            prediction: 예측값
            measurements: 측정값 리스트
            data_source: "measured", "transferred", "estimated"
            completeness: 데이터 완전성
            training_error: 모델 학습 오차
        
        Returns:
            UncertaintyComponents
        """
        
        # 1. 측정 불확실성
        meas_unc = UncertaintyQuantifier.calculate_measurement_uncertainty(
            measurements
        )
        
        # 2. 모델 불확실성
        model_unc = UncertaintyQuantifier.calculate_model_uncertainty(
            prediction,
            training_error
        )
        
        # 3. 결측 데이터 불확실성
        missing_unc = UncertaintyQuantifier.calculate_missing_data_uncertainty(
            prediction,
            data_source,
            completeness
        )
        
        # 4. 전체 불확실성 (분산의 합)
        total_unc = np.sqrt(
            meas_unc ** 2 +
            model_unc ** 2 +
            missing_unc ** 2
        )
        
        return UncertaintyComponents(
            measurement_uncertainty=meas_unc,
            model_uncertainty=model_unc,
            missing_data_uncertainty=missing_unc,
            total_uncertainty=total_unc
        )
    
    @staticmethod
    def confidence_interval(
        mean: float,
        std: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        신뢰 구간 계산
        
        Args:
            mean: 평균
            std: 표준편차
            confidence_level: 신뢰 수준 (0.95 = 95%)
        
        Returns:
            (lower_bound, upper_bound)
        """
        
        # 정규분포 가정
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        z = z_scores.get(confidence_level, 1.96)
        
        lower = mean - z * std
        upper = mean + z * std
        
        return lower, upper


def demo_uncertainty():
    """불확실성 정량화 데모"""
    
    print("="*70)
    print("불확실성 정량화 데모")
    print("="*70)
    
    quantifier = UncertaintyQuantifier()
    
    # 테스트 케이스
    test_cases = [
        {
            "name": "측정값 (높은 신뢰도)",
            "prediction": 100.0,
            "measurements": [95, 102, 98, 105, 100],
            "data_source": "measured",
            "completeness": 0.9
        },
        {
            "name": "전이 학습 (중간 신뢰도)",
            "prediction": 100.0,
            "measurements": [100],
            "data_source": "transferred",
            "completeness": 0.5
        },
        {
            "name": "추정값 (낮은 신뢰도)",
            "prediction": 100.0,
            "measurements": [],
            "data_source": "estimated",
            "completeness": 0.2
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['name']}")
        print("-"*70)
        
        unc = quantifier.quantify_total_uncertainty(
            prediction=case["prediction"],
            measurements=case["measurements"],
            data_source=case["data_source"],
            completeness=case["completeness"]
        )
        
        print(f"예측값: {case['prediction']:.1f}")
        print(f"\n불확실성 구성:")
        print(f"  측정 불확실성: {unc.measurement_uncertainty:.2f}")
        print(f"  모델 불확실성: {unc.model_uncertainty:.2f}")
        print(f"  결측 불확실성: {unc.missing_data_uncertainty:.2f}")
        print(f"  전체 불확실성: {unc.total_uncertainty:.2f}")
        
        # 신뢰 구간
        lower, upper = quantifier.confidence_interval(
            case["prediction"],
            unc.total_uncertainty,
            confidence_level=0.95
        )
        
        print(f"\n95% 신뢰 구간: [{lower:.1f}, {upper:.1f}]")
        print(f"상대 불확실성: {unc.total_uncertainty/case['prediction']*100:.1f}%")
    
    print("\n" + "="*70)
    print("불확실성 비교")
    print("="*70)
    
    print("\n데이터 소스별 불확실성:")
    print("  측정값 (measured):    ~15-20%")
    print("  전이 학습 (transferred): ~40-50%")
    print("  추정값 (estimated):    ~80-100%")
    
    print("\n불확실성 감소 전략:")
    print("  1. 더 많은 측정 데이터 수집")
    print("  2. 모델 정확도 향상")
    print("  3. 전이 학습 활용")
    print("  4. 앙상블 예측")
    
    print("\n" + "="*70)
    print("실용적 활용")
    print("="*70)
    
    print("\n예측값 보고 형식:")
    print("  kcat = 100 ± 20 s⁻¹ (95% CI: [60, 140])")
    print("  데이터 소스: transferred")
    print("  신뢰도: medium")
    
    print("\n의사결정 가이드:")
    print("  불확실성 < 20%: 높은 신뢰도 → 실험 진행")
    print("  불확실성 20-50%: 중간 신뢰도 → 추가 검증 필요")
    print("  불확실성 > 50%: 낮은 신뢰도 → 더 많은 데이터 필요")


if __name__ == "__main__":
    demo_uncertainty()
