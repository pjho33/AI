"""
Stage 2 평가 및 검증
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "models"))
from masked_loss import MaskedLoss


@dataclass
class EvaluationMetrics:
    """평가 메트릭"""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2: float  # R-squared
    coverage: float  # 신뢰 구간 커버리지
    calibration: float  # 불확실성 보정 정확도


class Stage2Evaluator:
    """
    Stage 2 평가기
    
    평가 항목:
    1. 예측 정확도 (MAE, RMSE, R²)
    2. 불확실성 보정 (신뢰 구간 커버리지)
    3. 데이터 소스별 성능
    """
    
    @staticmethod
    def evaluate_predictions(
        predictions: np.ndarray,
        actuals: np.ndarray,
        uncertainties: np.ndarray = None,
        mask: np.ndarray = None
    ) -> EvaluationMetrics:
        """
        예측 평가
        
        Args:
            predictions: 예측값 [N,]
            actuals: 실제값 [N,]
            uncertainties: 불확실성 [N,]
            mask: 마스크 [N,] (1=측정됨, 0=결측)
        
        Returns:
            EvaluationMetrics
        """
        
        # 마스크 적용
        if mask is not None:
            valid_idx = mask == 1
            predictions = predictions[valid_idx]
            actuals = actuals[valid_idx]
            if uncertainties is not None:
                uncertainties = uncertainties[valid_idx]
        
        if len(predictions) == 0:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0)
        
        # 1. MAE
        mae = np.mean(np.abs(predictions - actuals))
        
        # 2. RMSE
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # 3. MAPE
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        # 4. R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 5. 신뢰 구간 커버리지
        coverage = 0.0
        if uncertainties is not None:
            # 95% 신뢰 구간
            lower = predictions - 1.96 * uncertainties
            upper = predictions + 1.96 * uncertainties
            within_ci = (actuals >= lower) & (actuals <= upper)
            coverage = np.mean(within_ci)
        
        # 6. 불확실성 보정
        calibration = 0.0
        if uncertainties is not None:
            # 표준화된 오차
            z_scores = (predictions - actuals) / uncertainties
            # 정규분포 가정: 68% should be within ±1σ
            within_1sigma = np.abs(z_scores) <= 1
            calibration = np.mean(within_1sigma)
        
        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            coverage=coverage,
            calibration=calibration
        )
    
    @staticmethod
    def evaluate_by_source(
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        uncertainties: Dict[str, np.ndarray] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        데이터 소스별 평가
        
        Args:
            predictions: {"measured": [...], "transferred": [...], "estimated": [...]}
            actuals: {"measured": [...], "transferred": [...], "estimated": [...]}
            uncertainties: {"measured": [...], "transferred": [...], "estimated": [...]}
        
        Returns:
            {"measured": metrics, "transferred": metrics, "estimated": metrics}
        """
        
        results = {}
        
        for source in predictions.keys():
            pred = predictions[source]
            actual = actuals[source]
            unc = uncertainties.get(source) if uncertainties else None
            
            metrics = Stage2Evaluator.evaluate_predictions(pred, actual, unc)
            results[source] = metrics
        
        return results
    
    @staticmethod
    def compare_with_baseline(
        model_predictions: np.ndarray,
        baseline_predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        베이스라인과 비교
        
        Args:
            model_predictions: 모델 예측
            baseline_predictions: 베이스라인 예측 (평균 등)
            actuals: 실제값
        
        Returns:
            {"improvement_mae": 0.2, "improvement_rmse": 0.15}
        """
        
        model_mae = np.mean(np.abs(model_predictions - actuals))
        baseline_mae = np.mean(np.abs(baseline_predictions - actuals))
        
        model_rmse = np.sqrt(np.mean((model_predictions - actuals) ** 2))
        baseline_rmse = np.sqrt(np.mean((baseline_predictions - actuals) ** 2))
        
        improvement_mae = (baseline_mae - model_mae) / baseline_mae
        improvement_rmse = (baseline_rmse - model_rmse) / baseline_rmse
        
        return {
            "model_mae": model_mae,
            "baseline_mae": baseline_mae,
            "improvement_mae": improvement_mae,
            "model_rmse": model_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_rmse": improvement_rmse
        }


def demo_evaluation():
    """평가 시스템 데모"""
    
    print("="*70)
    print("Stage 2 평가 시스템 데모")
    print("="*70)
    
    # 시뮬레이션 데이터
    np.random.seed(42)
    n_samples = 50
    
    # 실제값
    actuals = np.random.uniform(50, 150, n_samples)
    
    # 예측값 (노이즈 추가)
    predictions = actuals + np.random.normal(0, 15, n_samples)
    
    # 불확실성
    uncertainties = np.random.uniform(10, 20, n_samples)
    
    # 마스크 (70% 측정)
    mask = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    print("\n1. 전체 평가")
    print("-"*70)
    
    metrics = Stage2Evaluator.evaluate_predictions(
        predictions,
        actuals,
        uncertainties,
        mask
    )
    
    print(f"MAE: {metrics.mae:.2f}")
    print(f"RMSE: {metrics.rmse:.2f}")
    print(f"MAPE: {metrics.mape:.1f}%")
    print(f"R²: {metrics.r2:.3f}")
    print(f"신뢰 구간 커버리지: {metrics.coverage:.1%}")
    print(f"불확실성 보정: {metrics.calibration:.1%}")
    
    print("\n2. 데이터 소스별 평가")
    print("-"*70)
    
    # 데이터 소스별로 분리
    n_per_source = n_samples // 3
    
    predictions_by_source = {
        "measured": predictions[:n_per_source] + np.random.normal(0, 5, n_per_source),
        "transferred": predictions[n_per_source:2*n_per_source] + np.random.normal(0, 15, n_per_source),
        "estimated": predictions[2*n_per_source:3*n_per_source] + np.random.normal(0, 30, n_per_source)
    }
    
    actuals_by_source = {
        "measured": actuals[:n_per_source],
        "transferred": actuals[n_per_source:2*n_per_source],
        "estimated": actuals[2*n_per_source:3*n_per_source]
    }
    
    uncertainties_by_source = {
        "measured": np.ones(n_per_source) * 10,
        "transferred": np.ones(n_per_source) * 20,
        "estimated": np.ones(n_per_source) * 40
    }
    
    source_metrics = Stage2Evaluator.evaluate_by_source(
        predictions_by_source,
        actuals_by_source,
        uncertainties_by_source
    )
    
    for source, metrics in source_metrics.items():
        print(f"\n{source}:")
        print(f"  MAE: {metrics.mae:.2f}")
        print(f"  RMSE: {metrics.rmse:.2f}")
        print(f"  R²: {metrics.r2:.3f}")
        print(f"  커버리지: {metrics.coverage:.1%}")
    
    print("\n3. 베이스라인 비교")
    print("-"*70)
    
    # 베이스라인: 평균값 예측
    baseline = np.ones(n_samples) * np.mean(actuals)
    
    comparison = Stage2Evaluator.compare_with_baseline(
        predictions,
        baseline,
        actuals
    )
    
    print(f"모델 MAE: {comparison['model_mae']:.2f}")
    print(f"베이스라인 MAE: {comparison['baseline_mae']:.2f}")
    print(f"개선율: {comparison['improvement_mae']:.1%}")
    
    print(f"\n모델 RMSE: {comparison['model_rmse']:.2f}")
    print(f"베이스라인 RMSE: {comparison['baseline_rmse']:.2f}")
    print(f"개선율: {comparison['improvement_rmse']:.1%}")
    
    print("\n" + "="*70)
    print("평가 기준")
    print("="*70)
    
    print("\n정확도:")
    print("  MAE < 20%: 우수")
    print("  MAE 20-30%: 양호")
    print("  MAE > 30%: 개선 필요")
    
    print("\n불확실성 보정:")
    print("  커버리지 > 90%: 잘 보정됨")
    print("  커버리지 80-90%: 보통")
    print("  커버리지 < 80%: 과소평가")
    
    print("\nR²:")
    print("  R² > 0.8: 우수")
    print("  R² 0.6-0.8: 양호")
    print("  R² < 0.6: 개선 필요")
    
    print("\n" + "="*70)
    print("Stage 2 성공 기준")
    print("="*70)
    
    print("\n최소 목표:")
    print("  MAE < 30% (kcat, Km)")
    print("  수율 MAE < 15%")
    print("  커버리지 > 70%")
    
    print("\n이상적 목표:")
    print("  MAE < 20% (kcat, Km)")
    print("  수율 MAE < 10%")
    print("  커버리지 > 85%")


if __name__ == "__main__":
    demo_evaluation()
