"""
Masked Loss 구현
결측 데이터 대응
"""

import numpy as np
from typing import Dict, List, Tuple


class MaskedLoss:
    """
    결측 데이터를 처리하는 손실 함수
    
    핵심 아이디어:
    - 측정된 값만 손실 계산에 포함
    - 결측값은 무시
    - 불확실성 증가
    """
    
    @staticmethod
    def masked_mse(
        predicted: np.ndarray,
        actual: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Masked Mean Squared Error
        
        Args:
            predicted: 예측값 [N,]
            actual: 실제값 [N,]
            mask: 마스크 [N,] (1=측정됨, 0=결측)
        
        Returns:
            float: MSE (측정값만)
        """
        
        if mask.sum() == 0:
            return 0.0  # 모두 결측
        
        error = (predicted - actual) ** 2
        masked_error = error * mask
        
        return masked_error.sum() / mask.sum()
    
    @staticmethod
    def masked_mae(
        predicted: np.ndarray,
        actual: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Masked Mean Absolute Error
        
        Args:
            predicted: 예측값 [N,]
            actual: 실제값 [N,]
            mask: 마스크 [N,] (1=측정됨, 0=결측)
        
        Returns:
            float: MAE (측정값만)
        """
        
        if mask.sum() == 0:
            return 0.0
        
        error = np.abs(predicted - actual)
        masked_error = error * mask
        
        return masked_error.sum() / mask.sum()
    
    @staticmethod
    def multi_target_masked_loss(
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        weights: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        여러 타겟에 대한 Masked Loss
        
        Args:
            predictions: {"kcat": [...], "Km": [...], "yield": [...]}
            actuals: {"kcat": [...], "Km": [...], "yield": [...]}
            masks: {"kcat": [...], "Km": [...], "yield": [...]}
            weights: 타겟별 가중치 (기본: 균등)
        
        Returns:
            (total_loss, individual_losses)
        """
        
        if weights is None:
            weights = {k: 1.0 for k in predictions.keys()}
        
        individual_losses = {}
        total_loss = 0.0
        total_weight = 0.0
        
        for target in predictions.keys():
            pred = predictions[target]
            actual = actuals[target]
            mask = masks[target]
            weight = weights.get(target, 1.0)
            
            # 타겟별 손실
            loss = MaskedLoss.masked_mse(pred, actual, mask)
            individual_losses[target] = loss
            
            # 가중 합산
            if mask.sum() > 0:
                total_loss += weight * loss
                total_weight += weight
        
        # 정규화
        if total_weight > 0:
            total_loss /= total_weight
        
        return total_loss, individual_losses
    
    @staticmethod
    def calculate_data_completeness(
        masks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        데이터 완전성 계산
        
        Args:
            masks: {"kcat": [...], "Km": [...]}
        
        Returns:
            {"kcat": 0.6, "Km": 0.4, "overall": 0.5}
        """
        
        completeness = {}
        
        for target, mask in masks.items():
            if len(mask) > 0:
                completeness[target] = mask.sum() / len(mask)
            else:
                completeness[target] = 0.0
        
        # 전체 완전성
        if masks:
            completeness["overall"] = np.mean(list(completeness.values()))
        else:
            completeness["overall"] = 0.0
        
        return completeness


def demo_masked_loss():
    """Masked Loss 데모"""
    
    print("="*70)
    print("Masked Loss 데모")
    print("="*70)
    
    # 시뮬레이션 데이터
    n_samples = 10
    
    # kcat 데이터 (60% 측정)
    kcat_actual = np.array([100, 120, 80, 150, 90, 110, 130, 95, 105, 140])
    kcat_predicted = kcat_actual + np.random.normal(0, 10, n_samples)
    kcat_mask = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])  # 6개 측정
    
    # Km 데이터 (40% 측정)
    Km_actual = np.array([0.5, 0.8, 0.3, 1.2, 0.6, 0.9, 0.4, 0.7, 1.0, 0.55])
    Km_predicted = Km_actual + np.random.normal(0, 0.1, n_samples)
    Km_mask = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])  # 4개 측정
    
    print("\n1. 단일 타겟 손실")
    print("-"*70)
    
    kcat_mse = MaskedLoss.masked_mse(kcat_predicted, kcat_actual, kcat_mask)
    kcat_mae = MaskedLoss.masked_mae(kcat_predicted, kcat_actual, kcat_mask)
    
    print(f"kcat:")
    print(f"  측정 데이터: {int(kcat_mask.sum())}/{len(kcat_mask)} ({kcat_mask.sum()/len(kcat_mask)*100:.0f}%)")
    print(f"  MSE: {kcat_mse:.2f}")
    print(f"  MAE: {kcat_mae:.2f}")
    
    Km_mse = MaskedLoss.masked_mse(Km_predicted, Km_actual, Km_mask)
    Km_mae = MaskedLoss.masked_mae(Km_predicted, Km_actual, Km_mask)
    
    print(f"\nKm:")
    print(f"  측정 데이터: {int(Km_mask.sum())}/{len(Km_mask)} ({Km_mask.sum()/len(Km_mask)*100:.0f}%)")
    print(f"  MSE: {Km_mse:.4f}")
    print(f"  MAE: {Km_mae:.4f}")
    
    print("\n2. 다중 타겟 손실")
    print("-"*70)
    
    predictions = {
        "kcat": kcat_predicted,
        "Km": Km_predicted
    }
    
    actuals = {
        "kcat": kcat_actual,
        "Km": Km_actual
    }
    
    masks = {
        "kcat": kcat_mask,
        "Km": Km_mask
    }
    
    total_loss, individual_losses = MaskedLoss.multi_target_masked_loss(
        predictions, actuals, masks
    )
    
    print(f"개별 손실:")
    for target, loss in individual_losses.items():
        print(f"  {target}: {loss:.4f}")
    
    print(f"\n전체 손실: {total_loss:.4f}")
    
    print("\n3. 데이터 완전성")
    print("-"*70)
    
    completeness = MaskedLoss.calculate_data_completeness(masks)
    
    for target, comp in completeness.items():
        print(f"  {target}: {comp*100:.1f}%")
    
    print("\n4. 결측 데이터의 영향")
    print("-"*70)
    
    # 모든 데이터 사용 (결측 무시)
    all_mask = np.ones(n_samples)
    full_mse = MaskedLoss.masked_mse(kcat_predicted, kcat_actual, all_mask)
    
    print(f"모든 데이터 사용: MSE = {full_mse:.2f}")
    print(f"측정값만 사용: MSE = {kcat_mse:.2f}")
    print(f"차이: {abs(full_mse - kcat_mse):.2f}")
    
    print("\n" + "="*70)
    print("Masked Loss의 장점:")
    print("="*70)
    print("1. 결측 데이터를 무시하고 측정값만 학습")
    print("2. 데이터 완전성을 정량화")
    print("3. 여러 타겟을 동시에 학습 가능")
    print("4. 불확실성 증가를 명시적으로 표현")


if __name__ == "__main__":
    demo_masked_loss()
