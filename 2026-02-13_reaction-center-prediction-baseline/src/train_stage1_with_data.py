"""
Stage 1 모델을 실제 데이터로 학습 및 평가
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.append(str(Path(__file__).parent))
from stage1_feasibility_predictor import Stage1FeasibilityPredictor, FeasibilityPrediction


def load_dataset(filepath: str) -> List[Dict]:
    """데이터셋 로드"""
    with open(filepath) as f:
        return json.load(f)


def evaluate_on_dataset(
    predictor: Stage1FeasibilityPredictor,
    dataset: List[Dict]
) -> Dict:
    """데이터셋에서 평가"""
    
    predictions = []
    ground_truth = []
    
    for entry in dataset:
        # 예측
        pred = predictor.predict_feasibility(
            substrate_smiles=entry["substrate"]["smiles"],
            reaction_type=entry["reaction_type"],
            enzyme_ec=entry["enzyme"]["primary_ec"],
            conditions=entry.get("conditions", {"pH": 7.4, "temperature": 37})
        )
        
        predictions.append(pred)
        ground_truth.append(entry["feasibility_label"])
    
    # 성능 평가
    performance = predictor.evaluate_performance(predictions, ground_truth)
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "performance": performance
    }


def analyze_errors(
    predictions: List[FeasibilityPrediction],
    ground_truth: List[bool],
    dataset: List[Dict],
    threshold: float = 0.7
) -> Dict:
    """오류 분석"""
    
    false_positives = []  # 불가능한데 가능하다고 예측
    false_negatives = []  # 가능한데 불가능하다고 예측
    
    for i, (pred, gt, data) in enumerate(zip(predictions, ground_truth, dataset)):
        predicted_positive = pred.P_feasible >= threshold
        
        if predicted_positive and not gt:
            false_positives.append({
                "index": i,
                "rhea_id": data["rhea_id"],
                "substrate": data["substrate"]["name"],
                "reason": data.get("reason", "unknown"),
                "P_feasible": pred.P_feasible,
                "limiting_factors": pred.limiting_factors
            })
        elif not predicted_positive and gt:
            false_negatives.append({
                "index": i,
                "rhea_id": data["rhea_id"],
                "substrate": data["substrate"]["name"],
                "P_feasible": pred.P_feasible,
                "limiting_factors": pred.limiting_factors
            })
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives)
    }


def main():
    """실제 데이터로 학습 및 평가"""
    
    print("="*70)
    print("Stage 1: 실제 데이터로 학습 및 평가")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1. 데이터셋 로드 중...")
    dataset_path = Path(__file__).parent.parent / "data" / "simulated_rhea_dataset.json"
    dataset = load_dataset(dataset_path)
    
    print(f"   총 {len(dataset)}개 반응")
    
    # 긍정/부정 분포
    positive = sum(1 for d in dataset if d["feasibility_label"])
    negative = len(dataset) - positive
    print(f"   긍정 예시: {positive}개 ({positive/len(dataset)*100:.1f}%)")
    print(f"   부정 예시: {negative}개 ({negative/len(dataset)*100:.1f}%)")
    
    # 2. 모델 초기화
    print("\n2. Stage 1 모델 초기화...")
    predictor = Stage1FeasibilityPredictor()
    
    # 3. 전체 데이터셋 평가
    print("\n3. 전체 데이터셋 평가 중...")
    results = evaluate_on_dataset(predictor, dataset)
    
    print("\n" + "="*70)
    print("성능 평가 결과")
    print("="*70)
    
    for threshold, metrics in results["performance"].items():
        if threshold.startswith("threshold"):
            print(f"\n{threshold}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    print(f"\n평균 신뢰도: {results['performance']['average_confidence']}")
    print(f"평균 P_feasible: {results['performance']['average_P_feasible']}")
    
    # 4. 오류 분석
    print("\n" + "="*70)
    print("오류 분석 (threshold=0.7)")
    print("="*70)
    
    errors = analyze_errors(
        results["predictions"],
        results["ground_truth"],
        dataset,
        threshold=0.7
    )
    
    print(f"\nFalse Positives: {errors['fp_count']}개")
    if errors["false_positives"]:
        print("\n상위 5개:")
        for fp in errors["false_positives"][:5]:
            print(f"  [{fp['rhea_id']}] {fp['substrate']}")
            print(f"    이유: {fp['reason']}")
            print(f"    P_feasible: {fp['P_feasible']:.3f}")
            print(f"    제한 요인: {', '.join(fp['limiting_factors'])}")
    
    print(f"\nFalse Negatives: {errors['fn_count']}개")
    if errors["false_negatives"]:
        print("\n상위 5개:")
        for fn in errors["false_negatives"][:5]:
            print(f"  [{fn['rhea_id']}] {fn['substrate']}")
            print(f"    P_feasible: {fn['P_feasible']:.3f}")
            print(f"    제한 요인: {', '.join(fn['limiting_factors'])}")
    
    # 5. 오류 패턴 분석
    print("\n" + "="*70)
    print("오류 패턴 분석")
    print("="*70)
    
    # FP 이유별 분포
    fp_reasons = {}
    for fp in errors["false_positives"]:
        reason = fp["reason"]
        fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
    
    print("\nFalse Positive 이유:")
    for reason, count in sorted(fp_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}개")
    
    # 6. 개선 제안
    print("\n" + "="*70)
    print("개선 제안")
    print("="*70)
    
    if errors['fp_count'] > 0:
        print("\n1. False Positive 감소:")
        if "extreme_pH" in fp_reasons:
            print("   • pH 페널티 더 강화 필요")
        if "extreme_temperature" in fp_reasons:
            print("   • 온도 페널티 더 강화 필요")
        if "enzyme_mismatch" in fp_reasons:
            print("   • 효소-반응 불일치 페널티 더 강화 필요")
    
    if errors['fn_count'] > 0:
        print("\n2. False Negative 감소:")
        print("   • 일부 가능한 반응을 너무 보수적으로 평가")
        print("   • 임계값 조정 또는 가중치 재조정 필요")
    
    # 7. 최적 임계값 찾기
    print("\n" + "="*70)
    print("최적 임계값 분석")
    print("="*70)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold_key, metrics in results["performance"].items():
        if threshold_key.startswith("threshold"):
            f1 = metrics["f1"]
            threshold = float(threshold_key.split("_")[1])
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    print(f"\n최적 임계값: {best_threshold}")
    print(f"최고 F1 점수: {best_f1}")
    
    # 8. 요약
    print("\n" + "="*70)
    print("요약")
    print("="*70)
    
    print(f"\n데이터셋: {len(dataset)}개 반응")
    print(f"최적 임계값: {best_threshold}")
    print(f"최고 성능: F1={best_f1:.3f}")
    print(f"오류: FP={errors['fp_count']}, FN={errors['fn_count']}")
    
    print("\n주요 개선 방향:")
    print("  1. 극한 조건 페널티 강화")
    print("  2. 효소-반응 불일치 페널티 강화")
    print("  3. 실제 Rhea 데이터로 확장")
    print("  4. ML 모델 도입 (규칙 → 학습)")


if __name__ == "__main__":
    main()
