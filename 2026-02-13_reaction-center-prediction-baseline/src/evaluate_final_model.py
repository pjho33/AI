"""
확장된 데이터셋으로 최종 평가
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train_stage1_with_data import load_dataset, evaluate_on_dataset, analyze_errors
from stage1_feasibility_predictor import Stage1FeasibilityPredictor


def main():
    """확장된 데이터셋으로 최종 평가"""
    
    print("="*70)
    print("Stage 1 최종 평가 - 확장 데이터셋")
    print("="*70)
    
    # 데이터 로드
    dataset_path = Path(__file__).parent.parent / "data" / "expanded_training_dataset.json"
    dataset = load_dataset(dataset_path)
    
    print(f"\n데이터셋: {len(dataset)}개 반응")
    
    # 긍정/부정 분포
    positive = sum(1 for d in dataset if d["feasibility_label"])
    negative = len(dataset) - positive
    print(f"  긍정 예시: {positive}개 ({positive/len(dataset)*100:.1f}%)")
    print(f"  부정 예시: {negative}개 ({negative/len(dataset)*100:.1f}%)")
    
    # 반응 유형별 분포
    reaction_types = {}
    for d in dataset:
        if d["feasibility_label"]:
            rt = d["reaction_type"]
            reaction_types[rt] = reaction_types.get(rt, 0) + 1
    
    print(f"\n긍정 예시 반응 유형:")
    for rt, count in sorted(reaction_types.items()):
        print(f"  {rt}: {count}개")
    
    # 모델 평가
    print("\n" + "="*70)
    print("모델 평가 중...")
    print("="*70)
    
    predictor = Stage1FeasibilityPredictor()
    results = evaluate_on_dataset(predictor, dataset)
    
    print("\n성능 평가 결과:")
    print("-"*70)
    
    for threshold, metrics in results["performance"].items():
        if threshold.startswith("threshold"):
            print(f"\n{threshold}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    print(f"\n평균 신뢰도: {results['performance']['average_confidence']}")
    print(f"평균 P_feasible: {results['performance']['average_P_feasible']}")
    
    # 오류 분석
    print("\n" + "="*70)
    print("오류 분석 (threshold=0.7)")
    print("="*70)
    
    errors = analyze_errors(
        results["predictions"],
        results["ground_truth"],
        dataset,
        threshold=0.7
    )
    
    print(f"\nFalse Positives: {errors['fp_count']}개 ({errors['fp_count']/negative*100:.1f}%)")
    print(f"False Negatives: {errors['fn_count']}개 ({errors['fn_count']/positive*100:.1f}%)")
    
    # FP 패턴
    if errors["false_positives"]:
        fp_reasons = {}
        for fp in errors["false_positives"]:
            reason = fp["reason"]
            fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
        
        print(f"\nFalse Positive 이유:")
        for reason, count in sorted(fp_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}개")
    
    # 최적 임계값
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
    print(f"최고 F1 점수: {best_f1:.3f}")
    
    # 최종 요약
    print("\n" + "="*70)
    print("최종 요약")
    print("="*70)
    
    print(f"\n데이터셋: {len(dataset)}개 반응")
    print(f"  긍정: {positive}개")
    print(f"  부정: {negative}개")
    
    print(f"\n최고 성능:")
    print(f"  임계값: {best_threshold}")
    print(f"  F1 점수: {best_f1:.3f}")
    print(f"  Accuracy: {results['performance'][f'threshold_{best_threshold}']['accuracy']:.3f}")
    print(f"  Precision: {results['performance'][f'threshold_{best_threshold}']['precision']:.3f}")
    print(f"  Recall: {results['performance'][f'threshold_{best_threshold}']['recall']:.3f}")
    
    print(f"\n오류:")
    print(f"  False Positives: {errors['fp_count']}개")
    print(f"  False Negatives: {errors['fn_count']}개")
    
    print("\n" + "="*70)
    print("Stage 1 완료!")
    print("="*70)
    
    print("\n✅ 달성한 것:")
    print("  • 화학적 가능성 예측 시스템")
    print("  • 반응 중심 예측")
    print("  • 제한 요인 식별")
    print("  • 신뢰도 정량화")
    print(f"  • F1 점수: {best_f1:.3f}")
    
    print("\n📊 학습 데이터:")
    print(f"  • {len(dataset)}개 반응")
    print(f"  • 산화: {reaction_types.get('oxidation', 0)}개")
    print(f"  • 이성질화: {reaction_types.get('isomerization', 0)}개")
    
    print("\n🎯 다음 단계 (Stage 2):")
    print("  • 성능 예측 (수율, kcat, Km)")
    print("  • 결측 데이터 대응")
    print("  • 불확실성 정량화")


if __name__ == "__main__":
    main()
