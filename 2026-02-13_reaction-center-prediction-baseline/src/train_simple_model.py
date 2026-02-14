"""
간단한 화학반응으로 AI 학습 시작
5개 기본 반응으로 패턴 학습 및 검증
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))

from rules.rule_based_predictor import RuleBasedPredictor, ReactionPrediction
from evaluation.evaluator import ReactionCenterEvaluator, TestSample
from rdkit import Chem
from rdkit.Chem import Draw


class SimpleReactionLearner:
    """간단한 반응으로 학습하는 시스템"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.reactions = self.load_reactions()
        self.predictor = RuleBasedPredictor()
        self.learned_patterns = defaultdict(list)
        
    def load_reactions(self) -> List[Dict]:
        """학습 데이터 로딩"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_reactions(self):
        """반응 분석 및 패턴 추출"""
        print("="*70)
        print("반응 분석 시작")
        print("="*70)
        
        for rxn in self.reactions:
            print(f"\n[{rxn['id']}] {rxn['name']}")
            print(f"설명: {rxn['description']}")
            print(f"반응: {rxn['substrate']['name']} → {rxn['product']['name']}")
            print(f"효소: {rxn['enzyme']['name']} (EC {rxn['enzyme']['ec_number']})")
            print(f"보조인자: {rxn['cofactor']['name']}")
            print(f"반응 중심: {rxn['reaction_center']['atom_description']}")
            print(f"변화 유형: {rxn['reaction_center']['change_type']}")
            
            # 패턴 저장
            pattern_key = rxn['reaction_center']['change_type']
            self.learned_patterns[pattern_key].append({
                'substrate': rxn['substrate']['smiles'],
                'product': rxn['product']['smiles'],
                'enzyme': rxn['enzyme']['ec_number'],
                'cofactor': rxn['cofactor']['name']
            })
            
            print("\n학습 포인트:")
            for point in rxn['learning_points']:
                print(f"  • {point}")
    
    def extract_learning_patterns(self):
        """학습한 패턴 요약"""
        print("\n" + "="*70)
        print("AI가 학습한 패턴 요약")
        print("="*70)
        
        print("\n1. 반응 유형별 패턴:")
        for pattern, examples in self.learned_patterns.items():
            print(f"\n  [{pattern}]")
            print(f"    발견 횟수: {len(examples)}개")
            
            enzymes = set(ex['enzyme'] for ex in examples)
            cofactors = set(ex['cofactor'] for ex in examples)
            
            print(f"    사용 효소: {', '.join(enzymes)}")
            print(f"    보조인자: {', '.join(cofactors)}")
        
        print("\n2. 효소 클래스별 분류:")
        ec_classes = defaultdict(list)
        for rxn in self.reactions:
            ec = rxn['enzyme']['ec_number']
            ec_class = ec.split('.')[0]
            ec_classes[ec_class].append(rxn['name'])
        
        for ec_class, rxn_names in ec_classes.items():
            print(f"\n  EC {ec_class}:")
            for name in rxn_names:
                print(f"    - {name}")
        
        print("\n3. 보조인자 요구사항:")
        cofactor_usage = defaultdict(int)
        for rxn in self.reactions:
            cofactor = rxn['cofactor']['name']
            if rxn['cofactor']['required']:
                cofactor_usage[cofactor] += 1
        
        for cofactor, count in cofactor_usage.items():
            print(f"  {cofactor}: {count}개 반응에서 필수")
    
    def test_predictions(self):
        """학습한 패턴으로 예측 테스트"""
        print("\n" + "="*70)
        print("예측 성능 테스트")
        print("="*70)
        
        test_samples = []
        
        for rxn in self.reactions:
            sample = TestSample(
                substrate_smiles=rxn['substrate']['smiles'],
                true_reaction_center=rxn['reaction_center']['atom_indices'],
                reaction_type=rxn['reaction_type'],
                ec_number=rxn['enzyme']['ec_number'],
                reaction_id=rxn['id']
            )
            test_samples.append(sample)
        
        evaluator = ReactionCenterEvaluator(test_samples)
        result = evaluator.evaluate(self.predictor)
        
        print(f"\n학습 데이터 재현 성능:")
        print(f"  Top-1 Accuracy: {result.top1_accuracy:.1f}%")
        print(f"  Top-3 Accuracy: {result.top3_accuracy:.1f}%")
        print(f"  Top-5 Accuracy: {result.top5_accuracy:.1f}%")
        
        print(f"\n반응 유형별 성능:")
        for rtype, stats in result.by_reaction_type.items():
            print(f"  {rtype}:")
            print(f"    Top-1: {stats['top1_accuracy']:.1f}%")
            print(f"    샘플 수: {stats['count']}")
        
        return result
    
    def test_new_molecule(self, smiles: str, name: str, reaction_type: str):
        """새로운 분자로 예측 테스트 (일반화 능력)"""
        print(f"\n새 분자 예측: {name}")
        print(f"SMILES: {smiles}")
        print(f"반응 유형: {reaction_type}")
        
        predictions = self.predictor.predict_reaction_centers(smiles, reaction_type)
        
        if predictions:
            print(f"\n예측 결과 (상위 3개):")
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. 원자 {pred.atom_indices}")
                print(f"     규칙: {pred.rule_name}")
                print(f"     신뢰도: {pred.confidence:.3f}")
                print(f"     효소: EC {pred.ec_class}")
                print(f"     보조인자: {pred.cofactor}")
        else:
            print("  예측 실패")
    
    def generate_learning_summary(self):
        """학습 결과 요약"""
        print("\n" + "="*70)
        print("AI 학습 결과 요약")
        print("="*70)
        
        print(f"\n총 학습 반응 수: {len(self.reactions)}개")
        
        print("\nAI가 배운 것:")
        print("\n1. 작용기 인식:")
        print("   • 1차 알코올 (R-CH2-OH) → 알데히드로 산화")
        print("   • 2차 알코올 (R-CH(OH)-R') → 케톤으로 산화")
        print("   • 알데히드 ⇌ 케톤 (이성질화)")
        
        print("\n2. 효소 요구사항:")
        print("   • 산화반응 → EC 1.1.1.x + NAD+ 필수")
        print("   • 이성질화 → EC 5.3.x + 보조인자 불필요")
        
        print("\n3. 위치 선택성:")
        print("   • 여러 OH기 중 특정 위치만 반응")
        print("   • 입체화학이 중요")
        print("   • 효소 특이성에 따라 결정")
        
        print("\n4. 효소 특이성:")
        oxidation_enzymes = [
            rxn for rxn in self.reactions 
            if rxn['reaction_type'] == 'oxidation'
        ]
        
        if oxidation_enzymes:
            print("   산화 효소:")
            for rxn in oxidation_enzymes:
                spec = rxn['enzyme']['specificity']
                print(f"   • EC {rxn['enzyme']['ec_number']}: {spec} specificity")
        
        print("\n다음 단계:")
        print("  1. 새로운 알코올 분자로 예측 테스트")
        print("  2. 더 많은 반응 추가 (10개 → 50개)")
        print("  3. ML 모델로 성능 향상")
        print("  4. 효소 활성도 예측 추가")


def main():
    """메인 실행 함수"""
    
    # 데이터 경로
    data_path = Path(__file__).parent.parent / "data" / "simple_reactions.json"
    
    if not data_path.exists():
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 학습 시작
    learner = SimpleReactionLearner(data_path)
    
    # 1. 반응 분석
    learner.analyze_reactions()
    
    # 2. 패턴 추출
    learner.extract_learning_patterns()
    
    # 3. 예측 테스트
    result = learner.test_predictions()
    
    # 4. 새 분자 테스트
    print("\n" + "="*70)
    print("일반화 능력 테스트 (학습 안 한 분자)")
    print("="*70)
    
    # 테스트 1: 자일리톨 (5탄당 알코올)
    learner.test_new_molecule(
        smiles="OCC(O)C(O)C(O)CO",
        name="xylitol",
        reaction_type="oxidation"
    )
    
    # 테스트 2: 1-부탄올
    learner.test_new_molecule(
        smiles="CCCCO",
        name="1-butanol",
        reaction_type="oxidation"
    )
    
    # 테스트 3: 2-부탄올
    learner.test_new_molecule(
        smiles="CCC(C)O",
        name="2-butanol",
        reaction_type="oxidation"
    )
    
    # 5. 학습 요약
    learner.generate_learning_summary()
    
    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)
    
    # 결과 저장
    results_path = Path(__file__).parent.parent / "data" / "learning_results.json"
    results_data = {
        "training_reactions": len(learner.reactions),
        "learned_patterns": len(learner.learned_patterns),
        "top1_accuracy": result.top1_accuracy,
        "top5_accuracy": result.top5_accuracy,
        "patterns": dict(learner.learned_patterns)
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_path}")


if __name__ == "__main__":
    main()
