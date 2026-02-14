"""
Evaluation framework for reaction center prediction.
Implements Top-K accuracy metrics and practical impact analysis.
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Results from evaluating a predictor."""
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mean_rank: float
    median_rank: float
    by_reaction_type: Dict[str, Dict[str, float]]
    by_ec_class: Dict[str, Dict[str, float]]
    total_samples: int


@dataclass
class TestSample:
    """Single test sample for evaluation."""
    substrate_smiles: str
    true_reaction_center: List[int]
    reaction_type: str
    ec_number: str
    reaction_id: str


class ReactionCenterEvaluator:
    """Evaluate reaction center prediction models."""
    
    def __init__(self, test_dataset: List[TestSample]):
        self.test_dataset = test_dataset
    
    def evaluate(self, predictor: Any, predict_fn: Callable = None) -> EvaluationResult:
        """
        Evaluate a predictor on the test dataset.
        
        Args:
            predictor: Model or predictor object
            predict_fn: Function that takes (predictor, smiles, reaction_type) 
                       and returns list of predictions sorted by confidence
                       
        Returns:
            EvaluationResult with comprehensive metrics
        """
        if predict_fn is None:
            predict_fn = lambda p, s, rt: p.predict_reaction_centers(s, rt)
        
        ranks = []
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        
        by_reaction_type = defaultdict(lambda: {"ranks": [], "top1": 0, "top3": 0, "top5": 0, "count": 0})
        by_ec_class = defaultdict(lambda: {"ranks": [], "top1": 0, "top3": 0, "top5": 0, "count": 0})
        
        for sample in self.test_dataset:
            predictions = predict_fn(predictor, sample.substrate_smiles, sample.reaction_type)
            
            if not predictions:
                rank = len(sample.substrate_smiles)
            else:
                rank = self._find_rank(predictions, sample.true_reaction_center)
            
            ranks.append(rank)
            
            if rank == 1:
                top1_correct += 1
                top3_correct += 1
                top5_correct += 1
            elif rank <= 3:
                top3_correct += 1
                top5_correct += 1
            elif rank <= 5:
                top5_correct += 1
            
            by_reaction_type[sample.reaction_type]["ranks"].append(rank)
            by_reaction_type[sample.reaction_type]["count"] += 1
            if rank == 1:
                by_reaction_type[sample.reaction_type]["top1"] += 1
            if rank <= 3:
                by_reaction_type[sample.reaction_type]["top3"] += 1
            if rank <= 5:
                by_reaction_type[sample.reaction_type]["top5"] += 1
            
            ec_main = sample.ec_number.split('.')[0] if sample.ec_number else "unknown"
            by_ec_class[ec_main]["ranks"].append(rank)
            by_ec_class[ec_main]["count"] += 1
            if rank == 1:
                by_ec_class[ec_main]["top1"] += 1
            if rank <= 3:
                by_ec_class[ec_main]["top3"] += 1
            if rank <= 5:
                by_ec_class[ec_main]["top5"] += 1
        
        n = len(self.test_dataset)
        
        by_reaction_type_summary = {}
        for rtype, stats in by_reaction_type.items():
            count = stats["count"]
            by_reaction_type_summary[rtype] = {
                "top1_accuracy": 100 * stats["top1"] / count,
                "top3_accuracy": 100 * stats["top3"] / count,
                "top5_accuracy": 100 * stats["top5"] / count,
                "mean_rank": np.mean(stats["ranks"]),
                "count": count
            }
        
        by_ec_class_summary = {}
        for ec, stats in by_ec_class.items():
            count = stats["count"]
            by_ec_class_summary[ec] = {
                "top1_accuracy": 100 * stats["top1"] / count,
                "top3_accuracy": 100 * stats["top3"] / count,
                "top5_accuracy": 100 * stats["top5"] / count,
                "mean_rank": np.mean(stats["ranks"]),
                "count": count
            }
        
        return EvaluationResult(
            top1_accuracy=100 * top1_correct / n,
            top3_accuracy=100 * top3_correct / n,
            top5_accuracy=100 * top5_correct / n,
            mean_rank=np.mean(ranks),
            median_rank=np.median(ranks),
            by_reaction_type=by_reaction_type_summary,
            by_ec_class=by_ec_class_summary,
            total_samples=n
        )
    
    def _find_rank(self, predictions: List[Any], true_center: List[int]) -> int:
        """
        Find the rank of the true reaction center in predictions.
        
        Args:
            predictions: List of predictions (must have atom_indices attribute)
            true_center: List of true reaction center atom indices
            
        Returns:
            Rank (1-indexed), or len(predictions)+1 if not found
        """
        true_set = set(true_center)
        
        for rank, pred in enumerate(predictions, 1):
            pred_atoms = set(pred.atom_indices) if hasattr(pred, 'atom_indices') else set(pred)
            
            if pred_atoms & true_set:
                return rank
        
        return len(predictions) + 1
    
    def practical_impact_analysis(self, result: EvaluationResult) -> Dict[str, Any]:
        """
        Translate accuracy metrics to real-world impact.
        
        Args:
            result: EvaluationResult from evaluate()
            
        Returns:
            Dictionary with practical impact metrics
        """
        baseline_screening_cost = 100
        baseline_time_weeks = 12
        
        if result.top5_accuracy >= 80:
            reduced_candidates = 5
        elif result.top3_accuracy >= 70:
            reduced_candidates = 10
        else:
            reduced_candidates = 20
        
        cost_reduction = 100 * (1 - reduced_candidates / baseline_screening_cost)
        time_reduction = 100 * (1 - reduced_candidates / baseline_screening_cost)
        
        return {
            "baseline_screening_cost": baseline_screening_cost,
            "reduced_candidates": reduced_candidates,
            "cost_reduction_percent": cost_reduction,
            "estimated_time_savings_weeks": baseline_time_weeks * time_reduction / 100,
            "top5_accuracy": result.top5_accuracy,
            "recommendation": self._get_recommendation(result)
        }
    
    def _get_recommendation(self, result: EvaluationResult) -> str:
        """Get recommendation based on performance."""
        if result.top5_accuracy >= 85:
            return "Production-ready: Can significantly reduce experimental screening"
        elif result.top5_accuracy >= 70:
            return "Useful: Can guide experimental design and prioritization"
        elif result.top5_accuracy >= 50:
            return "Promising: Needs improvement before practical use"
        else:
            return "Needs work: Performance below practical threshold"
    
    def print_report(self, result: EvaluationResult):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("REACTION CENTER PREDICTION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Performance (n={result.total_samples}):")
        print(f"  Top-1 Accuracy: {result.top1_accuracy:.1f}%")
        print(f"  Top-3 Accuracy: {result.top3_accuracy:.1f}%")
        print(f"  Top-5 Accuracy: {result.top5_accuracy:.1f}%")
        print(f"  Mean Rank: {result.mean_rank:.2f}")
        print(f"  Median Rank: {result.median_rank:.1f}")
        
        if result.by_reaction_type:
            print("\nPerformance by Reaction Type:")
            for rtype, stats in result.by_reaction_type.items():
                print(f"\n  {rtype.capitalize()} (n={stats['count']}):")
                print(f"    Top-1: {stats['top1_accuracy']:.1f}%")
                print(f"    Top-3: {stats['top3_accuracy']:.1f}%")
                print(f"    Top-5: {stats['top5_accuracy']:.1f}%")
        
        if result.by_ec_class:
            print("\nPerformance by EC Class:")
            for ec, stats in result.by_ec_class.items():
                print(f"\n  EC {ec} (n={stats['count']}):")
                print(f"    Top-1: {stats['top1_accuracy']:.1f}%")
                print(f"    Top-5: {stats['top5_accuracy']:.1f}%")
        
        impact = self.practical_impact_analysis(result)
        print("\nPractical Impact:")
        print(f"  Baseline screening: {impact['baseline_screening_cost']} candidates")
        print(f"  AI-guided screening: {impact['reduced_candidates']} candidates")
        print(f"  Cost reduction: {impact['cost_reduction_percent']:.0f}%")
        print(f"  Time savings: ~{impact['estimated_time_savings_weeks']:.1f} weeks")
        print(f"\n  {impact['recommendation']}")
        print("\n" + "="*60)


class BaselinePredictor:
    """Random baseline for comparison."""
    
    def predict_reaction_centers(self, smiles: str, reaction_type: str) -> List[Dict]:
        """Randomly predict reaction centers."""
        from rdkit import Chem
        import random
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
        
        random.shuffle(carbon_indices)
        
        return [{"atom_indices": [idx]} for idx in carbon_indices]


if __name__ == "__main__":
    test_samples = [
        TestSample(
            substrate_smiles="OCC(O)C(O)C(O)C(O)CO",
            true_reaction_center=[2],
            reaction_type="oxidation",
            ec_number="1.1.1.14",
            reaction_id="test_1"
        ),
    ]
    
    evaluator = ReactionCenterEvaluator(test_samples)
    
    baseline = BaselinePredictor()
    result = evaluator.evaluate(baseline)
    
    evaluator.print_report(result)
