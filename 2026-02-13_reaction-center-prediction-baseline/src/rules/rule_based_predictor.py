"""
Rule-based reaction center predictor.
Uses chemical transformation rules to identify possible reaction sites.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json


@dataclass
class ReactionRule:
    """Chemical transformation rule."""
    name: str
    smarts_substrate: str
    smarts_product: str
    ec_class: str
    cofactor: Optional[str]
    reversible: bool
    priority: int = 1


@dataclass
class ReactionPrediction:
    """Predicted reaction center."""
    atom_indices: List[int]
    rule_name: str
    ec_class: str
    confidence: float
    cofactor: Optional[str]


class RuleBasedPredictor:
    """Predict reaction centers using chemical transformation rules."""
    
    def __init__(self):
        self.rules = self._load_default_rules()
    
    def _load_default_rules(self) -> List[ReactionRule]:
        """Load default reaction rules for polyol/sugar transformations."""
        rules = [
            ReactionRule(
                name="primary_alcohol_oxidation",
                smarts_substrate="[CH2X4:1][OH1:2]",
                smarts_product="[CH1:1]=[O:2]",
                ec_class="1.1.1",
                cofactor="NAD+",
                reversible=True,
                priority=2
            ),
            ReactionRule(
                name="secondary_alcohol_oxidation",
                smarts_substrate="[CH1X4:1]([OH1:2])",
                smarts_product="[C:1](=[O:2])",
                ec_class="1.1.1",
                cofactor="NAD+",
                reversible=True,
                priority=2
            ),
            ReactionRule(
                name="polyol_dehydrogenation",
                smarts_substrate="[CH2:1][CH:2]([OH:3])[CH:4]",
                smarts_product="[CH2:1][C:2](=[O:3])[CH:4]",
                ec_class="1.1.1",
                cofactor="NAD+",
                reversible=True,
                priority=3
            ),
            ReactionRule(
                name="aldose_ketose_isomerization",
                smarts_substrate="[CH:1]=[O:2]",
                smarts_product="[C:1]([OH:2])",
                ec_class="5.3.1",
                cofactor=None,
                reversible=True,
                priority=2
            ),
            ReactionRule(
                name="ketose_aldose_isomerization",
                smarts_substrate="[C:1](=[O:2])[CH2:3][OH:4]",
                smarts_product="[C:1]([OH:2])[CH:3]=[O:4]",
                ec_class="5.3.1",
                cofactor=None,
                reversible=True,
                priority=2
            ),
        ]
        return rules
    
    def predict_reaction_centers(self, smiles: str, reaction_type: str = "oxidation") -> List[ReactionPrediction]:
        """
        Predict possible reaction centers in a molecule.
        
        Args:
            smiles: SMILES string of substrate
            reaction_type: "oxidation", "reduction", or "isomerization"
            
        Returns:
            List of predicted reaction centers, sorted by confidence
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        predictions = []
        
        relevant_rules = self._filter_rules_by_type(reaction_type)
        
        for rule in relevant_rules:
            matches = self._apply_rule(mol, rule)
            
            for match in matches:
                confidence = self._calculate_confidence(mol, match, rule)
                
                predictions.append(ReactionPrediction(
                    atom_indices=list(match),
                    rule_name=rule.name,
                    ec_class=rule.ec_class,
                    confidence=confidence,
                    cofactor=rule.cofactor
                ))
        
        if reaction_type == "isomerization":
            # 1. 알도스-케토스 이성질화 (알데히드 ⇌ 케톤)
            aldose_ketose = r"[CH](=O)[CH]([OH])"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aldose_ketose))
            for match in matches:
                predictions.append(ReactionPrediction(
                    atom_indices=list(match),
                    rule_name="aldose_ketose_isomerization",
                    ec_class="EC 5.3.1",
                    confidence=0.80,
                    cofactor=None
                ))
            
            # 2. 폴리올 이성질화 (여러 OH기 있는 경우)
            polyol_pattern = r"[CH]([OH])[CH]([OH])"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(polyol_pattern))
            for match in matches:
                predictions.append(ReactionPrediction(
                    atom_indices=list(match),
                    rule_name="polyol_isomerization",
                    ec_class="EC 5.3.1",
                    confidence=0.75,
                    cofactor=None
                ))
            
            # 3. 일반 이성질화 (C-O 결합 재배열)
            if "O" in Chem.MolToSmiles(mol) and "C" in Chem.MolToSmiles(mol):
                # 모든 탄소-산소 결합 찾기
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "C":
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == "O":
                                predictions.append(ReactionPrediction(
                                    atom_indices=[atom.GetIdx(), neighbor.GetIdx()],
                                    rule_name="general_isomerization",
                                    ec_class="EC 5.3.1",
                                    confidence=0.65,
                                    cofactor=None
                                ))
        
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def _filter_rules_by_type(self, reaction_type: str) -> List[ReactionRule]:
        """Filter rules based on reaction type."""
        if reaction_type == "oxidation":
            return [r for r in self.rules if "oxidation" in r.name or "dehydrogenation" in r.name]
        elif reaction_type == "reduction":
            return [r for r in self.rules if r.reversible and ("oxidation" in r.name or "dehydrogenation" in r.name)]
        elif reaction_type == "isomerization":
            return [r for r in self.rules if "isomerization" in r.name]
        else:
            return self.rules
    
    def _apply_rule(self, mol: Chem.Mol, rule: ReactionRule) -> List[Tuple[int, ...]]:
        """Apply a rule to find matching substructures."""
        pattern = Chem.MolFromSmarts(rule.smarts_substrate)
        if pattern is None:
            return []
        
        matches = mol.GetSubstructMatches(pattern)
        return matches
    
    def _calculate_confidence(self, mol: Chem.Mol, match: Tuple[int, ...], rule: ReactionRule) -> float:
        """
        Calculate confidence score for a rule match.
        
        Factors:
        - Rule priority
        - Steric accessibility
        - Electronic environment
        - Molecular context
        """
        base_score = rule.priority / 3.0
        
        if not match:
            return 0.0
        
        primary_atom_idx = match[0]
        atom = mol.GetAtomWithIdx(primary_atom_idx)
        
        accessibility_score = self._calculate_accessibility(mol, primary_atom_idx)
        base_score *= accessibility_score
        
        if self._has_electron_withdrawing_neighbor(atom):
            base_score *= 1.2
        
        if atom.IsInRing():
            base_score *= 0.8
        
        if self._is_terminal_position(atom):
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    def _calculate_accessibility(self, mol: Chem.Mol, atom_idx: int) -> float:
        """Calculate steric accessibility of an atom."""
        atom = mol.GetAtomWithIdx(atom_idx)
        
        degree = atom.GetDegree()
        
        if degree <= 2:
            return 1.0
        elif degree == 3:
            return 0.8
        else:
            return 0.6
    
    def _has_electron_withdrawing_neighbor(self, atom: Chem.Atom) -> bool:
        """Check if atom has electron-withdrawing groups nearby."""
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() in ['O', 'N', 'F', 'Cl']:
                return True
        return False
    
    def _is_terminal_position(self, atom: Chem.Atom) -> bool:
        """Check if atom is in a terminal position."""
        carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'C')
        return carbon_neighbors <= 1
    
    def add_custom_rule(self, rule: ReactionRule):
        """Add a custom reaction rule."""
        self.rules.append(rule)
    
    def save_rules(self, filepath: str):
        """Save rules to JSON file."""
        rules_data = [
            {
                "name": r.name,
                "smarts_substrate": r.smarts_substrate,
                "smarts_product": r.smarts_product,
                "ec_class": r.ec_class,
                "cofactor": r.cofactor,
                "reversible": r.reversible,
                "priority": r.priority
            }
            for r in self.rules
        ]
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def load_rules(self, filepath: str):
        """Load rules from JSON file."""
        with open(filepath, 'r') as f:
            rules_data = json.load(f)
        
        self.rules = [
            ReactionRule(**rule_dict)
            for rule_dict in rules_data
        ]


if __name__ == "__main__":
    predictor = RuleBasedPredictor()
    
    test_molecules = [
        ("OCC(O)C(O)C(O)C(O)CO", "sorbitol"),
        ("OCC(=O)C(O)C(O)C(O)CO", "fructose"),
        ("OCC(O)C(O)C(O)CO", "xylitol"),
    ]
    
    print("Rule-based Reaction Center Prediction\n")
    
    for smiles, name in test_molecules:
        print(f"\n{name}: {smiles}")
        
        for reaction_type in ["oxidation", "isomerization"]:
            predictions = predictor.predict_reaction_centers(smiles, reaction_type)
            
            if predictions:
                print(f"\n  {reaction_type.capitalize()}:")
                for i, pred in enumerate(predictions[:3], 1):
                    print(f"    {i}. Atoms {pred.atom_indices} - {pred.rule_name}")
                    print(f"       Confidence: {pred.confidence:.2f}, EC: {pred.ec_class}")
            else:
                print(f"  {reaction_type.capitalize()}: No predictions")
