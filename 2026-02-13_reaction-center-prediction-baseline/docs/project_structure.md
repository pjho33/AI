# Project Implementation Roadmap

## Data Schema Design

### Reaction Record
```python
{
    "reaction_id": "RHEA:12345",
    "substrate_smiles": "OCC(O)C(O)C(O)C(O)CO",  # sorbitol
    "product_smiles": "OCC(=O)C(O)C(O)C(O)CO",   # fructose
    "reaction_type": "oxidation",
    "ec_number": "1.1.1.14",
    "enzyme_name": "L-iditol 2-dehydrogenase",
    "cofactor": "NAD+",
    "reaction_center": {
        "atom_indices": [2],  # C2 position
        "change_type": "alcohol_to_ketone",
        "smarts_pattern": "[CH2:1][CH:2]([OH:3])[CH:4] >> [CH2:1][C:2](=[O:3])[CH:4]"
    },
    "atom_mapping": "...",  # Full atom-mapped SMILES
    "confidence": "experimental"
}
```

### Training Dataset Format
```python
{
    "X": {
        "smiles": str,
        "reaction_class": str,  # "oxidation" | "isomerization"
        "cofactor": str | None,
        "fingerprint": np.array,  # Morgan fingerprint
        "features": {
            "molecular_weight": float,
            "n_oh_groups": int,
            "n_carbons": int,
            ...
        }
    },
    "y": {
        "reaction_center_atom": int,  # Primary label
        "possible_atoms": List[int],  # Alternative valid centers
        "transformation_type": str
    },
    "metadata": {
        "ec_number": str,
        "source": str,  # "rhea" | "kegg" | "manual"
        "validation_status": str
    }
}
```

## Data Extraction Pipeline

### Step 1: Rhea Database Download
```bash
# Download Rhea reaction data
wget ftp://ftp.expasy.org/databases/rhea/rdf/rhea.rdf.gz
wget ftp://ftp.expasy.org/databases/rhea/ctfiles/rhea-reaction-smiles.tsv.gz
```

### Step 2: Filter Relevant Reactions
```python
# Filter criteria
EC_CLASSES = ["1.1.1", "5.3"]  # Oxidoreductase, Isomerase
KEYWORDS = ["polyol", "sugar", "sorbitol", "fructose", "glucose", "xylitol"]

# Filtering logic
def is_relevant_reaction(reaction):
    return (
        reaction.ec_number.startswith(tuple(EC_CLASSES))
        and any(kw in reaction.name.lower() for kw in KEYWORDS)
        and reaction.has_atom_mapping
        and reaction.is_balanced
    )
```

### Step 3: Extract Reaction Centers
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def extract_reaction_center(substrate_mol, product_mol, atom_mapping):
    """
    Compare atom-mapped substrate and product to identify changed atoms.
    
    Returns:
        List[int]: Indices of atoms where bond order or functional group changed
    """
    changed_atoms = []
    
    for atom_idx in range(substrate_mol.GetNumAtoms()):
        sub_atom = substrate_mol.GetAtomWithIdx(atom_idx)
        prod_atom = product_mol.GetAtomWithIdx(atom_idx)
        
        # Check bond order changes
        if get_bond_orders(sub_atom) != get_bond_orders(prod_atom):
            changed_atoms.append(atom_idx)
            
        # Check hybridization changes (sp3 ↔ sp2)
        if sub_atom.GetHybridization() != prod_atom.GetHybridization():
            changed_atoms.append(atom_idx)
            
        # Check formal charge changes
        if sub_atom.GetFormalCharge() != prod_atom.GetFormalCharge():
            changed_atoms.append(atom_idx)
    
    return list(set(changed_atoms))
```

### Step 4: Classify Transformation Type
```python
def classify_transformation(substrate_atom, product_atom):
    """
    Classify the type of chemical transformation.
    """
    sub_bonds = get_bond_summary(substrate_atom)
    prod_bonds = get_bond_summary(product_atom)
    
    # Oxidation: C-OH → C=O
    if sub_bonds.get('O', 0) == 1 and prod_bonds.get('O_double', 0) == 1:
        return "alcohol_to_carbonyl"
    
    # Reduction: C=O → C-OH
    if sub_bonds.get('O_double', 0) == 1 and prod_bonds.get('O', 0) == 1:
        return "carbonyl_to_alcohol"
    
    # Isomerization: position change without oxidation state change
    if get_oxidation_state(substrate_atom) == get_oxidation_state(product_atom):
        return "isomerization"
    
    return "other"
```

## Rule-Based Baseline System

### Reaction Rules (RetroRules format)
```python
OXIDATION_RULES = [
    {
        "name": "primary_alcohol_oxidation",
        "smarts": "[CH2:1][OH:2] >> [CH:1]=[O:2]",
        "ec_class": "1.1.1",
        "cofactor": "NAD+",
        "reversible": True
    },
    {
        "name": "secondary_alcohol_oxidation",
        "smarts": "[CH:1]([OH:2]) >> [C:1](=[O:2])",
        "ec_class": "1.1.1",
        "cofactor": "NAD+",
        "reversible": True
    }
]

ISOMERIZATION_RULES = [
    {
        "name": "aldose_ketose_isomerization",
        "smarts": "[CH:1]=[O:2].[CH2:3][OH:4] >> [CH:1][OH:2].[C:3](=[O:4])",
        "ec_class": "5.3.1",
        "reversible": True
    }
]
```

### Rule Application Engine
```python
class RuleBasedPredictor:
    def __init__(self, rules):
        self.rules = rules
        
    def predict_reaction_centers(self, smiles, reaction_type):
        """
        Apply rules to identify possible reaction centers.
        
        Returns:
            List[Dict]: Possible reaction centers with scores
        """
        mol = Chem.MolFromSmiles(smiles)
        candidates = []
        
        for rule in self.get_rules_for_type(reaction_type):
            matches = mol.GetSubstructMatches(rule.substrate_pattern)
            
            for match in matches:
                candidates.append({
                    "atom_indices": match,
                    "rule_name": rule.name,
                    "ec_class": rule.ec_class,
                    "confidence": self.calculate_rule_confidence(mol, match, rule)
                })
        
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    
    def calculate_rule_confidence(self, mol, match, rule):
        """
        Score rule applicability based on molecular context.
        """
        score = 1.0
        
        # Penalize steric hindrance
        if self.has_steric_clash(mol, match):
            score *= 0.5
            
        # Boost if cofactor binding site nearby
        if self.has_cofactor_binding_motif(mol, match):
            score *= 1.5
            
        # Consider pH/pKa if available
        # Consider known enzyme specificity
        
        return score
```

## Machine Learning Model (Phase 2)

### Feature Engineering
```python
def extract_atom_features(mol, atom_idx):
    """
    Extract features for a specific atom in context of the molecule.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    features = {
        # Atom properties
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "hybridization": atom.GetHybridization(),
        "is_aromatic": atom.GetIsAromatic(),
        "formal_charge": atom.GetFormalCharge(),
        
        # Local environment
        "num_h": atom.GetTotalNumHs(),
        "num_hetero_neighbors": count_hetero_neighbors(atom),
        "in_ring": atom.IsInRing(),
        "ring_size": get_smallest_ring_size(atom) if atom.IsInRing() else 0,
        
        # Chemical context
        "is_hydroxyl_carbon": is_hydroxyl_carbon(atom),
        "is_carbonyl_carbon": is_carbonyl_carbon(atom),
        "distance_to_nearest_oh": get_distance_to_functional_group(mol, atom_idx, "OH"),
        
        # Reactivity indicators
        "partial_charge": calculate_partial_charge(mol, atom_idx),
        "accessibility": calculate_steric_accessibility(mol, atom_idx),
    }
    
    return features
```

### Simple ML Model
```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class ReactionCenterPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
    def prepare_training_data(self, reactions):
        """
        Convert reaction dataset to ML format.
        """
        X, y = [], []
        
        for rxn in reactions:
            mol = Chem.MolFromSmiles(rxn.substrate_smiles)
            
            # Create one sample per carbon atom
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                
                # Only consider carbons for now
                if atom.GetAtomicNum() != 6:
                    continue
                
                features = extract_atom_features(mol, atom_idx)
                X.append(list(features.values()))
                
                # Label: 1 if this is the reaction center, 0 otherwise
                y.append(1 if atom_idx in rxn.reaction_center.atom_indices else 0)
        
        return np.array(X), np.array(y)
    
    def predict(self, smiles, reaction_type):
        """
        Predict reaction center probabilities for each atom.
        """
        mol = Chem.MolFromSmiles(smiles)
        predictions = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            
            if atom.GetAtomicNum() != 6:
                continue
                
            features = extract_atom_features(mol, atom_idx)
            prob = self.model.predict_proba([list(features.values())])[0][1]
            
            predictions.append({
                "atom_idx": atom_idx,
                "probability": prob
            })
        
        return sorted(predictions, key=lambda x: x["probability"], reverse=True)
```

## Evaluation Framework

### Metrics Implementation
```python
class ReactionCenterEvaluator:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        
    def evaluate(self, predictor):
        """
        Evaluate predictor on test set.
        """
        results = {
            "top1_accuracy": 0,
            "top3_accuracy": 0,
            "top5_accuracy": 0,
            "mean_rank": [],
            "by_reaction_type": {}
        }
        
        for rxn in self.test_dataset:
            predictions = predictor.predict(rxn.substrate_smiles, rxn.reaction_type)
            true_center = rxn.reaction_center.atom_indices[0]
            
            # Find rank of true center
            predicted_atoms = [p["atom_idx"] for p in predictions]
            rank = predicted_atoms.index(true_center) + 1 if true_center in predicted_atoms else len(predicted_atoms)
            
            results["mean_rank"].append(rank)
            
            if rank == 1:
                results["top1_accuracy"] += 1
            if rank <= 3:
                results["top3_accuracy"] += 1
            if rank <= 5:
                results["top5_accuracy"] += 1
        
        # Convert to percentages
        n = len(self.test_dataset)
        results["top1_accuracy"] = 100 * results["top1_accuracy"] / n
        results["top3_accuracy"] = 100 * results["top3_accuracy"] / n
        results["top5_accuracy"] = 100 * results["top5_accuracy"] / n
        results["mean_rank"] = np.mean(results["mean_rank"])
        
        return results
    
    def practical_impact_analysis(self, results):
        """
        Translate accuracy to real-world cost savings.
        """
        baseline_screening_cost = 100  # Test 100 enzymes
        
        if results["top5_accuracy"] > 80:
            reduced_cost = 5  # Only need to test top 5
            savings = 100 * (1 - reduced_cost / baseline_screening_cost)
            
            return f"Top-5 accuracy of {results['top5_accuracy']:.1f}% reduces screening cost by {savings:.0f}%"
```

### Benchmark Comparisons
```python
def compare_approaches():
    """
    Compare rule-based vs ML approaches.
    """
    evaluator = ReactionCenterEvaluator(test_data)
    
    # Baseline: Random
    random_predictor = RandomPredictor()
    random_results = evaluator.evaluate(random_predictor)
    
    # Rule-based
    rule_predictor = RuleBasedPredictor(OXIDATION_RULES + ISOMERIZATION_RULES)
    rule_results = evaluator.evaluate(rule_predictor)
    
    # ML
    ml_predictor = ReactionCenterPredictor()
    ml_predictor.train(training_data)
    ml_results = evaluator.evaluate(ml_predictor)
    
    # Hybrid (rules + ML)
    hybrid_predictor = HybridPredictor(rule_predictor, ml_predictor)
    hybrid_results = evaluator.evaluate(hybrid_predictor)
    
    return {
        "random": random_results,
        "rule_based": rule_results,
        "ml": ml_results,
        "hybrid": hybrid_results
    }
```

## Expected Performance Targets

### Phase 1 (Rule-based)
- Top-1: 30-40%
- Top-5: 60-70%
- **Already practically useful**

### Phase 2 (Simple ML)
- Top-1: 50-60%
- Top-5: 80-85%
- **Production-ready for screening**

### Phase 3 (Advanced)
- Top-1: 70%+
- Top-5: 90%+
- **Competitive with expert chemists**

## Implementation Priority

1. **Week 1-2**: Data extraction from Rhea
2. **Week 3**: Rule-based baseline
3. **Week 4**: Evaluation framework
4. **Week 5-6**: Simple ML model
5. **Week 7**: Integration and testing
6. **Week 8**: Documentation and expansion planning
