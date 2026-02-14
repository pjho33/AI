# Implementation Guide

## Quick Start

### 1. Setup Environment

```bash
cd /home/pjho3/projects/AI

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pilot Demo

```bash
# Start Jupyter
jupyter notebook notebooks/01_pilot_demo.ipynb
```

### 3. Test Rule-Based Predictor

```bash
# Test the predictor
python src/rules/rule_based_predictor.py

# Test evaluation framework
python src/evaluation/evaluator.py
```

## Core Components

### A. Data Extraction (`src/data_extraction/`)

**Purpose**: Extract and process reaction data from Rhea database

**Key Files**:
- `rhea_parser.py` - Download and parse Rhea reactions
- `reaction_center_extractor.py` - Extract reaction centers from atom-mapped SMILES

**Usage**:
```python
from data_extraction.rhea_parser import RheaParser

parser = RheaParser(cache_dir="data/raw")
reactions = parser.extract_reactions()
```

### B. Rule-Based Prediction (`src/rules/`)

**Purpose**: Predict reaction centers using chemical transformation rules

**Key Files**:
- `rule_based_predictor.py` - Rule engine and predictor

**Usage**:
```python
from rules.rule_based_predictor import RuleBasedPredictor

predictor = RuleBasedPredictor()
predictions = predictor.predict_reaction_centers(
    smiles="OCC(O)C(O)C(O)C(O)CO",
    reaction_type="oxidation"
)

for pred in predictions[:3]:
    print(f"Atoms: {pred.atom_indices}, Confidence: {pred.confidence:.2f}")
```

### C. Evaluation (`src/evaluation/`)

**Purpose**: Evaluate predictor performance with Top-K metrics

**Key Files**:
- `evaluator.py` - Evaluation framework and metrics

**Usage**:
```python
from evaluation.evaluator import ReactionCenterEvaluator, TestSample

test_data = [
    TestSample(
        substrate_smiles="OCC(O)C(O)C(O)C(O)CO",
        true_reaction_center=[2],
        reaction_type="oxidation",
        ec_number="1.1.1.14",
        reaction_id="test_1"
    )
]

evaluator = ReactionCenterEvaluator(test_data)
result = evaluator.evaluate(predictor)
evaluator.print_report(result)
```

## Development Workflow

### Phase 1: Rule-Based Baseline (Current)

**Goal**: Establish baseline performance with chemical rules

**Tasks**:
- ✓ Define reaction rules for oxidation/isomerization
- ✓ Implement rule matching engine
- ✓ Create evaluation framework
- ⧗ Test on known reactions
- ⧗ Benchmark performance

**Expected Performance**:
- Top-1: 30-40%
- Top-5: 60-70%

### Phase 2: Data Pipeline

**Goal**: Build labeled dataset from Rhea

**Tasks**:
1. Download Rhea reaction database
2. Filter polyol/sugar reactions (EC 1.1.1, 5.3)
3. Extract reaction centers from atom mapping
4. Validate labels manually (sample)
5. Split train/test sets

**Code**:
```python
from data_extraction.rhea_parser import RheaParser
from data_extraction.reaction_center_extractor import ReactionCenterExtractor

# Extract reactions
parser = RheaParser()
reactions = parser.extract_reactions()

# Extract reaction centers
extractor = ReactionCenterExtractor()
labeled_data = []

for rxn in reactions:
    if rxn.atom_mapped_smiles:
        center = extractor.extract_from_mapped_smiles(rxn.atom_mapped_smiles)
        if center:
            labeled_data.append({
                'reaction_id': rxn.reaction_id,
                'substrate': rxn.substrate_smiles,
                'product': rxn.product_smiles,
                'reaction_center': center.atom_indices,
                'change_type': center.change_type
            })
```

### Phase 3: Machine Learning Model

**Goal**: Improve prediction with ML

**Approach**:
1. Extract molecular features (Morgan fingerprints, atom features)
2. Train gradient boosting classifier
3. Predict reaction likelihood per atom
4. Compare with rule-based baseline

**Code Template**:
```python
from sklearn.ensemble import GradientBoostingClassifier
from rdkit.Chem import AllChem
import numpy as np

class MLPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5
        )
    
    def extract_features(self, mol, atom_idx):
        # Molecular fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        
        # Atom-specific features
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_features = [
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            # ... more features
        ]
        
        return np.concatenate([fp, atom_features])
    
    def train(self, labeled_reactions):
        X, y = [], []
        
        for rxn in labeled_reactions:
            mol = Chem.MolFromSmiles(rxn['substrate'])
            
            for atom_idx in range(mol.GetNumAtoms()):
                features = self.extract_features(mol, atom_idx)
                X.append(features)
                
                # Label: 1 if reaction center, 0 otherwise
                y.append(1 if atom_idx in rxn['reaction_center'] else 0)
        
        self.model.fit(X, y)
    
    def predict_reaction_centers(self, smiles, reaction_type):
        mol = Chem.MolFromSmiles(smiles)
        predictions = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            features = self.extract_features(mol, atom_idx)
            prob = self.model.predict_proba([features])[0][1]
            
            predictions.append({
                'atom_indices': [atom_idx],
                'confidence': prob
            })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
```

**Expected Performance**:
- Top-1: 50-60%
- Top-5: 80-85%

### Phase 4: Hybrid System

**Goal**: Combine rules and ML for best performance

**Strategy**:
1. Use rules to filter candidates
2. Use ML to rank filtered candidates
3. Ensemble predictions

## Testing Strategy

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

Test full pipeline:
1. Load molecule
2. Predict reaction centers
3. Validate predictions
4. Calculate metrics

### Validation

Manual validation of predictions:
- Sample 50 reactions
- Expert review of top-3 predictions
- Calculate agreement rate

## Performance Targets

### Minimum Viable Product (MVP)

- Top-5 accuracy: **≥70%**
- Processing time: **<1 second per molecule**
- Coverage: **≥100 reactions in test set**

### Production Ready

- Top-5 accuracy: **≥85%**
- Top-1 accuracy: **≥60%**
- Handles edge cases gracefully
- Clear confidence scores

### Research Grade

- Top-5 accuracy: **≥90%**
- Top-1 accuracy: **≥70%**
- Explainable predictions
- Uncertainty quantification

## Practical Impact

### Cost Reduction

**Baseline**: Screen 100 enzyme candidates
- Cost: $50,000
- Time: 12 weeks

**With AI (Top-5 accuracy = 85%)**:
- Screen: 5 candidates
- Cost: $2,500 (95% reduction)
- Time: 1 week (92% reduction)

### Use Cases

1. **Enzyme Discovery**: Identify promising enzymes for novel reactions
2. **Pathway Design**: Predict feasibility of biosynthetic routes
3. **Reaction Optimization**: Guide mutagenesis for improved selectivity
4. **Literature Mining**: Validate reported enzyme activities

## Expansion Roadmap

### Short Term (1-3 months)
- Complete Phase 1-2
- Validate on 100+ reactions
- Publish initial results

### Medium Term (3-6 months)
- Add EC 2 (Transferase) reactions
- Incorporate cofactor specificity
- Build enzyme recommendation system

### Long Term (6-12 months)
- Expand to C-C bond formation (EC 4, 6)
- Multi-step pathway prediction
- Integration with protein structure data

## Troubleshooting

### Common Issues

**Issue**: Low accuracy on certain reaction types
- **Solution**: Add more specific rules for that type
- **Solution**: Collect more training data

**Issue**: Slow prediction
- **Solution**: Cache molecular fingerprints
- **Solution**: Parallelize predictions

**Issue**: Poor generalization
- **Solution**: Use cross-validation
- **Solution**: Add regularization to ML model

## References

### Databases
- **Rhea**: https://www.rhea-db.org/
- **KEGG**: https://www.genome.jp/kegg/
- **RetroRules**: https://retrorules.org/

### Tools
- **RDKit**: https://www.rdkit.org/
- **scikit-learn**: https://scikit-learn.org/

### Literature
- Enzyme nomenclature: https://www.enzyme-database.org/
- Reaction classification: https://doi.org/10.1021/ci200379p
