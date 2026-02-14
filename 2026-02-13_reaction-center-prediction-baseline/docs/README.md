# Chemical Reaction Prediction & Enzyme Selection AI

## Project Overview

AI system for predicting chemical reaction centers and selecting appropriate enzymes for biosynthetic pathways.

### Core Philosophy

> "We don't try to predict all chemical reactions.  
> We conquer reactions that AI can learn well first,  
> then expand the framework to reactions of interest."

## Phase 0 Domain (Pilot)

**Target Reactions**: Polyol/sugar oxidation-reduction + isomerization

**Representative Reactions**:
- sorbitol ⇌ fructose
- glucose ⇌ fructose  
- xylitol ⇌ xylulose

**EC Classes**:
- EC 1.1.1.x (NAD⁺-dependent dehydrogenase)
- EC 5.3.x (isomerase)

**Why This Domain?**
- Clear reaction centers
- Abundant data (Rhea, KEGG)
- Simple cofactor patterns (NAD⁺/NADH)
- Direct relevance to biosynthesis research

## Reaction Classification (AI Perspective)

### A. Atom-conserving reactions ★☆☆☆☆
- Carbon skeleton preserved, functional groups change
- Examples: acid-base, isomerization, phosphorylation, hydrolysis
- **70%+ of enzyme reactions**
- **Primary AI target**

### B. Electron transfer reactions (Redox) ★★☆☆☆
- Skeleton preserved, oxidation state changes
- Examples: alcohol ⇌ aldehyde/ketone
- Cofactor-dependent (NAD⁺/NADH)
- **Secondary AI target**

### C. Bond rearrangement ★★★★☆
- C-C, C-N, C-S bond formation/cleavage
- Examples: aldol, decarboxylation, lyase reactions
- **Advanced stage**

### D. Radical/high-energy ★★★★★
- **Intentionally excluded** from current scope

## AI Problem Definition

### The Question AI Answers

> "In this molecule, which atom/bond is most likely to undergo reaction?"

**NOT**: "What is the product?"  
**NOT**: "What is the reaction pathway?"  
**YES**: "Where is the reaction center?"

### Input
```
- Substrate SMILES
- Reaction class (oxidation / isomerization)
- Cofactor (NAD⁺, optional)
```

### Output
```
- Reaction center atom index
- Transformation type (oxidation / isomerization / hydrolysis)
```

### Label (for training)
```
- Reaction center atom index (from atom-mapped reactions)
- EC number
- Enzyme name
```

## Implementation Strategy

### Phase 1: Rule-based Baseline
- Apply RetroRules / EC rules
- "Is this reaction type applicable to this molecule?"
- **Already reduces search space by 90%**
- **Already practically useful**

### Phase 2: Simple ML
- Input: Morgan fingerprint + reaction type
- Output: Reaction likelihood score per atom
- Model: Gradient boosting / simple NN
- **Graph Neural Networks are for later**

### Phase 3: Expansion
- Other polyols
- Other EC 1 classes
- Selective oxidation of specific carbons

## Evaluation Metrics

### Minimum
- **Top-1 accuracy**: Is the highest-scoring atom correct?
- **Top-3 accuracy**

### Practical
- **Top-5 accuracy**: If correct atom is in top 5, consider success
- **Real impact**: Reduces enzyme screening cost by >90%

## Project Structure

```
.
├── data/
│   ├── raw/              # Rhea, KEGG raw data
│   ├── processed/        # Extracted reaction centers
│   └── rules/            # RetroRules, EC rules
├── src/
│   ├── data_extraction/  # Rhea parser, atom mapping
│   ├── rules/            # Rule-based predictor
│   ├── models/           # ML models
│   └── evaluation/       # Metrics, benchmarks
├── notebooks/            # Exploratory analysis
├── tests/
└── configs/
```

## Success Criteria

This pilot succeeds when we have:
- ✓ Reaction classification system
- ✓ Reaction center definition method
- ✓ Data pipeline (Rhea → labeled dataset)
- ✓ Evaluation framework

These become **fixed assets** for expansion.

## Excluded Scope (Critical)

❌ C-C bond formation  
❌ Radical chemistry  
❌ Photoreactions  
❌ Multi-step cascades  

✅ Atom skeleton preservation  
✅ Single reaction center  
✅ Enzyme reactions  

## Next Steps

1. Extract Rhea data with atom mapping
2. Build reaction center labeling logic
3. Implement rule-based baseline
4. Develop simple ML model
5. Establish evaluation framework
