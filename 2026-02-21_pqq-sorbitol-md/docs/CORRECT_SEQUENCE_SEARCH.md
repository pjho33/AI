# Finding Correct Sorbitol Dehydrogenase Sequence

**Problem**: Q5FPE8 is TamB domain protein, NOT sorbitol dehydrogenase

---

## Correct Search Strategy

### Option 1: Direct Gene Name Search

**Search UniProt**:
```
organism:"Gluconobacter oxydans" AND gene:sldA
organism:"Gluconobacter oxydans" AND gene:sldB
```

### Option 2: Functional Search

**Search UniProt**:
```
organism:"Gluconobacter oxydans" AND name:"sorbitol dehydrogenase"
organism:"Gluconobacter oxydans" AND name:"PQQ" AND name:"dehydrogenase"
organism:"Gluconobacter" AND keyword:"PQQ" AND keyword:"sorbitol"
```

### Option 3: Use Known PDB Structures

**Known PQQ-DH structures**:
- **1WPQ**: Glucose dehydrogenase (Acinetobacter)
- **2CDU**: Methanol dehydrogenase (Methylobacterium)
- **3PQQ**: Quinoprotein glucose DH

**Strategy**: Use sequence from these PDBs directly

---

## Recommended Sequences

### Option A: Use PDB Structure Directly

**1WPQ - Glucose Dehydrogenase**
- Organism: Acinetobacter calcoaceticus
- Resolution: 1.8 Å
- Has: PQQ + Ca²⁺
- Length: ~550 aa
- **Advantage**: Structure already available, no AlphaFold needed

**How to use**:
```bash
# Download from PDB
wget https://files.rcsb.org/download/1WPQ.pdb

# Use directly as receptor_PQQ_Ca.pdb
# Skip AlphaFold step entirely
```

### Option B: Find Gluconobacter Sequence

**Search NCBI Protein**:
1. Go to: https://www.ncbi.nlm.nih.gov/protein
2. Search: `Gluconobacter oxydans sorbitol dehydrogenase`
3. Look for entries with:
   - "quinoprotein" or "PQQ"
   - "membrane-bound"
   - Gene name: sldA or sldB

### Option C: Use Related Organism

**Acetobacter aceti** or **Gluconobacter frateurii**
- Also have PQQ-dependent sorbitol DH
- May have better annotations

---

## Quick Solution: Use 1WPQ Directly

**Fastest approach** (skip AlphaFold):

```bash
cd structures/

# Download PDB structure
wget https://files.rcsb.org/download/1WPQ.pdb -O receptor_PQQ_Ca.pdb

# This already has:
# - Protein structure (glucose DH, similar to sorbitol DH)
# - PQQ cofactor
# - Ca²⁺ ion
# - Well-resolved structure

# Ready for Phase 2 (parameterization)
```

**Pros**:
- Immediate use
- Experimentally validated structure
- PQQ+Ca already in place
- No AlphaFold needed

**Cons**:
- Glucose DH, not sorbitol DH
- Different organism
- May have different substrate specificity

---

## Alternative: Search by EC Number

**EC 1.1.5.2**: Quinoprotein glucose dehydrogenase  
**EC 1.1.5.-**: Quinoprotein dehydrogenases

**Search**:
```
ec:1.1.5.2 AND organism:Gluconobacter
```

---

## What to Do Now

### Recommended Path

**Option 1 (Fastest)**: Use 1WPQ structure directly
- Download 1WPQ.pdb
- Use as receptor_PQQ_Ca.pdb
- Proceed to Phase 2

**Option 2 (More accurate)**: Find correct Gluconobacter sequence
- Search NCBI/UniProt more carefully
- Look for sldA gene specifically
- Run AlphaFold
- Transplant PQQ from 1WPQ

**Option 3 (Hybrid)**: Use 1WPQ sequence
- Extract sequence from 1WPQ.pdb
- Run AlphaFold for better model
- Transplant PQQ back

---

## Next Steps

**Tell me which option you prefer**:

1. **Use 1WPQ directly** (fastest, ready in 5 min)
2. **Find correct Gluconobacter sldA** (more accurate, need to search)
3. **Use different organism** (e.g., Acetobacter)

Or provide a different UniProt/NCBI ID if you found one!
