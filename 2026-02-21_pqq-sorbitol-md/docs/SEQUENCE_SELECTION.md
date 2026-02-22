# Sequence Selection Guide

**Critical**: Must use correct MB PQQ-dependent dehydrogenase, NOT yeast THR4

---

## Target Enzyme Characteristics

### Required Features
1. **Membrane-bound** PQQ-dependent dehydrogenase
2. **β-propeller** catalytic domain (periplasmic)
3. **Polyol/sorbitol** substrate specificity
4. **PQQ + Ca²⁺** cofactors

### Common Sources
- **Gluconobacter oxydans** (sldA, sldB)
- **Gluconobacter frateurii**
- **Acetobacter** species
- Other acetic acid bacteria

---

## Recommended Sequences

### Option 1: Gluconobacter oxydans SldA (RECOMMENDED)

**UniProt ID**: Q5FPE8  
**Gene**: sldA (sorbitol dehydrogenase large subunit)  
**Organism**: Gluconobacter oxydans 621H  
**Length**: ~700 aa (full length with TM domain)

**Catalytic domain**: Residues ~50-650 (exclude N-terminal anchor)

**Why this one**:
- Well-characterized PQQ-dependent sorbitol DH
- Known structure homologs available
- Proven sorbitol activity
- Good for L vs D comparison

### Option 2: Gluconobacter oxydans SldB

**UniProt ID**: Q5FPE7  
**Gene**: sldB (sorbitol dehydrogenase small subunit)  
**Note**: Often works with SldA, but SldA is the catalytic subunit

### Option 3: Generic PQQ-DH from Gluconobacter

Search UniProt:
```
organism:"Gluconobacter" AND keyword:"PQQ" AND keyword:"dehydrogenase"
```

---

## How to Get Sequence

### Method 1: UniProt (Recommended)

1. Go to https://www.uniprot.org
2. Search: `Q5FPE8` or `sldA Gluconobacter oxydans`
3. Download FASTA format
4. **Remove signal peptide and TM domain** (keep ~residues 50-650)

### Method 2: NCBI

1. Go to https://www.ncbi.nlm.nih.gov/protein
2. Search: `Gluconobacter oxydans sorbitol dehydrogenase`
3. Select appropriate entry
4. Download FASTA

### Method 3: PDB Homolog Search

1. Search PDB: `PQQ dehydrogenase`
2. Find structures with PQQ + Ca
3. Use sequence from PDB entry
4. Examples:
   - 1WPQ: Glucose dehydrogenase
   - 2CDU: Methanol dehydrogenase
   - 3PQQ: Quinoprotein glucose DH

---

## Sequence Preparation

### 1. Identify Domain Boundaries

**Full sequence structure**:
```
[Signal peptide] - [TM anchor] - [Catalytic domain (β-propeller)] - [C-term]
     ~1-30            ~30-50           ~50-650                      ~650-700
```

**Keep only**: Catalytic domain (~50-650)

### 2. Remove TM Regions

Use prediction tools:
- TMHMM: https://services.healthtech.dtu.dk/service.php?TMHMM-2.0
- Phobius: https://phobius.sbc.su.se/

**Goal**: Soluble periplasmic domain only

### 3. Save to File

```bash
cd structures/

# Create sequence.fasta
cat > sequence.fasta << 'EOF'
>Gluconobacter_oxydans_SldA_catalytic_domain
MKTLALAAVALSLPLLAGA...  (catalytic domain sequence)
EOF
```

---

## Example: Gluconobacter oxydans SldA

### Full Sequence (Q5FPE8)
```
>sp|Q5FPE8|SLDH_GLUOX Sorbitol dehydrogenase large subunit OS=Gluconobacter oxydans
MKTLALAAVALSLPLLAGAAPALAADPVTITGGSSGIGLAIARRLAAEGARVVVADRDP
ARLAEALAAAGVDRVLGVDVTDAAALVAAAVAAGGRAAALVGAGGRVDVAVNNAGVAPV
TPWDQAQQRWVDAAFAAAGPGDAALIVGSGPVGLLAAARGAADGVVNVVSSAAQRSPVG
...
```

### Catalytic Domain Only (for modeling)
```
>Gluconobacter_oxydans_SldA_catalytic
DPVTITGGSSGIGLAIARRLAAEGARVVVADRDPARLAEALAAAGVDRVLGVDVTDAAA
LVAAAVAAGGRAAALVGAGGRVDVAVNNAGVAPVTPWDQAQQRWVDAAFAAAGPGDAAA
LIVGSGPVGLLAAARGAADGVVNVVSSAAQRSPVG...
(~600 residues)
```

---

## Verification Checklist

Before proceeding to AlphaFold:

- [ ] Sequence is from MB PQQ-DH (NOT cytoplasmic enzyme)
- [ ] Signal peptide removed
- [ ] TM domain removed
- [ ] Length ~500-650 aa (catalytic domain)
- [ ] Contains conserved PQQ-binding motifs
- [ ] Organism is acetic acid bacteria (Gluconobacter/Acetobacter)

---

## Next Steps

1. **Save sequence** to `structures/sequence.fasta`
2. **Run AlphaFold** (see ALPHAFOLD_GUIDE.md)
3. **Verify structure** has β-propeller fold
4. **Proceed to PQQ transplant**

---

## Common Mistakes to Avoid

❌ Using yeast THR4 (completely different enzyme)  
❌ Including TM domain (will cause modeling issues)  
❌ Using cytoplasmic NAD-dependent DH  
❌ Forgetting to remove signal peptide  

✓ Use Gluconobacter PQQ-DH  
✓ Catalytic domain only  
✓ Verify β-propeller fold  
✓ Check for PQQ-binding residues  
