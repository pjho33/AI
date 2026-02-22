# PQQ+Ca Transplant Guide

**Goal**: Add PQQ and Ca²⁺ to AlphaFold structure using homolog alignment

---

## Prerequisites

- `structures/receptor_af.pdb` (AlphaFold structure)
- ChimeraX installed
- Homolog PDB with PQQ+Ca

---

## Step 1: Find Suitable Homolog

### Recommended PDB Structures

**Option 1: 1WPQ** (Glucose dehydrogenase)
- Organism: Acinetobacter calcoaceticus
- Resolution: 1.8 Å
- Contains: PQQ + Ca²⁺
- Good for: General PQQ-DH

**Option 2: 2CDU** (Methanol dehydrogenase)
- Organism: Methylobacterium extorquens
- Resolution: 1.9 Å
- Contains: PQQ + Ca²⁺

**Option 3: 3PQQ** (Quinoprotein glucose DH)
- Contains: PQQ + Ca²⁺
- Well-resolved cofactor

### Download Homolog

```bash
cd structures/

# Download from RCSB PDB
wget https://files.rcsb.org/download/1WPQ.pdb

# Or use ChimeraX: File → Fetch by ID → 1WPQ
```

---

## Step 2: ChimeraX Alignment & Transplant

### Method A: Automated (Recommended)

```bash
# Open ChimeraX
chimerax

# In ChimeraX command line:
open receptor_af.pdb
open 1WPQ.pdb

# Align structures (matchmaker)
matchmaker #1 to #2 pairing ss

# Select PQQ and Ca from homolog
select #2 & (ligand | :CA)

# Copy coordinates
combine #1,#2 close false modelId 3

# Delete homolog protein, keep only PQQ+Ca
delete #2 & protein

# Save combined structure
save receptor_PQQ_Ca.pdb #3
```

### Method B: Manual (More Control)

1. **Open both structures**
   ```
   open receptor_af.pdb
   open 1WPQ.pdb
   ```

2. **Align active sites**
   ```
   # Select conserved residues around PQQ binding site
   # Use matchmaker with specific residues
   matchmaker #1:100-150,200-250 to #2:100-150,200-250 pairing ss
   ```

3. **Verify alignment**
   ```
   # Check RMSD in log
   # Should be < 2 Å for active site
   ```

4. **Extract PQQ+Ca**
   ```
   select #2 & ligand
   select #2 & :CA
   # Copy selection
   ```

5. **Combine structures**
   ```
   combine #1,#2 close false
   delete #2 & protein
   save receptor_PQQ_Ca.pdb #3
   ```

---

## Step 3: Verify Transplant

### Check 1: Visual Inspection

```bash
# Open result
open receptor_PQQ_Ca.pdb

# Check:
# - PQQ in binding pocket?
# - Ca²⁺ coordinated properly?
# - No steric clashes?
# - Reasonable geometry?
```

### Check 2: Residue Coordination

**Expected PQQ coordination**:
- 2-3 conserved Trp/Tyr residues (π-stacking)
- Asp/Glu for Ca²⁺ coordination
- His/Asn for H-bonding

**Expected Ca²⁺ coordination**:
- 6-8 oxygen atoms
- From: Asp, Glu, PQQ oxygens, water

### Check 3: Distances

```python
# In ChimeraX
distance #1:PQQ@C5 #1:CA@CA
# Should be ~4-6 Å

# Check protein-PQQ contacts
select #1 & protein & <5 #1 & ligand
# Should show conserved residues
```

---

## Step 4: Clean Structure

### Remove Waters (Optional)

```bash
# In ChimeraX
delete solvent
save receptor_PQQ_Ca_clean.pdb
```

### Fix Residue Names

Ensure PQQ and Ca have correct names:
- PQQ: `PQQ` (residue name)
- Ca²⁺: `CA` (residue name)

```bash
# Check in text editor
grep "HETATM" receptor_PQQ_Ca.pdb | grep -E "PQQ|CA"
```

---

## Step 5: Validate Geometry

### Option 1: MolProbity

Upload to: http://molprobity.biochem.duke.edu/

Check:
- Clashscore
- Ramachandran outliers
- Rotamer outliers

### Option 2: PDBeFold

Check structural alignment quality:
https://www.ebi.ac.uk/msd-srv/ssm/

---

## Common Issues

### Issue: PQQ too far from active site
**Solution**: 
- Re-align using active site residues only
- Manually adjust PQQ position

### Issue: Ca²⁺ missing
**Solution**:
- Check homolog PDB has Ca
- Verify Ca wasn't deleted during combine

### Issue: Steric clashes
**Solution**:
- Run brief energy minimization
- Adjust side chain rotamers

### Issue: Wrong residue names
**Solution**:
```bash
# Edit PDB file
sed -i 's/OLD_NAME/PQQ    /g' receptor_PQQ_Ca.pdb
```

---

## Alternative: Manual Coordinate Copy

If ChimeraX alignment fails:

```python
# Python script to copy PQQ+Ca coordinates
import numpy as np

# 1. Load both PDBs
# 2. Find transformation matrix from alignment
# 3. Apply to PQQ+Ca coordinates
# 4. Write to new PDB

# (Detailed script available on request)
```

---

## Final Output

```
structures/
├── sequence.fasta
├── receptor_af.pdb              # AlphaFold model
├── 1WPQ.pdb                     # Homolog reference
└── receptor_PQQ_Ca.pdb         # Final structure ✓
```

---

## Verification Checklist

- [ ] PQQ in binding pocket
- [ ] Ca²⁺ present and coordinated
- [ ] No major steric clashes
- [ ] Conserved residues near PQQ
- [ ] Residue names correct (PQQ, CA)
- [ ] Structure saved as receptor_PQQ_Ca.pdb

---

## Next Step

Once `receptor_PQQ_Ca.pdb` is ready:
→ **Phase 2: Parameterization** (CGenFF for PQQ and sorbitol)

See: `docs/PARAMETERIZATION_GUIDE.md`
