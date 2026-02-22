# ChimeraX PQQ+Ca Transplant Instructions

**Goal**: Add PQQ and Ca²⁺ from 1WPQ to your AlphaFold structure

---

## 📁 Files Needed

✓ `structures/receptor_af.pdb` (your AlphaFold model)  
✓ `structures/1WPQ.pdb` (homolog with PQQ+Ca, downloaded)  
✓ `scripts/chimerax_transplant.cxc` (automated script)

---

## Method 1: Automated Script (Recommended)

### Step 1: Open ChimeraX

```bash
chimerax
```

### Step 2: Run Script

In ChimeraX command line:
```
cd /home/pjho3/projects/AI/2026-02-21_pqq-sorbitol-md/structures
open ../scripts/chimerax_transplant.cxc
```

Or use menu: **File → Open** → select `chimerax_transplant.cxc`

### Step 3: Verify Result

Check:
- ✓ PQQ in binding pocket
- ✓ Ca²⁺ present (yellow sphere)
- ✓ No major clashes
- ✓ File saved: `receptor_PQQ_Ca.pdb`

---

## Method 2: Manual (Step-by-Step)

### Step 1: Open Structures

```
open receptor_af.pdb
open 1WPQ.pdb
```

### Step 2: Align

```
matchmaker #1 to #2 pairing ss
```

**Check RMSD in log** (should be < 3 Å for good alignment)

### Step 3: Select PQQ and Ca from 1WPQ

```
select #2 & ligand
select #2 & :CA add
```

### Step 4: Combine Structures

```
combine #1,#2 close false modelId 3
```

### Step 5: Remove 1WPQ Protein

```
delete #2 & protein
```

### Step 6: Visualize

```
show #3 cartoons
show #3 & ligand sticks
color #3 & ligand green
show #3 & :CA spheres
color #3 & :CA yellow
```

### Step 7: Save

```
save receptor_PQQ_Ca.pdb #3
```

---

## Verification Checklist

After transplant, verify:

- [ ] PQQ is in central cavity (binding pocket)
- [ ] Ca²⁺ is near PQQ (~4-6 Å)
- [ ] No severe steric clashes (visual check)
- [ ] Conserved residues near PQQ (Trp, Tyr, Asp)
- [ ] File saved: `structures/receptor_PQQ_Ca.pdb`

---

## Visual Checks

### Good Signs:
- PQQ in pocket center
- Ca²⁺ coordinated by protein residues
- Aromatic residues (W, Y) stacking with PQQ
- Compact structure

### Bad Signs:
- PQQ outside protein
- Ca²⁺ floating in solvent
- Major atom overlaps (clashes)
- PQQ far from active site

---

## Troubleshooting

### Issue: Poor alignment (RMSD > 5 Å)
**Solution**: 
```
# Try aligning specific regions
matchmaker #1:100-200,300-400 to #2:100-200,300-400 pairing ss
```

### Issue: PQQ in wrong position
**Solution**:
- Check if 1WPQ has PQQ (should be residue PQQ or PQQ)
- Verify selection: `select #2 & ligand`

### Issue: Ca missing
**Solution**:
```
# Explicitly select Ca
select #2 & :CA
# Or by element
select #2 & element Ca
```

---

## Alternative: Use Different Homolog

If 1WPQ doesn't work well:

**2CDU** (Methanol dehydrogenase):
```
open 2CDU.pdb
matchmaker #1 to #3 pairing ss
```

**3PQQ** (Quinoprotein glucose DH):
```
open 3PQQ.pdb
matchmaker #1 to #4 pairing ss
```

---

## Expected Result

**File**: `structures/receptor_PQQ_Ca.pdb`

**Contents**:
- Protein from receptor_af.pdb
- PQQ cofactor from 1WPQ
- Ca²⁺ ion from 1WPQ
- Proper alignment and positioning

---

## Next Step

Once `receptor_PQQ_Ca.pdb` is ready:

→ **Phase 2: Parameterization**
- Generate PQQ parameters (CGenFF)
- Generate L-sorbitol parameters
- Generate D-sorbitol parameters
- Create PQQ position restraints

See: `docs/PARAMETERIZATION_GUIDE.md`

---

## Quick Commands Reference

```bash
# Open ChimeraX
chimerax

# In ChimeraX:
cd /home/pjho3/projects/AI/2026-02-21_pqq-sorbitol-md/structures
open ../scripts/chimerax_transplant.cxc

# Verify result
ls -lh receptor_PQQ_Ca.pdb

# Check file
grep "HETATM" receptor_PQQ_Ca.pdb | grep -E "PQQ|CA"
```
