# AlphaFold Structure Generation Guide

---

## Option 1: ColabFold (Recommended - Fast & Free)

### Step 1: Access ColabFold
https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

### Step 2: Upload Sequence
1. Open notebook
2. Paste sequence from `structures/sequence.fasta`
3. Set parameters:
   - `num_models`: 5
   - `use_amber`: True (for relaxation)
   - `template_mode`: none (or pdb70 if available)

### Step 3: Run
- Runtime → Run all
- Wait ~10-30 min (depending on sequence length)

### Step 4: Download Results
- Download best ranked model: `*_rank_1_*.pdb`
- Save as `structures/receptor_af.pdb`

---

## Option 2: AlphaFold Local (If Available)

### Requirements
- AlphaFold installed locally
- Genetic databases downloaded
- GPU with 16+ GB VRAM

### Command
```bash
python run_alphafold.py \
  --fasta_paths=structures/sequence.fasta \
  --output_dir=structures/alphafold_output \
  --model_preset=monomer \
  --max_template_date=2024-01-01 \
  --use_gpu_relax=True
```

### Output
- Best model: `ranked_0.pdb`
- Copy to: `structures/receptor_af.pdb`

---

## Option 3: AlphaFold Server (Easiest)

### Step 1: Access
https://alphafoldserver.com/

### Step 2: Submit
1. Paste sequence
2. Job name: "Gluconobacter_SldA_catalytic"
3. Submit

### Step 3: Wait
- Email notification when complete (~hours to days)

### Step 4: Download
- Download PDB file
- Save as `structures/receptor_af.pdb`

---

## Structure Verification

### 1. Visual Inspection (PyMOL/ChimeraX)

```bash
pymol structures/receptor_af.pdb

# Check:
# - β-propeller fold visible?
# - ~8 blades in propeller?
# - Compact globular structure?
# - No major gaps/breaks?
```

### 2. pLDDT Score Check

**Good structure**:
- pLDDT > 70 for most residues
- pLDDT > 90 in core regions
- pLDDT may be lower at termini (OK)

**In PyMOL**:
```python
# Color by pLDDT (stored in B-factor)
spectrum b, blue_white_red, minimum=50, maximum=100
```

### 3. Expected Features

**β-propeller characteristics**:
- 8 blades (β-sheets arranged radially)
- Central tunnel/cavity
- ~400-600 residues
- Compact fold

**PQQ binding site** (will be empty in AF model):
- Should be near propeller center
- Cavity/pocket visible
- Will add PQQ in next step

---

## Common Issues

### Issue: Low pLDDT scores
**Solution**: 
- Check sequence quality
- Remove disordered regions
- Try different AF version

### Issue: Unfolded structure
**Solution**:
- Verify sequence (no TM domains)
- Check for missing residues
- Re-run with different settings

### Issue: Multiple domains
**Solution**:
- Use only catalytic domain
- Remove linkers/extra domains

---

## Next Step: PQQ+Ca Transplant

Once you have `structures/receptor_af.pdb`:

1. Find homolog structure with PQQ+Ca (e.g., PDB: 1WPQ)
2. Use ChimeraX to align and transplant
3. See: `PQQ_TRANSPLANT_GUIDE.md`

---

## Files Generated

```
structures/
├── sequence.fasta              # Input sequence
├── receptor_af.pdb            # AlphaFold output ✓
└── alphafold_output/          # Full AF output (optional)
    ├── ranked_0.pdb
    ├── ranked_1.pdb
    └── ...
```

---

## Estimated Time

- **ColabFold**: 10-30 min
- **AF Server**: Hours to days
- **Local AF**: 30-60 min (if setup)

**Recommended**: Use ColabFold for speed
