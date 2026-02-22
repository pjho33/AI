# Complete Workflow - PQQ-Sorbitol MD

**Strategy**: "정공법" - PQQ 고정 → sorbitol 도킹 → MD → reactive fraction 분석

---

## Prerequisites

### Software Requirements
- GROMACS 2025.4+ (with CUDA)
- AutoDock Vina
- Python 3.8+ with MDAnalysis, NumPy
- ChimeraX (for PQQ transplant)
- CGenFF (for ligand parameterization)

### Critical Decisions
1. **Residue names** (must be consistent):
   - PQQ: `PQQ`
   - L-sorbitol: `LSO`
   - D-sorbitol: `DSO`
   - Ca²⁺: `CA`

2. **Reactive atom definition**:
   - Sorbitol: C2 (hydride donor)
   - PQQ: C5 (hydride acceptor, adjust based on structure)

---

## Phase 1: Receptor Preparation

### Step 1.1: Get Correct Sequence
**Target**: Membrane-bound PQQ-dependent polyol/sorbitol dehydrogenase
- **NOT** yeast THR4 (wrong protein)
- **YES** Gluconobacter sldA or similar MB PQQ-DH

**Example sources**:
- Gluconobacter oxydans sldA (large subunit)
- Search UniProt: "PQQ-dependent dehydrogenase" + "membrane"

**Output**: `structures/sequence.fasta`

### Step 1.2: AlphaFold Structure
```bash
# Use ColabFold or AlphaFold
# Input: sequence.fasta
# Output: receptor_af.pdb

# Remove TM/anchor regions if present
# Keep only periplasmic catalytic domain (β-propeller)
```

**Output**: `structures/receptor_af.pdb`

### Step 1.3: PQQ+Ca Transplant
**Method**: ChimeraX alignment with homolog

1. Find homolog with PQQ+Ca (e.g., PDB: 1WPQ, 2CDU)
2. Align active site regions
3. Copy PQQ + Ca²⁺ coordinates
4. Save combined structure

**Commands** (ChimeraX):
```
open receptor_af.pdb
open homolog_with_PQQ.pdb
matchmaker #1 to #2 pairing ss
combine #1,#2 close false
delete #2 & protein
save receptor_PQQ_Ca.pdb #1
```

**Output**: `structures/receptor_PQQ_Ca.pdb`

---

## Phase 2: Parameterization (CGenFF)

### Step 2.1: PQQ Parameters
```bash
# Generate CGenFF parameters for PQQ
# Input: PQQ structure (mol2 or pdb)
# Output: pqq.itp, pqq.prm

# Check penalty scores
# If penalty > 50, may need manual refinement
```

**Output**: `parameters/pqq.itp`

### Step 2.2: Sorbitol Parameters
```bash
# L-sorbitol
# Input: L-sorbitol.mol2
# Output: lsorbitol.itp

# D-sorbitol
# Input: D-sorbitol.mol2
# Output: dsorbitol.itp
```

**Output**: 
- `parameters/lsorbitol.itp`
- `parameters/dsorbitol.itp`

### Step 2.3: Verify Atom Names
**Critical**: Ensure C2 atom in sorbitol is named `C2` consistently

---

## Phase 3: Docking

### Step 3.1: Prepare Receptor
```bash
cd docking

# Convert to PDBQT
prepare_receptor4.py -r ../structures/receptor_PQQ_Ca.pdb -o receptor.pdbqt
```

### Step 3.2: Prepare Ligands
```bash
# L-sorbitol
prepare_ligand4.py -l L-sorbitol.mol2 -o L-sorbitol.pdbqt

# D-sorbitol
prepare_ligand4.py -l D-sorbitol.mol2 -o D-sorbitol.pdbqt
```

### Step 3.3: Define Grid (PQQ center)
```python
# Get PQQ center from receptor_PQQ_Ca.pdb
import numpy as np

pqq_atoms = []  # Extract PQQ coordinates
pqq_center = np.mean(pqq_atoms, axis=0)

# Grid config
center_x, center_y, center_z = pqq_center
size_x = size_y = size_z = 20  # Å
```

**Output**: `docking/config.txt`

### Step 3.4: Run Vina
```bash
# L-sorbitol
vina --receptor receptor.pdbqt \
     --ligand L-sorbitol.pdbqt \
     --config config.txt \
     --out L_poses.pdbqt \
     --num_modes 50

# D-sorbitol
vina --receptor receptor.pdbqt \
     --ligand D-sorbitol.pdbqt \
     --config config.txt \
     --out D_poses.pdbqt \
     --num_modes 50
```

### Step 3.5: Extract Poses
```bash
# Split multi-model PDBQT into individual PDB files
# pose_L_01.pdb, pose_L_02.pdb, ..., pose_L_50.pdb
# pose_D_01.pdb, pose_D_02.pdb, ..., pose_D_50.pdb
```

**Output**: `docking/pose_*_*.pdb` (100 files total)

---

## Phase 4: MD Screening (10-20 ns per pose)

### Step 4.1: Template System Setup
```bash
cd md_runs

# Create template with receptor + PQQ + Ca
gmx pdb2gmx -f ../structures/receptor_PQQ_Ca.pdb \
            -o protein.gro \
            -p topol.top \
            -ff charmm36-mar2019 \
            -water tip3p

# Add PQQ and sorbitol includes to topol.top
# #include "../parameters/pqq.itp"
# #include "../parameters/lsorbitol.itp"
```

### Step 4.2: Generate PQQ Position Restraints
```bash
# Create index group for PQQ heavy atoms
echo "r PQQ & ! a H*" | gmx make_ndx -f protein.gro -o index.ndx

# Generate position restraints
gmx genrestr -f protein.gro -n index.ndx -o posre_PQQ.itp -fc 1000 1000 1000
```

### Step 4.3: Automated Pose Insertion
**Use script**: `scripts/02_insert_poses.py`

For each pose:
1. Combine receptor + PQQ + Ca + sorbitol
2. Create system directory: `md_runs/pose_L_01/`
3. Copy topology and parameters
4. Insert sorbitol coordinates
5. Solvate and add ions

### Step 4.4: MD Protocol (per pose)
```bash
# 1. Energy minimization
gmx grompp -f em.mdp -c system.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# 2. NVT equilibration (500 ps, PQQ restrained)
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt

# 3. NPT equilibration (2 ns, PQQ restrained)
gmx grompp -f npt.mdp -c nvt.gro -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt

# 4. Production (10-20 ns, PQQ weakly restrained)
gmx grompp -f prod.mdp -c npt.gro -p topol.top -o prod.tpr
gmx mdrun -v -deffnm prod
```

### Step 4.5: Screening Analysis
For each pose, calculate:
1. RMSD (sorbitol stability)
2. Distance C2-PQQ
3. Binding persistence

**Select top 2-3 poses** with:
- Low RMSD
- Stable binding
- Reasonable C2-PQQ distance

---

## Phase 5: Production MD (100-200 ns)

### Step 5.1: Extended Runs
For each selected pose:
- 3 replicates (different random seeds)
- 100-200 ns each
- PQQ restraint: 100 kJ/mol/nm² or 0 (test both)

### Step 5.2: Directory Structure
```
md_runs/
├── production/
│   ├── pose_L_05_rep1/
│   ├── pose_L_05_rep2/
│   ├── pose_L_05_rep3/
│   ├── pose_L_12_rep1/
│   └── ...
```

---

## Phase 6: Reactive Fraction Analysis

### Step 6.1: Define Reactive State
**Criteria**:
- Distance: C2(sorbitol) - C5(PQQ) < 4.0 Å
- Optional: Angle constraint for hydride transfer geometry

### Step 6.2: Calculate Reactive Fraction
**Use script**: `scripts/04_analyze_reactive.py`

```python
import MDAnalysis as mda
import numpy as np

u = mda.Universe("prod.tpr", "traj.xtc")

sorb_C2 = u.select_atoms("resname LSO and name C2")
pqq_C5 = u.select_atoms("resname PQQ and name C5")

distances = []
for ts in u.trajectory:
    d = np.linalg.norm(sorb_C2.positions[0] - pqq_C5.positions[0])
    distances.append(d / 10.0)  # Convert to nm

distances = np.array(distances)
reactive_fraction = (distances < 0.4).mean()  # 4.0 Å cutoff

print(f"Reactive fraction: {reactive_fraction:.3f}")
print(f"Mean distance: {distances.mean():.2f} nm")
```

### Step 6.3: Compare L vs D
**Key metric**: Relative reactive fraction

```
RF_ratio = RF(L-sorbitol) / RF(D-sorbitol)
```

**Expected**: RF_ratio < 1.0 (L-sorbitol less reactive)

### Step 6.4: Statistical Analysis
- Mean ± SEM across replicates
- Histogram of distance distributions
- Time series of reactive state occupancy

---

## Critical Parameters

### MDP Settings

**em.mdp**:
```
integrator = steep
nsteps = 50000
emtol = 100.0
```

**nvt.mdp**:
```
integrator = md
dt = 0.002
nsteps = 250000  ; 500 ps
tcoupl = V-rescale
ref_t = 300
; PQQ restraints
define = -DPOSRES_PQQ
```

**npt.mdp**:
```
integrator = md
dt = 0.002
nsteps = 1000000  ; 2 ns
pcoupl = Parrinello-Rahman
ref_p = 1.0
; PQQ restraints
define = -DPOSRES_PQQ
```

**prod.mdp**:
```
integrator = md
dt = 0.002
nsteps = 50000000  ; 100 ns
; Weak PQQ restraints or none
define = -DPOSRES_PQQ_WEAK
```

### Position Restraints

**posre_PQQ.itp** (strong):
```
[ position_restraints ]
; atom  type  fx    fy    fz
  1     1     1000  1000  1000
  2     1     1000  1000  1000
  ...
```

**posre_PQQ_weak.itp** (weak):
```
[ position_restraints ]
; atom  type  fx    fy    fz
  1     1     100   100   100
  2     1     100   100   100
  ...
```

---

## Troubleshooting

### Issue: PQQ drifts away
**Solution**: Increase restraint force constant

### Issue: Sorbitol leaves binding site
**Solution**: Check docking pose quality, use tighter grid

### Issue: High penalty in CGenFF
**Solution**: Manual parameter refinement or switch to GAFF2

### Issue: Low reactive fraction for both L and D
**Solution**: Adjust cutoff distance, check PQQ reactive atom definition

---

## Expected Timeline

- Phase 1-2: 1-2 days (structure + parameters)
- Phase 3: 0.5 day (docking)
- Phase 4: 2-3 days (screening 50 poses × 10-20 ns)
- Phase 5: 5-7 days (production 6 runs × 100-200 ns)
- Phase 6: 1 day (analysis)

**Total**: ~2 weeks for complete workflow

---

## Success Criteria

1. ✓ PQQ remains stable (RMSD < 2 Å)
2. ✓ Sorbitol binding persists (> 50% of trajectory)
3. ✓ Reactive fraction measurable (> 0.01)
4. ✓ L vs D difference statistically significant (p < 0.05)
5. ✓ Results reproducible across replicates
