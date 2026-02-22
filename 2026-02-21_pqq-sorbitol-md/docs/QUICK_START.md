# Quick Start Guide

**Goal**: PQQ-fixed sorbitol MD → reactive fraction analysis

---

## Prerequisites Checklist

- [ ] GROMACS 2025.4+ with CUDA
- [ ] AutoDock Vina
- [ ] Python 3.8+ with MDAnalysis
- [ ] ChimeraX (for PQQ transplant)
- [ ] Correct MB PQQ-DH sequence (NOT yeast THR4)

---

## Step-by-Step Execution

### 1. Prepare Receptor (Day 1)

```bash
cd structures/

# 1.1 Get sequence
# Save to sequence.fasta

# 1.2 Run AlphaFold/ColabFold
# Output: receptor_af.pdb

# 1.3 Transplant PQQ+Ca using ChimeraX
# Open receptor_af.pdb and homolog
# Align, copy PQQ+Ca coordinates
# Save as receptor_PQQ_Ca.pdb
```

**Output**: `structures/receptor_PQQ_Ca.pdb`

---

### 2. Generate Parameters (Day 1)

```bash
cd parameters/

# 2.1 PQQ parameters (CGenFF)
# Use CGenFF server or local tool
# Output: pqq.itp

# 2.2 L-sorbitol parameters
# Output: lsorbitol.itp

# 2.3 D-sorbitol parameters
# Output: dsorbitol.itp

# 2.4 Generate PQQ restraints
cd ../scripts/
bash 01_generate_pqq_restraints.sh
```

**Output**: 
- `parameters/pqq.itp`
- `parameters/lsorbitol.itp`
- `parameters/dsorbitol.itp`
- `parameters/posre_PQQ_*.itp`

---

### 3. Docking (Day 1-2)

```bash
cd docking/

# 3.1 Prepare receptor
prepare_receptor4.py -r ../structures/receptor_PQQ_Ca.pdb -o receptor.pdbqt

# 3.2 Prepare ligands
prepare_ligand4.py -l L-sorbitol.mol2 -o L-sorbitol.pdbqt
prepare_ligand4.py -l D-sorbitol.mol2 -o D-sorbitol.pdbqt

# 3.3 Get PQQ center coordinates
# Extract from receptor_PQQ_Ca.pdb manually or with script

# 3.4 Create Vina config
cat > config.txt << EOF
center_x = X.XX
center_y = Y.YY
center_z = Z.ZZ
size_x = 20
size_y = 20
size_z = 20
exhaustiveness = 32
num_modes = 50
EOF

# 3.5 Run docking
vina --receptor receptor.pdbqt \
     --ligand L-sorbitol.pdbqt \
     --config config.txt \
     --out L_poses.pdbqt

vina --receptor receptor.pdbqt \
     --ligand D-sorbitol.pdbqt \
     --config config.txt \
     --out D_poses.pdbqt

# 3.6 Split poses
vina_split --input L_poses.pdbqt --ligand pose_L
vina_split --input D_poses.pdbqt --ligand pose_D

# Convert to PDB
for i in {01..50}; do
    obabel pose_L_${i}.pdbqt -O pose_L_${i}.pdb
    obabel pose_D_${i}.pdbqt -O pose_D_${i}.pdb
done
```

**Output**: `docking/pose_L_*.pdb`, `docking/pose_D_*.pdb`

---

### 4. Insert Poses (Day 2)

```bash
cd scripts/

# 4.1 Insert L-sorbitol poses
python 02_insert_poses.py --stereo L --n_poses 20

# 4.2 Insert D-sorbitol poses
python 02_insert_poses.py --stereo D --n_poses 20
```

**Output**: `md_runs/screening/pose_*_*/`

---

### 5. Prepare MD Systems (Day 2)

For each pose directory:

```bash
cd md_runs/screening/pose_L_01/

# 5.1 Generate topology
gmx pdb2gmx -f system.pdb -o system.gro -p topol.top \
            -ff charmm36-mar2019 -water tip3p

# 5.2 Edit topol.top to include ligand parameters
# Add after forcefield includes:
#include "../../parameters/pqq.itp"
#include "../../parameters/lsorbitol.itp"

# Add at end of [ molecules ] section:
PQQ    1
LSO    1
CA     1

# 5.3 Add position restraints
# Add after protein includes:
#ifdef POSRES_PQQ
#include "../../parameters/posre_PQQ_strong.itp"
#endif

#ifdef POSRES_PQQ_WEAK
#include "../../parameters/posre_PQQ_weak.itp"
#endif

#ifdef POSRES_CA
#include "../../parameters/posre_CA.itp"
#endif

# 5.4 Solvate and add ions
gmx editconf -f system.gro -o boxed.gro -c -d 1.2 -bt dodecahedron
gmx solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top

# Create ions.mdp (minimal)
cat > ions.mdp << EOF
integrator = steep
nsteps = 0
EOF

gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 1
gmx genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15
```

**Repeat for all poses** or automate with script

---

### 6. Run MD Screening (Day 2-4)

```bash
cd scripts/

# Run all poses (parallel if GNU parallel installed)
bash 03_run_screening.sh
```

**Output**: Trajectories in `md_runs/screening/pose_*/prod/`

---

### 7. Analyze and Rank Poses (Day 4)

```bash
cd analysis/

# For each pose, analyze:
# - RMSD (sorbitol stability)
# - Distance C2-PQQ
# - Binding persistence

# Select top 2-3 poses per stereoisomer
```

---

### 8. Production MD (Day 5-11)

For selected poses:

```bash
cd md_runs/production/

# Create replicate directories
mkdir -p pose_L_05_rep{1,2,3}

# For each replicate:
cd pose_L_05_rep1/

# Copy from screening
cp ../../screening/pose_L_05/npt/npt.gro ./
cp ../../screening/pose_L_05/topol.top ./

# Run 100-200 ns
gmx grompp -f prod_long.mdp -c npt.gro -p topol.top -o prod.tpr
gmx mdrun -v -deffnm prod -ntmpi 1
```

**Output**: Long trajectories for analysis

---

### 9. Reactive Fraction Analysis (Day 12)

```bash
cd analysis/

# Analyze each trajectory
python ../scripts/04_analyze_reactive.py \
    --tpr ../md_runs/production/pose_L_05_rep1/prod.tpr \
    --xtc ../md_runs/production/pose_L_05_rep1/prod.xtc \
    --stereo L \
    --cutoff 0.4 \
    --output pose_L_05_rep1

# Repeat for all replicates and D-sorbitol

# Compare L vs D
python compare_stereoisomers.py
```

**Output**: 
- Reactive fractions
- Distance distributions
- Statistical comparison

---

## Expected Results

### Success Criteria

1. **PQQ stability**: RMSD < 2 Å throughout simulation
2. **Sorbitol binding**: Stays in pocket > 50% of time
3. **Reactive fraction**: Measurable (> 0.01)
4. **L vs D difference**: Statistically significant

### Typical Values

- **D-sorbitol** (positive control): RF ~ 0.1-0.3
- **L-sorbitol**: RF ~ 0.01-0.1 (lower than D)
- **RF ratio** (L/D): < 1.0

---

## Troubleshooting

### PQQ drifts away
→ Increase restraint: 1000 → 5000 in posre file

### Sorbitol leaves pocket
→ Check docking quality, use tighter grid

### Low RF for both
→ Adjust cutoff, verify PQQ reactive atom

### High CGenFF penalty
→ Manual parameter refinement needed

---

## File Organization

```
2026-02-21_pqq-sorbitol-md/
├── structures/
│   ├── sequence.fasta
│   ├── receptor_af.pdb
│   └── receptor_PQQ_Ca.pdb ✓
├── parameters/
│   ├── pqq.itp ✓
│   ├── lsorbitol.itp ✓
│   ├── dsorbitol.itp ✓
│   └── posre_*.itp ✓
├── docking/
│   └── pose_*_*.pdb ✓
├── md_runs/
│   ├── screening/
│   │   └── pose_*_*/
│   └── production/
│       └── pose_*_rep*/
├── analysis/
│   └── reactive_*.txt
└── scripts/
    ├── 01_generate_pqq_restraints.sh ✓
    ├── 02_insert_poses.py ✓
    ├── 03_run_screening.sh ✓
    └── 04_analyze_reactive.py ✓
```

---

## Timeline

- **Day 1-2**: Structure + parameters + docking
- **Day 2-4**: MD screening (20 poses × 10-20 ns)
- **Day 5-11**: Production MD (6 runs × 100-200 ns)
- **Day 12**: Analysis and results

**Total**: ~2 weeks
