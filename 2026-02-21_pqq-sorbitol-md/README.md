# PQQ-Sorbitol MD Simulation Project

**Date**: February 21, 2026  
**Approach**: PQQ-fixed → Sorbitol docking → MD → Reactive fraction analysis

---

## Project Structure

```
2026-02-21_pqq-sorbitol-md/
├── structures/          # Protein structures, PQQ+Ca transplant
├── parameters/          # Force field parameters (CGenFF)
├── docking/            # Vina docking results (20-50 poses)
├── md_runs/            # MD simulations organized by pose
│   ├── pose_01/
│   ├── pose_02/
│   └── ...
├── analysis/           # Analysis results and plots
├── scripts/            # Automation scripts
└── docs/               # Documentation and protocols
```

---

## Workflow Overview

### Phase 1: Receptor Preparation
1. **Sequence**: Correct MB PQQ-dependent dehydrogenase
2. **Structure**: AlphaFold/ColabFold modeling
3. **PQQ+Ca**: Transplant from homolog structure
4. **Output**: `structures/receptor_PQQ_Ca.pdb`

### Phase 2: Parameterization
1. **Force field**: CHARMM36m (protein) + CGenFF (ligands)
2. **Ligands**: PQQ, L-sorbitol, D-sorbitol
3. **Output**: `parameters/*.itp`

### Phase 3: Docking
1. **Tool**: AutoDock Vina
2. **Grid center**: PQQ center
3. **Poses**: 20-50 per stereoisomer
4. **Output**: `docking/pose_*.pdb`

### Phase 4: MD Screening (10-20 ns)
1. **Purpose**: Filter out unstable poses
2. **Protocol**: EM → NVT → NPT → Production
3. **PQQ restraint**: Strong (1000 kJ/mol/nm²)
4. **Output**: Top 2-3 poses selected

### Phase 5: Production MD (100-200 ns)
1. **Replicates**: 3× per pose (different seeds)
2. **PQQ restraint**: Weak (100) or removed
3. **Output**: Trajectories for analysis

### Phase 6: Reactive Fraction Analysis
1. **Metric**: Distance C2(sorbitol) - C5(PQQ) < 4.0 Å
2. **Comparison**: L-sorbitol vs D-sorbitol
3. **Output**: Reactive fraction, distance distributions

---

## Key Files

### Input
- `structures/receptor_af.pdb` - AlphaFold structure
- `structures/receptor_PQQ_Ca.pdb` - With cofactors
- `parameters/pqq.itp` - PQQ topology
- `parameters/lsorbitol.itp` - L-sorbitol topology
- `parameters/dsorbitol.itp` - D-sorbitol topology

### Scripts
- `scripts/01_prepare_system.sh` - System setup automation
- `scripts/02_insert_poses.py` - Pose insertion automation
- `scripts/03_run_screening.sh` - MD screening launcher
- `scripts/04_analyze_reactive.py` - Reactive fraction analysis

### Output
- `analysis/reactive_fraction.txt` - Main results
- `analysis/distance_distributions.png` - Plots
- `analysis/pose_ranking.txt` - Pose stability ranking

---

## Current Status

- [ ] Phase 1: Receptor preparation
- [ ] Phase 2: Parameterization
- [ ] Phase 3: Docking
- [ ] Phase 4: MD screening
- [ ] Phase 5: Production MD
- [ ] Phase 6: Analysis

---

## Notes

- **PQQ restraints**: Critical for preventing cofactor drift
- **Residue names**: Define consistently (PQQ, LSO, DSO)
- **Reactive cutoff**: 4.0 Å (adjustable based on D-sorbitol control)
- **Replicates**: Essential for statistical significance
