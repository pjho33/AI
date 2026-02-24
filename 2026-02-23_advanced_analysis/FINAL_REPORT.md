# MD Feature Analysis Report: PQQ-Sorbitol Systems
**Date:** 2026-02-24  
**Systems:** LSO (L-sorbitol + PQQ) vs DSO (D-sorbitol + PQQ)  
**Simulation:** 10 ns production MD, CHARMM36-jul2022 FF + GAFF2 ligands, TIP3P water

---

## 1. Simulation Setup

| Parameter | LSO | DSO (v2) |
|---|---|---|
| Force field | CHARMM36-jul2022 | CHARMM36-jul2022 |
| Ligand FF | GAFF2 | GAFF2 |
| Water model | TIP3P | TIP3P |
| Box size | ~12 nm cubic | ~12 nm cubic |
| Ions | 0.15 M NaCl | 0.15 M NaCl |
| Simulation time | 10 ns | 10 ns |
| Frames analyzed | 1001 | 1001 |
| GPU | RTX 3090 | RTX 3090 |
| Performance | — | 126.9 ns/day |

**DSO v2 rebuild reason:** Original DSO simulation used incorrect docking box (center mismatch), causing ligand dissociation. Redocked with corrected box center [89.45, 104.61, 56.66] Å (same as LSO), affinity -5.229 kcal/mol.

---

## 2. A. Binding Stability

| Feature | LSO | DSO | Interpretation |
|---|---|---|---|
| Ligand RMSD mean (Å) | 1.32 | **1.20** | Both stable; DSO slightly lower fluctuation |
| Ligand RMSD std (Å) | 0.47 | **0.24** | DSO more rigid in pocket |
| Active-site RMSD mean (Å) | 6.24 | **5.05** | DSO pocket more stable |
| Residence time (%) | 100.0 | 100.0 | Both fully retained in pocket |
| Contact persistence (residues) | **40.7** | 22.0 | LSO contacts more residues |

**Key finding:** Both ligands remain stably bound throughout 10 ns. LSO engages more residues (40.7 vs 22.0), suggesting broader binding interface.

---

## 3. B. Reactive Geometry (Near-Attack Conformation)

Reactive atoms: LSO O1 ↔ PQQ C5 | DSO O2 ↔ PQQ C5

| Feature | LSO | DSO | Interpretation |
|---|---|---|---|
| Reactive distance mean (Å) | **5.73** | 7.44 | LSO closer to PQQ reactive center |
| NAC dist freq (<3.5 Å) (%) | 0.1 | **1.7** | DSO O2 transiently approaches C5 more often |
| NAC simultaneous freq (%) | 0.0 | 0.0 | Neither achieves full NAC geometry |
| NAC lifetime mean (frames) | 0.0 | 0.0 | NAC events too brief to measure lifetime |

**Key finding:** Neither system achieves sustained near-attack conformation. DSO O2 approaches C5 more frequently (1.7% vs 0.1%) but mean distance is larger. LSO O1 is on average closer to C5 (5.73 Å).

---

## 4. C. Hydrogen Bond Analysis

Ligand–protein H-bonds only (sorbitol ↔ protein residues)

| Feature | LSO | DSO | Interpretation |
|---|---|---|---|
| Unique H-bond pairs | 52 | **62** | DSO forms more distinct H-bond contacts |
| Mean occupancy (%) | **11.6** | 3.2 | LSO H-bonds more persistent |
| Max occupancy (%) | **86.3** | 70.2 | LSO has stronger dominant H-bond |
| Mean network size (H-bonds/frame) | **6.1** | 2.1 | LSO maintains larger H-bond network |

**Key finding:** LSO forms a more persistent and extensive H-bond network (mean 6.1 H-bonds/frame, max occupancy 86.3%). DSO has more unique contacts but lower occupancy, suggesting transient/dynamic interactions.

---

## 5. Summary for ML Feature Vector

| Feature | LSO | DSO |
|---|---|---|
| ligand_rmsd_mean | 1.32 | 1.20 |
| ligand_rmsd_std | 0.47 | 0.24 |
| active_site_rmsd_mean | 6.24 | 5.05 |
| residence_time_pct | 100.0 | 100.0 |
| contact_persistence_mean | 40.7 | 22.0 |
| reactive_distance_mean_A | 5.73 | 7.44 |
| nac_distance_freq_pct | 0.1 | 1.7 |
| nac_simultaneous_freq_pct | 0.0 | 0.0 |
| unique_hbond_pairs | 52 | 62 |
| mean_hbond_occupancy_pct | 11.6 | 3.2 |
| max_hbond_occupancy_pct | 86.3 | 70.2 |
| mean_hbond_network_size | 6.1 | 2.1 |

---

## 6. Output Files

| File | Description |
|---|---|
| `ABC_analysis_results.json` | Full numerical results for all features |
| `ABC_analysis_visualization.png` | Time-series plots for A, B, C analyses |
| `gromacs_dso_v2/md.xtc` | DSO v2 10 ns trajectory (633 MB) |
| `gromacs_lso/md.xtc` | LSO 10 ns trajectory |
