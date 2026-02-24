#!/usr/bin/env python3
"""
MD Feature Extraction for ML: A, B, C Analysis
- A. Binding Stability (Ligand RMSD, Active-site RMSD, Contact persistence)
- B. Reactive Geometry (Near-attack distance/angle frequency)
- C. H-bond/Electrostatic Interactions (H-bond occupancy, persistence)
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, distances, contacts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path("/home/pjho3tr/projects/AI/2026-02-22_pqq-sorbitol-md-phase3")
OUTPUT_DIR = Path("/home/pjho3tr/projects/AI/2026-02-23_advanced_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# PQQ = resname UNL (resid 714), LSO/DSO = resname MOL (resid 715)
# Reactive atom on sorbitol:
#   LSO (L-sorbitol): C1-OH → O1 is oxidized by PQQ
#   DSO (D-sorbitol): C2-OH → O2 is oxidized by PQQ (D-sorbitol oxidation at C2)
SYSTEMS = {
    'LSO': {
        'gro': BASE_DIR / 'gromacs_lso/md.gro',
        'xtc': BASE_DIR / 'gromacs_lso/md.xtc',
        'sorbitol_sel':  'resname MOL',
        'pqq_sel':       'resname UNL',
        'ligand_atoms':  'resname MOL',
        'sorb_react_atom': 'O1',   # C1-OH oxygen
        'pqq_react_atom':  'C5',   # PQQ carbonyl carbon
    },
    'DSO': {
        'gro': BASE_DIR / 'gromacs_dso_v2/md.gro',
        'xtc': BASE_DIR / 'gromacs_dso_v2/md.xtc',
        'sorbitol_sel':  'resname MOL',
        'pqq_sel':       'resname UNL',
        'ligand_atoms':  'resname MOL',
        'sorb_react_atom': 'O2',   # C2-OH oxygen (D-sorbitol oxidation site)
        'pqq_react_atom':  'C5',   # PQQ carbonyl carbon
    }
}

# Results storage
results = {
    'A_binding_stability': {},
    'B_reactive_geometry': {},
    'C_hbond_analysis': {}
}


def analyze_binding_stability(u, ligand_sel, system_name):
    """
    A. Binding Stability Analysis
    - Ligand RMSD (mean, std)
    - Active-site RMSD
    - Ligand residence time
    - Contact persistence
    """
    print(f"\n{'='*60}")
    print(f"A. Binding Stability Analysis: {system_name}")
    print(f"{'='*60}")
    
    ligand = u.select_atoms(ligand_sel)
    protein = u.select_atoms('protein')

    # Define active site: residues within 5 Å of ligand (use last frame for stable pose)
    u.trajectory[-1]
    active_site_resids = set(
        protein.select_atoms(f'protein and around 5.0 ({ligand_sel})').resids
    )
    # also check first frame
    u.trajectory[0]
    active_site_resids |= set(
        protein.select_atoms(f'protein and around 5.0 ({ligand_sel})').resids
    )
    active_site_sel = 'protein and resid ' + ' '.join(str(r) for r in sorted(active_site_resids)) if active_site_resids else 'protein'
    active_site_residues = u.select_atoms(active_site_sel)
    print(f"Active site residues: {len(active_site_residues.residues)} residues (union of frame 0 & last)")
    
    # 1. Ligand RMSD (self-fit)
    ligand_rmsd = []
    ref_ligand = ligand.positions.copy()
    
    for ts in u.trajectory:
        # Align ligand to itself
        current = ligand.positions.copy()
        rmsd_val = rms.rmsd(current, ref_ligand, superposition=True)
        ligand_rmsd.append(rmsd_val)
    
    ligand_rmsd = np.array(ligand_rmsd)
    
    # 2. Active-site RMSD
    active_rmsd = []
    ref_active = active_site_residues.positions.copy()
    u.trajectory[0]
    
    for ts in u.trajectory:
        current = active_site_residues.positions.copy()
        rmsd_val = rms.rmsd(current, ref_active, superposition=True)
        active_rmsd.append(rmsd_val)
    
    active_rmsd = np.array(active_rmsd)
    
    # 3. Ligand residence time (RMSD < 2.0 Å threshold)
    residence_threshold = 2.0  # Å
    residence_frames = np.sum(ligand_rmsd < residence_threshold)
    residence_time = residence_frames / len(ligand_rmsd) * 100  # percentage
    
    # 4. Contact persistence (within 3.5 Å)
    contact_cutoff = 3.5  # Å
    contact_persistence = []
    
    for ts in u.trajectory:
        dist_array = distances.distance_array(ligand.positions, active_site_residues.positions)
        min_distances = np.min(dist_array, axis=0)
        in_contact = np.sum(min_distances < contact_cutoff)
        contact_persistence.append(in_contact)
    
    contact_persistence = np.array(contact_persistence)
    
    # Store results
    results['A_binding_stability'][system_name] = {
        'ligand_rmsd_mean': float(np.mean(ligand_rmsd)),
        'ligand_rmsd_std': float(np.std(ligand_rmsd)),
        'active_site_rmsd_mean': float(np.mean(active_rmsd)),
        'active_site_rmsd_std': float(np.std(active_rmsd)),
        'residence_time_percent': float(residence_time),
        'contact_persistence_mean': float(np.mean(contact_persistence)),
        'contact_persistence_std': float(np.std(contact_persistence)),
        'n_active_site_residues': len(active_site_residues.residues)
    }
    
    # Print summary
    print(f"\nLigand RMSD: {np.mean(ligand_rmsd):.2f} ± {np.std(ligand_rmsd):.2f} Å")
    print(f"Active-site RMSD: {np.mean(active_rmsd):.2f} ± {np.std(active_rmsd):.2f} Å")
    print(f"Residence time (RMSD < {residence_threshold} Å): {residence_time:.1f}%")
    print(f"Contact persistence: {np.mean(contact_persistence):.1f} ± {np.std(contact_persistence):.1f} residues")
    
    return ligand_rmsd, active_rmsd, contact_persistence


def analyze_reactive_geometry(u, sorbitol_sel, pqq_sel, system_name,
                              sorb_react_atom='O1', pqq_react_atom='C5'):
    """
    B. Reactive Geometry Analysis
    PQQ-dependent alcohol dehydrogenase (sldA) 반응 메커니즘:
      - Sorbitol C1-OH (O1) → PQQ C5 (carbonyl carbon) hydride transfer
      - NAC 기준: O1(sorbitol) ↔ C5(PQQ) 거리 < 3.5 Å
      - 동시에 O1-H-C5 각도 > 120° (in-line geometry)
    """
    print(f"\n{'='*60}")
    print(f"B. Reactive Geometry Analysis: {system_name}")
    print(f"{'='*60}")

    pqq     = u.select_atoms(pqq_sel)
    sorbitol = u.select_atoms(sorbitol_sel)
    print(f"PQQ atoms: {len(pqq)}, Sorbitol atoms: {len(sorbitol)}")

    # --- 특정 반응 원자 선택 (per-system 설정 사용) ---
    pqq_C5   = u.select_atoms(f'{pqq_sel} and name {pqq_react_atom}')
    sorb_O1  = u.select_atoms(f'{sorbitol_sel} and name {sorb_react_atom}')
    # H on reactive O (이름 규칙: HO1, HO2, H1, H2 등 다양)
    sorb_HO1 = u.select_atoms(f'{sorbitol_sel} and (name HO{sorb_react_atom[-1]} or name H{sorb_react_atom[-1]})')

    use_specific = (len(pqq_C5) > 0 and len(sorb_O1) > 0)
    if use_specific:
        print(f"Using specific atoms: PQQ {pqq_react_atom} ({len(pqq_C5)}) "
              f"↔ Sorbitol {sorb_react_atom} ({len(sorb_O1)})")
    else:
        print(f"WARNING: {pqq_react_atom}/{sorb_react_atom} not found → min O-C distance proxy")
        print(f"  PQQ atom names:     {list(pqq.names)}")
        print(f"  Sorbitol atom names: {list(sorbitol.names)}")

    # NAC 기준
    nac_dist_cutoff  = 3.5   # Å
    nac_angle_cutoff = 120.0 # degrees

    dist_list, angle_list = [], []
    nac_dist_frames, nac_both_frames = [], []

    for ts in u.trajectory:
        if use_specific:
            d = float(distances.dist(pqq_C5, sorb_O1)[2][0])
        else:
            # minimum distance between any sorbitol O and any PQQ C
            sorb_O = u.select_atoms(f'{sorbitol_sel} and name O*')
            pqq_C  = u.select_atoms(f'{pqq_sel} and name C*')
            if len(sorb_O) == 0 or len(pqq_C) == 0:
                sorb_O = sorbitol
                pqq_C  = pqq
            d_arr = distances.distance_array(sorb_O.positions, pqq_C.positions)
            d = float(np.min(d_arr))
        dist_list.append(d)

        # angle: O1-H···C5 (if H available)
        ang = np.nan
        if use_specific and len(sorb_HO1) > 0:
            v1 = sorb_O1.positions[0] - sorb_HO1.positions[0]
            v2 = pqq_C5.positions[0]  - sorb_HO1.positions[0]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            ang = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        angle_list.append(ang)

        if d < nac_dist_cutoff:
            nac_dist_frames.append(ts.frame)
            if np.isnan(ang) or ang > nac_angle_cutoff:  # angle criterion met or no H data
                nac_both_frames.append(ts.frame)

    dist_arr  = np.array(dist_list)
    angle_arr = np.array(angle_list)
    n_frames  = len(u.trajectory)
    dt_ns     = u.trajectory.dt / 1000.0  # ps → ns

    nac_dist_freq = len(nac_dist_frames) / n_frames * 100
    nac_both_freq = len(nac_both_frames) / n_frames * 100

    # NAC lifetime (consecutive frames)
    def calc_lifetimes(frame_list):
        if not frame_list:
            return []
        lifetimes, cur = [], 1
        for i in range(1, len(frame_list)):
            if frame_list[i] == frame_list[i-1] + 1:
                cur += 1
            else:
                lifetimes.append(cur)
                cur = 1
        lifetimes.append(cur)
        return lifetimes

    nac_lifetimes = calc_lifetimes(nac_both_frames)

    results['B_reactive_geometry'][system_name] = {
        'sorb_react_atom':            sorb_react_atom,
        'pqq_react_atom':             pqq_react_atom,
        'reactive_distance_mean_A':   float(np.mean(dist_arr)),
        'reactive_distance_std_A':    float(np.std(dist_arr)),
        'reactive_distance_min_A':    float(np.min(dist_arr)),
        'nac_dist_cutoff_A':          nac_dist_cutoff,
        'nac_angle_cutoff_deg':       nac_angle_cutoff,
        'nac_distance_freq_pct':      float(nac_dist_freq),
        'nac_simultaneous_freq_pct':  float(nac_both_freq),
        'nac_lifetime_mean_frames':   float(np.mean(nac_lifetimes)) if nac_lifetimes else 0.0,
        'nac_lifetime_mean_ns':       float(np.mean(nac_lifetimes)) * dt_ns if nac_lifetimes else 0.0,
        'nac_lifetime_max_frames':    float(np.max(nac_lifetimes)) if nac_lifetimes else 0.0,
        'used_specific_atoms':        use_specific,
    }

    print(f"\nReactive distance ({sorb_react_atom}↔{pqq_react_atom}): {np.mean(dist_arr):.2f} ± {np.std(dist_arr):.2f} Å")
    print(f"Minimum distance:              {np.min(dist_arr):.2f} Å")
    print(f"NAC distance freq (<{nac_dist_cutoff}Å):     {nac_dist_freq:.1f}%")
    print(f"NAC simultaneous freq:      {nac_both_freq:.1f}%  (dist+angle)")
    if nac_lifetimes:
        print(f"NAC lifetime: {np.mean(nac_lifetimes):.1f} ± {np.std(nac_lifetimes):.1f} frames "
              f"({np.mean(nac_lifetimes)*dt_ns*1000:.0f} ps avg)")

    return dist_arr, angle_arr


def analyze_hbonds(u, ligand_sel, system_name):
    """
    C. H-bond and Electrostatic Interactions
    - H-bond occupancy
    - H-bond persistence
    - H-bond network size
    """
    print(f"\n{'='*60}")
    print(f"C. H-bond Analysis: {system_name}")
    print(f"{'='*60}")
    
    from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

    ligand  = u.select_atoms(ligand_sel)
    protein = u.select_atoms('protein')
    print(f"Ligand atoms: {len(ligand)}, Protein atoms: {len(protein)}")

    # H-bond: ligand ↔ protein 간만
    # charge 정보 없으므로 donors/acceptors 명시적으로 지정
    donors_sel    = f'(({ligand_sel}) or protein) and (name O* N*)'
    hydrogens_sel = f'(({ligand_sel}) or protein) and (name H*)'
    acceptors_sel = f'(({ligand_sel}) or protein) and (name O* N*)'

    hbonds = HydrogenBondAnalysis(
        universe=u,
        donors_sel=donors_sel,
        hydrogens_sel=hydrogens_sel,
        acceptors_sel=acceptors_sel,
        d_h_cutoff=1.2,
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=120
    )

    print("Running H-bond analysis (this may take a few minutes)...")
    hbonds.run(verbose=True)

    # Get results
    hbond_results = hbonds.results.hbonds

    if len(hbond_results) == 0:
        print("WARNING: No H-bonds detected")
        results['C_hbond_analysis'][system_name] = {
            'total_hbonds': 0, 'unique_hbonds': 0, 'mean_occupancy': 0.0
        }
        return None

    # ligand 원자 인덱스 집합 → ligand 관련 H-bond만 필터링
    ligand_ag  = u.select_atoms(ligand_sel)
    ligand_idx = set(ligand_ag.indices)

    filtered = [hb for hb in hbond_results
                if int(hb[1]) in ligand_idx or int(hb[3]) in ligand_idx]
    print(f"Total H-bonds: {len(hbond_results)} → ligand-involved: {len(filtered)}")

    if len(filtered) == 0:
        print("WARNING: No ligand-protein H-bonds detected")
        results['C_hbond_analysis'][system_name] = {
            'total_hbonds': 0, 'unique_hbonds': 0, 'mean_occupancy': 0.0,
            'mean_network_size': 0.0, 'max_network_size': 0
        }
        return None

    # Calculate occupancy for each unique H-bond pair
    unique_hbonds = {}
    for hbond in filtered:
        frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle = hbond
        key = (int(donor_idx), int(acceptor_idx))
        if key not in unique_hbonds:
            unique_hbonds[key] = []
        unique_hbonds[key].append(frame)

    n_frames = len(u.trajectory)
    occupancies = [len(frames) / n_frames * 100 for frames in unique_hbonds.values()]

    # H-bond network size per frame (ligand-involved only)
    hbonds_per_frame = {}
    for hbond in filtered:
        frame = int(hbond[0])
        hbonds_per_frame[frame] = hbonds_per_frame.get(frame, 0) + 1

    network_sizes = list(hbonds_per_frame.values())
    
    # Store results
    results['C_hbond_analysis'][system_name] = {
        'total_hbonds_detected': len(hbond_results),
        'unique_hbond_pairs': len(unique_hbonds),
        'mean_occupancy_percent': float(np.mean(occupancies)),
        'max_occupancy_percent': float(np.max(occupancies)),
        'mean_network_size': float(np.mean(network_sizes)),
        'max_network_size': float(np.max(network_sizes)),
    }
    
    print(f"\nTotal H-bonds detected: {len(hbond_results)}")
    print(f"Unique H-bond pairs: {len(unique_hbonds)}")
    print(f"Mean occupancy: {np.mean(occupancies):.1f}%")
    print(f"Max occupancy: {np.max(occupancies):.1f}%")
    print(f"Mean network size: {np.mean(network_sizes):.1f} H-bonds/frame")
    print(f"Max network size: {np.max(network_sizes)} H-bonds/frame")
    
    return occupancies, network_sizes


def main():
    print("="*80)
    print("MD Feature Extraction: A, B, C Analysis")
    print("="*80)
    
    all_data = {}
    
    for system_name, paths in SYSTEMS.items():
        print(f"\n\nProcessing {system_name} system...")
        print("-"*80)
        
        # Load universe
        u = mda.Universe(str(paths['gro']), str(paths['xtc']))
        print(f"Loaded trajectory: {len(u.trajectory)} frames")
        print(f"Time range: 0 - {u.trajectory[-1].time/1000:.1f} ns")
        
        ligand_sel   = paths['ligand_atoms']
        sorbitol_sel = paths['sorbitol_sel']
        pqq_sel      = paths['pqq_sel']

        # Run analyses
        ligand_rmsd, active_rmsd, contact_pers = analyze_binding_stability(u, ligand_sel, system_name)
        dist_arr, angle_arr = analyze_reactive_geometry(
            u, sorbitol_sel, pqq_sel, system_name,
            sorb_react_atom=paths['sorb_react_atom'],
            pqq_react_atom=paths['pqq_react_atom']
        )
        hbond_data = analyze_hbonds(u, ligand_sel, system_name)

        all_data[system_name] = {
            'ligand_rmsd':        ligand_rmsd,
            'active_rmsd':        active_rmsd,
            'contact_persistence': contact_pers,
            'distances':          dist_arr,
            'angles':             angle_arr,
        }
    
    # Save results to JSON
    output_json = OUTPUT_DIR / 'ABC_analysis_results.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_json}")
    
    # Create visualization
    create_visualizations(all_data)
    
    # Print summary table
    print_summary_table()


def create_visualizations(all_data):
    """Create comprehensive visualization of A, B, C analyses"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('MD Feature Analysis: A, B, C', fontsize=16, fontweight='bold')
    
    systems = list(all_data.keys())
    colors = {'LSO': '#2E86AB', 'DSO': '#A23B72'}
    
    # A1: Ligand RMSD
    ax = axes[0, 0]
    for sys in systems:
        time = np.arange(len(all_data[sys]['ligand_rmsd'])) * 0.01  # 10 ps timestep
        ax.plot(time, all_data[sys]['ligand_rmsd'], label=sys, color=colors[sys], alpha=0.7)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Ligand RMSD (Å)')
    ax.set_title('A1. Ligand RMSD (Binding Stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # A2: Active-site RMSD
    ax = axes[0, 1]
    for sys in systems:
        time = np.arange(len(all_data[sys]['active_rmsd'])) * 0.01
        ax.plot(time, all_data[sys]['active_rmsd'], label=sys, color=colors[sys], alpha=0.7)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Active-site RMSD (Å)')
    ax.set_title('A2. Active-site RMSD (Induced Fit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # A3: Contact Persistence
    ax = axes[1, 0]
    for sys in systems:
        time = np.arange(len(all_data[sys]['contact_persistence'])) * 0.01
        ax.plot(time, all_data[sys]['contact_persistence'], label=sys, color=colors[sys], alpha=0.7)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Number of contacts')
    ax.set_title('A3. Contact Persistence (<3.5 Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # B1: Sorbitol O1 ↔ PQQ C5 Distance
    ax = axes[1, 1]
    for sys in systems:
        if all_data[sys]['distances'] is not None:
            time = np.arange(len(all_data[sys]['distances'])) * 0.01
            ax.plot(time, all_data[sys]['distances'], label=sys, color=colors[sys], alpha=0.7)
    ax.axhline(y=3.5, color='red', linestyle='--', alpha=0.5, label='NAC cutoff (3.5Å)')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Distance (Å)')
    ax.set_title('B1. Reactive Distance O1↔C5 (NAC)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary bar plots
    # A: RMSD comparison
    ax = axes[2, 0]
    x = np.arange(len(systems))
    width = 0.35
    
    ligand_means = [results['A_binding_stability'][sys]['ligand_rmsd_mean'] for sys in systems]
    ligand_stds = [results['A_binding_stability'][sys]['ligand_rmsd_std'] for sys in systems]
    active_means = [results['A_binding_stability'][sys]['active_site_rmsd_mean'] for sys in systems]
    active_stds = [results['A_binding_stability'][sys]['active_site_rmsd_std'] for sys in systems]
    
    ax.bar(x - width/2, ligand_means, width, yerr=ligand_stds, label='Ligand RMSD', 
           color=[colors[s] for s in systems], alpha=0.7)
    ax.bar(x + width/2, active_means, width, yerr=active_stds, label='Active-site RMSD',
           color=[colors[s] for s in systems], alpha=0.4)
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('A. RMSD Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # B: NAC frequency
    ax = axes[2, 1]
    nac_freqs = []
    for sys in systems:
        if 'nac_frequency_percent' in results['B_reactive_geometry'][sys]:
            nac_freqs.append(results['B_reactive_geometry'][sys]['nac_frequency_percent'])
        else:
            nac_freqs.append(0)
    
    ax.bar(systems, nac_freqs, color=[colors[s] for s in systems], alpha=0.7)
    ax.set_ylabel('NAC Frequency (%)')
    ax.set_title('B. Near-Attack Conformation Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_fig = OUTPUT_DIR / 'ABC_analysis_visualization.png'
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_fig}")
    plt.close()


def print_summary_table():
    """Print formatted summary table"""
    
    print("\n" + "="*80)
    print("SUMMARY TABLE: MD-derived Features for ML")
    print("="*80)
    
    print("\n### A. Binding Stability")
    print("-"*80)
    print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
    print("-"*80)
    
    for sys in ['LSO', 'DSO']:
        data = results['A_binding_stability'][sys]
    
    print(f"{'Ligand RMSD (mean)':<30} "
          f"{results['A_binding_stability']['LSO']['ligand_rmsd_mean']:>15.2f} "
          f"{results['A_binding_stability']['DSO']['ligand_rmsd_mean']:>15.2f} {'Å':>10}")
    print(f"{'Ligand RMSD (std)':<30} "
          f"{results['A_binding_stability']['LSO']['ligand_rmsd_std']:>15.2f} "
          f"{results['A_binding_stability']['DSO']['ligand_rmsd_std']:>15.2f} {'Å':>10}")
    print(f"{'Active-site RMSD (mean)':<30} "
          f"{results['A_binding_stability']['LSO']['active_site_rmsd_mean']:>15.2f} "
          f"{results['A_binding_stability']['DSO']['active_site_rmsd_mean']:>15.2f} {'Å':>10}")
    print(f"{'Residence time':<30} "
          f"{results['A_binding_stability']['LSO']['residence_time_percent']:>15.1f} "
          f"{results['A_binding_stability']['DSO']['residence_time_percent']:>15.1f} {'%':>10}")
    print(f"{'Contact persistence (mean)':<30} "
          f"{results['A_binding_stability']['LSO']['contact_persistence_mean']:>15.1f} "
          f"{results['A_binding_stability']['DSO']['contact_persistence_mean']:>15.1f} {'residues':>10}")
    
    print("\n### B. Reactive Geometry")
    print("-"*80)
    print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
    print("-"*80)
    
    lso_b = results['B_reactive_geometry']['LSO']
    dso_b = results['B_reactive_geometry']['DSO']
    if 'error' not in lso_b and 'error' not in dso_b:
        print(f"{'Reactive distance (mean)':<30} "
              f"{lso_b['reactive_distance_mean_A']:>15.2f} "
              f"{dso_b['reactive_distance_mean_A']:>15.2f} {'Å':>10}")
        print(f"{'NAC dist freq (<3.5Å)':<30} "
              f"{lso_b['nac_distance_freq_pct']:>15.1f} "
              f"{dso_b['nac_distance_freq_pct']:>15.1f} {'%':>10}")
        print(f"{'NAC simultaneous freq':<30} "
              f"{lso_b['nac_simultaneous_freq_pct']:>15.1f} "
              f"{dso_b['nac_simultaneous_freq_pct']:>15.1f} {'%':>10}")
        print(f"{'NAC lifetime (mean)':<30} "
              f"{lso_b['nac_lifetime_mean_frames']:>15.1f} "
              f"{dso_b['nac_lifetime_mean_frames']:>15.1f} {'frames':>10}")
    
    print("\n### C. H-bond Analysis")
    print("-"*80)
    print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
    print("-"*80)
    
    print(f"{'Unique H-bond pairs':<30} "
          f"{results['C_hbond_analysis']['LSO']['unique_hbond_pairs']:>15} "
          f"{results['C_hbond_analysis']['DSO']['unique_hbond_pairs']:>15} {'pairs':>10}")
    print(f"{'Mean occupancy':<30} "
          f"{results['C_hbond_analysis']['LSO']['mean_occupancy_percent']:>15.1f} "
          f"{results['C_hbond_analysis']['DSO']['mean_occupancy_percent']:>15.1f} {'%':>10}")
    print(f"{'Mean network size':<30} "
          f"{results['C_hbond_analysis']['LSO']['mean_network_size']:>15.1f} "
          f"{results['C_hbond_analysis']['DSO']['mean_network_size']:>15.1f} {'H-bonds':>10}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
