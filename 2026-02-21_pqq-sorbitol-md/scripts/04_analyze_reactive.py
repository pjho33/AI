#!/usr/bin/env python3
"""
Reactive Fraction Analysis for PQQ-Sorbitol MD

Calculates:
1. Reactive fraction (distance < cutoff)
2. Distance distributions
3. H-bond occupancy
4. Residence time in reactive state

Usage:
    python 04_analyze_reactive.py --tpr prod.tpr --xtc traj.xtc --stereo L
    python 04_analyze_reactive.py --tpr prod.tpr --xtc traj.xtc --stereo D
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances
except ImportError:
    print("Error: MDAnalysis not installed")
    print("Install with: pip install MDAnalysis")
    exit(1)

def calculate_reactive_fraction(u, sorb_res='LSO', pqq_res='PQQ', 
                                  sorb_atom='C2', pqq_atom='C5',
                                  cutoff_nm=0.4):
    """
    Calculate reactive fraction based on distance criterion
    
    Parameters:
    -----------
    u : MDAnalysis.Universe
        Trajectory universe
    sorb_res : str
        Sorbitol residue name (LSO or DSO)
    pqq_res : str
        PQQ residue name
    sorb_atom : str
        Sorbitol reactive atom (C2)
    pqq_atom : str
        PQQ reactive atom (C5 or adjust)
    cutoff_nm : float
        Distance cutoff for reactive state (nm)
    
    Returns:
    --------
    dict : Analysis results
    """
    
    # Select atoms
    try:
        sorb_C2 = u.select_atoms(f"resname {sorb_res} and name {sorb_atom}")
        pqq_ref = u.select_atoms(f"resname {pqq_res} and name {pqq_atom}")
    except:
        print(f"Error: Could not select atoms")
        print(f"  Sorbitol: resname {sorb_res} and name {sorb_atom}")
        print(f"  PQQ: resname {pqq_res} and name {pqq_atom}")
        return None
    
    if len(sorb_C2) == 0:
        print(f"Error: No sorbitol {sorb_atom} atom found")
        print(f"Available sorbitol atoms:")
        sorb_all = u.select_atoms(f"resname {sorb_res}")
        if len(sorb_all) > 0:
            print(f"  {set(sorb_all.names)}")
        return None
    
    if len(pqq_ref) == 0:
        print(f"Error: No PQQ {pqq_atom} atom found")
        print(f"Available PQQ atoms:")
        pqq_all = u.select_atoms(f"resname {pqq_res}")
        if len(pqq_all) > 0:
            print(f"  {set(pqq_all.names)}")
        return None
    
    print(f"\nAtom selection:")
    print(f"  Sorbitol {sorb_atom}: {len(sorb_C2)} atom(s)")
    print(f"  PQQ {pqq_atom}: {len(pqq_ref)} atom(s)")
    
    # Calculate distances over trajectory
    distances_list = []
    times = []
    
    print(f"\nAnalyzing {len(u.trajectory)} frames...")
    
    for ts in u.trajectory:
        # Distance in Angstroms, convert to nm
        d = np.linalg.norm(sorb_C2.positions[0] - pqq_ref.positions[0]) / 10.0
        distances_list.append(d)
        times.append(ts.time / 1000.0)  # Convert ps to ns
    
    distances = np.array(distances_list)
    times = np.array(times)
    
    # Calculate reactive fraction
    reactive = distances < cutoff_nm
    reactive_fraction = reactive.mean()
    
    # Calculate statistics
    mean_dist = distances.mean()
    std_dist = distances.std()
    min_dist = distances.min()
    max_dist = distances.max()
    
    # Calculate residence time
    residence_times = []
    in_reactive = False
    current_residence = 0
    
    for is_reactive in reactive:
        if is_reactive:
            if not in_reactive:
                in_reactive = True
                current_residence = 1
            else:
                current_residence += 1
        else:
            if in_reactive:
                residence_times.append(current_residence)
                in_reactive = False
                current_residence = 0
    
    if in_reactive:
        residence_times.append(current_residence)
    
    # Convert residence times to ns (assuming frame interval)
    if len(times) > 1:
        dt = times[1] - times[0]
        residence_times_ns = np.array(residence_times) * dt
        mean_residence = residence_times_ns.mean() if len(residence_times_ns) > 0 else 0
    else:
        mean_residence = 0
    
    results = {
        'distances': distances,
        'times': times,
        'reactive_fraction': reactive_fraction,
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'n_reactive_frames': reactive.sum(),
        'n_total_frames': len(reactive),
        'residence_times': residence_times,
        'mean_residence_ns': mean_residence,
        'cutoff_nm': cutoff_nm
    }
    
    return results

def plot_results(results, output_prefix, stereo):
    """Generate plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distance vs time
    ax = axes[0, 0]
    ax.plot(results['times'], results['distances'], alpha=0.5, linewidth=0.5)
    ax.axhline(results['cutoff_nm'], color='r', linestyle='--', 
               label=f'Cutoff ({results["cutoff_nm"]:.1f} nm)')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Distance (nm)')
    ax.set_title(f'{stereo}-Sorbitol C2 - PQQ Distance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Distance distribution
    ax = axes[0, 1]
    ax.hist(results['distances'], bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(results['cutoff_nm'], color='r', linestyle='--', 
               label=f'Cutoff ({results["cutoff_nm"]:.1f} nm)')
    ax.axvline(results['mean_distance'], color='g', linestyle='--',
               label=f'Mean ({results["mean_distance"]:.2f} nm)')
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Reactive state occupancy
    ax = axes[1, 0]
    reactive_binary = (results['distances'] < results['cutoff_nm']).astype(int)
    ax.plot(results['times'], reactive_binary, linewidth=0.5)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Reactive State (0/1)')
    ax.set_title(f'Reactive State Occupancy ({results["reactive_fraction"]*100:.1f}%)')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    
    # 4. Residence time distribution
    ax = axes[1, 1]
    if len(results['residence_times']) > 0:
        # Convert to ns
        dt = results['times'][1] - results['times'][0] if len(results['times']) > 1 else 1
        residence_ns = np.array(results['residence_times']) * dt
        ax.hist(residence_ns, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residence Time (ns)')
        ax.set_ylabel('Count')
        ax.set_title(f'Reactive State Residence Time\n(Mean: {results["mean_residence_ns"]:.2f} ns)')
    else:
        ax.text(0.5, 0.5, 'No reactive events', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reactive State Residence Time')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300)
    print(f"\n✓ Plot saved: {output_prefix}_analysis.png")
    
    plt.close()

def save_results(results, output_file, stereo):
    """Save results to text file"""
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Reactive Fraction Analysis - {stereo}-Sorbitol\n")
        f.write("="*70 + "\n\n")
        
        f.write("Parameters:\n")
        f.write(f"  Distance cutoff: {results['cutoff_nm']:.2f} nm\n")
        f.write(f"  Total frames: {results['n_total_frames']}\n")
        f.write(f"  Simulation time: {results['times'][-1]:.2f} ns\n\n")
        
        f.write("Results:\n")
        f.write(f"  Reactive fraction: {results['reactive_fraction']:.4f} ({results['reactive_fraction']*100:.2f}%)\n")
        f.write(f"  Reactive frames: {results['n_reactive_frames']}/{results['n_total_frames']}\n")
        f.write(f"  Mean distance: {results['mean_distance']:.3f} ± {results['std_distance']:.3f} nm\n")
        f.write(f"  Min distance: {results['min_distance']:.3f} nm\n")
        f.write(f"  Max distance: {results['max_distance']:.3f} nm\n")
        f.write(f"  Mean residence time: {results['mean_residence_ns']:.2f} ns\n")
        f.write(f"  Number of reactive events: {len(results['residence_times'])}\n")
    
    print(f"✓ Results saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze reactive fraction from MD trajectory')
    parser.add_argument('--tpr', type=str, required=True, help='TPR file')
    parser.add_argument('--xtc', type=str, required=True, help='XTC trajectory file')
    parser.add_argument('--stereo', type=str, required=True, choices=['L', 'D'],
                        help='Stereoisomer (L or D)')
    parser.add_argument('--cutoff', type=float, default=0.4,
                        help='Distance cutoff for reactive state (nm, default: 0.4)')
    parser.add_argument('--sorb_atom', type=str, default='C2',
                        help='Sorbitol reactive atom name (default: C2)')
    parser.add_argument('--pqq_atom', type=str, default='C5',
                        help='PQQ reactive atom name (default: C5)')
    parser.add_argument('--output', type=str, default='reactive_analysis',
                        help='Output prefix for files')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Reactive Fraction Analysis")
    print("="*70)
    
    # Load trajectory
    print(f"\nLoading trajectory...")
    print(f"  TPR: {args.tpr}")
    print(f"  XTC: {args.xtc}")
    
    try:
        u = mda.Universe(args.tpr, args.xtc)
    except Exception as e:
        print(f"\n✗ Error loading trajectory: {e}")
        return
    
    print(f"  ✓ Loaded {len(u.trajectory)} frames")
    print(f"  Total atoms: {len(u.atoms)}")
    
    # Determine residue name
    sorb_res = 'LSO' if args.stereo == 'L' else 'DSO'
    
    # Analyze
    results = calculate_reactive_fraction(
        u, 
        sorb_res=sorb_res,
        pqq_res='PQQ',
        sorb_atom=args.sorb_atom,
        pqq_atom=args.pqq_atom,
        cutoff_nm=args.cutoff
    )
    
    if results is None:
        print("\n✗ Analysis failed")
        return
    
    # Print results
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"\nReactive Fraction: {results['reactive_fraction']:.4f} ({results['reactive_fraction']*100:.2f}%)")
    print(f"Reactive frames: {results['n_reactive_frames']}/{results['n_total_frames']}")
    print(f"Mean distance: {results['mean_distance']:.3f} ± {results['std_distance']:.3f} nm")
    print(f"Distance range: {results['min_distance']:.3f} - {results['max_distance']:.3f} nm")
    print(f"Mean residence time: {results['mean_residence_ns']:.2f} ns")
    print(f"Number of reactive events: {len(results['residence_times'])}")
    
    # Save results
    output_prefix = f"{args.output}_{args.stereo}"
    save_results(results, f"{output_prefix}_results.txt", args.stereo)
    
    # Plot
    plot_results(results, output_prefix, args.stereo)
    
    # Save distance time series
    np.savetxt(f"{output_prefix}_distances.txt",
               np.column_stack([results['times'], results['distances']]),
               header="Time(ns) Distance(nm)",
               fmt='%.6f')
    print(f"✓ Distance data saved: {output_prefix}_distances.txt")
    
    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
