#!/usr/bin/env python3
"""
Automated pose insertion for GROMACS MD screening

Takes docking poses and creates complete MD-ready systems:
1. Combines receptor + PQQ + Ca + sorbitol pose
2. Creates directory structure
3. Copies topology and parameters
4. Prepares for solvation and MD

Usage:
    python 02_insert_poses.py --stereo L --n_poses 20
    python 02_insert_poses.py --stereo D --n_poses 20
"""

import argparse
import shutil
from pathlib import Path
import numpy as np

def parse_pdb(filename):
    """Parse PDB file"""
    atoms = []
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atoms.append(line)
    return atoms

def write_pdb(atoms, filename):
    """Write PDB file"""
    with open(filename, 'w') as f:
        for line in atoms:
            f.write(line)
        f.write("END\n")

def combine_structures(receptor_pdb, pose_pdb, output_pdb, stereo='L'):
    """Combine receptor (with PQQ+Ca) and sorbitol pose"""
    
    # Read receptor (protein + PQQ + Ca)
    receptor_atoms = []
    with open(receptor_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Skip any existing sorbitol
                res_name = line[17:20].strip()
                if res_name not in ['LSO', 'DSO', 'SBT', 'L-s', 'D-s']:
                    receptor_atoms.append(line)
    
    # Read sorbitol pose
    sorbitol_atoms = []
    with open(pose_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Change residue name to LSO or DSO
                new_res = 'LSO' if stereo == 'L' else 'DSO'
                new_line = line[:17] + f"{new_res:>3}" + line[20:]
                sorbitol_atoms.append(new_line)
    
    # Combine
    combined = receptor_atoms + sorbitol_atoms
    
    # Renumber serials
    renumbered = []
    serial = 1
    for line in combined:
        new_line = line[:6] + f"{serial:>5}" + line[11:]
        renumbered.append(new_line)
        serial += 1
    
    # Write
    write_pdb(renumbered, output_pdb)
    
    return len(receptor_atoms), len(sorbitol_atoms)

def create_md_directory(pose_num, stereo, base_dir):
    """Create MD directory structure for a pose"""
    
    pose_dir = base_dir / f"pose_{stereo}_{pose_num:02d}"
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (pose_dir / "em").mkdir(exist_ok=True)
    (pose_dir / "nvt").mkdir(exist_ok=True)
    (pose_dir / "npt").mkdir(exist_ok=True)
    (pose_dir / "prod").mkdir(exist_ok=True)
    
    return pose_dir

def copy_template_files(pose_dir, template_dir):
    """Copy topology and MDP files to pose directory"""
    
    # Copy topology template
    if (template_dir / "topol_template.top").exists():
        shutil.copy(template_dir / "topol_template.top", pose_dir / "topol.top")
    
    # Copy MDP files
    for mdp in ['em.mdp', 'nvt.mdp', 'npt.mdp', 'prod.mdp']:
        if (template_dir / mdp).exists():
            shutil.copy(template_dir / mdp, pose_dir / mdp)

def create_run_script(pose_dir, pose_num, stereo):
    """Create automated run script for this pose"""
    
    script = f"""#!/bin/bash
#
# MD screening run for pose {stereo}_{pose_num:02d}
#

set -e

POSE="pose_{stereo}_{pose_num:02d}"

echo "============================================================"
echo "MD Screening: ${{POSE}}"
echo "============================================================"

# 1. Energy Minimization
echo ""
echo "[1/4] Energy Minimization..."
cd em
gmx grompp -f ../em.mdp -c ../system.gro -p ../topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em -ntmpi 1
cd ..

# 2. NVT Equilibration
echo ""
echo "[2/4] NVT Equilibration (500 ps)..."
cd nvt
gmx grompp -f ../nvt.mdp -c ../em/em.gro -p ../topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt -ntmpi 1
cd ..

# 3. NPT Equilibration
echo ""
echo "[3/4] NPT Equilibration (2 ns)..."
cd npt
gmx grompp -f ../npt.mdp -c ../nvt/nvt.gro -p ../topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt -ntmpi 1
cd ..

# 4. Production MD
echo ""
echo "[4/4] Production MD (10-20 ns)..."
cd prod
gmx grompp -f ../prod.mdp -c ../npt/npt.gro -p ../topol.top -o prod.tpr -maxwarn 1
gmx mdrun -v -deffnm prod -ntmpi 1
cd ..

echo ""
echo "============================================================"
echo "Screening complete: ${{POSE}}"
echo "============================================================"
echo ""
echo "Next: Analyze trajectory for stability and binding"
"""
    
    script_file = pose_dir / "run_md.sh"
    with open(script_file, 'w') as f:
        f.write(script)
    
    script_file.chmod(0o755)

def main():
    parser = argparse.ArgumentParser(description='Insert docking poses into MD systems')
    parser.add_argument('--stereo', type=str, required=True, choices=['L', 'D'],
                        help='Stereoisomer (L or D)')
    parser.add_argument('--n_poses', type=int, default=20,
                        help='Number of poses to process')
    parser.add_argument('--receptor', type=str, 
                        default='../structures/receptor_PQQ_Ca.pdb',
                        help='Receptor PDB file')
    parser.add_argument('--docking_dir', type=str,
                        default='../docking',
                        help='Directory containing docking poses')
    parser.add_argument('--output_dir', type=str,
                        default='../md_runs/screening',
                        help='Output directory for MD systems')
    parser.add_argument('--template_dir', type=str,
                        default='../md_runs/template',
                        help='Template directory with topology and MDPs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Automated Pose Insertion for MD Screening")
    print("="*70)
    print(f"\nStereoisomer: {args.stereo}-sorbitol")
    print(f"Number of poses: {args.n_poses}")
    
    # Check inputs
    receptor_pdb = Path(args.receptor)
    if not receptor_pdb.exists():
        print(f"\n✗ Error: Receptor not found: {receptor_pdb}")
        return
    
    docking_dir = Path(args.docking_dir)
    if not docking_dir.exists():
        print(f"\n✗ Error: Docking directory not found: {docking_dir}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    template_dir = Path(args.template_dir)
    
    # Process each pose
    print(f"\n{'='*70}")
    print("Processing Poses")
    print(f"{'='*70}")
    
    successful = 0
    
    for i in range(1, args.n_poses + 1):
        pose_file = docking_dir / f"pose_{args.stereo}_{i:02d}.pdb"
        
        if not pose_file.exists():
            print(f"\n⚠ Pose {i:02d}: File not found, skipping")
            continue
        
        print(f"\nPose {i:02d}:")
        
        # Create MD directory
        pose_dir = create_md_directory(i, args.stereo, output_dir)
        print(f"  Directory: {pose_dir}")
        
        # Combine structures
        combined_pdb = pose_dir / "system.pdb"
        n_receptor, n_sorbitol = combine_structures(
            receptor_pdb, pose_file, combined_pdb, args.stereo
        )
        print(f"  Receptor atoms: {n_receptor}")
        print(f"  Sorbitol atoms: {n_sorbitol}")
        
        # Convert to GRO (if pdb2gmx not needed)
        # For now, just copy PDB - user will run pdb2gmx separately
        
        # Copy template files
        copy_template_files(pose_dir, template_dir)
        print(f"  ✓ Template files copied")
        
        # Create run script
        create_run_script(pose_dir, i, args.stereo)
        print(f"  ✓ Run script created")
        
        successful += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"\nSuccessfully processed: {successful}/{args.n_poses} poses")
    print(f"Output directory: {output_dir}")
    
    print(f"\nNext steps:")
    print(f"  1. For each pose directory:")
    print(f"     cd {output_dir}/pose_{args.stereo}_XX")
    print(f"     gmx pdb2gmx -f system.pdb -o system.gro -p topol.top")
    print(f"     gmx editconf -f system.gro -o boxed.gro -c -d 1.2 -bt dodecahedron")
    print(f"     gmx solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top")
    print(f"     gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr")
    print(f"     gmx genion -s ions.tpr -o system.gro -p topol.top -neutral -conc 0.15")
    print(f"  2. Run MD: bash run_md.sh")
    
    print(f"\nOr use batch script: scripts/03_run_screening.sh")

if __name__ == "__main__":
    main()
