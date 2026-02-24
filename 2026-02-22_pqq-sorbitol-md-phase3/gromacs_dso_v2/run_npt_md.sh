#!/bin/bash
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drug-md
export GMXLIB=/home/pjho3tr/miniforge3/envs/drug-md/share/gromacs/top
cd /home/pjho3tr/projects/AI/2026-02-22_pqq-sorbitol-md-phase3/gromacs_dso_v2

echo "=== NPT grompp ==="
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 2

echo "=== NPT mdrun ==="
gmx mdrun -v -deffnm npt -ntmpi 1 -ntomp 4

echo "=== MD grompp ==="
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md.tpr -maxwarn 2

echo "=== Production MD mdrun ==="
gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 4

echo "=== ALL DONE ==="
