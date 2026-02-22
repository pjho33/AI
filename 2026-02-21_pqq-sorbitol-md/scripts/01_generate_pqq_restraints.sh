#!/bin/bash
#
# Generate PQQ position restraints for GROMACS
# Creates both strong (1000) and weak (100) restraint files
#

set -e

echo "============================================================"
echo "PQQ Position Restraint Generation"
echo "============================================================"

# Check input
if [ ! -f "../structures/receptor_PQQ_Ca.pdb" ]; then
    echo "Error: receptor_PQQ_Ca.pdb not found"
    echo "Please prepare the receptor structure first"
    exit 1
fi

WORKDIR="restraints"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Copy structure
cp ../../structures/receptor_PQQ_Ca.pdb ./

echo ""
echo "[1/4] Creating index file for PQQ heavy atoms..."

# Create index group for PQQ heavy atoms (exclude hydrogens)
cat > make_pqq_index.txt << 'EOF'
r PQQ & ! a H*
name 20 PQQ_heavy
q
EOF

gmx make_ndx -f receptor_PQQ_Ca.pdb -o pqq_index.ndx < make_pqq_index.txt > make_ndx.log 2>&1

if [ ! -f pqq_index.ndx ]; then
    echo "Error: Failed to create index file"
    cat make_ndx.log
    exit 1
fi

echo "  ✓ Index file created"

# Count PQQ heavy atoms
N_PQQ=$(grep -A1 "PQQ_heavy" pqq_index.ndx | tail -1 | wc -w)
echo "  PQQ heavy atoms: ${N_PQQ}"

echo ""
echo "[2/4] Generating strong position restraints (1000 kJ/mol/nm²)..."

# Generate strong restraints
gmx genrestr -f receptor_PQQ_Ca.pdb \
             -n pqq_index.ndx \
             -o posre_PQQ_strong.itp \
             -fc 1000 1000 1000 << EOF > genrestr_strong.log 2>&1
PQQ_heavy
EOF

if [ ! -f posre_PQQ_strong.itp ]; then
    echo "Error: Failed to generate strong restraints"
    cat genrestr_strong.log
    exit 1
fi

echo "  ✓ posre_PQQ_strong.itp created"

echo ""
echo "[3/4] Generating weak position restraints (100 kJ/mol/nm²)..."

# Generate weak restraints
gmx genrestr -f receptor_PQQ_Ca.pdb \
             -n pqq_index.ndx \
             -o posre_PQQ_weak.itp \
             -fc 100 100 100 << EOF > genrestr_weak.log 2>&1
PQQ_heavy
EOF

if [ ! -f posre_PQQ_weak.itp ]; then
    echo "Error: Failed to generate weak restraints"
    cat genrestr_weak.log
    exit 1
fi

echo "  ✓ posre_PQQ_weak.itp created"

echo ""
echo "[4/4] Creating Ca²⁺ position restraints..."

# Create index for Ca
cat > make_ca_index.txt << 'EOF'
r CA
name 20 Calcium
q
EOF

gmx make_ndx -f receptor_PQQ_Ca.pdb -o ca_index.ndx < make_ca_index.txt > make_ca_ndx.log 2>&1

# Generate Ca restraints (strong)
gmx genrestr -f receptor_PQQ_Ca.pdb \
             -n ca_index.ndx \
             -o posre_CA.itp \
             -fc 1000 1000 1000 << EOF > genrestr_ca.log 2>&1
Calcium
EOF

echo "  ✓ posre_CA.itp created"

# Copy to parameters directory
echo ""
echo "Copying restraint files to parameters directory..."
cp posre_PQQ_strong.itp ../../parameters/
cp posre_PQQ_weak.itp ../../parameters/
cp posre_CA.itp ../../parameters/

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo ""
echo "Generated restraint files:"
echo "  • posre_PQQ_strong.itp  (1000 kJ/mol/nm²)"
echo "  • posre_PQQ_weak.itp    (100 kJ/mol/nm²)"
echo "  • posre_CA.itp          (1000 kJ/mol/nm²)"
echo ""
echo "Files saved to: ../parameters/"
echo ""
echo "Usage in topology file (topol.top):"
echo ""
echo "  ; Include position restraints"
echo "  #ifdef POSRES_PQQ"
echo "  #include \"posre_PQQ_strong.itp\""
echo "  #endif"
echo ""
echo "  #ifdef POSRES_PQQ_WEAK"
echo "  #include \"posre_PQQ_weak.itp\""
echo "  #endif"
echo ""
echo "  #ifdef POSRES_CA"
echo "  #include \"posre_CA.itp\""
echo "  #endif"
echo ""
echo "Usage in MDP file:"
echo ""
echo "  ; For NVT/NPT (strong restraints)"
echo "  define = -DPOSRES_PQQ -DPOSRES_CA"
echo ""
echo "  ; For production (weak restraints)"
echo "  define = -DPOSRES_PQQ_WEAK -DPOSRES_CA"
echo ""
echo "  ; For production (no PQQ restraints)"
echo "  define = -DPOSRES_CA"
echo ""
echo "============================================================"

cd ..
