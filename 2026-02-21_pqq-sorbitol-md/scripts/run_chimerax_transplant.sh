#!/bin/bash
#
# Automated PQQ+Ca transplant using ChimeraX
# Runs in Drug-MD conda environment
#

set -e

echo "============================================================"
echo "PQQ+Ca Transplant - Automated Script"
echo "============================================================"

# Activate Drug-MD environment
echo ""
echo "[1/5] Activating Drug-MD conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Drug-MD

# Check ChimeraX is available
if ! command -v chimerax &> /dev/null; then
    echo "✗ Error: ChimeraX not found in Drug-MD environment"
    echo "  Install with: conda install -c conda-forge chimerax"
    exit 1
fi

echo "  ✓ ChimeraX found: $(which chimerax)"

# Navigate to structures directory
cd /home/pjho3/projects/AI/2026-02-21_pqq-sorbitol-md/structures

# Check required files
echo ""
echo "[2/5] Checking required files..."

if [ ! -f "receptor_af.pdb" ]; then
    echo "✗ Error: receptor_af.pdb not found"
    exit 1
fi
echo "  ✓ receptor_af.pdb found"

if [ ! -f "4CVB.pdb" ]; then
    echo "✗ Error: 4CVB.pdb not found"
    echo "  Downloading..."
    wget -q https://files.rcsb.org/download/4CVB.pdb
    echo "  ✓ 4CVB.pdb downloaded"
else
    echo "  ✓ 4CVB.pdb found"
fi

if [ ! -f "../scripts/chimerax_transplant.cxc" ]; then
    echo "✗ Error: chimerax_transplant.cxc not found"
    exit 1
fi
echo "  ✓ chimerax_transplant.cxc found"

# Run ChimeraX with script
echo ""
echo "[3/5] Running ChimeraX transplant..."
echo "  This will:"
echo "    - Align receptor_af.pdb with 4CVB.pdb"
echo "    - Extract PQQ and Ca from 4CVB"
echo "    - Transplant to receptor_af"
echo "    - Save as receptor_PQQ_Ca.pdb"
echo ""

# Run ChimeraX in nogui mode with script
chimerax --nogui --script ../scripts/chimerax_transplant.cxc > chimerax_transplant.log 2>&1

# Check if output was created
echo ""
echo "[4/5] Verifying output..."

if [ ! -f "receptor_PQQ_Ca.pdb" ]; then
    echo "✗ Error: receptor_PQQ_Ca.pdb not created"
    echo "  Check log: structures/chimerax_transplant.log"
    exit 1
fi

echo "  ✓ receptor_PQQ_Ca.pdb created"

# Verify PQQ and Ca are present
PQQ_COUNT=$(awk '/^HETATM/ && $4=="PQQ"' receptor_PQQ_Ca.pdb | wc -l)
CA_COUNT=$(awk '/^HETATM/ && $3=="CA" && $4=="CA"' receptor_PQQ_Ca.pdb | wc -l)

echo "  PQQ atoms found: $PQQ_COUNT"
echo "  Ca atoms found: $CA_COUNT"

if [ "$PQQ_COUNT" -eq 0 ]; then
    echo "  ⚠ Warning: No PQQ atoms found"
fi

if [ "$CA_COUNT" -eq 0 ]; then
    echo "  ⚠ Warning: No Ca atoms found"
fi

# Summary
echo ""
echo "[5/5] Summary"
echo "============================================================"

FILE_SIZE=$(ls -lh receptor_PQQ_Ca.pdb | awk '{print $5}')
TOTAL_ATOMS=$(grep -cE "^(ATOM|HETATM)" receptor_PQQ_Ca.pdb || true)

echo ""
echo "Output file: receptor_PQQ_Ca.pdb"
echo "  Size: $FILE_SIZE"
echo "  Total atoms: $TOTAL_ATOMS"
echo "  PQQ atoms: $PQQ_COUNT"
echo "  Ca atoms: $CA_COUNT"
echo ""

if [ "$PQQ_COUNT" -gt 0 ] && [ "$CA_COUNT" -gt 0 ]; then
    echo "✓ Transplant successful!"
    echo ""
    echo "Next steps:"
    echo "  1. Visual check (optional):"
    echo "     chimerax receptor_PQQ_Ca.pdb"
    echo ""
    echo "  2. Proceed to Phase 2 (Parameterization):"
    echo "     cd ../scripts"
    echo "     # Generate CGenFF parameters for PQQ and sorbitol"
    echo ""
else
    echo "⚠ Transplant completed with warnings"
    echo "  Check structures/chimerax_transplant.log for details"
    echo ""
fi

echo "============================================================"

# Return to project root
cd ..
