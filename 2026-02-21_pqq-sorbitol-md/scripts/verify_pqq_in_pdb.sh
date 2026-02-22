#!/bin/bash
#
# Verify PQQ presence in PDB structure before using
#

PDB_ID=$1

if [ -z "$PDB_ID" ]; then
    echo "Usage: bash verify_pqq_in_pdb.sh <PDB_ID>"
    echo "Example: bash verify_pqq_in_pdb.sh 2CDU"
    exit 1
fi

echo "============================================================"
echo "Verifying PQQ in PDB: $PDB_ID"
echo "============================================================"

# Download PDB file
echo ""
echo "[1/3] Downloading $PDB_ID.pdb..."
wget -q "https://files.rcsb.org/download/${PDB_ID}.pdb" -O "/tmp/${PDB_ID}.pdb"

if [ ! -f "/tmp/${PDB_ID}.pdb" ]; then
    echo "✗ Failed to download $PDB_ID.pdb"
    exit 1
fi

echo "  ✓ Downloaded"

# Check for PQQ
echo ""
echo "[2/3] Checking for PQQ cofactor..."
PQQ_COUNT=$(grep "PQQ" "/tmp/${PDB_ID}.pdb" | grep "HETATM" | wc -l)

if [ "$PQQ_COUNT" -gt 0 ]; then
    echo "  ✓ PQQ found: $PQQ_COUNT atoms"
    grep "PQQ" "/tmp/${PDB_ID}.pdb" | grep "HETATM" | head -3
else
    echo "  ✗ No PQQ found"
    echo ""
    echo "Available ligands:"
    grep "HETATM" "/tmp/${PDB_ID}.pdb" | awk '{print $4}' | sort -u | head -10
    exit 1
fi

# Check for Ca
echo ""
echo "[3/3] Checking for Ca²⁺..."
CA_COUNT=$(grep "HETATM" "/tmp/${PDB_ID}.pdb" | grep " CA " | wc -l)

if [ "$CA_COUNT" -gt 0 ]; then
    echo "  ✓ Ca found: $CA_COUNT atoms"
else
    echo "  ⚠ No Ca found (may need to add separately)"
fi

# Summary
echo ""
echo "============================================================"
echo "RESULT: $PDB_ID"
echo "============================================================"
echo "  PQQ atoms: $PQQ_COUNT"
echo "  Ca atoms: $CA_COUNT"

if [ "$PQQ_COUNT" -gt 0 ]; then
    echo ""
    echo "✓ This structure is suitable for PQQ transplant"
    echo ""
    echo "To use this structure:"
    echo "  mv /tmp/${PDB_ID}.pdb ../structures/"
    echo "  # Update chimerax_transplant.cxc to use $PDB_ID.pdb"
else
    echo ""
    echo "✗ This structure is NOT suitable (no PQQ)"
    rm "/tmp/${PDB_ID}.pdb"
fi

echo "============================================================"
