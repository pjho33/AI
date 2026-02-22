# PDB Structures with PQQ Cofactor - Verification

## Known PQQ-containing structures

### Method: Check RCSB PDB ligand search

Search URL: https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text_chem%22%2C%22parameters%22%3A%7B%22value%22%3A%22PQQ%22%7D%7D%7D

### Candidate structures to verify:

1. **2CDU** - Methanol dehydrogenase
2. **1G6K** - Glucose dehydrogenase  
3. **1KB0** - Glucose dehydrogenase
4. **3PQQ** - Quinoprotein glucose dehydrogenase

## Verification Process

For each candidate:
1. Download PDB file
2. Check for PQQ in HETATM records: `grep "PQQ" file.pdb`
3. Check for Ca: `grep "CA " file.pdb | grep "HETATM"`
4. If both present → use for transplant
5. If not → try next candidate

## Current Status

- ❌ 1WPQ: Has NAD, not PQQ
- ❌ 1QBI: No PQQ in structure
- ⏳ Testing: 2CDU, 1G6K, 1KB0, 3PQQ
