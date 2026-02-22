#!/usr/bin/env python3
"""
PQQ+Ca transplant using BioPython
Aligns receptor_af.pdb with 1WPQ.pdb and transplants PQQ+Ca
"""

from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
import numpy as np
import sys

class ProteinSelect(Select):
    """Select only protein atoms"""
    def accept_residue(self, residue):
        return residue.id[0] == ' '  # Standard residues only

class LigandSelect(Select):
    """Select only hetero atoms (ligands)"""
    def accept_residue(self, residue):
        return residue.id[0] != ' '  # HETATM only

def align_structures(mobile_structure, target_structure):
    """Align mobile structure to target using CA atoms"""
    
    # Get CA atoms from both structures
    mobile_atoms = []
    target_atoms = []
    
    for mobile_chain in mobile_structure[0]:
        for mobile_res in mobile_chain:
            if mobile_res.id[0] == ' ' and 'CA' in mobile_res:
                mobile_atoms.append(mobile_res['CA'])
    
    for target_chain in target_structure[0]:
        for target_res in target_chain:
            if target_res.id[0] == ' ' and 'CA' in target_res:
                target_atoms.append(target_res['CA'])
    
    # Use minimum length
    min_len = min(len(mobile_atoms), len(target_atoms))
    mobile_atoms = mobile_atoms[:min_len]
    target_atoms = target_atoms[:min_len]
    
    print(f"  Aligning {min_len} CA atoms...")
    
    # Superimpose
    super_imposer = Superimposer()
    super_imposer.set_atoms(target_atoms, mobile_atoms)
    
    # Apply transformation to entire mobile structure
    super_imposer.apply(mobile_structure[0].get_atoms())
    
    rmsd = super_imposer.rms
    print(f"  RMSD: {rmsd:.2f} Å")
    
    return rmsd

def extract_ligands(structure, ligand_names=['PQQ', 'CA']):
    """Extract specific ligands from structure"""
    ligands = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.resname.strip()
                if res_name in ligand_names or residue.id[0] != ' ':
                    # Check if it's PQQ or CA
                    if 'PQQ' in res_name or res_name == 'CA':
                        ligands.append(residue)
    
    return ligands

def main():
    print("="*70)
    print("PQQ+Ca Transplant - Python Script")
    print("="*70)
    
    # File paths
    receptor_file = "../structures/receptor_af.pdb"
    homolog_file = "../structures/1WPQ.pdb"
    output_file = "../structures/receptor_PQQ_Ca.pdb"
    
    # Parse structures
    print("\n[1/5] Loading structures...")
    parser = PDBParser(QUIET=True)
    
    try:
        receptor = parser.get_structure('receptor', receptor_file)
        print(f"  ✓ Loaded {receptor_file}")
    except Exception as e:
        print(f"  ✗ Error loading receptor: {e}")
        sys.exit(1)
    
    try:
        homolog = parser.get_structure('homolog', homolog_file)
        print(f"  ✓ Loaded {homolog_file}")
    except Exception as e:
        print(f"  ✗ Error loading homolog: {e}")
        sys.exit(1)
    
    # Align structures
    print("\n[2/5] Aligning structures...")
    try:
        rmsd = align_structures(homolog, receptor)
        if rmsd > 5.0:
            print(f"  ⚠ Warning: High RMSD ({rmsd:.2f} Å)")
            print(f"    Structures may not align well")
    except Exception as e:
        print(f"  ✗ Error during alignment: {e}")
        sys.exit(1)
    
    # Extract ligands from homolog
    print("\n[3/5] Extracting PQQ and Ca from homolog...")
    ligands = extract_ligands(homolog)
    
    if not ligands:
        print("  ✗ No ligands found in homolog")
        sys.exit(1)
    
    print(f"  Found {len(ligands)} ligand residue(s):")
    for lig in ligands:
        print(f"    - {lig.resname.strip()} (chain {lig.parent.id}, {len(lig)} atoms)")
    
    # Add ligands to receptor
    print("\n[4/5] Transplanting ligands to receptor...")
    
    # Get first chain of receptor
    receptor_chain = list(receptor[0])[0]
    
    for ligand in ligands:
        # Create a copy of the ligand
        new_ligand = ligand.copy()
        new_ligand.detach_parent()
        
        # Add to receptor
        try:
            receptor_chain.add(new_ligand)
            print(f"  ✓ Added {ligand.resname.strip()}")
        except Exception as e:
            print(f"  ⚠ Warning adding {ligand.resname.strip()}: {e}")
    
    # Save combined structure
    print("\n[5/5] Saving combined structure...")
    io = PDBIO()
    io.set_structure(receptor)
    
    try:
        io.save(output_file)
        print(f"  ✓ Saved to {output_file}")
    except Exception as e:
        print(f"  ✗ Error saving: {e}")
        sys.exit(1)
    
    # Verify output
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    # Count atoms
    total_atoms = sum(1 for _ in receptor[0].get_atoms())
    protein_atoms = sum(1 for chain in receptor[0] for res in chain if res.id[0] == ' ' for _ in res.get_atoms())
    ligand_atoms = total_atoms - protein_atoms
    
    print(f"\nOutput file: {output_file}")
    print(f"  Total atoms: {total_atoms}")
    print(f"  Protein atoms: {protein_atoms}")
    print(f"  Ligand atoms: {ligand_atoms}")
    print(f"  Alignment RMSD: {rmsd:.2f} Å")
    
    # Check for PQQ and Ca
    has_pqq = False
    has_ca = False
    
    for chain in receptor[0]:
        for res in chain:
            if 'PQQ' in res.resname:
                has_pqq = True
            if res.resname.strip() == 'CA':
                has_ca = True
    
    print(f"\nLigands present:")
    print(f"  PQQ: {'✓' if has_pqq else '✗'}")
    print(f"  Ca: {'✓' if has_ca else '✗'}")
    
    if has_pqq and has_ca:
        print("\n✓ Transplant successful!")
        print("\nNext steps:")
        print("  1. Visual check (optional):")
        print("     pymol structures/receptor_PQQ_Ca.pdb")
        print("  2. Proceed to Phase 2 (Parameterization)")
    else:
        print("\n⚠ Transplant completed with warnings")
        print("  Some ligands may be missing")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
