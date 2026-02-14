"""
Extract reaction centers from atom-mapped reactions.
Identifies which atoms undergo changes during the reaction.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import re


@dataclass
class ReactionCenter:
    """Represents the reaction center of a chemical transformation."""
    atom_indices: List[int]
    change_type: str
    smarts_pattern: Optional[str] = None
    confidence: str = "experimental"


@dataclass
class LabeledReaction:
    """Reaction with identified reaction center."""
    reaction_id: str
    substrate_smiles: str
    product_smiles: str
    reaction_type: str
    ec_number: Optional[str]
    enzyme_name: Optional[str]
    cofactor: Optional[str]
    reaction_center: ReactionCenter
    metadata: Dict


class ReactionCenterExtractor:
    """Extract reaction centers from atom-mapped SMILES."""
    
    def __init__(self):
        self.transformation_patterns = {
            'alcohol_to_carbonyl': (
                lambda s, p: self._is_alcohol(s) and self._is_carbonyl(p)
            ),
            'carbonyl_to_alcohol': (
                lambda s, p: self._is_carbonyl(s) and self._is_alcohol(p)
            ),
            'isomerization': (
                lambda s, p: self._same_oxidation_state(s, p)
            )
        }
    
    def extract_from_mapped_smiles(self, reaction_smiles: str) -> Optional[ReactionCenter]:
        """
        Extract reaction center from atom-mapped reaction SMILES.
        
        Args:
            reaction_smiles: Atom-mapped SMILES (e.g., "[CH2:1][OH:2]>>[CH:1]=[O:2]")
            
        Returns:
            ReactionCenter object or None if extraction fails
        """
        parts = reaction_smiles.split('>>')
        if len(parts) != 2:
            return None
            
        substrate_smiles, product_smiles = parts
        
        substrate_mol = Chem.MolFromSmiles(substrate_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)
        
        if substrate_mol is None or product_mol is None:
            return None
        
        atom_map = self._extract_atom_mapping(substrate_mol, product_mol)
        if not atom_map:
            return None
        
        changed_atoms = self._find_changed_atoms(substrate_mol, product_mol, atom_map)
        
        if not changed_atoms:
            return None
        
        change_type = self._classify_transformation(
            substrate_mol, product_mol, changed_atoms, atom_map
        )
        
        return ReactionCenter(
            atom_indices=changed_atoms,
            change_type=change_type,
            smarts_pattern=None,
            confidence="experimental"
        )
    
    def extract_without_mapping(self, substrate_smiles: str, product_smiles: str) -> Optional[ReactionCenter]:
        """
        Extract reaction center without atom mapping using maximum common substructure.
        Less reliable but useful when mapping is unavailable.
        """
        substrate_mol = Chem.MolFromSmiles(substrate_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)
        
        if substrate_mol is None or product_mol is None:
            return None
        
        mcs = Chem.rdFMCS.FindMCS([substrate_mol, product_mol])
        
        if mcs is None:
            return None
        
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        substrate_match = substrate_mol.GetSubstructMatch(mcs_mol)
        
        changed_atoms = []
        for i in range(substrate_mol.GetNumAtoms()):
            if i not in substrate_match:
                changed_atoms.append(i)
        
        if not changed_atoms:
            return None
        
        return ReactionCenter(
            atom_indices=changed_atoms,
            change_type="unknown",
            confidence="inferred"
        )
    
    def _extract_atom_mapping(self, substrate_mol: Chem.Mol, product_mol: Chem.Mol) -> Dict[int, int]:
        """Extract atom mapping from molecules with atom map numbers."""
        substrate_map = {}
        product_map = {}
        
        for atom in substrate_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                substrate_map[map_num] = atom.GetIdx()
        
        for atom in product_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                product_map[map_num] = atom.GetIdx()
        
        atom_mapping = {}
        for map_num in substrate_map:
            if map_num in product_map:
                atom_mapping[substrate_map[map_num]] = product_map[map_num]
        
        return atom_mapping
    
    def _find_changed_atoms(self, substrate_mol: Chem.Mol, product_mol: Chem.Mol, 
                           atom_map: Dict[int, int]) -> List[int]:
        """Identify atoms that changed during reaction."""
        changed_atoms = []
        
        for sub_idx, prod_idx in atom_map.items():
            sub_atom = substrate_mol.GetAtomWithIdx(sub_idx)
            prod_atom = product_mol.GetAtomWithIdx(prod_idx)
            
            if self._atom_changed(sub_atom, prod_atom, substrate_mol, product_mol):
                changed_atoms.append(sub_idx)
        
        return changed_atoms
    
    def _atom_changed(self, sub_atom: Chem.Atom, prod_atom: Chem.Atom,
                     sub_mol: Chem.Mol, prod_mol: Chem.Mol) -> bool:
        """Check if an atom underwent a change."""
        if sub_atom.GetHybridization() != prod_atom.GetHybridization():
            return True
        
        if sub_atom.GetFormalCharge() != prod_atom.GetFormalCharge():
            return True
        
        if sub_atom.GetTotalNumHs() != prod_atom.GetTotalNumHs():
            return True
        
        sub_bonds = self._get_bond_summary(sub_atom, sub_mol)
        prod_bonds = self._get_bond_summary(prod_atom, prod_mol)
        
        if sub_bonds != prod_bonds:
            return True
        
        return False
    
    def _get_bond_summary(self, atom: Chem.Atom, mol: Chem.Mol) -> Dict:
        """Get summary of bonds for an atom."""
        summary = {}
        
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            bond_type = str(bond.GetBondType())
            other_symbol = other_atom.GetSymbol()
            
            key = f"{other_symbol}_{bond_type}"
            summary[key] = summary.get(key, 0) + 1
        
        return summary
    
    def _classify_transformation(self, substrate_mol: Chem.Mol, product_mol: Chem.Mol,
                                changed_atoms: List[int], atom_map: Dict[int, int]) -> str:
        """Classify the type of transformation."""
        if not changed_atoms:
            return "no_change"
        
        primary_atom_idx = changed_atoms[0]
        sub_atom = substrate_mol.GetAtomWithIdx(primary_atom_idx)
        prod_atom_idx = atom_map.get(primary_atom_idx)
        
        if prod_atom_idx is None:
            return "unknown"
        
        prod_atom = product_mol.GetAtomWithIdx(prod_atom_idx)
        
        sub_bonds = self._get_bond_summary(sub_atom, substrate_mol)
        prod_bonds = self._get_bond_summary(prod_atom, product_mol)
        
        if 'O_SINGLE' in sub_bonds and 'O_DOUBLE' in prod_bonds:
            return "alcohol_to_carbonyl"
        
        if 'O_DOUBLE' in sub_bonds and 'O_SINGLE' in prod_bonds:
            return "carbonyl_to_alcohol"
        
        if sub_atom.GetHybridization() != prod_atom.GetHybridization():
            return "hybridization_change"
        
        return "isomerization"
    
    def _is_alcohol(self, atom: Chem.Atom) -> bool:
        """Check if atom is part of alcohol group."""
        if atom.GetSymbol() != 'C':
            return False
        
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.SINGLE:
                if other.GetTotalNumHs() >= 1:
                    return True
        return False
    
    def _is_carbonyl(self, atom: Chem.Atom) -> bool:
        """Check if atom is part of carbonyl group."""
        if atom.GetSymbol() != 'C':
            return False
        
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
                return True
        return False
    
    def _same_oxidation_state(self, sub_atom: Chem.Atom, prod_atom: Chem.Atom) -> bool:
        """Check if atoms have same oxidation state."""
        return sub_atom.GetHybridization() == prod_atom.GetHybridization()


if __name__ == "__main__":
    extractor = ReactionCenterExtractor()
    
    test_reactions = [
        "[CH2:1][OH:2]>>[CH:1]=[O:2]",
        "[CH:1]([OH:2])>>[C:1](=[O:2])",
    ]
    
    for rxn_smiles in test_reactions:
        print(f"\nReaction: {rxn_smiles}")
        center = extractor.extract_from_mapped_smiles(rxn_smiles)
        if center:
            print(f"  Changed atoms: {center.atom_indices}")
            print(f"  Change type: {center.change_type}")
        else:
            print("  Could not extract reaction center")
