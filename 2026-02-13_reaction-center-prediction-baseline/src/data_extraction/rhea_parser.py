"""
Rhea database parser for extracting enzyme-catalyzed reactions.
Focuses on polyol/sugar oxidation-reduction and isomerization reactions.
"""

import requests
import gzip
import io
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RheaReaction:
    """Structured representation of a Rhea reaction."""
    reaction_id: str
    equation: str
    ec_numbers: List[str]
    substrate_smiles: Optional[str]
    product_smiles: Optional[str]
    atom_mapped_smiles: Optional[str]
    is_balanced: bool
    direction: str


class RheaParser:
    """Parser for Rhea reaction database."""
    
    RHEA_SMILES_URL = "ftp://ftp.expasy.org/databases/rhea/ctfiles/rhea-reaction-smiles.tsv.gz"
    RHEA_DIRECTIONS_URL = "ftp://ftp.expasy.org/databases/rhea/tsv/rhea-directions.tsv"
    
    EC_CLASSES_OF_INTEREST = ["1.1.1", "5.3"]
    KEYWORDS = ["polyol", "sugar", "sorbitol", "fructose", "glucose", "xylitol", 
                "xylulose", "dehydrogenase", "isomerase", "alcohol", "ketose", "aldose"]
    
    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_rhea_smiles(self) -> Path:
        """Download Rhea SMILES file if not cached."""
        cache_file = self.cache_dir / "rhea-reaction-smiles.tsv.gz"
        
        if cache_file.exists():
            print(f"Using cached file: {cache_file}")
            return cache_file
            
        print(f"Downloading Rhea SMILES from {self.RHEA_SMILES_URL}")
        response = requests.get(self.RHEA_SMILES_URL, stream=True)
        response.raise_for_status()
        
        with open(cache_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded to {cache_file}")
        return cache_file
    
    def parse_smiles_file(self, filepath: Path) -> List[Dict]:
        """Parse Rhea SMILES TSV file."""
        reactions = []
        
        with gzip.open(filepath, 'rt') as f:
            header = f.readline().strip().split('\t')
            
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue
                    
                reaction_data = dict(zip(header, fields))
                reactions.append(reaction_data)
        
        print(f"Parsed {len(reactions)} reactions from Rhea")
        return reactions
    
    def filter_relevant_reactions(self, reactions: List[Dict]) -> List[RheaReaction]:
        """Filter reactions by EC class and keywords."""
        filtered = []
        
        for rxn_data in reactions:
            rhea_id = rxn_data.get('RHEA_ID', '')
            equation = rxn_data.get('EQUATION', '')
            smiles = rxn_data.get('REACTION_SMILES', '')
            
            if not smiles or smiles == 'null':
                continue
            
            equation_lower = equation.lower()
            if not any(kw in equation_lower for kw in self.KEYWORDS):
                continue
            
            parts = smiles.split('>>')
            if len(parts) != 2:
                continue
                
            substrate_smiles = parts[0].strip()
            product_smiles = parts[1].strip()
            
            reaction = RheaReaction(
                reaction_id=rhea_id,
                equation=equation,
                ec_numbers=[],
                substrate_smiles=substrate_smiles,
                product_smiles=product_smiles,
                atom_mapped_smiles=smiles if ':' in smiles else None,
                is_balanced=True,
                direction='bidirectional'
            )
            
            filtered.append(reaction)
        
        print(f"Filtered to {len(filtered)} relevant reactions")
        return filtered
    
    def extract_reactions(self) -> List[RheaReaction]:
        """Main pipeline: download, parse, and filter Rhea reactions."""
        smiles_file = self.download_rhea_smiles()
        all_reactions = self.parse_smiles_file(smiles_file)
        relevant_reactions = self.filter_relevant_reactions(all_reactions)
        
        return relevant_reactions


if __name__ == "__main__":
    parser = RheaParser()
    reactions = parser.extract_reactions()
    
    print(f"\nExtracted {len(reactions)} reactions")
    print("\nSample reactions:")
    for rxn in reactions[:5]:
        print(f"\n{rxn.reaction_id}: {rxn.equation}")
        print(f"  Substrate: {rxn.substrate_smiles[:80]}...")
        print(f"  Product: {rxn.product_smiles[:80]}...")
