#!/usr/bin/env python3
"""
Analyze Q5FPE8 sequence to identify domain boundaries
"""

sequence = """
MRVRRILLRVAAGLVIVPVGLVAVAVTGVLVGINISPGQRLIERKIGPLTGDMVEISGLS
GFLPHHLAVAKLLIKDTKGPWIELDNADLRWSPLSLLHLDAKITNLSASRVAVLRKMVSE
PEKTPAKPTTTSTAPSKLHLRVDLASLHVGRLEIGPDYTVTPTAFSLDGHAHIHSIAPFL
DGVTMKTLPVMDVALALKRLDQPGSLTLNVQTPKKRIAVGIRFQEGDNGFATTLGQMPQL
DPLDLHLNLNGPRTASLLDFGLGAGPIKASATGTMNLLTRVGDLHVKANAPSMTLRPGIA
WNAIALDTDLHGPLTAPNGQGVLDIDALTAAGAGIGHLHAQFEGVESDQPNDATGHLTAS
LDGLRIPGSQPTLLAAAPLTLDVLAHPLAPSAPIVAKLDHPLFHGTATADLKPAAKGHLD
LDLPDLHPLAAMGSTDLKGNAGLHADFAMPVTKRDDLTLKSTGTLAIIGGQAQAVNLIGK
TGTFSLDLTKSPANVLTLKSFGLDGAALHMLVSSVIDMAHGNRMQTKASVQLPDLAKASP
AILGNTTLTATADGPTDDLAVKADLNGDFGTKEVAKGPVELHADFQHLPSHPDGTLTAKG
TLDHAPLVLDTALQQDDADAYHLDLNTLGWNSLTGKGRLRLPKGAKVPLGDLDVSIRNLG
DFQRLIGQAISGHLTLGLHTTEAENAPPVVKLGLDGMLSMAQAAVQSLKVNGTITNPIDA
PEPALVLDLAGMRYQAMTGKAHATIKGPQTAMAIALNAAFQNVMDAPANIDTALVLNAPE
KTVRLGQLTALAKGESLRLSRPAVVSFGKTMGVDHLLATVAPQGVAPATIDVVGTLKPAL
ALTARLDHITPAIAKPFAPDLSATGAISLNAKLGGTLAAPTGTVSLTGRDLRMRTGPAAS
LPAAQILANVGLAASSAKVDATLGAGPSVALAVRGTAPLSSTGAMALATTGHVDLSVGNA
VLGASGMGVAGKVGINLNVAGTAAQPRATGQVTLENASFDHYAQGVHLNHINGALVASGD
SIAVNHIVAHAGPGTIVLEGTVGAFRPDLPVDLHITSEKARPVSSDLLTATINTDLHIHG
QATTRLDVDGKVNIPNATINIPDSMPASVPQLDVIRPGQKPPSSSSSSLIIGLGVDVISP
GEFFVRGHGVFAEMQGRLRVRGTSAEPAVSGGFDLKRGNFNLGGINLNFTNGRVAFNGSG
VNHKLDPTLDFRADRNASGTLASLLVTGYASAPKIDFASTPSLPRDQVLSILLFGTDSHS
LSTTQLAELGAAVVQLAGGSAFDPLSKVRNLLGLDRLAVGGGSGVDNGGTSVEAGKYVMK
GVYVGAKQATSGSGTQAQVQIDLTKRLKFNTTVGTGGQVTGFTTPENDPGSSIGLSYGYS
Y
""".replace('\n', '').strip()

print("="*70)
print("Q5FPE8 Sequence Analysis")
print("="*70)

print(f"\nFull sequence length: {len(sequence)} aa")

# Analyze N-terminus for signal peptide and TM domain
print("\nN-terminal analysis (first 100 residues):")
print(sequence[:100])

# Look for hydrophobic stretches (TM domain indicators)
print("\n" + "="*70)
print("Hydrophobic stretch analysis (potential TM domains)")
print("="*70)

hydrophobic = set('AILMFVW')
window = 20

for i in range(0, min(150, len(sequence) - window)):
    segment = sequence[i:i+window]
    hydro_count = sum(1 for aa in segment if aa in hydrophobic)
    hydro_percent = hydro_count / window * 100
    
    if hydro_percent > 60:  # >60% hydrophobic suggests TM
        print(f"Position {i+1}-{i+window}: {hydro_percent:.1f}% hydrophobic")
        print(f"  Sequence: {segment}")

# Identify likely domain boundaries
print("\n" + "="*70)
print("Domain Boundary Prediction")
print("="*70)

print("\nN-terminal region (1-60):")
print("  Contains: Signal peptide + TM domain")
print("  Characteristics: High hydrophobicity")

print("\nCatalytic domain (likely starts ~60-100):")
print("  Should contain β-propeller fold")
print("  Length: ~500-600 residues")

# Check for β-propeller motifs (WD40-like repeats)
print("\n" + "="*70)
print("Looking for β-propeller motifs")
print("="*70)

# Count Trp (W) residues (common in β-propellers)
w_count = sequence.count('W')
print(f"\nTryptophan (W) count: {w_count}")
print(f"Tryptophan positions: ", end="")
for i, aa in enumerate(sequence):
    if aa == 'W':
        print(f"{i+1} ", end="")
print()

# Suggested catalytic domain extraction
print("\n" + "="*70)
print("RECOMMENDED DOMAIN EXTRACTION")
print("="*70)

# Conservative approach: skip first 60 residues (signal + TM)
start = 60
end = len(sequence)  # Use full length, or trim C-terminus if needed

catalytic_domain = sequence[start:end]

print(f"\nCatalytic domain: residues {start+1}-{end}")
print(f"Length: {len(catalytic_domain)} aa")
print(f"\nFirst 60 residues of catalytic domain:")
print(catalytic_domain[:60])
print(f"\nLast 60 residues of catalytic domain:")
print(catalytic_domain[-60:])

# Save catalytic domain
print("\n" + "="*70)
print("Saving catalytic domain sequence")
print("="*70)

with open('../structures/sequence.fasta', 'w') as f:
    f.write(">Gluconobacter_oxydans_Q5FPE8_catalytic_domain\n")
    # Write in 60 character lines
    for i in range(0, len(catalytic_domain), 60):
        f.write(catalytic_domain[i:i+60] + "\n")

print(f"\n✓ Saved to: structures/sequence.fasta")
print(f"  Length: {len(catalytic_domain)} aa")
print(f"  Residues: {start+1}-{end}")

# Alternative: More aggressive trimming
print("\n" + "="*70)
print("ALTERNATIVE: More aggressive trimming")
print("="*70)

start_alt = 80
end_alt = 1200  # Trim potential C-terminal extensions

if end_alt > len(sequence):
    end_alt = len(sequence)

catalytic_alt = sequence[start_alt:end_alt]

print(f"\nAlternative catalytic domain: residues {start_alt+1}-{end_alt}")
print(f"Length: {len(catalytic_alt)} aa")

with open('../structures/sequence_alt.fasta', 'w') as f:
    f.write(">Gluconobacter_oxydans_Q5FPE8_catalytic_domain_alt\n")
    for i in range(0, len(catalytic_alt), 60):
        f.write(catalytic_alt[i:i+60] + "\n")

print(f"\n✓ Saved to: structures/sequence_alt.fasta")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nUse: structures/sequence.fasta (residues 61-end)")
print("  - Conservative approach")
print("  - Removes signal peptide + TM domain")
print("  - Keeps full catalytic domain")
print("\nIf AlphaFold gives low pLDDT at termini:")
print("  - Try: structures/sequence_alt.fasta (residues 81-1200)")
print("  - More aggressive trimming")
print("\n" + "="*70)
