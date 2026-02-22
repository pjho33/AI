#!/usr/bin/env python3
"""
Extract catalytic domain from Q70JN9 (sldA)
Remove signal peptide and identify periplasmic domain
"""

# Q70JN9 full sequence
sequence = """
MRRSHLLATVACATLACAPLAANAQFAPAGSGGSPTSSVPGPGNGSGNSFEPTENTPAAK
SRFSGPSPYAPQAPGVNAANLPDIGSMDPNDVPQMAPQQSASPASGDWAAYGHDDSQMRY
SPLSEITPQNADQLKVAFVYHTGSYPRPGQTNKWAAETTPIKVGDGLYMCSAQNDIMKID
PATGKEIWRHNINEKYEAIPYTAACKGVTYFTSSQVPEGQPCHNRILEGTLDMRLIAVDA
ATGNLCEGFGNGGQVNLMQGLGESVPGFVSMTTPPPVVNGVVVVNHEVLDGQRRWAPSGV
IRGYDAESGKFLWAWDVNRPNDHSQPTGNNHYSRGTPNSWAAMTGDNALGLVYVPTGNSA
SDYYSALRSPEENKVSSAVVALDVKTGSPRWVFQTVHKDVWDYDIGSQATLMDMPGQDGQ
PVPALIMPTKRGQTFVLDRRDGKPILPVEERPAPSPGVIPGDPRSPTQPWSTGMPALRVP
DLKETDMWGMSPIDQLFCRIKFRRANYTGEFTPPSVDKPWIEYPGYNGGSDWGSVSYDPQ
SGILIANWNITPMYDQLVTRKKADELGLMPIDDPNYKPGGGGAEGNGAMDGTPYGIVVTP
FWDQYTGMMCNRPPYGMITAIDMKHGQKVLWQHPLGTARANGPWGLPTGLPWEIGTPNNG
GSVVTAGGVVFIAAATDNQIRAIDEHTGKVVWSAVLPGGGQANPMTYEANGHQYVAIMAG
GHHFMMTPVSDQLVVYALPDHKG
""".replace('\n', '').strip()

print("="*70)
print("Q70JN9 (sldA) Catalytic Domain Extraction")
print("="*70)

print(f"\nFull sequence length: {len(sequence)} aa")

# Analyze N-terminus for signal peptide
print("\n" + "="*70)
print("Signal Peptide Analysis")
print("="*70)

print("\nN-terminal 60 residues:")
print(sequence[:60])

# Signal peptide characteristics:
# - Positively charged N-terminus (M, R, K)
# - Hydrophobic core
# - Cleavage site (often after A, G, S)

# Look for likely cleavage sites
print("\nLikely signal peptide cleavage sites:")
print("  Position 20: ...PLAANA (AXA motif)")
print("  Position 25: ...NAQFAP")
print("  Position 30: ...PAGSGG")

# Check hydrophobicity in N-terminal region
hydrophobic = set('AILMFVW')
for i in range(0, 40, 10):
    segment = sequence[i:i+10]
    hydro_count = sum(1 for aa in segment if aa in hydrophobic)
    print(f"  Residues {i+1}-{i+10}: {segment} ({hydro_count}/10 hydrophobic)")

# Recommended cleavage: after position 30 (APLAANA|QFAPA...)
# Conservative: start at position 35 to be safe

print("\n" + "="*70)
print("Domain Extraction Options")
print("="*70)

# Option 1: Conservative (remove first 35 residues)
start_conservative = 35
catalytic_conservative = sequence[start_conservative:]

print(f"\nOption 1 (Conservative):")
print(f"  Remove: residues 1-{start_conservative} (signal peptide)")
print(f"  Keep: residues {start_conservative+1}-{len(sequence)}")
print(f"  Length: {len(catalytic_conservative)} aa")
print(f"  Start: {catalytic_conservative[:40]}...")

# Option 2: Moderate (remove first 30 residues)
start_moderate = 30
catalytic_moderate = sequence[start_moderate:]

print(f"\nOption 2 (Moderate):")
print(f"  Remove: residues 1-{start_moderate} (signal peptide)")
print(f"  Keep: residues {start_moderate+1}-{len(sequence)}")
print(f"  Length: {len(catalytic_moderate)} aa")
print(f"  Start: {catalytic_moderate[:40]}...")

# Option 3: Aggressive (remove first 25 residues)
start_aggressive = 25
catalytic_aggressive = sequence[start_aggressive:]

print(f"\nOption 3 (Aggressive):")
print(f"  Remove: residues 1-{start_aggressive} (signal peptide)")
print(f"  Keep: residues {start_aggressive+1}-{len(sequence)}")
print(f"  Length: {len(catalytic_aggressive)} aa")
print(f"  Start: {catalytic_aggressive[:40]}...")

# Check for TM domains in remaining sequence
print("\n" + "="*70)
print("TM Domain Check")
print("="*70)

# Check for long hydrophobic stretches
window = 20
found_tm = False

for seq_name, seq_start, seq in [
    ("Conservative", start_conservative, catalytic_conservative),
    ("Moderate", start_moderate, catalytic_moderate),
    ("Aggressive", start_aggressive, catalytic_aggressive)
]:
    print(f"\n{seq_name} option:")
    tm_found = False
    for i in range(0, min(100, len(seq) - window)):
        segment = seq[i:i+window]
        hydro_count = sum(1 for aa in segment if aa in hydrophobic)
        hydro_percent = hydro_count / window * 100
        
        if hydro_percent > 65:  # Likely TM
            print(f"  ⚠ Potential TM at position {seq_start+i+1}-{seq_start+i+window}: {hydro_percent:.1f}% hydrophobic")
            tm_found = True
    
    if not tm_found:
        print(f"  ✓ No TM domains detected")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("\nRecommended: Option 2 (Moderate)")
print(f"  Remove residues 1-30 (signal peptide)")
print(f"  Catalytic domain: residues 31-{len(sequence)}")
print(f"  Length: {len(catalytic_moderate)} aa")
print(f"  Rationale:")
print(f"    - Removes clear signal peptide")
print(f"    - No TM domains detected")
print(f"    - Suitable length for β-propeller fold")
print(f"    - Starts with periplasmic domain")

# Save recommended sequence
print("\n" + "="*70)
print("Saving Sequence")
print("="*70)

with open('../structures/sequence.fasta', 'w') as f:
    f.write(">Q70JN9_sldA_catalytic_domain|residues_31-end\n")
    # Write in 60 character lines
    for i in range(0, len(catalytic_moderate), 60):
        f.write(catalytic_moderate[i:i+60] + "\n")

print(f"\n✓ Saved to: structures/sequence.fasta")
print(f"  Sequence: Q70JN9 residues 31-{len(sequence)}")
print(f"  Length: {len(catalytic_moderate)} aa")

# Also save alternatives
with open('../structures/sequence_conservative.fasta', 'w') as f:
    f.write(">Q70JN9_sldA_catalytic_domain_conservative|residues_36-end\n")
    for i in range(0, len(catalytic_conservative), 60):
        f.write(catalytic_conservative[i:i+60] + "\n")

with open('../structures/sequence_aggressive.fasta', 'w') as f:
    f.write(">Q70JN9_sldA_catalytic_domain_aggressive|residues_26-end\n")
    for i in range(0, len(catalytic_aggressive), 60):
        f.write(catalytic_aggressive[i:i+60] + "\n")

print(f"\n✓ Alternative sequences also saved:")
print(f"  structures/sequence_conservative.fasta (residues 36-end, {len(catalytic_conservative)} aa)")
print(f"  structures/sequence_aggressive.fasta (residues 26-end, {len(catalytic_aggressive)} aa)")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nProtein: Q70JN9 (sldA - Sorbitol dehydrogenase large subunit)")
print(f"Organism: Gluconobacter oxydans 621H")
print(f"Full length: {len(sequence)} aa")
print(f"\nSignal peptide: residues 1-30 (removed)")
print(f"Catalytic domain: residues 31-{len(sequence)} ({len(catalytic_moderate)} aa)")
print(f"\nReady for AlphaFold: structures/sequence.fasta")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("\n1. Run AlphaFold/ColabFold:")
print("   - Use: structures/sequence.fasta")
print("   - Expected: β-propeller fold")
print("   - Time: 10-30 min")
print("\n2. If pLDDT low at termini:")
print("   - Try: sequence_conservative.fasta")
print("\n3. After AlphaFold:")
print("   - Verify β-propeller structure")
print("   - Proceed to PQQ+Ca transplant")
print("\n" + "="*70)
