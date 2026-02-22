# PyMOL script for PQQ+Ca transplant
# Load structures
load ../structures/receptor_af.pdb, receptor
load ../structures/1WPQ.pdb, homolog

# Align structures
align receptor, homolog

# Extract PQQ and Ca from homolog
select pqq_ca, homolog and (resn PQQ or resn CA or name CA)

# Create combined structure
create combined, receptor
create ligands, pqq_ca

# Get transformation matrix and apply to ligands
# This is done automatically by PyMOL during align

# Combine receptor with ligands
copy_to combined, ligands

# Save result
save ../structures/receptor_PQQ_Ca.pdb, combined

# Print info
print "Transplant complete"
print "Saved to: structures/receptor_PQQ_Ca.pdb"

quit
