import json

results = json.load(open("ABC_analysis_results.json"))

print("="*80)
print("SUMMARY TABLE: MD-derived Features for ML")
print("="*80)

print("\n### A. Binding Stability")
print("-"*80)
print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
print("-"*80)
A = results["A_binding_stability"]
print(f"{'Ligand RMSD (mean)':<30} {A['LSO']['ligand_rmsd_mean']:>15.2f} {A['DSO']['ligand_rmsd_mean']:>15.2f} {'A':>10}")
print(f"{'Ligand RMSD (std)':<30} {A['LSO']['ligand_rmsd_std']:>15.2f} {A['DSO']['ligand_rmsd_std']:>15.2f} {'A':>10}")
print(f"{'Active-site RMSD (mean)':<30} {A['LSO']['active_site_rmsd_mean']:>15.2f} {A['DSO']['active_site_rmsd_mean']:>15.2f} {'A':>10}")
print(f"{'Residence time':<30} {A['LSO']['residence_time_percent']:>15.1f} {A['DSO']['residence_time_percent']:>15.1f} {'%':>10}")
print(f"{'Contact persistence':<30} {A['LSO']['contact_persistence_mean']:>15.1f} {A['DSO']['contact_persistence_mean']:>15.1f} {'residues':>10}")

print("\n### B. Reactive Geometry")
print("-"*80)
print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
print("-"*80)
B = results["B_reactive_geometry"]
print(f"{'Reactive distance (mean)':<30} {B['LSO']['reactive_distance_mean_A']:>15.2f} {B['DSO']['reactive_distance_mean_A']:>15.2f} {'A':>10}")
print(f"{'NAC dist freq (<3.5A)':<30} {B['LSO']['nac_distance_freq_pct']:>15.1f} {B['DSO']['nac_distance_freq_pct']:>15.1f} {'%':>10}")
print(f"{'NAC simultaneous freq':<30} {B['LSO']['nac_simultaneous_freq_pct']:>15.1f} {B['DSO']['nac_simultaneous_freq_pct']:>15.1f} {'%':>10}")
print(f"{'NAC lifetime (mean)':<30} {B['LSO']['nac_lifetime_mean_frames']:>15.1f} {B['DSO']['nac_lifetime_mean_frames']:>15.1f} {'frames':>10}")

print("\n### C. H-bond Analysis")
print("-"*80)
print(f"{'Feature':<30} {'LSO':>15} {'DSO':>15} {'Unit':>10}")
print("-"*80)
C = results["C_hbond_analysis"]
print(f"{'Unique H-bond pairs':<30} {C['LSO']['unique_hbond_pairs']:>15} {C['DSO']['unique_hbond_pairs']:>15} {'pairs':>10}")
print(f"{'Mean occupancy':<30} {C['LSO']['mean_occupancy_percent']:>15.1f} {C['DSO']['mean_occupancy_percent']:>15.1f} {'%':>10}")
print(f"{'Max occupancy':<30} {C['LSO']['max_occupancy_percent']:>15.1f} {C['DSO']['max_occupancy_percent']:>15.1f} {'%':>10}")
print(f"{'Mean network size':<30} {C['LSO']['mean_network_size']:>15.1f} {C['DSO']['mean_network_size']:>15.1f} {'H-bonds':>10}")
print("="*80)
