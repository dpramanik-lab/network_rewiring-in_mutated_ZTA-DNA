# ============================================================
# Script Name   : protein_dna_hbond_analysis.py
# Purpose       : Quantification of protein–DNA hydrogen bonds
#                 from molecular dynamics simulations
#
# Author        : Boobalan Duraisamy, Debabrata Pramanik (corresponding author)
# Affiliation   : SRM University-AP, Andhra pradesh
#
# Software Used :
#   - MDAnalysis
#   - ProLIF
#   - NumPy, Pandas
#   - Matplotlib, Seaborn
#
# Input Files   :
#   - md_f.tpr        : GROMACS topology with explicit hydrogens
#   - md__fit.xtc     : PBC-corrected, fitted trajectory
#
# Output Files  :
#   - AllRes_DNA_Hbond_Frequency_AthenB_AllResidues.png
#   - AllRes_DNA_Hbond_Fingerprint_AthenB_AllResidues.png
#
# Analysis Type :
#   - Frame-wise hydrogen bond detection
#   - Occurrence frequency calculation (% frames)
#
# Notes :
#   - Hydrogen bonds are defined using ProLIF default criteria
#
# ============================================================
import MDAnalysis as mda
import prolif as plf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------------------------------
# Load trajectory
# -------------------------------------------------
u = mda.Universe("md_f.tpr", "md_fit.xtc")

# -------------------------------------------------
# Select protein and DNA
# -------------------------------------------------
protein = u.select_atoms("protein")
dna = u.select_atoms("nucleic")

print(f"Protein residues: {len(protein.residues)} | DNA residues: {len(dna.residues)}")

# -------------------------------------------------
# Choose correct hydrogen-bond definition
# -------------------------------------------------
available = plf.Fingerprint.list_available()
if "HydrogenBond" in available:
    fp = plf.Fingerprint(interactions=["HydrogenBond"])
else:
    fp = plf.Fingerprint(interactions=["HBAcceptor", "HBDonor"])

# -------------------------------------------------
# Run ProLIF hydrogen-bond fingerprint
# -------------------------------------------------
fp.run(u.trajectory[40000:50000:1], protein, dna)

df = fp.to_dataframe().astype(int)
df.columns = [f"{lig}-{prot}" for lig, prot, itype in df.columns]
df = df.loc[:, df.sum() > 0]  # keep only pairs that ever form H-bonds

# -------------------------------------------------
# Map residue numbering for monomer A/B
# -------------------------------------------------
map_a = {i: 177 + i for i in range(1, 60)}      # monomer A: 1–59 → 178–236
map_b = {i: 177 + (i - 59) for i in range(60, 119)}  # monomer B: 60–118 → 178–236
res_map = {**map_a, **map_b}

def clean_res_label(label):
    """Clean protein label like ARG12 → AARG179 or BARG179."""
    label = label.split(":")[0]
    m = re.search(r"([A-Z]+)(\d+)", label)
    if not m:
        return label
    resname, resid = m.groups()
    resid = int(resid)
    if resid in res_map:
        if resid <= 59:
            return f"A{resname}{res_map[resid]}"
        else:
            return f"B{resname}{res_map[resid]}"
    return f"{resname}{resid}"

def clean_dna_label(label):
    """Clean DNA label like DA121:Protein in water → A121."""
    label = label.split(":")[0]
    m = re.match(r"D([A-Z])(\d+)", label)
    if m:
        base, resid = m.groups()
        return f"{base}{resid}"
    return label

# -------------------------------------------------
# Apply clean labels
# -------------------------------------------------
new_cols = []
for c in df.columns:
    parts = c.split("-")
    prot_part = parts[0]
    dna_part = parts[1] if len(parts) > 1 else "DNA"
    prot_label = clean_res_label(prot_part)
    dna_label = clean_dna_label(dna_part)
    new_cols.append(f"{prot_label}-{dna_label}")
df.columns = new_cols

# -------------------------------------------------
# Compute H-bond occurrence frequency (% frames)
# -------------------------------------------------
interaction_freq = df.mean(axis=0) * 100
pairs = interaction_freq.index.str.split("-")
interaction_freq = pd.DataFrame({
    "PROT": [p[0] for p in pairs],
    "DNA": [p[1] for p in pairs],
    "Frequency (%)": interaction_freq.values
})

# -------------------------------------------------
# Sorting utilities
# -------------------------------------------------
def extract_resnum(label):
    """Extract residue number (works for AARG178, BGLU182, etc.)."""
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 9999

def extract_chain(label):
    """Return 'A' or 'B' for sorting chain first."""
    m = re.match(r"([AB])", label)
    return m.group(1) if m else "Z"

def extract_dna_num(label):
    """Extract DNA residue number (e.g., A121 → 121)."""
    m = re.search(r"[ACGT](\d+)", label)
    return int(m.group(1)) if m else 9999

# -------------------------------------------------
# Sort both protein and DNA residues
# -------------------------------------------------
interaction_freq = interaction_freq.sort_values(
    by=["PROT", "DNA"],
    key=lambda col: col.map(
        lambda x: (extract_chain(x), extract_resnum(x))
        if col.name == "PROT"
        else extract_dna_num(x)
    )
)

pivot = interaction_freq.pivot_table(
    index="PROT", columns="DNA", values="Frequency (%)", aggfunc="mean", fill_value=0
)

# -------------------------------------------------
# Add missing residues (even if zero)
# -------------------------------------------------
missing_residues = [179, 183, 187, 190]
for chain in ["A", "B"]:
    for resnum in missing_residues:
        label = f"{chain}RES{resnum}"  # dummy placeholder
        # If residue doesn't exist, add zero row
        if not any(p.startswith(f"{chain}") and str(resnum) in p for p in pivot.index):
            pivot.loc[f"{chain}RES{resnum}"] = 0

# Sort all again (A first, then B, by residue number)
a_chain = sorted([x for x in pivot.index if x.startswith("A")], key=extract_resnum)
b_chain = sorted([x for x in pivot.index if x.startswith("B")], key=extract_resnum)
pivot = pivot.reindex(a_chain + b_chain)
pivot = pivot[sorted(pivot.columns, key=extract_dna_num)]

# -------------------------------------------------
# Plot 1: Frequency Heatmap (A then B, includes zeros)
# -------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot,
    cmap="YlOrBr",
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    cbar_kws={"label": "H-bond Occurrence (%)"}
)
plt.title("Protein–DNA Hydrogen Bond Frequency (A→B, includes zero interactions)", fontsize=14)
plt.xlabel("DNA Residue", fontsize=12)
plt.ylabel("Protein Residue (A then B)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("AllRes_DNA_Hbond_Frequency_AthenB_AllResidues.png", dpi=600, bbox_inches="tight")
plt.close()
print("✅ Saved heatmap (A→B, includes zero residues) → AllRes_DNA_Hbond_Frequency_AthenB_AllResidues.png")

# -------------------------------------------------
# Plot 2: Barcode Fingerprint (A then B, includes zero residues)
# -------------------------------------------------
df_t = df.T.copy()
if df_t.index.duplicated().any():
    df_t = df_t[~df_t.index.duplicated(keep='first')]

# Add zero rows for missing residues to match same order
for chain in ["A", "B"]:
    for resnum in missing_residues:
        label = f"{chain}RES{resnum}"
        if label not in df_t.index:
            df_t.loc[label] = 0

# Sort by chain and residue number
a_chain = sorted([x for x in df_t.index if x.startswith("A")], key=extract_resnum)
b_chain = sorted([x for x in df_t.index if x.startswith("B")], key=extract_resnum)
df_t = df_t.reindex(a_chain + b_chain)

matrix = df_t.values

plt.figure(figsize=(14, 7))
plt.imshow(matrix, aspect='auto', cmap='Greys', interpolation='nearest')
plt.yticks(range(len(df_t.index)), df_t.index, fontsize=5)
plt.xticks(np.linspace(0, len(df_t.columns), 10).astype(int))
plt.xlabel("Trajectory Frames", fontsize=10)
plt.ylabel("Protein–DNA Pairs (A then B, all residues)", fontsize=10)
plt.title("Hydrogen Bond Fingerprint (– = bond present, includes zero-residues)", fontsize=12)
plt.tight_layout()
plt.savefig("AllRes_DNA_Hbond_Fingerprint_AthenB_AllResidues.png", dpi=600, bbox_inches="tight")
plt.close()
print("✅ Saved barcode (A→B, includes zero-residues) → AllRes_DNA_Hbond_Fingerprint_AthenB_AllResidues.png")
