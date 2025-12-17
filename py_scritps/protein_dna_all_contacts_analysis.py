# ============================================================
# Script Name   : protein_dna_all_contacts_analysis.py
#
# Purpose       :
#   Comprehensive analysis of all protein–DNA contacts
#   from molecular dynamics simulations.
# Author        : Boobalan Duraisamy, Debabrata Pramanik (corresponding author)
# Affiliation   : SRM University-AP, Andhra pradesh
#
#
# Software Used :
#   - MDAnalysis
#   - ProLIF
#   - NumPy, Pandas
#   - Matplotlib, Seaborn
#
# Input Files   :
#   - md_f.tpr     : GROMACS topology with explicit hydrogens
#   - md__fit.xtc  : PBC-corrected, fitted trajectory
#
# Output Files  :
#   - AllRes_DNA_Contact_Frequency_AthenB_AllResidues.png
#   - AllRes_DNA_Contact_Fingerprint_AthenB_AllResidues.png
#
# Analysis Type :
#   - Frame-wise protein–DNA interaction detection
#   - Contact occurrence frequency (% frames)
#
# Interaction Definitions (ProLIF defaults) :
# Analysis Window :
#   - Trajectory frames analysed: 40000–50000
#
# Notes :
#   - Ionic, metal-mediated, and halogen-bond interactions
#     were excluded from this analysis.
#   - No user-defined distance cutoffs were applied.
#   - Default ProLIF interaction definitions were used
#     to ensure reproducibility.
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
u = mda.Universe("md_f.tpr", "md__fit.xtc")

# -------------------------------------------------
# Select protein and DNA
# -------------------------------------------------
protein = u.select_atoms("protein")
dna = u.select_atoms("nucleic")

print(f"Protein residues: {len(protein.residues)} | DNA residues: {len(dna.residues)}")

# -------------------------------------------------
# Use all possible interaction types as CONTACTS
# -------------------------------------------------
available = plf.Fingerprint.list_available()
selected_interactions = [i for i in available if i not in ["MetalAcceptor", "XBond", "Ionic"]]

fp = plf.Fingerprint(interactions=selected_interactions)

# -------------------------------------------------
# Run ProLIF contact fingerprint
# -------------------------------------------------
fp.run(u.trajectory[40000:50000:1], protein, dna)

# Convert to binary DataFrame
df = fp.to_dataframe().astype(int)

# FIX Pandas FutureWarning
df = df.T.groupby(level=[0, 1]).max().T

# Keep only columns that ever form contact
df.columns = [f"{lig}-{prot}" for lig, prot in df.columns]
df = df.loc[:, df.sum() > 0]

# -------------------------------------------------
# Residue renumbering
# -------------------------------------------------
map_a = {i: 177 + i for i in range(1, 60)}
map_b = {i: 177 + (i - 59) for i in range(60, 119)}
res_map = {**map_a, **map_b}

def clean_res_label(label):
    """Clean protein label like ARG12 -> AARG178 or BARG178."""
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
# Apply cleaned names
# -------------------------------------------------
new_cols = []
for c in df.columns:
    parts = c.split("-")
    prot = clean_res_label(parts[0])
    dna_label = clean_dna_label(parts[1])
    new_cols.append(f"{prot}-{dna_label}")

df.columns = new_cols

# -------------------------------------------------
# Contact frequency (%)
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
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 9999

def extract_chain(label):
    m = re.match(r"([AB])", label)
    return m.group(1) if m else "Z"

def extract_dna_num(label):
    m = re.search(r"[ACGT](\d+)", label)
    return int(m.group(1)) if m else 9999

# -------------------------------------------------
# Sort rows & columns
# -------------------------------------------------
interaction_freq = interaction_freq.sort_values(
    by=["PROT", "DNA"],
    key=lambda col: col.map(
        lambda x: (extract_chain(x), extract_resnum(x))
        if col.name == "PROT" else extract_dna_num(x)
    )
)

pivot = interaction_freq.pivot_table(
    index="PROT", columns="DNA", values="Frequency (%)",
    aggfunc="mean", fill_value=0
)

# -------------------------------------------------
# Add missing residues (zero-contact)
# -------------------------------------------------
missing_residues = [179, 183, 187, 190]

for chain in ["A", "B"]:
    for resnum in missing_residues:
        label = f"{chain}RES{resnum}"
        if label not in pivot.index:
            pivot.loc[label] = 0

# Re-sort
a_chain = sorted([x for x in pivot.index if x.startswith("A")], key=extract_resnum)
b_chain = sorted([x for x in pivot.index if x.startswith("B")], key=extract_resnum)

pivot = pivot.reindex(a_chain + b_chain)
pivot = pivot[sorted(pivot.columns, key=extract_dna_num)]

# =================================================
# PLOT 1 — ULTRA-HD CONTACT HEATMAP (DPI 1200)
# =================================================
plt.figure(figsize=(14, 12))

sns.heatmap(
    pivot,
    cmap="YlGnBu",
    annot=True,
    fmt=".1f",
    linewidths=0.6,
    annot_kws={"size": 10},
    cbar_kws={"label": "Contact Occurrence (%)", "shrink": 0.8}
)

plt.title("Protein–DNA Contact Frequency (A→B, all residues)", fontsize=26)
plt.xlabel("DNA Residue", fontsize=22)
plt.ylabel("Protein Residue (A then B)", fontsize=22)

plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()
plt.savefig("AllRes_DNA_Contact_Frequency_AthenB_AllResidues.png", dpi=1200, bbox_inches="tight")
plt.close()
print(" Saved: AllRes_DNA_Contact_Frequency_AthenB_AllResidues.png")

# =================================================
# PLOT 2 — ULTRA-HD CONTACT FINGERPRINT (DPI 1200)
# =================================================
df_t = df.T.copy()
df_t = df_t[~df_t.index.duplicated(keep='first')]

# Add missing rows
for chain in ["A", "B"]:
    for resnum in missing_residues:
        label = f"{chain}RES{resnum}"
        if label not in df_t.index:
            df_t.loc[label] = 0

a_chain = sorted([x for x in df_t.index if x.startswith("A")], key=extract_resnum)
b_chain = sorted([x for x in df_t.index if x.startswith("B")], key=extract_resnum)
df_t = df_t.reindex(a_chain + b_chain)

matrix = df_t.values

plt.figure(figsize=(16, 14))
plt.imshow(matrix, aspect='auto', cmap='Greys', interpolation='nearest')

plt.yticks(range(len(df_t.index)), df_t.index, fontsize=8)
plt.xticks(
    np.linspace(0, len(df_t.columns), 12).astype(int),
    fontsize=22
)

plt.xlabel("Trajectory Frames", fontsize=22)
plt.ylabel("Protein–DNA Pairs (A→B)", fontsize=22)
plt.title("Protein–DNA Contact Fingerprint", fontsize=26)

plt.tight_layout()
plt.savefig("AllRes_DNA_Contact_Fingerprint_AthenB_AllResidues.png", dpi=1200, bbox_inches="tight")
plt.close()
print(" Saved: AllRes_DNA_Contact_Fingerprint_AthenB_AllResidues.png")

