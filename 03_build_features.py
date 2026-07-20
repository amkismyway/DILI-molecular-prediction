"""Corrected feature builder: preserves stereochemistry and handles duplicate structures safely."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds",
    "RingCount", "AromaticRingCount", "FractionCSP3", "HeavyAtomCount",
    "HeteroatomCount", "FormalCharge", "MolMR"
]

def standardize_preserving_stereo(smiles: str):
    """Standardize parent structure without deliberately removing stereochemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol, None
    except Exception as exc:
        return None, type(exc).__name__

def descriptors(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRingCount": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "HeteroatomCount": Lipinski.NumHeteroatoms(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "MolMR": Crippen.MolMR(mol),
    }

def main() -> None:
    labels = pd.read_csv(INTERIM / "dilirank_labels.csv")
    structures = pd.read_csv(INTERIM / "pubchem_structures.csv")

    # Use PubChem's isomeric SMILES first; fall back only if unavailable.
    structure_cols = [
        "ltkb_id", "status", "cid", "canonical_smiles",
        "isomeric_smiles", "inchikey"
    ]
    for col in structure_cols:
        if col not in structures.columns:
            structures[col] = np.nan

    data = labels.merge(structures[structure_cols], on="ltkb_id", how="left")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )

    records, desc_rows, fp_rows, exclusions = [], [], [], []

    for _, row in data.iterrows():
        if row.get("status") != "matched":
            exclusions.append({**row.to_dict(), "exclusion_reason": "no_pubchem_structure"})
            continue

        source_smiles = row.get("isomeric_smiles")
        if pd.isna(source_smiles) or not str(source_smiles).strip():
            source_smiles = row.get("canonical_smiles")

        if pd.isna(source_smiles) or not str(source_smiles).strip():
            exclusions.append({**row.to_dict(), "exclusion_reason": "missing_smiles"})
            continue

        mol, error = standardize_preserving_stereo(str(source_smiles))
        if mol is None:
            exclusions.append({**row.to_dict(), "exclusion_reason": error})
            continue

        stereo_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        nonstereo_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        )
        full_inchikey = Chem.MolToInchiKey(mol)
        connectivity_key = full_inchikey.split("-")[0] if full_inchikey else ""

        # Scaffold is intentionally connectivity-based; chirality is preserved in features.
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.uint8)

        records.append({
            "ltkb_id": row["ltkb_id"],
            "compound_name": row["compound_name"],
            "concern_group": row["concern_group"],
            "primary_label": row["primary_label"],
            "broad_label": row["broad_label"],
            "is_new": bool(row["is_new"]),
            "pubchem_cid": row.get("cid"),
            "standardized_isomeric_smiles": stereo_smiles,
            "standardized_nonisomeric_smiles": nonstereo_smiles,
            "standardized_inchikey": full_inchikey,
            "connectivity_key": connectivity_key,
            "murcko_scaffold": scaffold,
        })
        desc_rows.append(descriptors(mol))
        fp_rows.append(fp)

    meta = pd.DataFrame(records)
    desc = pd.DataFrame(desc_rows)
    fps = np.vstack(fp_rows) if fp_rows else np.empty((0, 2048), dtype=np.uint8)

    # Audit exact stereochemistry-preserving duplicates.
    duplicate_mask = meta.duplicated("standardized_inchikey", keep=False)
    duplicate_audit = meta.loc[duplicate_mask].copy()
    duplicate_audit["duplicate_group_size"] = (
        duplicate_audit.groupby("standardized_inchikey")["ltkb_id"].transform("size")
    )
    duplicate_audit["primary_label_nunique"] = (
        duplicate_audit.groupby("standardized_inchikey")["primary_label"]
        .transform(lambda s: s.dropna().nunique())
    )
    duplicate_audit["conflicting_primary_labels"] = (
        duplicate_audit["primary_label_nunique"] > 1
    )
    duplicate_audit.to_csv(
        PROCESSED / "duplicate_structures_stereo_preserved_audit.csv", index=False
    )

    # Exact duplicate structures with conflicting primary labels are excluded rather
    # than resolved arbitrarily. Same-label duplicates retain one representative.
    conflict_keys = set(
        duplicate_audit.loc[
            duplicate_audit["conflicting_primary_labels"], "standardized_inchikey"
        ].dropna()
    )
    conflict_mask = meta["standardized_inchikey"].isin(conflict_keys)

    if conflict_mask.any():
        conflict_rows = meta.loc[conflict_mask].copy()
        conflict_rows["exclusion_reason"] = "exact_structure_conflicting_primary_labels"
        exclusions.extend(conflict_rows.to_dict("records"))

    keep_mask = ~conflict_mask
    surviving = meta.loc[keep_mask].copy()
    same_structure_duplicate = surviving.duplicated(
        "standardized_inchikey", keep="first"
    )
    final_keep = keep_mask.copy()
    final_keep.loc[keep_mask] = ~same_structure_duplicate.values

    meta = meta.loc[final_keep].reset_index(drop=True)
    desc = desc.loc[final_keep.values].reset_index(drop=True)
    fps = fps[final_keep.values]

    # Separate audit showing stereoisomers that share connectivity but remain distinct.
    stereo_family_mask = meta.duplicated("connectivity_key", keep=False)
    stereo_families = meta.loc[stereo_family_mask].sort_values(
        ["connectivity_key", "compound_name"]
    )
    stereo_families.to_csv(
        PROCESSED / "stereoisomer_family_audit.csv", index=False
    )

    meta.to_csv(PROCESSED / "molecule_metadata.csv", index=False)
    desc.to_csv(PROCESSED / "molecular_descriptors.csv", index=False)
    np.savez_compressed(PROCESSED / "morgan_fingerprints.npz", X=fps)
    pd.DataFrame(exclusions).to_csv(
        PROCESSED / "structure_exclusions.csv", index=False
    )
    (PROCESSED / "feature_manifest.json").write_text(json.dumps({
        "descriptor_names": DESCRIPTOR_NAMES,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "morgan_include_chirality": True,
        "structure_identity": "full standardized InChIKey",
        "stereochemistry_preserved": True,
        "conflicting_exact_duplicates_excluded": True,
        "n_molecules": len(meta),
    }, indent=2))

    print(f"Usable unique stereochemistry-preserved molecules: {len(meta)}")
    print(f"Structure-processing exclusions: {len(exclusions)}")
    print(f"Exact duplicate rows audited: {len(duplicate_audit)}")
    print(f"Conflicting exact-structure groups excluded: {len(conflict_keys)}")
    print(f"Connectivity-sharing stereoisomer-family rows retained: {len(stereo_families)}")

if __name__ == "__main__":
    main()
