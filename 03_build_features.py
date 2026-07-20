"""Standardize molecules and generate descriptors, Morgan fingerprints, and scaffolds."""
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

def standardize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    try:
        mol = rdMolStandardize.Cleanup(mol)
        fragments = rdMolStandardize.LargestFragmentChooser().choose(mol)
        uncharger = rdMolStandardize.Uncharger()
        parent = uncharger.uncharge(fragments)
        Chem.SanitizeMol(parent)
        return parent, None
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
    data = labels.merge(
        structures[["ltkb_id", "status", "cid", "canonical_smiles", "inchikey"]],
        on="ltkb_id", how="left"
    )

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    metadata, desc_rows, fp_rows, exclusions = [], [], [], []

    for _, row in data.iterrows():
        if row.get("status") != "matched" or pd.isna(row.get("canonical_smiles")):
            exclusions.append({**row.to_dict(), "exclusion_reason": "no_pubchem_structure"})
            continue
        mol, error = standardize(str(row["canonical_smiles"]))
        if mol is None:
            exclusions.append({**row.to_dict(), "exclusion_reason": error})
            continue
        std_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        metadata.append({
            "ltkb_id": row["ltkb_id"],
            "compound_name": row["compound_name"],
            "concern_group": row["concern_group"],
            "primary_label": row["primary_label"],
            "broad_label": row["broad_label"],
            "is_new": bool(row["is_new"]),
            "pubchem_cid": row.get("cid"),
            "standardized_smiles": std_smiles,
            "murcko_scaffold": scaffold,
        })
        desc_rows.append(descriptors(mol))
        fp_rows.append(fp)

    meta = pd.DataFrame(metadata)
    desc = pd.DataFrame(desc_rows)
    fps = np.vstack(fp_rows) if fp_rows else np.empty((0, 2048), dtype=np.uint8)

    # Duplicate parent structures are retained in an audit file, then one representative is kept.
    duplicates = meta[meta.duplicated("standardized_smiles", keep=False)].copy()
    duplicates.to_csv(PROCESSED / "duplicate_structures_audit.csv", index=False)
    keep_mask = ~meta.duplicated("standardized_smiles", keep="first")
    meta = meta.loc[keep_mask].reset_index(drop=True)
    desc = desc.loc[keep_mask.values].reset_index(drop=True)
    fps = fps[keep_mask.values]

    meta.to_csv(PROCESSED / "molecule_metadata.csv", index=False)
    desc.to_csv(PROCESSED / "molecular_descriptors.csv", index=False)
    np.savez_compressed(PROCESSED / "morgan_fingerprints.npz", X=fps)
    pd.DataFrame(exclusions).to_csv(PROCESSED / "structure_exclusions.csv", index=False)
    (PROCESSED / "feature_manifest.json").write_text(json.dumps({
        "descriptor_names": DESCRIPTOR_NAMES,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "n_molecules": len(meta),
    }, indent=2))
    print(f"Usable unique molecules: {len(meta)}")
    print(f"Excluded before deduplication: {len(exclusions)}")
    print(f"Duplicate parent structures logged: {len(duplicates)}")

if __name__ == "__main__":
    main()
