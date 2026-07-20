"""Fetch chemical structures from PubChem with caching and an audit log."""
from pathlib import Path
import json
import time
import urllib.parse
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "data" / "interim" / "dilirank_labels.csv"
OUT = ROOT / "data" / "interim"
CACHE = OUT / "pubchem_cache"
CACHE.mkdir(parents=True, exist_ok=True)

PROPERTY_FIELDS = "ConnectivitySMILES,SMILES,InChIKey,MolecularFormula,MolecularWeight"

def query_pubchem(name: str, session: requests.Session, retries: int = 4) -> dict:
    cache_file = CACHE / f"{urllib.parse.quote(name, safe='')}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    encoded = urllib.parse.quote(name, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/"
        f"property/{PROPERTY_FIELDS}/JSON"
    )
    result = {"query_name": name, "status": "failed", "error": None}
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=45)
            if response.status_code == 200:
                props = response.json()["PropertyTable"]["Properties"][0]
                result.update({
                    "status": "matched",
                    "cid": props.get("CID"),
                    "canonical_smiles": props.get("ConnectivitySMILES") or props.get("CanonicalSMILES"),
                    "isomeric_smiles": props.get("SMILES") or props.get("IsomericSMILES"),
                    "inchikey": props.get("InChIKey"),
                    "molecular_formula_pubchem": props.get("MolecularFormula"),
                    "molecular_weight_pubchem": props.get("MolecularWeight"),
                })
                break
            if response.status_code == 404:
                result["error"] = "not_found"
                break
            result["error"] = f"http_{response.status_code}"
        except requests.RequestException as exc:
            result["error"] = type(exc).__name__
        time.sleep(2 ** attempt)

    cache_file.write_text(json.dumps(result, indent=2))
    time.sleep(0.22)
    return result

def main() -> None:
    labels = pd.read_csv(INFILE)
    session = requests.Session()
    session.headers.update({"User-Agent": "DILI-molecular-research/1.0 (academic project)"})
    records = []
    total = len(labels)
    for i, row in labels.iterrows():
        name = str(row["compound_name"])
        record = query_pubchem(name, session)
        record["ltkb_id"] = row.get("ltkb_id")
        record["compound_name"] = name
        records.append(record)
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"Processed {i + 1}/{total}")
    structures = pd.DataFrame(records)
    structures.to_csv(OUT / "pubchem_structures.csv", index=False)
    structures[structures["status"] != "matched"].to_csv(
        OUT / "pubchem_unresolved_audit.csv", index=False
    )
    print(structures["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
