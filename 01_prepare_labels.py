"""Prepare FDA DILIrank 2.0 and DILIst label files."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)

def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_concern(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower().replace("^v", "").replace("vmost", "most")
    text = text.replace("vless", "less").replace("vno", "no")
    if "ambiguous" in text:
        return "ambiguous"
    if "most" in text:
        return "most"
    if "less" in text:
        return "less"
    if "no-dili" in text or text.startswith("no"):
        return "no"
    return "unknown"

def main() -> None:
    rank = pd.read_excel(RAW / "DILIrank_2_0.xlsx", sheet_name="version 2", skiprows=1)
    rank.columns = [str(c).strip() for c in rank.columns]
    rank = rank.rename(columns={
        "CompoundName": "compound_name",
        "vDILI-Concern": "dilirank_concern",
        "SeverityClass": "severity_class",
        "LabelSection": "label_section",
        "Comment": "comment",
        "LTKBID": "ltkb_id",
    })
    needed = ["ltkb_id", "compound_name", "severity_class", "label_section",
              "dilirank_concern", "comment"]
    rank = rank[[c for c in needed if c in rank.columns]].copy()
    rank["compound_name_normalized"] = rank["compound_name"].map(normalize_name)
    rank["concern_group"] = rank["dilirank_concern"].map(normalize_concern)
    rank["primary_label"] = rank["concern_group"].map({"most": 1, "no": 0})
    rank["broad_label"] = rank["concern_group"].map({"most": 1, "less": 1, "no": 0})
    rank["is_new"] = rank["comment"].astype(str).str.strip().str.lower().eq("new")
    rank.to_csv(OUT / "dilirank_labels.csv", index=False)

    dilist = pd.read_excel(RAW / "DILIst.xlsx", sheet_name="DILIst")
    dilist.columns = [str(c).strip() for c in dilist.columns]
    dilist = dilist.rename(columns={
        "DILIST_ID": "dilist_id",
        "CompoundName": "compound_name",
        "DILIst Classification": "dilist_label",
        "Routs of Administration": "route",
    })
    # Handle trailing spaces or slight source-header differences.
    for col in list(dilist.columns):
        if col.lower().startswith("dilist classification"):
            dilist = dilist.rename(columns={col: "dilist_label"})
        if col.lower().startswith("routs of administration"):
            dilist = dilist.rename(columns={col: "route"})
    dilist["compound_name_normalized"] = dilist["compound_name"].map(normalize_name)
    dilist["dilist_label"] = pd.to_numeric(dilist["dilist_label"], errors="coerce")
    dilist.to_csv(OUT / "dilist_labels.csv", index=False)

    print("DILIrank counts:")
    print(rank["concern_group"].value_counts(dropna=False))
    print("\nPrimary endpoint:", rank["primary_label"].notna().sum(), "drugs before structure filtering")
    print("DILIst labels:", dilist["dilist_label"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()
