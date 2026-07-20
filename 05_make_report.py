"""Create descriptive summary tables after feature generation/modeling."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

def main():
    meta = pd.read_csv(DATA / "molecule_metadata.csv")
    desc = pd.read_csv(DATA / "molecular_descriptors.csv")
    joined = pd.concat([meta.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)

    counts = (
        joined.groupby("concern_group", dropna=False)
        .agg(n=("ltkb_id", "count"), new_drugs=("is_new", "sum"))
        .reset_index()
    )
    counts.to_csv(TABLES / "dataset_counts.csv", index=False)

    descriptor_cols = list(desc.columns)
    primary = joined[joined["primary_label"].notna()].copy()
    summaries = []
    for label, group in primary.groupby("primary_label"):
        for col in descriptor_cols:
            summaries.append({
                "primary_label": int(label),
                "descriptor": col,
                "n": group[col].notna().sum(),
                "median": group[col].median(),
                "q1": group[col].quantile(0.25),
                "q3": group[col].quantile(0.75),
            })
    pd.DataFrame(summaries).to_csv(TABLES / "descriptor_summary.csv", index=False)
    print(counts)

if __name__ == "__main__":
    main()
