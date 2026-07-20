"""Train models and compare random, scaffold, and pseudo-temporal validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "tables"
MODELS = ROOT / "supplementary" / "models"
TABLES.mkdir(exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SEED = 20260720

def metric_row(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "prevalence": float(np.mean(y)),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "average_precision": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, p),
    }

def bootstrap_ci(y, p, metric, n_boot=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[sample])) < 2 and metric == "roc_auc":
            continue
        if metric == "roc_auc":
            vals.append(roc_auc_score(y[sample], p[sample]))
        else:
            vals.append(average_precision_score(y[sample], p[sample]))
    return np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)

def random_split(y):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    return next(splitter.split(np.zeros(len(y)), y))

def scaffold_split(scaffolds, y, test_fraction=0.25):
    groups = {}
    for idx, scaffold in enumerate(scaffolds):
        key = scaffold if isinstance(scaffold, str) and scaffold else f"NO_SCAFFOLD_{idx}"
        groups.setdefault(key, []).append(idx)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    target = int(round(len(y) * test_fraction))
    test, train = [], []
    counts = {0: 0, 1: 0}
    totals = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    for group in ordered:
        group_y = y[group]
        can_add = len(test) + len(group) <= target * 1.15
        # Prefer groups that keep both classes represented and approach target size.
        if can_add:
            test.extend(group)
            for cls in (0, 1):
                counts[cls] += int(np.sum(group_y == cls))
        else:
            train.extend(group)
    if not train:
        train = [i for i in range(len(y)) if i not in set(test)]
    return np.array(train), np.array(test)

def descriptor_models():
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "descriptor_logistic": Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))
        ]),
        "descriptor_random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=800, min_samples_leaf=3, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1
            ))
        ]),
        "descriptor_hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED
            ))
        ]),
    }

def evaluate_split(name, train_idx, test_idx, Xd, Xf, y, meta):
    results = []
    models = descriptor_models()
    models["fingerprint_logistic"] = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=0.2, random_state=SEED
    )
    for model_name, model in models.items():
        X = Xf if model_name == "fingerprint_logistic" else Xd
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        p = fitted.predict_proba(X[test_idx])[:, 1]
        row = {"split": name, "model": model_name, **metric_row(y[test_idx], p)}
        lo, hi = bootstrap_ci(y[test_idx], p, "roc_auc")
        row["roc_auc_ci_low"], row["roc_auc_ci_high"] = lo, hi
        lo, hi = bootstrap_ci(y[test_idx], p, "average_precision")
        row["ap_ci_low"], row["ap_ci_high"] = lo, hi
        results.append(row)
        dump(fitted, MODELS / f"{name}_{model_name}.joblib")
        pd.DataFrame({
            "ltkb_id": meta.iloc[test_idx]["ltkb_id"].values,
            "compound_name": meta.iloc[test_idx]["compound_name"].values,
            "y_true": y[test_idx],
            "probability": p,
        }).to_csv(TABLES / f"predictions_{name}_{model_name}.csv", index=False)
    return results

def main():
    meta = pd.read_csv(DATA / "molecule_metadata.csv")
    desc = pd.read_csv(DATA / "molecular_descriptors.csv")
    fps = np.load(DATA / "morgan_fingerprints.npz")["X"]

    eligible = meta["primary_label"].notna().values
    m = meta.loc[eligible].reset_index(drop=True)
    Xd = desc.loc[eligible].to_numpy(dtype=float)
    Xf = fps[eligible]
    y = m["primary_label"].astype(int).to_numpy()

    results = []
    tr, te = random_split(y)
    results.extend(evaluate_split("random", tr, te, Xd, Xf, y, m))

    tr, te = scaffold_split(m["murcko_scaffold"].fillna("").to_numpy(), y)
    results.extend(evaluate_split("scaffold", tr, te, Xd, Xf, y, m))

    train = np.flatnonzero(~m["is_new"].astype(bool).to_numpy())
    test = np.flatnonzero(m["is_new"].astype(bool).to_numpy())
    if len(test) >= 20 and len(np.unique(y[test])) == 2:
        results.extend(evaluate_split("pseudo_temporal", train, test, Xd, Xf, y, m))
    else:
        print("Pseudo-temporal split skipped: insufficient new drugs or only one class.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES / "model_performance.csv", index=False)
    print(results_df[["split", "model", "n", "roc_auc", "average_precision", "brier"]])

if __name__ == "__main__":
    main()
