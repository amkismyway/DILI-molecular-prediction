# DILI-molecular-prediction
# Molecular Prediction of Drug-Induced Liver Injury

This repository contains the code used for the study:

"Interpretable Machine Learning Prediction of Drug-Induced Liver Injury
From Molecular Structure Using FDA DILIrank 2.0"

## Data sources

- FDA DILIrank 2.0
- FDA DILIst
- PubChem PUG REST

Raw source data are not redistributed in this repository.
Instructions for downloading them are provided below.

## Reproduction

1. Install Python 3.11.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the FDA DILIrank 2.0 and DILIst files and place them in `data/raw/`.

4. Run:

```bash
python scripts/01_prepare_labels.py
python scripts/02_fetch_pubchem.py
python scripts/03_build_features.py
python scripts/04_train_models.py
python scripts/05_make_report.py
```

## Citation

Archived release: https://doi.org/10.5281/zenodo.21461204
## Main endpoint

Most-DILI-concern versus No-DILI-concern.

## Software

The analysis uses RDKit, pandas, NumPy, and scikit-learn.

## License

Code is released under the MIT License.


