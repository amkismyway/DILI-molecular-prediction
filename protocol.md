# Prespecified Research Protocol

## Working title
Interpretable Machine Learning Prediction of Drug-Induced Liver Injury From Molecular Structure Using FDA DILIrank 2.0

## Background
Drug-induced liver injury is an important cause of medication-related morbidity and drug withdrawal. Chemical structure may encode properties associated with hepatotoxic risk, but model performance can be inflated when structurally similar compounds appear in both training and test sets. This study will compare interpretable molecular descriptors and structural fingerprints using random, scaffold-based, and pseudo-temporal validation.

## Primary objective
To compare the discrimination and calibration of descriptor-based and fingerprint-based machine-learning models for classifying FDA drugs as Most-DILI-concern versus No-DILI-concern.

## Primary hypothesis
Fingerprint-based models will show higher random-split discrimination than descriptor-only models but a larger decline under scaffold and pseudo-temporal validation.

## Data sources
1. FDA DILIrank 2.0
2. FDA DILIst for secondary label comparison
3. PubChem for canonical molecular structures

## Eligibility
### Include
- Small-molecule drug with a resolvable PubChem structure
- FDA DILIrank 2.0 label
- Structure successfully standardized by RDKit

### Exclude
- Biologics, peptides, vaccines, polymers, mixtures, and poorly defined substances
- Compounds without a resolvable structure
- Duplicate standardized parent structures
- Primary endpoint: Less-DILI-concern and Ambiguous-DILI-concern

All exclusions will be logged with a reason.

## Outcomes
### Primary outcome
Most-DILI-concern (1) versus No-DILI-concern (0).

### Secondary outcomes
- Most or Less concern versus No concern
- Agreement and performance against DILIst binary classifications
- Performance on drugs marked `New` in DILIrank 2.0 after training on `Unchanged` drugs

## Molecular representations
### Interpretable descriptors
Molecular weight, MolLogP, TPSA, H-bond donors, H-bond acceptors, rotatable bonds, ring count, aromatic ring count, fraction Csp3, heavy atoms, heteroatoms, formal charge, and molar refractivity.

### Structural representations
- Morgan fingerprints, radius 2, 2048 bits
- Optional MACCS keys

## Models
1. Dummy classifier
2. Regularized logistic regression
3. Random forest
4. Histogram gradient boosting for descriptor data
5. Logistic regression on Morgan fingerprints

## Validation

1. Held-out stratified random split
2. Bemis-Murcko scaffold split
3. Pseudo-temporal split using unchanged drugs for development and new drugs for locked testing

Hyperparameter selection will use training data only.

## Metrics
- ROC-AUC
- Average precision / PR-AUC
- Balanced accuracy
- Sensitivity
- Specificity
- Precision
- F1
- Matthews correlation coefficient
- Brier score
- Calibration slope/intercept where estimable
- Bootstrap 95% confidence intervals

## Statistical analysis
Continuous descriptors will be summarized by median and interquartile range. Models will be compared primarily by ROC-AUC and average precision, with emphasis on validation-set uncertainty rather than isolated p-values. Bootstrap confidence intervals will use resampling within the test set.

## Interpretability
Descriptor coefficients, permutation importance, and SHAP-compatible analyses may be used. Associations will not be interpreted as causal toxicologic mechanisms.

## Reproducibility
Random seeds, software versions, data-cleaning decisions, exclusions, and model settings will be saved. Code and nonrestricted derived data will be released with the manuscript where permitted.

## Ethics
The project uses public, non-patient-level drug and chemical data. Institutional review or a formal non-human-subject determination should be confirmed with the supervising institution.

## Deviations
Any departure from this protocol will be recorded in `supplementary/protocol_deviations.md` before final analysis.
