# BioBridge

Code for the article "Training event count shapes the value of biological knowledge in breast cancer survival prediction across transcriptomic platforms" (under review).

## What is included

- The locked analysis protocol (`config/protocol.yaml`), frozen gene sets and data-source manifest
- `PROTOCOL_LOCK.json`: SHA-256 fingerprints of the protocol and of **every** analysis file, including files not yet released, with the dated amendment history
- `data/raw/DATA_LOCK.json`: source addresses and SHA-256 hashes of the public input files
- Data download, cohort preparation, cross-platform harmonisation (within-patient ranks) and module scoring (`src/biobridge/prepare.py`, `features.py`)
- All survival model families (`src/biobridge/models.py`), including:
  - **K2**, the pre-specified knowledge-first two-stage model (boosting on biological modules, then on genes from the module-based prediction, with an inner-holdout gate)
  - **XGB-AFT** with initialisation at the censored maximum-likelihood intercept instead of the library default
  - Cox, Coxnet, random survival forest, extra survival trees, GBM-Cox, XGB-Cox and DeepSurv
- **F2**, the exploratory equal-weight fusion of gene- and module-based XGB-AFT (`src/biobridge/paper.py`, `scripts/08_exploratory_fusion.py`)
- Evaluation metrics and paired bootstrap (`src/biobridge/metrics.py`)
- Documentation of the protocol, implementation notes and the mapping from scripts to the article

## What is released on publication

Experiment runners for the development settings, external validation, learning curve, clinical utility, interpretation and additional neural comparators, unit tests and continuous integration. The complete code is provided to the reviewers and will be published at github.com/ishakpacall/BioBridge with an archived DOI.

Because `PROTOCOL_LOCK.json` already records the SHA-256 hash of every withheld file, anyone can verify after publication that the released code is identical to the code fingerprinted here.

## Note

This preview is for inspection. The scripts verify the full protocol lock before running and will stop until the complete release is available.

## Contact

Ishak Pacal, Igdir University, ishakpacal@igdir.edu.tr
