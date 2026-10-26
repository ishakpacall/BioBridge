# Protocol lock and amendments

## How the lock works

`scripts/00_lock_protocol.py` computes SHA-256 hashes of:

- `config/protocol.yaml` (cohorts, endpoint, covariates, representations, model grids, validation settings, seeds, metrics, primary hypothesis)
- `config/genesets_frozen.json` and `config/sources.json`
- every Python file in `src/biobridge/` and `scripts/`

and writes them, with package versions and a UTC timestamp, to `PROTOCOL_LOCK.json`. Every analysis script calls `verify_lock()` before running and stops if any of these files differs from the lock. Files in `tools/`, `docs/`, `tests/` and the repository root are not part of the lock.

A deliberate change is recorded with:

```bash
python scripts/00_lock_protocol.py --amend "reason for the change"
```

which appends a dated entry (time, changed files, reason) to the history in `PROTOCOL_LOCK.json` and to `PROTOCOL_AMENDMENTS.md`.

## Timeline

The initial lock was written on 10 October 2026 at 13:04 UTC, before the raw data for this study were downloaded. A pilot on the same development cohorts preceded the lock and motivated the primary hypothesis; SCAN-B and GSE20685 were scored in that pilot with other models and are therefore treated as previously examined external cohorts.

Seven amendment entries followed: a process-isolation fix for DeepSurv made before the full development run; the addition of the exploratory fusion analysis (recorded twice); the learning-curve, clinical-utility and interpretation analyses with hypotheses H1 and H2; recalibration, supplementary tables and the confirmatory script for future cohorts; figure formatting; and five additional neural comparators. None changed the cohorts, folds, features, original model grids or the primary hypothesis. Each analysis was recorded before its results were computed. The authoritative record, with exact times and changed files, is `PROTOCOL_LOCK.json` (Supplementary Table S13 of the article).

## Verifying a checkout

```bash
python tools/check_release.py
```

reports whether the checked-out code matches the lock and whether any data or result files are tracked by git.
