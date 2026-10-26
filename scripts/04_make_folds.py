"""Create the development folds once (stratified by event within cohort; fixed seed from the protocol)."""
import numpy as np, pandas as pd
import _bootstrap  # noqa: F401
from biobridge.utils import PREPARED, load_protocol, verify_lock, sha256, atomic_json, read_json
from biobridge.prepare import load_prepared

import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
verify_lock(a.smoke)
P = load_protocol(); out = PREPARED / 'folds.csv'
if out.exists():
    raise SystemExit(f'{out} zaten var; fold dosyası bir kez oluşturulur.')
rng = np.random.default_rng(P['validation']['fold_seed']); recs = []
for c in P['cohorts']['development']:
    rows, _ = load_prepared(c)
    fold = np.zeros(len(rows), int)
    for ev in (0, 1):
        idx = np.flatnonzero(rows.event.to_numpy() == ev); rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % P['validation']['folds'] + 1
    recs.append(pd.DataFrame(dict(cohort=c, PATIENT_ID=rows.PATIENT_ID, fold=fold)))
    print(c, pd.crosstab(fold, rows.event).to_dict())
pd.concat(recs).to_csv(out, index=False)
atomic_json(PREPARED / 'folds.json', dict(sha256=sha256(out), fold_seed=P['validation']['fold_seed']))
print('COMPLETE', out)
