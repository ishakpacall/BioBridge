"""Prepare cohorts: patients.csv + within-patient percentile ranks on a frozen common gene universe.

The universe = genes present in all development and pre-specified external cohorts (no outcomes used).
Custom external cohorts added later (data/raw/custom/<name>/) are mapped onto the frozen universe:
  python scripts/02_prepare.py --custom NAME
"""
import argparse
import _bootstrap  # noqa: F401
from biobridge.utils import PREPARED, atomic_json, read_json, sha256, log
from biobridge import prepare as pr

ap = argparse.ArgumentParser(); ap.add_argument('--custom', nargs='*', default=[]); a = ap.parse_args()
PREPARED.mkdir(parents=True, exist_ok=True)
uni_path = PREPARED / 'universe.json'
if not uni_path.exists():
    loaded = {}
    for name, loader in pr.LOADERS.items():
        log('LOAD', name); loaded[name] = loader()
    sets = [set(v[2]) for v in loaded.values()]
    universe = sorted(set.intersection(*sets))
    if len(universe) < 5000:
        raise SystemExit(f'Ortak gen sayısı beklenmedik derecede düşük ({len(universe)}); eşlemeyi inceleyin.')
    atomic_json(uni_path, dict(genes=universe, n=len(universe), cohorts=list(loaded)))
    for name, (rows, x, genes, audit) in loaded.items():
        (PREPARED / name).mkdir(exist_ok=True)
        ranks, present = pr.percentile_ranks(x, genes, universe)
        audit['universe_genes_present'] = present
        pr.save_cohort(name, rows, ranks, audit)
else:
    log('Universe frozen:', read_json(uni_path)['n'], 'genes (delete data/prepared to rebuild everything)')
universe = pr.universe()
for name in a.custom:
    rows, x, genes, audit = pr.load_custom(name)
    (PREPARED / name).mkdir(exist_ok=True)
    ranks, present = pr.percentile_ranks(x, genes, universe)
    audit['universe_genes_present'] = present
    if present < 0.8 * len(universe):
        log(f'UYARI: {name} evrendeki genlerin yalnızca {present}/{len(universe)} tanesini içeriyor')
    pr.save_cohort(name, rows, ranks, audit)
print('COMPLETE')
