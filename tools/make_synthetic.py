"""Write a SYNTHETIC data/prepared folder for functional tests only (never for results).

  python tools/make_synthetic.py --out data_synthetic/prepared
  BIOBRIDGE_PREPARED=data_synthetic/prepared python scripts/05_develop.py --smoke
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'data_synthetic' / 'prepared'))
    ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import biobridge.utils as U
    U.PREPARED = out
    from biobridge import prepare as pr
    pr.PREPARED = out
    spec = json.loads((ROOT / 'config' / 'genesets_frozen.json').read_text())
    genes = []
    for entries in spec['sets'].values():
        genes += [e['symbols'][0] for e in entries]
    genes += [g['symbols'][0] for g in spec['pam50']['genes']]
    genes = sorted({g.upper() for g in genes}); genes += [f'FILLER{i}' for i in range(1500)]
    universe = sorted(genes)
    U.atomic_json(out / 'universe.json', dict(genes=universe, n=len(universe), cohorts=['synthetic']))
    rng = np.random.default_rng(a.seed)
    prolif = {e['symbols'][0].upper(): e['sign'] for e in spec['sets']['mod_AURKA']}
    er = {e['symbols'][0].upper(): e['sign'] for e in spec['sets']['mod_ESR1']}
    base = rng.normal(0, 2, len(universe))
    for name, n, shift, cens in [('metabric', 900, 0.0, 200), ('tcga', 500, 1.0, 60),
                                 ('scanb', 600, 0.5, 70), ('gse20685', 200, -0.5, 150)]:
        p = rng.normal(size=n); erl = rng.normal(size=n) - 0.5 * p
        x = base + rng.normal(0, 1, (n, len(universe))) + shift * rng.normal(0, 1, len(universe))
        for j, g in enumerate(universe):
            if g in prolif:
                x[:, j] += 1.2 * prolif[g] * p
            if g in er:
                x[:, j] += 1.0 * er[g] * erl
        age = rng.normal(60, 12, n); node = (rng.random(n) < 0.45).astype(float)
        lp = 0.7 * p - 0.3 * erl + 0.6 * node + 0.03 * (age - 60)
        tt = rng.exponential(np.exp(-lp) * 120); cc = rng.exponential(cens, n)
        rows = pd.DataFrame(dict(PATIENT_ID=[f'{name}_{i}' for i in range(n)], SAMPLE_ID=[f'{name}_{i}' for i in range(n)],
                                 time=np.maximum(np.minimum(tt, cc), 0.1), event=(tt <= cc).astype(int),
                                 AGE_AT_DIAGNOSIS=np.where(rng.random(n) < .03, np.nan, age),
                                 NODE_POSITIVE=np.where(rng.random(n) < .05, np.nan, node),
                                 SITE=[f'{name}_S{k}' for k in rng.integers(0, 4, n)]))
        (out / name).mkdir(exist_ok=True)
        ranks, present = pr.percentile_ranks(x.astype(np.float32), np.array(universe), universe)
        pr.save_cohort(name, rows, ranks, dict(synthetic=True))
    print('SYNTHETIC data written to', out)


if __name__ == '__main__':
    main()
