"""EXPLORATORY analysis (added after the locked primary analysis was completed and reported).

Plan, fixed in this file BEFORE the fusion results are computed (the file hash is recorded by
`00_lock_protocol.py --amend` before running):

Idea taken from Witowski et al., Nat Commun 2026: late fusion of models trained on different inputs with
EQUAL weights (no learned weights). No model is retrained; existing predictions are combined.

  F2 = "xgbfuse2__aft"    : XGB-AFT(genes) + XGB-AFT(modules).
                             Risk = mean of the two risks (both are -log-time, same scale);
                             S(t) = mean of the two predicted survival probabilities.
  F4 = "xgbfuse4__aftcox" : XGB-AFT(genes), XGB-AFT(modules), XGB-Cox(genes), XGB-Cox(modules).
                             Risk = mean of within-test-set percentile ranks (label-free; Cox and AFT
                             scales differ); S(t) = mean of the four predicted survival probabilities.

Comparison of interest (exploratory, reported unadjusted AND Holm-adjusted): F2 vs XGB-AFT(genes);
F4 vs XGB-AFT(genes); both vs every other model. Development: same units/folds/seeds and paired bootstrap
as the main report. External: SCAN-B and GSE20685 seed-averaged predictions, both training sources.
Because SCAN-B/GSE20685 were already used, these external numbers are NOT confirmatory; any claim needs
a new, untouched cohort. Nothing in this analysis is tuned; there are no free parameters.

  python scripts/08_exploratory_fusion.py
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
import _bootstrap  # noqa: F401
from biobridge.utils import RESULTS, load_protocol, verify_lock, atomic_json, atomic_npz, log
from biobridge.metrics import harrell, secondary, PairedBootstrap
from biobridge.report import summarize, holm

F2 = ('xgbfuse2__aft', ['xgb_aft__genes', 'xgb_aft__modules'], 'mean')
F4 = ('xgbfuse4__aftcox', ['xgb_aft__genes', 'xgb_aft__modules', 'xgb_cox__genes', 'xgb_cox__modules'], 'rank')
FUSIONS = [F2, F4]


def fuse(parts, how):
    risks = [np.asarray(p['risk'], float) for p in parts]
    if how == 'rank':
        risks = [rankdata(r) / len(r) for r in risks]
    return np.mean(risks, 0), np.mean([np.asarray(p['surv'], float) for p in parts], 0)


def development(P, src, dst, rep_dir, B):
    for setting in sorted(p for p in src.iterdir() if p.is_dir()):
        for unit in sorted(p for p in setting.iterdir() if p.is_dir()):
            out = dst / setting.name / unit.name
            out.mkdir(parents=True, exist_ok=True)
            for f in list(unit.glob('*__*.npz')) + list(unit.glob('*__*.json')) + [unit / 'labels.npz', unit / 'unit.json']:
                if not f.name.endswith('.error.json') and not f.name.endswith('.skipped.json'):
                    shutil.copy2(f, out / f.name)
            lab = np.load(unit / 'labels.npz'); e, t = lab['event'], lab['time']
            for key, comps, how in FUSIONS:
                parts = [np.load(unit / f'{c}.npz') for c in comps]
                risk, surv = fuse(parts, how)
                sec, notes = secondary(e, t, risk, surv, P['metrics']['horizons_months'], P['metrics']['uno_tau_months'])
                atomic_npz(out / f'{key}.npz', risk=risk, surv=surv)
                atomic_json(out / f'{key}.json', dict(key=key, harrell=harrell(e, t, risk), **sec, notes=notes,
                                                      components=comps, combination=how))
    log('development fusion predictions written; computing paired bootstrap report')
    s, _, p2 = summarize(dst, rep_dir / 'report_F2', B, ('xgbfuse2__aft', 'xgb_aft__genes'))
    _, _, p4 = summarize(dst, rep_dir / 'report_F4', B, ('xgbfuse4__aftcox', 'xgb_aft__genes'))
    return s, p2, p4


def external(P, ext, B):
    rows, con = [], []
    for cohort in sorted(p for p in ext.iterdir() if p.is_dir() and (p / 'labels.npz').exists()):
        lab = np.load(cohort / 'labels.npz'); e, t = lab['event'], lab['time']
        pb = PairedBootstrap(e, t, B, 2027)
        for source in ['metabric', 'pooled']:
            preds = {}
            for f in cohort.glob(f'{source}__*.npz'):
                d = np.load(f); preds[f.stem.split('__', 1)[1]] = dict(risk=d['risk'], surv=d['surv'])
            if not all(c in preds for _, comps, _ in FUSIONS for c in comps):
                continue
            for key, comps, how in FUSIONS:
                r, s = fuse([preds[c] for c in comps], how); preds[key] = dict(risk=r, surv=s)
            boot = {k: pb.cindex(v['risk']) for k, v in preds.items()}
            for k, v in preds.items():
                sec, _ = secondary(e, t, v['risk'], v['surv'], P['metrics']['horizons_months'], P['metrics']['uno_tau_months'])
                rows.append(dict(cohort=cohort.name, source=source, model=k, cindex=harrell(e, t, v['risk']),
                                 ci_low=np.quantile(boot[k], .025), ci_high=np.quantile(boot[k], .975),
                                 brier_60=sec.get('brier_60')))
            for key, _, _ in FUSIONS:
                for other in preds:
                    if other == key:
                        continue
                    d = boot[key] - boot[other]
                    con.append(dict(cohort=cohort.name, source=source, fusion=key, comparator=other,
                                    delta=harrell(e, t, preds[key]['risk']) - harrell(e, t, preds[other]['risk']),
                                    ci_low=np.quantile(d, .025), ci_high=np.quantile(d, .975),
                                    p_one_sided=float((np.sum(d <= 0) + 1) / (B + 1))))
    return pd.DataFrame(rows), pd.DataFrame(con)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=None); a = ap.parse_args()
    verify_lock(a.smoke); P = load_protocol(); B = a.bootstrap or P['metrics']['bootstrap']
    sfx = '_smoke' if a.smoke else ''
    src, dst = RESULTS / f'development{sfx}', RESULTS / f'exploratory_fusion_development{sfx}'
    if not (src / 'RUN_STATUS.json').exists() or not json.loads((src / 'RUN_STATUS.json').read_text())['complete']:
        raise SystemExit('Geliştirme koşusu tamamlanmamış.')
    out = RESULTS / f'exploratory_fusion{sfx}'; out.mkdir(exist_ok=True)
    s, p2, p4 = development(P, src, dst, out, B)
    prim = pd.concat([p2.assign(fusion='F2'), p4.assign(fusion='F4')], ignore_index=True)
    prim['p_holm_all12'] = holm(prim.p_one_sided)
    prim.to_csv(out / 'development_fusion_vs_xgb_aft_genes.csv', index=False)
    s.to_csv(out / 'development_summary_with_fusion.csv', index=False)
    ext_dir = RESULTS / f'external{sfx}'
    if ext_dir.exists():
        m, c = external(P, ext_dir, B)
        m.to_csv(out / 'external_metrics_with_fusion.csv', index=False); c.to_csv(out / 'external_contrasts_fusion.csv', index=False)
    print('\n== GELİŞTİRME: C-index (ana modeller + füzyonlar)')
    keep = ['xgb_aft__genes', 'xgb_aft__modules', 'xgb_cox__genes', 'xgb_cox__modules', 'gbm_cox__genes',
            'coxnet__modules', 'k2__k2', 'xgbfuse2__aft', 'xgbfuse4__aftcox']
    t = s[s.model.isin(keep)].assign(txt=lambda d: d.apply(lambda r: f"{r.cindex:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
    print(t.pivot(index='model', columns='setting', values='txt').to_string())
    print('\n== GELİŞTİRME: füzyon − XGB-AFT(genler) (keşifsel; Holm 12 karşılaştırma üzerinden)')
    print(prim[['fusion', 'setting', 'delta', 'ci_low', 'ci_high', 'p_one_sided', 'p_holm_all12']].round(4).to_string(index=False))
    if ext_dir.exists():
        print('\n== DIŞ KOHORTLAR (daha önce kullanıldı; doğrulayıcı DEĞİL)')
        mm = m[m.model.isin(keep)].assign(txt=lambda d: d.apply(lambda r: f"{r.cindex:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
        print(mm.pivot_table(index='model', columns=['cohort', 'source'], values='txt', aggfunc='first').to_string())
        cc = c[c.comparator.isin(['xgb_aft__genes', 'xgb_cox__genes', 'gbm_cox__genes'])]
        print(cc[['cohort', 'source', 'fusion', 'comparator', 'delta', 'ci_low', 'ci_high']].round(4).to_string(index=False))
    print('\nCOMPLETE', out)


if __name__ == '__main__':
    main()
