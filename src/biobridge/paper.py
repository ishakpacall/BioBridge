"""Helpers for the manuscript analyses (clinical utility, ER subgroups, figures). No model is trained here."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from .metrics import km
from .utils import RAW, RESULTS


def fuse(parts, how='mean'):
    risks = [np.asarray(p[0], float) for p in parts]
    if how == 'rank':
        risks = [rankdata(r) / len(r) for r in risks]
    return np.mean(risks, 0), np.mean([np.asarray(p[1], float) for p in parts], 0)


MODELS = {  # name -> (components, how); single models have one component
    'Clinical CoxPH': (['coxph_clinical__clinical'], None),
    'XGB-AFT (genes)': (['xgb_aft__genes'], None),
    'XGB-AFT (modules)': (['xgb_aft__modules'], None),
    'GBM-Cox (genes)': (['gbm_cox__genes'], None),
    'F2 fusion': (['xgb_aft__genes', 'xgb_aft__modules'], 'mean'),
}


def cv_patient_predictions(setting='cv_metabric', dev=None):
    """One prediction per patient: CV test-fold predictions, averaged over seeds (risk and S(t))."""
    dev = Path(dev or RESULTS / 'development') / setting
    rows = {}
    for unit in sorted(p for p in dev.iterdir() if p.is_dir()):
        lab = np.load(unit / 'labels.npz')
        for name, (comps, how) in MODELS.items():
            parts = [(np.load(unit / f'{c}.npz')['risk'], np.load(unit / f'{c}.npz')['surv']) for c in comps]
            if how == 'rank':
                raise ValueError('rank fusion is test-set specific; not pooled across folds')
            r, s = fuse(parts) if how else parts[0]
            for i, pid in enumerate(lab['patient_id']):
                d = rows.setdefault((name, str(pid)), dict(event=int(lab['event'][i]), time=float(lab['time'][i]), r=[], s=[]))
                d['r'].append(r[i]); d['s'].append(s[i])
    out = {}
    for (name, pid), d in rows.items():
        out.setdefault(name, []).append(dict(PATIENT_ID=pid, event=d['event'], time=d['time'],
                                             risk=np.mean(d['r']), surv36=np.mean([x[0] for x in d['s']]),
                                             surv60=np.mean([x[1] for x in d['s']])))
    return {k: pd.DataFrame(v).sort_values('PATIENT_ID').reset_index(drop=True) for k, v in out.items()}


def external_patient_predictions(cohort, source='metabric', ext=None):
    folder = Path(ext or RESULTS / 'external') / cohort
    lab = np.load(folder / 'labels.npz'); out = {}
    for name, (comps, how) in MODELS.items():
        parts = [(np.load(folder / f'{source}__{c}.npz')['risk'], np.load(folder / f'{source}__{c}.npz')['surv']) for c in comps]
        r, s = fuse(parts, how) if how else parts[0]
        out[name] = pd.DataFrame(dict(PATIENT_ID=lab['patient_id'].astype(str), event=lab['event'], time=lab['time'],
                                      risk=r, surv36=s[:, 0], surv60=s[:, 1]))
    return out


def calibration(df, horizon=60, groups=10):
    """Grouped calibration at the horizon: mean predicted risk vs 1 - KM within risk deciles."""
    pred = 1 - df[f'surv{horizon}'].to_numpy()
    q = pd.qcut(pd.Series(pred).rank(method='first'), groups, labels=False)
    rows = []
    for g in range(groups):
        m = (q == g).to_numpy()
        rows.append(dict(group=g + 1, n=int(m.sum()), predicted=float(pred[m].mean()),
                         observed=float(1 - km(df.event.to_numpy()[m], df.time.to_numpy()[m], horizon))))
    cal = pd.DataFrame(rows)
    ici = float(np.average(np.abs(cal.observed - cal.predicted), weights=cal.n))
    citl = float(pred.mean() - (1 - km(df.event.to_numpy(), df.time.to_numpy(), horizon)))
    return cal, dict(ici=ici, calibration_in_the_large=citl)


def decision_curve(df, horizon=60, thresholds=np.arange(0.02, 0.41, 0.01)):
    """Net benefit with KM-estimated event probability among patients above each threshold (Vickers et al.)."""
    pred = 1 - df[f'surv{horizon}'].to_numpy(); e, t = df.event.to_numpy(), df.time.to_numpy()
    n = len(df); s_all = km(e, t, horizon); out = []
    for pt in thresholds:
        hi = pred >= pt; p_hi = hi.mean()
        if hi.sum() == 0:
            nb = 0.0
        else:
            s_hi = km(e[hi], t[hi], horizon)
            nb = p_hi * (1 - s_hi) - p_hi * s_hi * pt / (1 - pt)
        out.append(dict(threshold=pt, net_benefit=nb, treat_all=(1 - s_all) - s_all * pt / (1 - pt)))
    return pd.DataFrame(out)


def er_status(cohort):
    """ER status (1/0) by PATIENT_ID from the raw files; returns empty Series if unavailable."""
    try:
        if cohort == 'metabric':
            s = pd.read_csv(RAW / 'metabric_sample.txt', sep='\t', comment='#')
            p = pd.read_csv(RAW / 'metabric_patient.txt', sep='\t', comment='#')
            d = s.merge(p, on='PATIENT_ID', how='left')
            col = next((c for c in ['ER_STATUS', 'ER_IHC'] if c in d.columns), None)
            if col is None:
                return pd.Series(dtype=float)
            v = d[col].astype(str).str.lower()
            return pd.Series(np.where(v.str.startswith('pos'), 1.0, np.where(v.str.startswith('neg'), 0.0, np.nan)),
                             index=d.PATIENT_ID.astype(str)).dropna()
        if cohort == 'scanb':
            from .prepare import geo_metadata
            g = pd.concat([geo_metadata(RAW / 'scanb_gpl11154.txt.gz'), geo_metadata(RAW / 'scanb_gpl18573.txt.gz')])
            col = next((c for c in g.columns if c.lower().replace('_', ' ') in ('er status',)), None)
            if col is None:
                return pd.Series(dtype=float)
            v = pd.to_numeric(g[col], errors='coerce')
            return pd.Series(v.to_numpy(), index=g.SAMPLE_ID.astype(str)).dropna()
    except FileNotFoundError:
        pass
    return pd.Series(dtype=float)
