"""Evaluation metrics. IPCW metrics use the censoring distribution of the evaluated cohort itself
(marginal Kaplan-Meier; independent censoring assumed), identically for every model."""
import numpy as np
from .models import fast_cindex, surv_y


def harrell(e, t, risk):
    return fast_cindex(e, t, risk)


def secondary(e, t, risk, surv_h, horizons, tau):
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score
    y = surv_y(e, t); out, notes = {}, {}
    try:
        out['uno_cindex_tau'] = float(concordance_index_ipcw(y, y, risk, tau=tau)[0])
    except Exception as exc:
        out['uno_cindex_tau'] = np.nan; notes['uno_cindex_tau'] = repr(exc)[:200]
    for j, h in enumerate(horizons):
        ok = (t > h).any() and ((t <= h) & (e == 1)).any() and h < t.max()
        for name, fn in [(f'auc_{h}', lambda: cumulative_dynamic_auc(y, y, risk, [h])[0][0]),
                         (f'brier_{h}', lambda: brier_score(y, y, surv_h[:, j:j + 1], [h])[1][0])]:
            if not ok or surv_h is None and name.startswith('brier'):
                out[name] = np.nan; notes[name] = 'horizon not supported by follow-up'; continue
            try:
                out[name] = float(fn())
            except Exception as exc:
                out[name] = np.nan; notes[name] = repr(exc)[:200]
    for j, h in enumerate(horizons):          # calibration-in-the-large: mean predicted S(h) vs KM S(h)
        if surv_h is not None:
            out[f'mean_pred_surv_{h}'] = float(np.mean(surv_h[:, j]))
            out[f'km_surv_{h}'] = float(km(e, t, h))
    return out, notes


def km(e, t, h):
    s = 1.0
    for u in np.unique(t[(e == 1) & (t <= h)]):
        s *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
    return s


class PairedBootstrap:
    """Harrell C under patient resampling, computed as weighted pair counts so that every model and seed is
    evaluated on the same resampled patients (stratified by event within each fold)."""

    def __init__(self, e, t, B, seed):
        self.e, self.t = np.asarray(e, int), np.asarray(t, float)
        rng = np.random.default_rng(seed); n = len(e)
        W = np.zeros((B, n), np.float32)
        for g in (0, 1):
            idx = np.flatnonzero(self.e == g)
            if len(idx):
                draws = rng.choice(idx, size=(B, len(idx)))
                for b in range(B):
                    W[b] += np.bincount(draws[b], minlength=n)
        self.W = W
        self.ev = np.flatnonzero(self.e == 1)
        self.comp = (self.t[self.ev, None] < self.t[None, :]).astype(np.float32)

    def cindex(self, risk):
        r = np.asarray(risk, float)
        num = ((r[self.ev, None] > r[None, :]) + 0.5 * (r[self.ev, None] == r[None, :])).astype(np.float32) * self.comp
        We = self.W[:, self.ev]
        return np.einsum('be,en,bn->b', We, num, self.W, optimize=True) / \
            np.einsum('be,en,bn->b', We, self.comp, self.W, optimize=True)
