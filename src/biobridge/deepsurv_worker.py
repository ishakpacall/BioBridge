"""DeepSurv in an isolated child process.

On macOS, PyTorch, XGBoost and scikit-learn each ship their own OpenMP runtime; using several of them with
multiple threads in one process can deadlock. The main process therefore never imports torch. This worker
imports torch first and only numpy besides it; inputs/outputs are exchanged through temporary files, and the
parent enforces a timeout so a stuck model is recorded as an error instead of blocking the whole run.

Model and selection are unchanged from the locked protocol: MLP (2 hidden layers), configs hidden in {32, 64}
x dropout in {0.1, 0.4}, Adam lr 1e-3, weight decay 1e-4, full-batch Cox partial likelihood, early stopping on
the stratified inner holdout (checked every 5 epochs, patience 10 checks), refit on all training data.
"""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np

CONFIGS = [dict(hidden=h, dropout=d) for h in (32, 64) for d in (0.1, 0.4)]
TIMEOUT_SECONDS = 1800


# ---------------------------------------------------------------- numpy helpers (no sklearn / xgboost here)

def _cindex(e, t, r):
    e = np.asarray(e, bool); t = np.asarray(t, float); r = np.asarray(r, float)
    i = np.flatnonzero(e); num = den = 0.0
    for a in range(0, len(i), 512):
        ii = i[a:a + 512]; comp = t[ii, None] < t[None, :]
        num += ((r[ii, None] > r[None, :]) & comp).sum() + 0.5 * ((r[ii, None] == r[None, :]) & comp).sum()
        den += comp.sum()
    return num / den if den else np.nan


def _holdout(strata, seed, frac):
    rng = np.random.default_rng(seed); val = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s); rng.shuffle(idx)
        val[idx[:int(round(frac * len(idx)))]] = True
    return val


def _breslow(eta, e, t):
    order = np.argsort(t); t_s, e_s, w = t[order], e[order], np.exp(eta[order] - eta.max())
    at_risk = np.cumsum(w[::-1])[::-1]; ut = np.unique(t_s[e_s == 1])
    h = np.array([e_s[t_s == u].sum() / at_risk[np.searchsorted(t_s, u, 'left')] for u in ut])
    return ut, np.cumsum(h), eta.max()


def _survival(base, eta, times):
    ut, H, shift = base; pos = np.searchsorted(ut, times, 'right') - 1
    H0 = np.where(pos >= 0, H[np.clip(pos, 0, None)], 0.0)
    return np.exp(-np.exp(eta - shift)[:, None] * H0[None, :])


# ---------------------------------------------------------------- torch part (child process only)

def _fit(torch, X, e, t, cfg, seed, threads, val=None, epochs=None):
    torch.manual_seed(seed); torch.set_num_threads(threads)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    h, d = cfg['hidden'], cfg['dropout']
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Dropout(d),
                              torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Dropout(d), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    order = np.argsort(-t)
    xt = torch.tensor(((X - mu) / sd)[order], dtype=torch.float32)
    et = torch.tensor(e[order], dtype=torch.float32)

    def risk(Z):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(1).numpy().astype(float)
    best, best_ep, state, bad = -1.0, 0, None, 0
    for ep in range(1, (epochs or 500) + 1):
        net.train(); opt.zero_grad()
        r = net(xt).squeeze(1)
        loss = -((r - torch.logcumsumexp(r, 0)) * et).sum() / et.sum().clamp(min=1)
        loss.backward(); opt.step()
        if epochs is None and (ep % 5 == 0 or ep == 1):
            c = _cindex(val[1], val[2], risk(val[0]))
            if c > best:
                best, best_ep, bad = c, ep, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
    if epochs is None and state is not None:
        net.load_state_dict(state)
    return risk, (epochs or best_ep), best


def _child(in_dir):
    import torch                                  # first and only OpenMP runtime in this process
    a = np.load(Path(in_dir) / 'input.npz')
    meta = json.loads((Path(in_dir) / 'meta.json').read_text())
    X, e, t, strata = a['X'], a['e'].astype(int), a['t'].astype(float), a['strata']
    seed, threads, frac = meta['seed'], meta['threads'], meta['frac']
    configs = CONFIGS[:1] if meta.get('smoke') else CONFIGS
    val = _holdout(strata, seed, frac); trace, best = [], None
    for cfg in configs:
        risk, ep, _ = _fit(torch, X[~val], e[~val], t[~val], cfg, seed, threads, val=(X[val], e[val], t[val]))
        c = _cindex(e[val], t[val], risk(X[val]))
        trace.append(dict(cfg=cfg, inner_c=c, iterations=ep))
        if np.isfinite(c) and (best is None or c > best[0]):
            best = (c, cfg, ep)
    if best is None:
        raise RuntimeError(f'DeepSurv: no valid configuration {trace}')
    risk, ep, _ = _fit(torch, X, e, t, best[1], seed, threads, epochs=max(1, best[2]))
    base = _breslow(risk(X), e, t)
    horizons = np.array(meta['horizons'], float); out = {}
    for k in meta['tests']:
        r = risk(a[k]); out[f'risk_{k}'] = r; out[f'surv_{k}'] = _survival(base, r, horizons)
    np.savez(Path(in_dir) / 'output.npz', **out)
    (Path(in_dir) / 'selection.json').write_text(json.dumps(dict(inner_c=best[0], cfg=best[1], iterations=best[2],
                                                                 trace=trace), default=float))


# ---------------------------------------------------------------- parent-side API

def run(X, e, t, strata, tests, horizons, seed, threads, P):
    """tests: dict name -> test matrix. Returns ({name: (risk, surv)}, selection)."""
    with tempfile.TemporaryDirectory(prefix='deepsurv_') as d:
        np.savez(Path(d) / 'input.npz', X=np.asarray(X, np.float32), e=np.asarray(e), t=np.asarray(t, float),
                 strata=np.asarray(strata, str), **{k: np.asarray(v, np.float32) for k, v in tests.items()})
        (Path(d) / 'meta.json').write_text(json.dumps(dict(seed=int(seed), threads=int(threads), tests=list(tests),
                                                           frac=P['validation']['inner_holdout_fraction'],
                                                           horizons=[float(h) for h in horizons],
                                                           smoke=bool(P.get('smoke')))))
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, '-m', 'biobridge.deepsurv_worker', d], env=env,
                                  capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f'DeepSurv worker exceeded {TIMEOUT_SECONDS}s and was stopped')
        if proc.returncode != 0 or not (Path(d) / 'output.npz').exists():
            raise RuntimeError(f'DeepSurv worker failed (code {proc.returncode}):\n{proc.stderr[-3000:]}')
        o = np.load(Path(d) / 'output.npz')
        res = {k: (o[f'risk_{k}'], o[f'surv_{k}']) for k in tests}
        sel = json.loads((Path(d) / 'selection.json').read_text()); sel['worker_seconds'] = time.time() - t0
        return res, sel


if __name__ == '__main__':
    try:
        _child(sys.argv[1])
    except Exception:
        traceback.print_exc(); sys.exit(1)
