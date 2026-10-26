"""Survival models. Common contract: fit(X, e, t) -> self; risk(X) (higher = earlier death);
survival(X, times) -> S(t) matrix. Hyper-parameters are chosen on a stratified inner holdout of the
training patients only, then the model is refit on all training patients.
"""
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- helpers

def fast_cindex(e, t, r):
    """Harrell C (ties in risk count 1/2; comparable if t_i < t_j and e_i = 1)."""
    e = np.asarray(e, bool); t = np.asarray(t, float); r = np.asarray(r, float)
    i = np.flatnonzero(e)
    if not len(i):
        return np.nan
    num = den = 0.0
    for a in range(0, len(i), 512):
        ii = i[a:a + 512]
        comp = t[ii, None] < t[None, :]
        num += ((r[ii, None] > r[None, :]) & comp).sum() + 0.5 * ((r[ii, None] == r[None, :]) & comp).sum()
        den += comp.sum()
    return num / den if den else np.nan


def holdout(strata, seed, frac):
    rng = np.random.default_rng(seed); val = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s); rng.shuffle(idx)
        val[idx[:int(round(frac * len(idx)))]] = True
    return val


def breslow(eta_train, e, t):
    order = np.argsort(t); t_s, e_s, w = t[order], e[order], np.exp(eta_train[order] - eta_train.max())
    at_risk = np.cumsum(w[::-1])[::-1]
    ut = np.unique(t_s[e_s == 1])
    h = np.array([e_s[t_s == u].sum() / at_risk[np.searchsorted(t_s, u, 'left')] for u in ut])
    return ut, np.cumsum(h), eta_train.max()


def breslow_survival(base, eta, times):
    ut, H, shift = base
    pos = np.searchsorted(ut, times, 'right') - 1
    H0 = np.where(pos >= 0, H[np.clip(pos, 0, None)], 0.0)
    return np.exp(-np.exp(eta - shift)[:, None] * H0[None, :])


def aft_nll(mu, e, t, dist, sigma):
    z = (np.log(t) - mu) / sigma
    if dist == 'normal':
        ev, ce = -norm.logpdf(z) + np.log(sigma * t), -norm.logsf(z)
    elif dist == 'logistic':
        ev, ce = -(z - 2 * np.logaddexp(0, z)) + np.log(sigma * t), np.logaddexp(0, z)
    else:
        raise ValueError(dist)
    return np.where(e == 1, ev, ce)


def aft_init(e, t, cfg):
    """Censored constant-model MLE of the AFT location (log-months). XGBoost's default base_score=0.5 means an
    initial margin of log(0.5) ~ -0.69, i.e. ~0.5 months, far from the data; with early stopping this can
    freeze models near a constant. The fitted location is passed as base_score=exp(mu)."""
    f = lambda mu: aft_nll(np.full(len(t), mu), e, t, cfg['aft_loss_distribution'],
                           cfg['aft_loss_distribution_scale']).sum()
    lt = np.log(t)
    return float(minimize_scalar(f, bounds=(lt.min() - 2, lt.max() + 5), method='bounded').x)


def aft_survival(mu, times, dist, sigma):
    z = (np.log(np.maximum(times, 1e-6))[None, :] - mu[:, None]) / sigma
    return norm.sf(z) if dist == 'normal' else 1 / (1 + np.exp(np.clip(z, -50, 50)))


def step_eval(functions, times):
    """Evaluate sksurv step functions, clipping horizons to each function's domain."""
    out = []
    for f in functions:
        lo, hi = f.domain
        out.append(f(np.clip(times, lo, hi)))
    return np.vstack(out)


def surv_y(e, t):
    from sksurv.util import Surv
    return Surv.from_arrays(np.asarray(e, bool), np.asarray(t, float))


# ----------------------------------------------------------------------------- base

class Model:
    configs = [{}]
    early_stopping = False

    def __init__(self, cfg, seed, threads, P):
        self.cfg, self.seed, self.threads, self.P = cfg, seed, threads, P

    # fit(X,e,t,val=None) with val=(Xv,ev,tv) for early stopping; returns self and sets self.selected_iter
    def selected(self):
        return dict(self.cfg, **({'iterations': self.iterations} if hasattr(self, 'iterations') else {}))


def select_and_fit(cls, X, e, t, strata, seed, threads, P, configs=None):
    configs = configs or (cls.configs[:1] if P.get('smoke') else cls.configs)
    val = holdout(strata, seed, P['validation']['inner_holdout_fraction'])
    best, trace = None, []
    for cfg in configs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                m = cls(cfg, seed, threads, P).fit(X[~val], e[~val], t[~val], val=(X[val], e[val], t[val]),
                                                   selection=True)
                c = fast_cindex(e[val], t[val], m.risk(X[val]))
            except Exception as exc:          # recorded, never silently selected
                trace.append(dict(cfg=cfg, error=repr(exc)[:300])); continue
        it = getattr(m, 'iterations', None)
        trace.append(dict(cfg=cfg, inner_c=c, iterations=it))
        if np.isfinite(c) and (best is None or c > best[0]):
            best = (c, cfg, it)
    if best is None:
        raise RuntimeError(f'{cls.__name__}: no valid configuration: {trace}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final = cls(best[1], seed, threads, P).fit(X, e, t, iterations=best[2])
    final.selection = dict(inner_c=best[0], cfg=best[1], iterations=best[2], trace=trace)
    return final


# ----------------------------------------------------------------------------- linear / sksurv

class CoxPH(Model):
    configs = [dict(alpha=a) for a in (0.01, 0.1, 1.0, 10.0)]

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        self.sc = StandardScaler().fit(X)
        self.m = CoxPHSurvivalAnalysis(alpha=self.cfg['alpha'], ties='breslow', n_iter=200).fit(self.sc.transform(X), surv_y(e, t))
        return self

    def risk(self, X):
        return self.m.predict(self.sc.transform(X))

    def survival(self, X, times):
        return step_eval(self.m.predict_survival_function(self.sc.transform(X)), times)


class Coxnet(CoxPH):
    configs = [dict(l1_ratio=l, alpha=a) for l in (0.1, 0.5) for a in (0.02, 0.1)]

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        self.sc = StandardScaler().fit(X)
        self.m = CoxnetSurvivalAnalysis(l1_ratio=self.cfg['l1_ratio'], alphas=[self.cfg['alpha']],
                                        fit_baseline_model=not selection, max_iter=100000)
        self.m.fit(self.sc.transform(X), surv_y(e, t))
        return self


class Forest(Model):
    configs = [dict(min_samples_leaf=l, max_features=f) for l in (10, 25) for f in ('sqrt', 0.3)]
    kind = 'rsf'

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
        cls = RandomSurvivalForest if self.kind == 'rsf' else ExtraSurvivalTrees
        self.m = cls(n_estimators=self.P['models']['forest_trees'], min_samples_split=2 * self.cfg['min_samples_leaf'],
                     random_state=self.seed, n_jobs=self.threads, low_memory=selection, **self.cfg)
        self.m.fit(X, surv_y(e, t))
        return self

    def risk(self, X):
        return self.m.predict(X)

    def survival(self, X, times):
        return step_eval(self.m.predict_survival_function(X), times)


class ExtraTrees(Forest):
    kind = 'et'


class GBMCox(Model):
    configs = [dict(max_depth=d, n_estimators=n) for d in (1, 2) for n in (200, 600)]

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        self.m = GradientBoostingSurvivalAnalysis(loss='coxph', learning_rate=0.05, subsample=0.8,
                                                  min_samples_leaf=10, random_state=self.seed, **self.cfg)
        self.m.fit(X, surv_y(e, t))
        return self

    def risk(self, X):
        return self.m.predict(X)

    def survival(self, X, times):
        return step_eval(self.m.predict_survival_function(X), times)


# ----------------------------------------------------------------------------- XGBoost

class XGB(Model):
    early_stopping = True

    def _params(self, base=None, **extra):
        p = dict(tree_method='hist', learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                 min_child_weight=10, seed=self.seed, nthread=self.threads, verbosity=0)
        p.update(self.cfg); p.update(extra)
        if base is not None:
            p['base_score'] = base
        return p

    def _train(self, params, dtr, val, iterations, metric_sign):
        import xgboost as xgb
        P = self.P['models']
        if iterations is not None:
            return xgb.train(params, dtr, int(iterations)), int(iterations)
        dva = self._dmat(*val)
        ev, tv = val[1], val[2]

        def metric(pred, _):
            return 'c', fast_cindex(ev, tv, metric_sign * pred)
        b = xgb.train(params, dtr, P['xgb_max_rounds'], evals=[(dva, 'v')], custom_metric=metric, maximize=True,
                      early_stopping_rounds=P['xgb_patience'], verbose_eval=False)
        return b, b.best_iteration + 1


class XGBAFT(XGB):
    configs = [dict(max_depth=d, reg_lambda=l, aft_loss_distribution=dist, aft_loss_distribution_scale=1.0)
               for d, l, dist in ((2, 10, 'normal'), (2, 30, 'logistic'), (3, 10, 'normal'), (3, 30, 'logistic'))]

    def _dmat(self, X, e, t, margin=None):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        d.set_float_info('label_lower_bound', t); d.set_float_info('label_upper_bound', np.where(e == 1, t, np.inf))
        if margin is not None:
            d.set_base_margin(margin)
        return d

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        self.base = aft_init(e, t, self.cfg)
        # XGBoost's survival:aft takes base_score on the TIME scale (initial margin = log(base_score)).
        self.b, self.iterations = self._train(self._params(float(np.exp(self.base)), objective='survival:aft'),
                                              self._dmat(X, e, t), val, iterations, -1)
        return self

    def mu(self, X):
        import xgboost as xgb
        return self.b.predict(xgb.DMatrix(X), output_margin=True)

    def risk(self, X):
        return -self.mu(X)

    def survival(self, X, times):
        return aft_survival(self.mu(X), times, self.cfg['aft_loss_distribution'], self.cfg['aft_loss_distribution_scale'])


class XGBCox(XGB):
    configs = [dict(max_depth=d, min_child_weight=c, reg_lambda=l) for d, c, l in ((2, 10, 10), (2, 20, 30), (3, 10, 10), (3, 20, 30))]

    def _dmat(self, X, e, t):
        import xgboost as xgb
        d = xgb.DMatrix(X); d.set_label(np.where(e == 1, t, -t)); return d

    def fit(self, X, e, t, val=None, selection=False, iterations=None):
        self.b, self.iterations = self._train(self._params(objective='survival:cox'), self._dmat(X, e, t), val, iterations, 1)
        if not selection:
            self.base = breslow(self.risk(X), e, t)
        return self

    def risk(self, X):
        import xgboost as xgb
        return self.b.predict(xgb.DMatrix(X), output_margin=True)

    def survival(self, X, times):
        return breslow_survival(self.base, self.risk(X), times)


# ----------------------------------------------------------------------------- DeepSurv
# DeepSurv runs in an isolated child process (see deepsurv_worker.py); torch is never imported here.


# ----------------------------------------------------------------------------- proposed: K2

class K2:
    """Knowledge-first two-stage XGBoost-AFT.
    Stage 1: XGB-AFT on clinical + biological module scores (selected exactly like XGBAFT).
    Stage 2: boosting continues from the stage-1 margin (base_margin) on cohort-standardized gene ranks,
    with stronger regularization; dropped if it does not improve the inner-holdout C-index."""

    def __init__(self, seed, threads, P):
        self.seed, self.threads, self.P = seed, threads, P

    def fit(self, Xm, Xg, e, t, strata):
        import xgboost as xgb
        P, K = self.P, self.P['k2']
        val = holdout(strata, self.seed, P['validation']['inner_holdout_fraction'])
        s1 = select_and_fit(XGBAFT, Xm, e, t, strata, self.seed, self.threads, P)
        cfg, r1 = s1.selection['cfg'], s1.selection['iterations']
        inner1 = XGBAFT(cfg, self.seed, self.threads, P).fit(Xm[~val], e[~val], t[~val], iterations=r1)
        m_tr, m_va = inner1.mu(Xm[~val]), inner1.mu(Xm[val])
        c1 = fast_cindex(e[val], t[val], -m_va)
        best, trace = (c1, None, 0), []
        for lam in K['stage2_reg_lambda']:
            st = XGBAFT(dict(cfg), self.seed, self.threads, P)
            params = st._params(None, objective='survival:aft', max_depth=K['stage2_depth'],
                                learning_rate=K['stage2_learning_rate'], colsample_bytree=K['stage2_colsample'],
                                subsample=K['stage2_subsample'], min_child_weight=K['stage2_min_child_weight'],
                                reg_lambda=lam)
            params.pop('base_score', None)
            dva = st._dmat(Xg[val], e[val], t[val], m_va); ev, tv = e[val], t[val]

            def metric(pred, _):
                return 'c', fast_cindex(ev, tv, -pred)
            b = xgb.train(params, st._dmat(Xg[~val], e[~val], t[~val], m_tr), P['models']['xgb_max_rounds'],
                          evals=[(dva, 'v')], custom_metric=metric, maximize=True,
                          early_stopping_rounds=P['models']['xgb_patience'], verbose_eval=False)
            trace.append(dict(reg_lambda=lam, inner_c=b.best_score, rounds=b.best_iteration + 1))
            if b.best_score > best[0] or not K['stage2_drop_if_no_inner_gain']:
                best = (b.best_score, lam, b.best_iteration + 1); self._params2 = params
        self.stage1, self.cfg = s1, cfg
        self.stage2 = None
        if best[2] > 0:
            p = dict(self._params2); p['reg_lambda'] = best[1]
            self.stage2 = xgb.train(p, s1._dmat(Xg, e, t, s1.mu(Xm)), best[2])
        self.selection = dict(stage1=s1.selection, stage1_inner_c=c1, stage2_trace=trace,
                              stage2_reg_lambda=best[1], stage2_rounds=best[2], inner_c=best[0])
        return self

    def mu(self, Xm, Xg):
        import xgboost as xgb
        m = self.stage1.mu(Xm)
        if self.stage2 is None:
            return m
        d = xgb.DMatrix(Xg); d.set_base_margin(m)
        return self.stage2.predict(d, output_margin=True)

    def risk(self, Xm, Xg):
        return -self.mu(Xm, Xg)

    def survival(self, Xm, Xg, times):
        return aft_survival(self.mu(Xm, Xg), times, self.cfg['aft_loss_distribution'], self.cfg['aft_loss_distribution_scale'])


REGISTRY = dict(coxph_clinical=CoxPH, coxnet=Coxnet, rsf=Forest, extra_trees=ExtraTrees, gbm_cox=GBMCox,
                xgb_cox=XGBCox, xgb_aft=XGBAFT)
