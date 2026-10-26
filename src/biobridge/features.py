"""Feature construction. Outcomes are never read here.

Every cohort chunk carries a label-free reference matrix (its own training patients, or the whole cohort when
it is a transfer/external target). Module scores and genes are standardized within cohort with that reference.
"""
from dataclasses import dataclass
import numpy as np, pandas as pd
from .utils import CONFIG, read_json

CLINICAL = ['AGE_AT_DIAGNOSIS', 'NODE_POSITIVE']
CLIN_NAMES = ['age', 'age_missing', 'node', 'node_missing']


@dataclass
class Chunk:
    rows: pd.DataFrame
    ranks: np.ndarray          # patients x universe (may be a memmap)
    ref: np.ndarray            # label-free reference rows (same cohort) for standardization


class GeneSets:
    def __init__(self, universe):
        spec = read_json(CONFIG / 'genesets_frozen.json')
        pos = {str(g).upper(): j for j, g in enumerate(universe)}       # prepare.collapse upper-cases symbols
        self.sets, self.coverage = {}, {}
        for name, entries in spec['sets'].items():
            idx, sign = [], []
            for e in entries:
                hit = next((s.upper() for s in e['symbols'] if s.upper() in pos), None)
                if hit is not None and pos[hit] not in idx:
                    idx.append(pos[hit]); sign.append(e['sign'])
            self.coverage[name] = (len(idx), len(entries))
            if len(idx) >= 2:
                self.sets[name] = (np.array(idx), np.array(sign) > 0)
        p_idx, cen = [], []
        for e in spec['pam50']['genes']:
            hit = next((s.upper() for s in e['symbols'] if s.upper() in pos), None)
            if hit is not None:
                p_idx.append(pos[hit]); cen.append(e['centroid'])
        self.pam_idx, self.pam_cen, self.subtypes = np.array(p_idx), np.array(cen), spec['pam50']['subtypes']
        self.coverage['pam50'] = (len(p_idx), len(spec['pam50']['genes']))
        self.module_names = list(self.sets) + ['pam50_' + s for s in self.subtypes]
        self.all_idx = np.unique(np.concatenate([i for i, _ in self.sets.values()] + [self.pam_idx]))

    def raw_scores(self, ranks):
        out = []
        for idx, pos in self.sets.values():
            a = np.asarray(ranks[:, idx], np.float32)
            a = np.where(pos[None, :], a, 1 - a)
            out.append(np.nanmean(a, axis=1))
        return np.column_stack(out)

    def pam50(self, ranks, ref):
        center = np.nanmedian(np.asarray(ref[:, self.pam_idx], np.float32), axis=0)
        a = np.asarray(ranks[:, self.pam_idx], np.float32) - center
        a = np.where(np.isfinite(a), a, 0.0)
        ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
        rc = np.argsort(np.argsort(self.pam_cen, axis=0), axis=0).astype(float)
        ra -= ra.mean(1, keepdims=True); rc -= rc.mean(0, keepdims=True)
        return (ra @ rc) / np.sqrt((ra ** 2).sum(1, keepdims=True) * (rc ** 2).sum(0, keepdims=True))

    def modules(self, chunk, standardize=True):
        m = self.raw_scores(chunk.ranks)
        if standardize:
            r = self.raw_scores(chunk.ref)
            sd = np.nanstd(r, 0); sd[~(sd > 0)] = 1
            m = (m - np.nanmean(r, 0)) / sd
        m = np.where(np.isfinite(m), m, 0.0)
        return np.column_stack([m, self.pam50(chunk.ranks, chunk.ref)])


def clinical(chunks, train_rows):
    med = {c: np.nanmedian(pd.to_numeric(train_rows[c], errors='coerce').to_numpy(float)) for c in CLINICAL}
    out = []
    for ch in chunks:
        cols = []
        for c in CLINICAL:
            v = pd.to_numeric(ch.rows[c], errors='coerce').to_numpy(float)
            cols += [np.where(np.isnan(v), med[c], v), np.isnan(v).astype(float)]
        out.append(np.column_stack(cols))
    return out


def top_variance(train_chunks, k):
    """Top-k genes by variance across all training patients (pooled), computed without outcomes."""
    n = sum(len(c.rows) for c in train_chunks)
    s1 = s2 = cnt = 0
    for c in train_chunks:
        a = np.asarray(c.ranks, np.float32)
        fin = np.isfinite(a)
        s1 = s1 + np.where(fin, a, 0).sum(0); s2 = s2 + np.where(fin, a * a, 0).sum(0); cnt = cnt + fin.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        var = s2 / cnt - (s1 / cnt) ** 2
    var[(cnt < 0.9 * n) | ~np.isfinite(var)] = -1        # require >=90% observed in training
    return np.sort(np.argsort(var, kind='stable')[-k:])


def genes(chunk, idx, standardize=True):
    a = np.asarray(chunk.ranks[:, idx], np.float32)
    ref = np.asarray(chunk.ref[:, idx], np.float32)
    med = np.nanmedian(ref, 0); med = np.where(np.isfinite(med), med, 0.5)
    a = np.where(np.isfinite(a), a, med)
    if standardize:
        ref = np.where(np.isfinite(ref), ref, med)
        sd = ref.std(0); sd[~(sd > 0)] = 1
        a = (a - ref.mean(0)) / sd
    return a


class Design:
    """Builds all representations for one analysis unit (train chunks -> test chunk)."""

    def __init__(self, train, test, gs, n_genes, standardize=True):
        self.names = {}
        rows_tr = pd.concat([c.rows for c in train], ignore_index=True)
        clin = clinical(train + [test], rows_tr)
        mods = [gs.modules(c, standardize) for c in train + [test]]
        self.gene_idx = top_variance(train, n_genes)
        gen = [genes(c, self.gene_idx, standardize) for c in train + [test]]
        stack = lambda blocks: (np.vstack(blocks[:-1]).astype(np.float32), blocks[-1].astype(np.float32))
        self.clin_tr, self.clin_te = stack(clin)
        self.mod_tr, self.mod_te = stack(mods)
        self.gen_tr, self.gen_te = stack(gen)
        self.module_names = gs.module_names

    def get(self, rep):
        if rep == 'clinical':
            return self.clin_tr, self.clin_te, CLIN_NAMES
        if rep == 'modules':
            return (np.hstack([self.clin_tr, self.mod_tr]), np.hstack([self.clin_te, self.mod_te]),
                    CLIN_NAMES + self.module_names)
        if rep == 'genes':
            return (np.hstack([self.clin_tr, self.gen_tr]), np.hstack([self.clin_te, self.gen_te]),
                    CLIN_NAMES + [f'g{i}' for i in self.gene_idx])
        raise ValueError(rep)
