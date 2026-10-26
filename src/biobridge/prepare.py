"""Raw tables -> per-cohort patients.csv + within-patient percentile ranks on a common gene universe.

Loaders are adapted from the pilot's verified 06_prepare_full.py. Nothing here looks at survival outcomes
except the basic eligibility rule (valid OS time > 0 and status in {0,1}), applied identically to all cohorts.
"""
import csv, gzip, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from .utils import RAW, PREPARED, atomic_json, atomic_npy, read_json, sha256, log

COLUMNS = ['PATIENT_ID', 'SAMPLE_ID', 'time', 'event', 'AGE_AT_DIAGNOSIS', 'NODE_POSITIVE', 'SITE']


def geo_metadata(path):
    titles, ids, characteristics = None, None, []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            vals = next(csv.reader([line], delimiter='\t'))
            if not vals:
                continue
            if vals[0] == '!Sample_title':
                titles = vals[1:]
            elif vals[0] == '!Sample_geo_accession':
                ids = vals[1:]
            elif vals[0] == '!Sample_characteristics_ch1':
                characteristics.append(vals[1:])
    if titles is None or ids is None or len(titles) != len(ids):
        raise RuntimeError(f'Invalid GEO metadata: {path}')
    records = [dict(SAMPLE_ID=t, GEO_ID=g) for t, g in zip(titles, ids)]
    for values in characteristics:
        if len(values) != len(records):
            raise RuntimeError('GEO characteristics length mismatch')
        for rec, text in zip(records, values):
            if ': ' in text:
                k, v = text.split(': ', 1)
                if k in rec and rec[k] != v:
                    raise RuntimeError(f'Conflicting GEO characteristic {k}')
                rec[k] = v
    return pd.DataFrame(records)


def node_stage(value):
    s = str(value).strip().upper()
    if re.match(r'^N?0(?:\b|$|\s|\()', s):
        return 0.0
    if re.match(r'^N?[123]', s):
        return 1.0
    return np.nan


def gpl570_mapping():
    with gzip.open(RAW / 'gpl570.annot.gz', 'rt') as f:
        for line in f:
            if line.startswith('ID\t'):
                header = next(csv.reader([line], delimiter='\t')); break
        else:
            raise RuntimeError('GPL570 annotation header missing')
        pos = header.index('Gene symbol'); mapping, ambiguous = {}, 0
        for vals in csv.reader(f, delimiter='\t'):
            if len(vals) <= pos or vals[0].startswith('!'):
                continue
            syms = {s.strip().upper() for s in vals[pos].split('///') if s.strip() not in {'', '---'}}
            if len(syms) == 1:
                mapping[vals[0]] = next(iter(syms))
            else:
                ambiguous += 1
    return mapping, ambiguous


def collapse(x, symbols):
    """Mean of replicate probes per gene, per patient (no cross-patient information)."""
    groups = {}
    for i, s in enumerate(symbols):
        s = str(s).strip().upper()
        if s not in {'', 'NAN', 'NONE', '---'}:
            groups.setdefault(s, []).append(i)
    genes = np.array(sorted(groups), dtype=str)
    rr, cc = [], []
    for j, s in enumerate(genes):
        rr += groups[s]; cc += [j] * len(groups[s])
    m = sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(x.shape[1], len(genes)))
    out = np.empty((len(x), len(genes)), np.float32)
    for a0 in range(0, len(x), 128):
        a = x[a0:a0 + 128]
        num = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) @ m
        den = np.isfinite(a).astype(np.float32) @ m
        out[a0:a0 + len(a)] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out, genes


def read_cbio_expression(path):
    header = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
    blocks, symbols = [], []
    for chunk in pd.read_csv(path, sep='\t', dtype={k: np.float32 for k in header[2:]}, chunksize=1000):
        blocks.append(chunk.iloc[:, 2:].to_numpy(np.float32))
        symbols += chunk.iloc[:, 0].fillna('').astype(str).tolist()
    x = np.concatenate(blocks).T.copy(); x[~np.isfinite(x)] = np.nan
    return x, np.array(header[2:], dtype=str), symbols


def read_geo_expression(path, mapping):
    blocks, symbols = [], []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
        header = next(csv.reader([next(f)], delimiter='\t'))
        rows, names = [], []
        for line in f:
            if line.startswith('!series_matrix_table_end'):
                break
            vals = next(csv.reader([line], delimiter='\t'))
            sym = mapping.get(vals[0]) if vals else None
            if sym is None:
                continue
            rows.append([np.nan if v.lower() in {'null', 'na', 'nan', ''} else float(v) for v in vals[1:]])
            names.append(sym)
            if len(rows) == 1000:
                blocks.append(np.array(rows, np.float32)); symbols += names; rows, names = [], []
        if rows:
            blocks.append(np.array(rows, np.float32)); symbols += names
    return np.concatenate(blocks).T.copy(), np.array(header[1:], dtype=str), symbols


def align(rows, x, ids):
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate expression sample IDs')
    valid = rows.event.isin([0, 1]) & np.isfinite(rows.time) & (rows.time > 0)
    audit = dict(invalid_os_excluded=int((~valid).sum()))
    rows = rows.loc[valid]
    lookup = {s: j for j, s in enumerate(ids)}
    audit['valid_os_without_expression'] = int((~rows.SAMPLE_ID.isin(lookup)).sum())
    rows = rows.loc[rows.SAMPLE_ID.isin(lookup)].sort_values('PATIENT_ID').reset_index(drop=True)
    if rows.PATIENT_ID.duplicated().any():
        raise RuntimeError('Multiple samples per patient remain')
    x = x[[lookup[s] for s in rows.SAMPLE_ID]]
    rows['event'] = rows.event.astype(int)
    return rows, x, audit


def _status(series):
    return pd.to_numeric(series.astype(str).str.split(':').str[0], errors='coerce')


def load_metabric():
    p = pd.read_csv(RAW / 'metabric_patient.txt', sep='\t', comment='#')
    s = pd.read_csv(RAW / 'metabric_sample.txt', sep='\t', comment='#')
    rows = s.merge(p, on='PATIENT_ID', validate='one_to_one')
    rows['time'] = pd.to_numeric(rows.OS_MONTHS, errors='coerce'); rows['event'] = _status(rows.OS_STATUS)
    rows['AGE_AT_DIAGNOSIS'] = pd.to_numeric(rows.AGE_AT_DIAGNOSIS, errors='coerce')
    n = pd.to_numeric(rows.LYMPH_NODES_EXAMINED_POSITIVE, errors='coerce')
    rows['NODE_POSITIVE'] = np.where(n.notna() & (n >= 0), (n > 0).astype(float), np.nan)
    rows['SITE'] = 'MB' + pd.to_numeric(rows.COHORT, errors='coerce').fillna(0).astype(int).astype(str)
    x, ids, sym = read_cbio_expression(RAW / 'metabric_expression.txt')
    rows, x, audit = align(rows, x, ids)
    x, genes = collapse(x, sym)
    return rows, x, genes, audit


def load_tcga():
    p = pd.read_csv(RAW / 'tcga_patient.txt', sep='\t', comment='#')
    s = pd.read_csv(RAW / 'tcga_sample.txt', sep='\t', comment='#')
    s = s[s.SAMPLE_TYPE.astype(str).str.lower().eq('primary')].sort_values('SAMPLE_ID')
    dup = int(s.PATIENT_ID.duplicated().sum()); s = s.drop_duplicates('PATIENT_ID', keep='first')
    rows = s[['PATIENT_ID', 'SAMPLE_ID']].merge(p, on='PATIENT_ID', validate='one_to_one')
    rows['time'] = pd.to_numeric(rows.OS_MONTHS, errors='coerce'); rows['event'] = _status(rows.OS_STATUS)
    rows['AGE_AT_DIAGNOSIS'] = pd.to_numeric(rows.AGE, errors='coerce')
    rows['NODE_POSITIVE'] = rows.PATH_N_STAGE.map(node_stage)
    rows['SITE'] = 'TS' + rows.PATIENT_ID.astype(str).str.slice(5, 7)
    x, ids, sym = read_cbio_expression(RAW / 'tcga_expression.txt')
    rows, x, audit = align(rows, x, ids); audit['extra_primary_samples_excluded'] = dup
    x, genes = collapse(x, sym)
    return rows, x, genes, audit


def load_scanb():
    rows = pd.concat([geo_metadata(RAW / 'scanb_gpl11154.txt.gz'), geo_metadata(RAW / 'scanb_gpl18573.txt.gz')],
                     ignore_index=True)
    rep = rows.SAMPLE_ID.str.contains('repl', case=False, regex=False)
    rows = rows.loc[~rep].copy()
    if not rows.SAMPLE_ID.str.fullmatch(r'F\d+').all():
        raise RuntimeError('Unexpected SCAN-B sample IDs; inspect rather than guess')
    rows['PATIENT_ID'] = rows.SAMPLE_ID
    rows['time'] = pd.to_numeric(rows['overall survival days'], errors='coerce') / (365.25 / 12)
    rows['event'] = pd.to_numeric(rows['overall survival event'], errors='coerce')
    rows['AGE_AT_DIAGNOSIS'] = pd.to_numeric(rows['age at diagnosis'], errors='coerce')
    rows['NODE_POSITIVE'] = rows['lymph node status'].map({'NodeNegative': 0.0, 'NodePositive': 1.0})
    rows['SITE'] = 'SCANB'
    header = pd.read_csv(RAW / 'scanb_expression.csv.gz', nrows=0).columns.tolist()
    blocks, sym = [], []
    for ch in pd.read_csv(RAW / 'scanb_expression.csv.gz', chunksize=1000, dtype={k: np.float32 for k in header[1:]}):
        blocks.append(ch.iloc[:, 1:].to_numpy(np.float32)); sym += ch.iloc[:, 0].fillna('').astype(str).tolist()
    x = np.concatenate(blocks).T.copy()
    rows, x, audit = align(rows, x, np.array(header[1:], dtype=str)); audit['technical_replicates_excluded'] = int(rep.sum())
    x, genes = collapse(x, sym)
    return rows, x, genes, audit


def load_gse20685():
    rows = geo_metadata(RAW / 'gse20685_matrix.txt.gz')
    rows['PATIENT_ID'] = rows.SAMPLE_ID
    rows['time'] = pd.to_numeric(rows['follow_up_duration (years)'], errors='coerce') * 12
    rows['event'] = pd.to_numeric(rows['event_death'], errors='coerce')
    rows['AGE_AT_DIAGNOSIS'] = pd.to_numeric(rows['age at diagnosis'], errors='coerce')
    rows['NODE_POSITIVE'] = rows['n_stage'].map(node_stage)
    rows['SITE'] = 'GSE20685'
    mapping, amb = gpl570_mapping()
    x, ids, sym = read_geo_expression(RAW / 'gse20685_matrix.txt.gz', mapping)
    rows['SAMPLE_ID'] = rows.GEO_ID
    rows, x, audit = align(rows, x, ids); audit['gpl570_ambiguous_probes_excluded'] = amb
    x, genes = collapse(x, sym)
    return rows, x, genes, audit


def load_custom(name):
    """Generic external cohort: data/raw/custom/<name>/clinical.csv + expression.csv.

    clinical.csv: PATIENT_ID, time_months, event (0/1), age, node_positive (0/1/blank)
    expression.csv: first column gene symbol, remaining columns PATIENT_ID (any scale; ranks are used)."""
    folder = RAW / 'custom' / name
    c = pd.read_csv(folder / 'clinical.csv')
    rows = pd.DataFrame(dict(PATIENT_ID=c.PATIENT_ID.astype(str), SAMPLE_ID=c.PATIENT_ID.astype(str),
                             time=pd.to_numeric(c.time_months, errors='coerce'),
                             event=pd.to_numeric(c.event, errors='coerce'),
                             AGE_AT_DIAGNOSIS=pd.to_numeric(c.age, errors='coerce'),
                             NODE_POSITIVE=pd.to_numeric(c.node_positive, errors='coerce'), SITE=name))
    e = pd.read_csv(folder / 'expression.csv')
    x = e.iloc[:, 1:].to_numpy(np.float32).T
    rows, x, audit = align(rows, x, np.array(e.columns[1:], dtype=str))
    x, genes = collapse(x, e.iloc[:, 0].astype(str).tolist())
    return rows, x, genes, audit


LOADERS = dict(metabric=load_metabric, tcga=load_tcga, scanb=load_scanb, gse20685=load_gse20685)


def percentile_ranks(x, genes, universe):
    """Within-patient percentile ranks over the universe; genes absent from a cohort stay NaN."""
    lookup = {g: j for j, g in enumerate(genes)}
    out = np.full((len(x), len(universe)), np.nan, np.float32)
    present = np.array([g in lookup for g in universe])
    idx = np.array([lookup[g] for g in np.asarray(universe)[present]], dtype=int)
    for a0 in range(0, len(x), 64):
        a = np.asarray(x[a0:a0 + 64][:, idx], np.float32)
        fin = np.isfinite(a); a[~fin] = np.nan
        den = np.maximum(fin.sum(1, keepdims=True), 1)
        r = rankdata(a, axis=1, method='average', nan_policy='omit') / den
        block = np.full((len(a), len(universe)), np.nan, np.float32); block[:, present] = r
        out[a0:a0 + 64] = block
    return out, int(present.sum())


def save_cohort(name, rows, ranks, audit):
    folder = PREPARED / name
    rows[COLUMNS].to_csv(folder / 'patients.csv', index=False)
    atomic_npy(folder / 'ranks.npy', ranks)
    audit.update(patients=len(rows), deaths=int(rows.event.sum()), median_followup_months=float(rows.time.median()),
                 missing_age=float(rows.AGE_AT_DIAGNOSIS.isna().mean()),
                 missing_node=float(rows.NODE_POSITIVE.isna().mean()))
    atomic_json(folder / 'prepared.json', dict(audit=audit, files={n: sha256(folder / n)
                                                                    for n in ['patients.csv', 'ranks.npy']}))
    log('PREPARED', name, audit)


def load_prepared(name):
    folder = PREPARED / name
    meta = read_json(folder / 'prepared.json')
    for n, h in meta['files'].items():
        if sha256(folder / n) != h:
            raise SystemExit(f'{folder / n} hazırlandıktan sonra değişmiş.')
    rows = pd.read_csv(folder / 'patients.csv', dtype={'PATIENT_ID': str, 'SAMPLE_ID': str, 'SITE': str})
    return rows, np.load(folder / 'ranks.npy', mmap_mode='r')


def universe():
    return np.array(read_json(PREPARED / 'universe.json')['genes'], dtype=str)
