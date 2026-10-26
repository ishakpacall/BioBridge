"""Build config/genesets_frozen.json from genefu (bhklab/genefu, data/*.rda) and HGNC.

Already run once by the project author; the frozen JSON ships with the project. Re-running is only
for reproducibility checks (needs: pip install rdata). Outcome data are never used here.
"""
import io, json, urllib.request, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENEFU = 'https://raw.githubusercontent.com/bhklab/genefu/master/data/'
HGNC = ('https://raw.githubusercontent.com/cBioPortal/datahub-study-curation-tools/master/'
        'gene-table-update/build-input-for-importer/hgnc_complete_set.txt')


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    import rdata, tempfile
    sources, blobs = {}, {}
    for name in ['mod1', 'sig.ggi', 'sig.oncotypedx', 'pam50']:
        blobs[name] = fetch(GENEFU + name + '.rda'); sources[name] = hashlib.sha256(blobs[name]).hexdigest()
    hg = fetch(HGNC); sources['hgnc_complete_set'] = hashlib.sha256(hg).hexdigest()
    h = pd.read_csv(io.BytesIO(hg), sep='\t', dtype=str, usecols=['symbol', 'entrez_id', 'prev_symbol', 'alias_symbol'])

    official = set(h.symbol.dropna())

    def cands(r):
        # Own approved symbol first; previous/alias symbols only if they are not another gene's approved symbol.
        c = [r.symbol]
        for col in ['prev_symbol', 'alias_symbol']:
            if isinstance(r[col], str):
                c += [s.strip() for s in r[col].replace('"', '').split('|')
                      if s.strip() and s.strip() not in official and s.strip() not in c]
        return c
    by_entrez, by_symbol = {}, {}
    for _, r in h.iterrows():
        cs = cands(r)
        if isinstance(r.entrez_id, str):
            by_entrez[str(int(float(r.entrez_id)))] = cs
        for s in cs:
            by_symbol.setdefault(s, cs)

    def read(name):
        with tempfile.NamedTemporaryFile(suffix='.rda') as f:
            f.write(blobs[name]); f.flush()
            return rdata.read_rda(f.name)[name]

    def ent(x):
        return str(x).replace('.0', '')
    sets = {}
    for k, v in read('mod1').items():
        g = {}
        for e, c in zip(v['EntrezGene.ID'], v['coefficient']):
            if ent(e) in by_entrez:
                g.setdefault(ent(e), []).append(float(c))
        sets['mod_' + str(k)] = [dict(entrez=e, symbols=by_entrez[e], sign=float(np.sign(np.mean(c))))
                                 for e, c in g.items() if np.mean(c) != 0]
    ggi = read('sig.ggi'); g = {}
    for e, gr in zip(ggi['EntrezGene.ID'], ggi['grade']):
        if ent(e) in by_entrez:
            g.setdefault(ent(e), []).append(1.0 if float(gr) == 3 else -1.0)
    sets['ggi'] = [dict(entrez=e, symbols=by_entrez[e], sign=float(np.sign(np.mean(c)))) for e, c in g.items() if np.mean(c) != 0]
    onc = read('sig.oncotypedx')
    for grp in ['proliferation', 'her2', 'estrogen']:
        sub = onc[onc.group == grp]
        sets['onc_' + grp] = [dict(entrez=ent(e), symbols=by_entrez.get(ent(e), [s]), sign=1.0)
                              for e, s in zip(sub['EntrezGene.ID'], sub['symbol'])]
    p = read('pam50')['centroids']
    cen = pd.DataFrame(np.asarray(p.values), index=[str(s) for s in p.coords['dim_0'].values],
                       columns=[str(s) for s in p.coords['dim_1'].values])
    pam = dict(subtypes=cen.columns.tolist(),
               genes=[dict(symbols=by_symbol.get(s, [s]), centroid=cen.loc[s].round(6).tolist()) for s in cen.index])
    out = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               sources=dict(genefu=GENEFU, hgnc=HGNC, sha256=sources),
               scoring='signed mean of within-patient percentile ranks; probes collapsed to genes by mean coefficient sign',
               sets=sets, pam50=pam)
    (ROOT / 'config' / 'genesets_frozen.json').write_text(json.dumps(out, indent=1))
    print({k: len(v) for k, v in sets.items()}, 'pam50', len(pam['genes']))


if __name__ == '__main__':
    main()
