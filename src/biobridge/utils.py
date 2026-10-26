import hashlib, json, os, sys, time, platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config'
DATA = ROOT / 'data'
RAW = DATA / 'raw'
# Environment overrides exist only for synthetic functional tests (tools/make_synthetic.py).
PREPARED = Path(os.environ.get('BIOBRIDGE_PREPARED', DATA / 'prepared')).resolve()
RESULTS = Path(os.environ.get('BIOBRIDGE_RESULTS', ROOT / 'results')).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    raise TypeError(type(o))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(value, indent=2, default=_default))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.part.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_npy(path, array):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.part.npy')
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def load_protocol():
    import yaml
    return yaml.safe_load((CONFIG / 'protocol.yaml').read_text())


def code_files():
    files = sorted((ROOT / 'src' / 'biobridge').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    return [CONFIG / 'protocol.yaml', CONFIG / 'genesets_frozen.json', CONFIG / 'sources.json'] + files


def code_fingerprint():
    return {str(p.relative_to(ROOT)): sha256(p) for p in code_files()}


def verify_lock(smoke=False):
    """Every analysis script calls this: protocol + code must match the locked versions."""
    if smoke:
        return None
    lock = ROOT / 'PROTOCOL_LOCK.json'
    if not lock.exists():
        raise SystemExit('PROTOCOL_LOCK.json yok. Önce: python scripts/00_lock_protocol.py')
    saved = read_json(lock)
    now = code_fingerprint()
    changed = sorted(k for k in set(saved['files']) | set(now) if saved['files'].get(k) != now.get(k))
    if changed:
        raise SystemExit('Kilitli protokolden sonra değişen dosyalar: ' + ', '.join(changed) +
                         '\nDeğişiklik bilinçliyse gerekçesiyle: '
                         'python scripts/00_lock_protocol.py --amend "gerekçe"')
    return saved


def environment():
    import importlib
    out = dict(python=sys.version.split()[0], platform=platform.platform())
    for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'sksurv', 'xgboost', 'torch']:
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = None
    return out
