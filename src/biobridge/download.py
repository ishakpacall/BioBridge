"""Resumable HTTPS downloads with hash enforcement (trust-on-first-download for unpinned sources)."""
import ssl, time, urllib.request, urllib.error
from pathlib import Path
from .utils import RAW, CONFIG, sha256, read_json, atomic_json, log

LOCK = RAW / 'DATA_LOCK.json'


def _fetch(url, dest, retries=5):
    part = dest.with_name(dest.name + '.part')
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'BioBridge/2.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
                append = offset > 0 and r.status == 206
                total = r.headers.get('Content-Length')
                total = (int(total) + (offset if append else 0)) if total else None
                done = offset if append else 0
                last = time.monotonic()
                with part.open('ab' if append else 'wb') as out:
                    for block in iter(lambda: r.read(1 << 20), b''):
                        out.write(block); done += len(block)
                        if time.monotonic() - last > 15:
                            log(f'  {dest.name}: {done / 1e6:.0f}' + (f'/{total / 1e6:.0f}' if total else '') + ' MB')
                            last = time.monotonic()
            if total and part.stat().st_size != total:
                raise ConnectionError('incomplete download')
            part.replace(dest)
            return
        except (ssl.SSLCertVerificationError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if 'CERTIFICATE_VERIFY_FAILED' in str(exc):
                raise SystemExit('macOS sertifika hatası: "/Applications/Python 3.x/Install Certificates.command" '
                                 'çalıştırın veya: pip install certifi') from exc
            if attempt + 1 == retries:
                raise
            log(f'RETRY {attempt + 1} {dest.name}: {type(exc).__name__}')
            time.sleep(3 * (attempt + 1))


def sources():
    return read_json(CONFIG / 'sources.json')['sources']


def import_legacy(path):
    """Take exact URL/hash pairs from the pilot's full_data_sources.json (matched by file basename)."""
    legacy = read_json(path)['sources']
    by_base = {Path(it['relative_path']).name: it for it in legacy}
    out = {}
    for s in sources():
        it = by_base.get(s['file'])
        if it:
            out[s['name']] = dict(url=it['url'], sha256=it['sha256'])
    log('Legacy kaynaklardan eşleşen:', ', '.join(out) or '(yok)')
    return out


def download_all(only=None, legacy=None):
    RAW.mkdir(parents=True, exist_ok=True)
    lock = read_json(LOCK) if LOCK.exists() else {}
    for s in sources():
        if only and s['name'] not in only:
            continue
        url, expected = s['url'], s['sha256']
        if legacy and s['name'] in legacy:
            url, expected = legacy[s['name']]['url'], legacy[s['name']]['sha256']
        expected = expected or lock.get(s['name'], {}).get('sha256')
        dest = RAW / s['file']
        if not dest.exists():
            log('DOWNLOAD', s['name'], url)
            _fetch(url, dest)
        digest = sha256(dest)
        if expected and digest != expected:
            raise SystemExit(f'{dest}: SHA-256 beklenenden farklı. Kaynak sürümü değişmiş olabilir; dosyayı '
                             'kenara alıp inceleyin. Farklı veri sürümleri sessizce karıştırılmaz.')
        lock[s['name']] = dict(file=s['file'], url=url, sha256=digest, bytes=dest.stat().st_size,
                               pinned=bool(expected))
        atomic_json(LOCK, lock)
        log('VERIFIED' if expected else 'LOCKED (first download)', s['name'])
    return lock
