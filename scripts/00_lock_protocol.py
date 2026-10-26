"""Lock protocol + code BEFORE any modelling. Writes PROTOCOL_LOCK.json (upload it, e.g. to OSF, with a timestamp).

  python scripts/00_lock_protocol.py                    # first lock
  python scripts/00_lock_protocol.py --amend "reason"   # documented amendment after a deliberate change
"""
import argparse, datetime
import _bootstrap  # noqa: F401
from biobridge.utils import ROOT, code_fingerprint, atomic_json, read_json, sha256, environment, load_protocol

ap = argparse.ArgumentParser(); ap.add_argument('--amend', default=None); a = ap.parse_args()
lock = ROOT / 'PROTOCOL_LOCK.json'
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
load_protocol()                                   # must parse
if lock.exists() and not a.amend:
    raise SystemExit('Protokol zaten kilitli. Bilinçli bir değişiklik için --amend "gerekçe" kullanın.')
history = read_json(lock)['history'] if lock.exists() else []
files = code_fingerprint()
history.append(dict(time_utc=now, kind='amendment' if a.amend else 'initial_lock', reason=a.amend,
                    changed=sorted(k for k in files if not lock.exists() or read_json(lock)['files'].get(k) != files[k])))
atomic_json(lock, dict(locked_utc=now, files=files, history=history, environment=environment()))
if a.amend:
    with open(ROOT / 'PROTOCOL_AMENDMENTS.md', 'a') as f:
        f.write(f'\n## {now}\n{a.amend}\nChanged files: {", ".join(history[-1]["changed"]) or "-"}\n')
print('LOCKED', now, 'sha256(PROTOCOL_LOCK.json)=', sha256(lock))
