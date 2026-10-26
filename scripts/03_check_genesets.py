"""Report how many genes of each frozen biological gene set are found in the universe (no outcomes)."""
import _bootstrap  # noqa: F401
from biobridge.prepare import universe
from biobridge.features import GeneSets
gs = GeneSets(universe())
for k, (found, total) in gs.coverage.items():
    print(f'{k:22s} {found:4d}/{total:<4d}' + ('   <-- düşük kapsam' if found < 0.5 * total else ''))
print('Kullanılan modül özellikleri:', len(gs.module_names))
