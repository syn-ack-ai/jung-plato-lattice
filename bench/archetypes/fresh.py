"""Fresh item set (README addendum 9): items3/w0.py - w3.py, written by four Claude subagents from items3/BRIEF.md and
the pattern list in items3/assign.py, before any model saw them. Exposes the same shapes as the original sets:
  A, B: (id, pattern, anchor, true, surface)    A2, B2: id -> (lure, foil)    BRK: id -> "cause" | "link"
  WRITER: id -> writer number. Order: as in items3/assign.py.
FRESH_DIR (environment) points at another folder of w*.py files, for code tests only."""
import importlib.util, os, sys
from pathlib import Path

DIR = Path(os.environ.get("FRESH_DIR", Path(__file__).parent / "items3"))
A, B, A2, B2, BRK, WRITER = [], [], {}, {}, {}, {}
for k, path in enumerate(sorted(DIR.glob("w*.py"))):
    spec = importlib.util.spec_from_file_location(f"fresh_{path.stem}", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for dst, ex, items in ((A, A2, m.A), (B, B2, m.B)):
        for iid, pat, anchor, true, surface, lure, foil, brk in items:
            assert iid not in BRK, iid
            dst.append((iid, pat, anchor, true, surface)); ex[iid] = (lure, foil); BRK[iid] = brk; WRITER[iid] = k
if "FRESH_DIR" not in os.environ:
    sys.path.insert(0, str(Path(__file__).parent / "items3"))
    from assign import MECHANISMS, MYTHS  # noqa: E402
    for dst, ref in ((A, MYTHS), (B, MECHANISMS)):
        order = {i: n for n, (i, _) in enumerate(ref)}
        assert sorted(order) == sorted(it[0] for it in dst), "fresh set ids differ from items3/assign.py"
        dst.sort(key=lambda it: order[it[0]])
