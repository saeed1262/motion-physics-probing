"""Regenerate p2_labels_funcs.py and p2_recorder.py from build_nb.py.

Phase 2 reuses the exact label and recording code of phase 1. Run this after
editing those cells in build_nb.py, then rebuild both notebooks:
    python tools/extract_shared.py && python tools/build_nb.py && python tools/build_nb2.py
"""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("b", os.path.join(HERE, "build_nb.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
codes = [src for kind, tag, src in b.cells if kind == "code"]
lab = next(c for c in codes if "from shapely.geometry import" in c)
open(os.path.join(HERE, "p2_labels_funcs.py"), "w").write(lab[:lab.index("t0 = time.time()\nLAB = labels_for")])
open(os.path.join(HERE, "p2_recorder.py"), "w").write(next(c for c in codes if "class Recorder" in c))
print("extracted p2_labels_funcs.py and p2_recorder.py")
