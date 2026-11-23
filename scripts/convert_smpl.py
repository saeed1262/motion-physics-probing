"""Convert the official SMPL v1.1.0 neutral model (Python 2 pickle with chumpy objects) into a plain
numpy pickle that the `smplx` package and MDM's joints-to-SMPL fitter can load in Python 3.

Loading uses a restricted unpickler: only numpy array types and scipy sparse matrices are allowed,
chumpy objects are replaced by a stub that keeps their stored array, and anything else is refused.

    .venv/bin/python scripts/convert_smpl.py vendor/smpl_raw/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl

Writes vendor/mdm/body_models/smpl/SMPL_NEUTRAL.pkl (MDM's expected location).
"""
import importlib, os, pickle, sys
import numpy as np
import scipy.sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "vendor", "mdm", "body_models", "smpl", "SMPL_NEUTRAL.pkl")


class ChumpyStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"),
           ("numpy", "dtype"), ("scipy.sparse.csc", "csc_matrix"), ("scipy.sparse._csc", "csc_matrix"),
           ("scipy.sparse.compressed", "_cs_matrix"), ("copy_reg", "_reconstructor"), ("__builtin__", "object"), ("__builtin__", "set")}


class SafeUnpickler(pickle.Unpickler):
    seen = set()

    def find_class(self, module, name):
        self.seen.add((module, name))
        if module.startswith("chumpy"):
            return ChumpyStub
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refused to load {module}.{name}")
        if module == "copy_reg":
            module = "copyreg"
        if module == "__builtin__":
            module = "builtins"
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core")
        if module == "scipy.sparse.csc":
            return scipy.sparse.csc_matrix
        return getattr(importlib.import_module(module), name)


def to_array(v):
    if isinstance(v, ChumpyStub):
        st = v._state
        for key in ("x", "a"):
            if isinstance(st, dict) and key in st and isinstance(st[key], np.ndarray):
                return np.asarray(st[key])
        raise ValueError(f"chumpy object without a stored array: {list(st) if isinstance(st, dict) else type(st)}")
    return v


with open(sys.argv[1], "rb") as f:
    data = SafeUnpickler(f, encoding="latin1").load()
print("classes referenced:", sorted(SafeUnpickler.seen))
out = {k: to_array(v) for k, v in data.items()}
for k, v in out.items():
    print(f"  {k}: {type(v).__name__} {getattr(v, 'shape', v if isinstance(v, str) else '')}")
need = ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"]
assert all(k in out for k in need), [k for k in need if k not in out]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(out, f, protocol=4)
print("wrote", OUT)
