"""Rebuild public/, the content of the public GitHub repository, from this working tree, then scan it.

    python3 scripts/export_public.py               # rebuild public/ from scratch and scan it
    python3 scripts/export_public.py --self-test   # check that the scan catches planted bad files (writes nothing here)

The working tree is the only copy maintained by hand; public/ is generated. Upload or push public/'s content as it is.
What goes in is the explicit manifest RULES below (gitignore-like: last matching rule wins, default out). It was
derived from the public copy as uploaded so far: the initial upload (scripts/build_upload.sh, every non-ignored file,
in _upload/batches/) plus the hand-maintained updates in _upload/sync/ (README.md, docs/, paper/, results/, scripts/).
Every top-level entry of the repository must be either published (a '+' rule's first folder) or listed in
NOT_PUBLISHED, so a new top-level folder stops the export until it is decided.

The paper is public on arXiv, so the author's name may appear. The scan fails the export (exit 1, public/ left in
place for inspection, not safe to publish) on: SMPL / SMPL+H / AMASS / HumanML3D data files, checkpoints and model
weights, archives (they can hide either), secrets and tokens, absolute local paths (/Users/..., /private/tmp, ...),
files over 50 MB, virtualenvs, .git, __pycache__ and other junk. The only rewrite: in text files, this repository's
absolute path becomes relative (run logs print it). Standard library only.
"""
import fnmatch, os, re, shutil, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public")
MAX_BYTES = 50 * 1024 * 1024

# ------------------------------------------------------------------------------------------------ manifest
# (+ include / - exclude, pattern): fnmatch on the repository-relative POSIX path, where '*' also crosses '/'.
RULES = [
    # what the public repository holds: the folders of _upload/sync and the rest of the initial upload
    ("+", "README.md"),             # overview, setup, pipeline table, licences
    ("+", "LICENSE"),               # MIT, for this repository's own code
    ("+", "requirements.txt"),      # pinned packages
    ("+", ".gitignore"),            # the public repository ignores the same regenerable outputs
    ("+", "docs/*"),                # FINDINGS, HANDOFF, RESEARCH_PLAN, math_notes, reviews/ (moved from the top and paper/ on Nov 18)
    ("+", "paper/*"),               # LaTeX sources, tables, figures, figures_web, supplement videos, README, submission texts
    ("+", "results/*"),             # result CSV/TXT/JSON, figures, the two small probe-result .npz files
    ("+", "scripts/*"),             # every pipeline, analysis, figure and build script (this one too)
    ("+", "tools/*"),               # notebook builders and the shared label functions
    ("+", "notebooks/*"),           # the two notebooks
    ("+", "prompts/*"),             # the jump-heavy prompt list
    ("+", "tests/*"),               # synthetic-data tests
    ("+", "video_v2/charts.py"),    # the revised presentation's code and narration text (the supplement ships the code too);
    ("+", "video_v2/make_video.py"),  # its voice recordings, candidates, rejected voice-cloning script and the
    ("+", "video_v2/script.md"),    # built videos (28 MB, over GitHub's web-upload limit) stay out
    ("+", "video_v2/public.json"),  # the credit line make_video.py --public prints (name and arXiv id)

    # inside those, left out (mirrors .gitignore)
    ("-", "*/__pycache__/*"), ("-", "*.pyc"), ("-", "*.DS_Store"),
    ("-", "docs/HANDOFF.md"),                             # the private working log: scratch paths, reviewer and account notes
    ("-", "docs/reviews/*"),                              # the simulated pre-submission reviews (user, Nov 23: not published)
    ("-", "paper/submission/submission_form.md"),         # the TMLR form text with action-editor candidates (same)
    ("-", "results/*_smoke/*"),                           # smoke-test outputs
    ("-", "results/*/preds/*"),                           # per-sample predictions, about 290 MB (--save-preds)
    ("-", "tests/*.npy"),                                 # the HumanML3D clip tests/real_clip_check.py downloads
    ("-", "paper/build/*"), ("+", "paper/build/main.pdf"),  # LaTeX logs out, the compiled paper in
    ("-", "paper/submission/supplementary_material.zip"),  # the anonymous supplement (scripts/build_supplement.sh)...
    ("-", "paper/submission/supplementary_material/*"),   # ...and its unzipped build folder
    ("-", "paper/supplement/voice_sample_*.wav"),         # narration voice auditions
]
NOT_PUBLISHED = {   # top-level entries that are never walked, and why
    ".git": "version control", ".venv": "Python environment", "vendor": "third-party code, checkpoints, SMPL, AMASS",
    "data": "generated motions, activations, caches (37 GB)", "_upload": "upload packages and the old hand-made sync copy",
    ".claude": "local tool configuration", ".DS_Store": "Finder metadata", "public": "this script's output",
    "personal_site_entry": "personal-site material, published through the site's own repository",
    "project_page": "superseded standalone project page", "site_page": "the project page's source for the site",
    "site_page 2": "a duplicate of site_page", "site_page.zip": "the project page's upload package",
    "video_v2": "only the four files named in RULES are published",
}


def wanted(rel):
    keep = False
    for sign, pat in RULES:
        if fnmatch.fnmatchcase(rel, pat):
            keep = sign == "+"
    return keep


def manifest():
    roots = {pat.split("/")[0].rstrip("*") for sign, pat in RULES if sign == "+"}
    undecided = sorted(set(os.listdir(ROOT)) - roots - set(NOT_PUBLISHED))
    if undecided:
        sys.exit(f"EXPORT FAILED: top-level entries in neither RULES nor NOT_PUBLISHED: {undecided}")
    files = []
    for top in sorted(roots):
        path = os.path.join(ROOT, top)
        if os.path.isfile(path):
            files += [top] if wanted(top) else []
            continue
        for d, dirs, names in os.walk(path):
            dirs.sort()
            for n in sorted(names):
                rel = os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, "/")
                if wanted(rel):
                    files.append(rel)
    return files


# ------------------------------------------------------------------------------------------------ scan
JUNK_DIRS = {".git", ".venv", "venv", "site-packages", "__pycache__", "node_modules", ".ipynb_checkpoints", ".pytest_cache"}
JUNK_FILES = {".DS_Store", "pyvenv.cfg", ".env", ".netrc", "id_rsa", "id_ed25519", "credentials.json"}
WEIGHTS = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl", ".pickle", ".h5", ".hdf5", ".onnx", ".tflite", ".pb",
           ".bvh", ".fbx", ".c3d", ".mat", ".pyc", ".pem", ".key"}
ARCHIVES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
DATA_NAME = re.compile(r"smpl|amass|bmlmovi|humanml|_poses|^(mean|std)\.npy$|new_joint|joint_vecs", re.I)  # on .npz/.npy names
AMASS_KEYS = {"poses", "trans", "betas"}                        # an AMASS *_poses.npz
BODY_KEYS = {"shapedirs", "posedirs", "J_regressor", "v_template"}  # an SMPL / SMPL+H / SMPL-X model file
SECRETS = [r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY", r"\bsk-[A-Za-z0-9_\-]{20,}", r"\bgh[pousr]_[A-Za-z0-9]{30,}",
           r"github_pat_[A-Za-z0-9_]{30,}", r"\bhf_[A-Za-z0-9]{30,}", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", r"AIza[0-9A-Za-z_\-]{35}",
           r"(api[_-]?key|secret|password|passwd|access[_-]?token)\s*[:=]\s*['\"][^'\"\s]{8,}"]
# a home folder needs a name after it, so scanner patterns (build_supplement.sh's, this file's) do not match themselves
LOCAL_PATHS = [r"/Users/[A-Za-z][\w.-]*", r"/home/[a-z][\w.-]*/", r"/private/(tmp|var)/", r"/(private/)?var/folders/", r"[A-Z]:\\Users\\"]
TEXT = {".md", ".txt", ".csv", ".json", ".py", ".sh", ".tex", ".bib", ".html", ".js", ".css", ".ipynb", ".srt", ".cfg", ".yml"}


def npz_keys(path):
    """Array names of an .npz without unpickling anything (it is a ZIP of <name>.npy members)."""
    try:
        with zipfile.ZipFile(path) as z:
            return {n[:-4] if n.endswith(".npy") else n for n in z.namelist()}
    except zipfile.BadZipFile:
        return set()


def scan(root):
    """Every problem in the folder, as 'path: reason' lines (empty when clean)."""
    secrets = re.compile("|".join(f"(?:{p})" for p in SECRETS), re.I)
    local = re.compile("|".join(f"(?:{p})" for p in LOCAL_PATHS).encode())
    hits = []
    for d, dirs, names in os.walk(root):
        for x in sorted(set(dirs) & JUNK_DIRS):
            hits.append(f"{os.path.relpath(os.path.join(d, x), root)}/: virtualenv, VCS or cache folder")
        for n in sorted(names):
            path = os.path.join(d, n)
            rel = os.path.relpath(path, root)
            ext = os.path.splitext(n)[1].lower()
            size = os.path.getsize(path)
            if size > MAX_BYTES:
                hits.append(f"{rel}: {size / 2**20:.1f} MB, over 50 MB")
            if n in JUNK_FILES or n.endswith(".egg-info"):
                hits.append(f"{rel}: environment, secret or junk file")
            if ext in WEIGHTS:
                hits.append(f"{rel}: checkpoint, body model, motion-capture or compiled file ({ext})")
            if ext in ARCHIVES:
                hits.append(f"{rel}: archive (could hold data or checkpoints)")
            if ext in {".npz", ".npy"} and DATA_NAME.search(n):
                hits.append(f"{rel}: file name of SMPL / AMASS / HumanML3D data")
            if ext == ".npz":
                keys = npz_keys(path)
                if AMASS_KEYS <= keys or "mocap_framerate" in keys or keys & BODY_KEYS:
                    hits.append(f"{rel}: AMASS motion or SMPL body-model arrays ({sorted(keys)[:6]})")
            body = open(path, "rb").read()
            found = [(body.count(b"\n", 0, m.start()) + 1, m.group(0)[:40].decode(errors="replace")) for m in local.finditer(body)]
            if found:
                hits.append(f"{rel}:{found[0][0]}: absolute local path {found[0][1]!r}"
                            + (f" (and {len(found) - 1} more)" if len(found) > 1 else ""))
            for m in secrets.finditer(body.decode("latin-1")):
                hits.append(f"{rel}: possible secret {m.group(0)[:24]!r}")
    return hits


def self_test():
    """Plant one bad file per rule in a temporary folder; the scan must name each, and pass a clean file."""
    with tempfile.TemporaryDirectory() as t:
        def put(rel, data=b"x"):
            os.makedirs(os.path.dirname(os.path.join(t, rel)), exist_ok=True)
            open(os.path.join(t, rel), "wb").write(data)
        put("ok/clean.csv", b"a,b\n1,2\n")
        assert scan(t) == [], scan(t)
        put("big.dat"); os.truncate(os.path.join(t, "big.dat"), MAX_BYTES + 1)
        put(".venv/pyvenv.cfg"); put("s/__pycache__/m.cpython-312.pyc"); put(".git/HEAD")
        put("ckpt/model000475000.pt"); put("body/SMPL_NEUTRAL.pkl"); put("hml/Mean.npy")
        with zipfile.ZipFile(os.path.join(t, "clip.npz"), "w") as z:           # an AMASS-like clip under an innocent name
            for k in ("poses", "trans", "betas"):
                z.writestr(f"{k}.npy", b"")
        put("a.zip"); put("notes.md", b"see /" + b"Users/someone/data and\napi_" + b"key = 'abcdefghij12345'\nghp_" + b"a" * 36)
        found = "\n".join(scan(t))
        for want in ("big.dat: ", ".venv/", "pyvenv.cfg", "__pycache__/", ".git/", "model000475000.pt", "SMPL_NEUTRAL.pkl",
                     "Mean.npy", "clip.npz: AMASS", "a.zip: archive", "notes.md:1: absolute local path",
                     "notes.md: possible secret \"api_key", "notes.md: possible secret 'ghp_"):
            assert want in found, f"self-test: scan missed {want!r}\n{found}"
    print("self-test passed: the scan catches each planted file and passes a clean one")


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()
    if os.path.exists(os.path.join(OUT, ".git")):
        sys.exit("EXPORT FAILED: public/ holds a .git; move it out of the way first (this script deletes public/)")
    files = manifest()
    shutil.rmtree(OUT, ignore_errors=True)
    here, relativized = ROOT.encode(), 0
    for rel in files:
        os.makedirs(os.path.dirname(os.path.join(OUT, rel)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(OUT, rel))
        body = open(os.path.join(OUT, rel), "rb").read()
        if os.path.splitext(rel)[1] in TEXT and here in body:                 # run logs: repository paths made relative
            open(os.path.join(OUT, rel), "wb").write(body.replace(here + b"/", b"").replace(here, b"."))
            relativized += 1
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in files)
    print(f"wrote public/: {len(files)} files, {size / 2**20:.1f} MB; repository paths made relative in {relativized}")
    hits = scan(OUT)
    if hits:
        print(f"SCAN FAILED ({len(hits)}): public/ is NOT safe to publish; fix the source in the working tree and rerun")
        print("  " + "\n  ".join(hits))
        sys.exit(1)
    print("scan clean: no licensed data, checkpoints, archives, secrets, local paths, files over 50 MB or junk")


if __name__ == "__main__":
    main()
