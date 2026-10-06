"""Step 12C read-only probe. Makes NO changes to the real project."""
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\dalvi\Downloads\data analusis\customer_churn_project")
OUT = {}


def sh(args, cwd, timeout=2400):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, errors="replace")


# ---------- 1. fixtures ----------
fixtures = {}
for name in ["test_prediction_service.py", "test_input_contract.py",
             "test_model_bundle.py", "test_shadow_audit.py",
             "test_shap_explainability.py", "conftest.py"]:
    f = ROOT / "tests" / name
    if not f.exists():
        continue
    src = f.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = []
            for d in node.decorator_list:
                s = getattr(d, "id", None) or getattr(getattr(d, "attr", None), "__str__", lambda: None)()
                if s is None:
                    try:
                        s = ast.unparse(d)
                    except Exception:
                        s = "?"
                decs.append(s)
            if any("fixture" in d for d in decs) or node.name in (
                    "pointer_dir", "bundles", "bundle", "service_v1", "service_v2"):
                fixtures[f"tests/{name}:{node.lineno}"] = {
                    "name": node.name, "decorators": decs,
                    "source": ast.get_source_segment(src, node) or "",
                }
OUT["fixtures"] = fixtures

# ---------- 2. real baseline ----------
r = sh([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
OUT["baseline"] = {"rc": r.returncode,
                   "tail": r.stdout.strip().splitlines()[-6:]}

# ---------- 3. v2 temp copy ----------
tmp = Path(tempfile.mkdtemp(prefix="s12c_"))
dst = tmp / "proj"
shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
    "__pycache__", ".git", "venv", ".venv", "*.pyc", ".pytest_cache"))
(dst / "models" / "active_bundle.json").write_text(
    json.dumps({"active_version": "v2"}), encoding="utf-8")

r2 = sh([sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:randomly"], cwd=dst)
text = r2.stdout
OUT["v2_rc"] = r2.returncode
OUT["v2_tail"] = text.strip().splitlines()[-4:]

fails = []
cur = None
for line in text.splitlines():
    m = re.match(r"^(tests/[^\s:]+\.py)::(\S+)", line.strip())
    if line.startswith("FAILED ") or line.startswith("ERROR "):
        fails.append(line.strip())
    m2 = re.match(r"^/.*: AssertionError: (.*)$", line.strip())
    if m2 and cur:
        cur["assert"] = m2.group(1)[:200]
    if re.match(r"^_{5,}\s", line):
        cur = None
# pair node id with the tb=line reason
ids = re.findall(r"^FAILED (\S+)", text, re.M)
reasons = re.findall(r"^/[\w\\\.:\-]+:(\d+): (\w+Error[^\n]*)", text, re.M)
OUT["failed_ids"] = ids
OUT["tb_reasons"] = [{"line": a, "reason": b[:180]} for a, b in reasons]
OUT["n_failed"] = len(ids)

# ---------- 4. full traceback per failing test ----------
per = {}
for nid in ids:
    r3 = sh([sys.executable, "-m", "pytest", "-q", "--tb=short", nid], cwd=dst)
    body = r3.stdout
    sel = [l for l in body.splitlines()
           if l.startswith("E ") or l.strip().startswith(">")
           or "assert" in l]
    per[nid] = sel[:12]
OUT["per_test"] = per

shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(OUT, indent=1)[:60000])
