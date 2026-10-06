"""Read-only diagnostic: find tests coupled to the ACTIVE bundle pointer.

Copies the project to a temp dir, flips the pointer to v2 there, runs the
suite, and reports the failing node IDs. The real project is untouched.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = r"C:\Users\dalvi\Downloads\data analusis\customer_churn_project"
SKIP = {"__pycache__", ".git", ".pytest_cache"}


def copy_project(dst):
    for name in os.listdir(ROOT):
        if name in SKIP:
            continue
        s = os.path.join(ROOT, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns(*SKIP))
        else:
            shutil.copy2(s, d)


def main():
    tmp = tempfile.mkdtemp(prefix="s12b_")
    proj = os.path.join(tmp, "proj")
    os.makedirs(proj)
    copy_project(proj)

    ptr = os.path.join(proj, "models", "active_bundle.json")
    with open(ptr, "w", encoding="utf-8") as fh:
        json.dump({"active_version": "v2"}, fh, indent=2)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=proj, capture_output=True, text=True,
    )
    out = r.stdout or ""
    print("RC =", r.returncode)
    for line in out.splitlines():
        if line.startswith("FAILED") or " passed" in line or " failed" in line:
            print(line)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
