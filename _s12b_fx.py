import ast
from pathlib import Path

ROOT = Path(r"C:\Users\dalvi\Downloads\data analusis\customer_churn_project")
T = ROOT / "tests"

for f in sorted(T.glob("test_*.py")):
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fixtures = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decs = " ".join(ast.unparse(d) for d in n.decorator_list)
        if "fixture" not in decs:
            continue
        body = ast.unparse(n)
        flags = []
        if "get_active_bundle" in body:
            flags.append("ACTIVE")
        if "copytree" in body or "shutil" in body:
            flags.append("COPIES")
        if "active_bundle" in body:
            flags.append("PTRFILE")
        fixtures.append((n.lineno, n.name, [a.arg for a in n.args.args], flags))
    if fixtures:
        print("== " + f.name)
        for line, name, params, flags in sorted(fixtures):
            print("   line %-5d %-34s params=%-40s %s"
                  % (line, name, ",".join(params), ",".join(flags) or "-"))
