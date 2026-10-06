import ast
from pathlib import Path

# ... snip ...
T = Path(r"C:\Users\dalvi\Downloads\data analusis\customer_churn_project\tests")
TOKENS = ['== "v1"', "== 'v1'", '"LogisticRegression"', 'estimator_only', '== 30', '== 45', 'GradientBoosting', '== "pipeline"', '"v2"', "'v2'"]

out = []
for f in sorted(T.glob("test_*.py")):
    lines = f.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
            body = "\n".join(lines[n.lineno - 1:n.end_lineno])
            args = [a.arg for a in n.args.args]
            hits = [t for t in TOKENS if t in body]
            coupled = [a for a in args if a in ("service", "pointer_dir")]
            if hits or coupled:
                out.append(f"{f.name}::{n.name}")
                out.append(f"  args={args} dict_hits={hits}")
print("\n".join(out))