"""Scoring for generated robot code (pure Python, no GPU needed).

- exact match: the generated text, stripped, equals the reference exactly.
- valid code: the text parses as Python and every statement is a single call
  ``robot.<method>(<literal args>)`` with a method from ``ROBOT_METHODS``.
  This is "would a strict executor accept it", not "is it correct".
"""

from __future__ import annotations

import ast

ROBOT_METHODS = {"move", "rotate", "grab", "navigate", "stop", "scan"}
FENCE = "`" * 3


def clean_generation(text: str) -> str:
    """Strip whitespace and a surrounding code fence if the model added one."""
    t = text.strip()
    if t.startswith(FENCE):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit(FENCE, 1)[0]
    return t.strip()


def is_valid_robot_code(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not tree.body:
        return False
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return False
        fn = stmt.value.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == "robot" and fn.attr in ROBOT_METHODS):
            return False
        if stmt.value.keywords:
            return False
        for a in stmt.value.args:
            try:
                ast.literal_eval(a)
            except ValueError:
                return False
    return True


def command_type(output: str) -> str:
    """'move' | 'rotate' | 'grab' | 'complex' from a reference output."""
    if "\n" in output:
        return "complex"
    return output.split("(")[0].replace("robot.", "")


def score(preds, refs, unseen=None):
    """Exact-match / valid-code rates: overall, per command type, and on the unseen subset."""
    rows = [(clean_generation(p), r) for p, r in zip(preds, refs)]

    def agg(idx):
        idx = list(idx)
        if not idx:
            return {"n": 0}
        em = sum(rows[i][0] == rows[i][1] for i in idx)
        vc = sum(is_valid_robot_code(rows[i][0]) for i in idx)
        return {"n": len(idx), "exact_match": round(em / len(idx), 4),
                "valid_code": round(vc / len(idx), 4)}

    out = {"overall": agg(range(len(rows)))}
    types = sorted({command_type(r) for _, r in rows})
    out["by_type"] = {t: agg(i for i, (_, r) in enumerate(rows) if command_type(r) == t) for t in types}
    if unseen is not None:
        out["unseen_instruction"] = agg(i for i, u in enumerate(unseen) if u)
    return out
