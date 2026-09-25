import ast
import json
import re

from jetson_llm.data import generate_dataset, generate_samples, make_splits, unseen_mask


def test_generate_samples_is_deterministic():
    assert generate_samples(200, seed=7) == generate_samples(200, seed=7)
    assert generate_samples(200, seed=7) != generate_samples(200, seed=8)


def test_generate_samples_does_not_touch_global_rng():
    import random
    random.seed(123)
    expected = random.random()
    random.seed(123)
    generate_samples(50, seed=0)
    assert random.random() == expected


def test_sample_format():
    for s in generate_samples(500, seed=0):
        assert set(s) == {"instruction", "output"}
        assert s["instruction"].strip()
        for line in s["output"].split("\n"):
            assert re.fullmatch(r"robot\.(move|rotate|grab|navigate)\(.*\)", line), line
        ast.parse(s["output"])


def test_output_matches_instruction_values():
    for s in generate_samples(500, seed=1):
        out, ins = s["output"], s["instruction"]
        if out.startswith("robot.move"):
            dist = re.search(r"(\d+)\)", out).group(1)
            assert f"{dist}cm" in ins
            assert 10 <= int(dist) <= 500
        elif out.startswith("robot.grab"):
            assert re.search(r"'(\w+)'", out).group(1) in ins


def test_splits_are_disjoint_draws():
    train, test = make_splits(300, 50, seed=0)
    assert len(train) == 300 and len(test) == 50
    assert (train, test) == make_splits(300, 50, seed=0)
    mask = unseen_mask(train, test)
    seen = {s["instruction"] for s in train}
    assert all((s["instruction"] not in seen) == m for s, m in zip(test, mask))


def test_generate_dataset_writes_json(tmp_path):
    path = tmp_path / "d.json"
    data = generate_dataset(target_count=20, output_file=str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_rotate_label_is_deterministic_given_instruction():
    """v2: an instruction string always maps to one rotate label."""
    labels = {}
    for s in generate_samples(3000, seed=2):
        if s["output"].startswith("robot.rotate"):
            labels.setdefault(s["instruction"], set()).add(s["output"])
    assert labels and all(len(v) == 1 for v in labels.values())
    for ins, (out,) in labels.items():
        angle = int(re.search(r"\((-?\d+)\)", out).group(1))
        if not any(d in ins for d in ("left", "right", "front", "back")):
            assert angle > 0, (ins, out)
        elif any(d in ins for d in ("left", "back")):
            assert angle < 0, (ins, out)


def test_legacy_rotate_reproduces_v1():
    v1 = generate_samples(3000, seed=2, legacy_rotate=True)
    v2 = generate_samples(3000, seed=2)
    assert [s["instruction"] for s in v1] == [s["instruction"] for s in v2]
    diff = [(a, b) for a, b in zip(v1, v2) if a != b]
    assert diff and all(a["output"] == b["output"].replace("(", "(-", 1) for a, b in diff)
    ambiguous = {}
    for s in v1:
        if s["output"].startswith("robot.rotate"):
            ambiguous.setdefault(s["instruction"], set()).add(s["output"])
    assert any(len(v) > 1 for v in ambiguous.values())
