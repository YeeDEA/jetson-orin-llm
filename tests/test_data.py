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
