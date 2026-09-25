from jetson_llm.evaluate import clean_generation, command_type, is_valid_robot_code, score


def test_valid_code():
    assert is_valid_robot_code("robot.move('left', 30)")
    assert is_valid_robot_code("robot.navigate('kitchen')\nrobot.grab('cup')")
    assert not is_valid_robot_code("robot.fly(3)")
    assert not is_valid_robot_code("import os")
    assert not is_valid_robot_code("robot.move(x)")
    assert not is_valid_robot_code("robot.move('left', 30")
    assert not is_valid_robot_code("")


def test_clean_generation_strips_fence():
    fence = "`" * 3
    assert clean_generation(f"{fence}python\nrobot.grab('cup')\n{fence}") == "robot.grab('cup')"
    assert clean_generation("  robot.stop()\n") == "robot.stop()"


def test_command_type():
    assert command_type("robot.move('left', 30)") == "move"
    assert command_type("robot.navigate('kitchen')\nrobot.grab('cup')") == "complex"


def test_score():
    refs = ["robot.move('left', 30)", "robot.grab('cup')"]
    s = score(["robot.move('left', 30)", "robot.grab('mug')"], refs, unseen=[True, False])
    assert s["overall"] == {"n": 2, "exact_match": 0.5, "valid_code": 1.0}
    assert s["unseen_instruction"]["exact_match"] == 1.0
    assert s["by_type"]["grab"]["exact_match"] == 0.0
