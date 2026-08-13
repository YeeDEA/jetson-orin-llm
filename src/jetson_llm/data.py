"""Synthetic robot-command dataset generation.

Extracted verbatim from notebooks/01-dataset-synthesis.ipynb (single code cell),
reorganized into functions. Generates instruction/output pairs mapping
natural-language commands to robot control code (e.g. "Go left for 30cm"
-> "robot.move('left', 30)").
"""

import json
import os
import random

# Robot action vocabulary (from the notebook; `actions` was defined but unused there too)
actions = ["move", "rotate", "grab", "stop", "scan"]
directions = ["front", "back", "left", "right"]
objects = ["cup", "bottle", "trash", "book", "remote", "smartphone"]
places = ["kitchen", "living room", "bedroom", "bathroom", "entrance"]
adverbs = ["slowly", "quickly", "carefully", "immediately"]  # NOTE: defined but never used in the notebook

# Natural-language templates (multiple patterns per action type for diversity)
templates = [
    # movement
    {
        "type": "move",
        "patterns": [
            "Go {direction} for {dist}cm",
            "Move {dist}cm to the {direction}",
            "{direction} {dist}cm, now",
            "Walk {dist}cm {direction}",
            "Please advance {dist}cm {direction}",
        ],
    },
    # rotation
    {
        "type": "rotate",
        "patterns": [
            "Turn {angle} degrees",
            "Rotate {angle} degrees to the {direction}",
            "Spin {angle} degrees",
            "Make a {angle} degree turn",
        ],
    },
    # grabbing
    {
        "type": "grab",
        "patterns": [
            "Pick up the {obj}",
            "Grab that {obj}",
            "Hold the {obj}",
            "Get the {obj} for me",
        ],
    },
    # compound commands (move + grab)
    {
        "type": "complex",
        "patterns": [
            "Go to the {place} and bring me the {obj}",
            "Navigate to {place}, then grab the {obj}",
            "Find the {obj} in the {place}",
        ],
    },
]


def generate_sample():
    """Generate one {"instruction", "output"} pair. Verbatim from the notebook."""
    template = random.choice(templates)

    # random values
    direction = random.choice(directions)
    dist = random.randint(10, 500)
    angle = random.choice([45, 90, 180, 270, 360])
    obj = random.choice(objects)
    place = random.choice(places)

    # instruction
    pattern = random.choice(template["patterns"])
    instruction = pattern.format(direction=direction, dist=dist, angle=angle, obj=obj, place=place)

    # output (Python code)
    if template["type"] == "move":
        output = f"robot.move('{direction}', {dist})"
    elif template["type"] == "rotate":
        dir_param = 1 if direction in ['right', 'front'] else -1  # example logic (notebook comment: 예시 로직)
        output = f"robot.rotate({angle * dir_param})"
    elif template["type"] == "grab":
        output = f"robot.grab('{obj}')"
    elif template["type"] == "complex":
        # compound command calls two functions
        output = f"robot.navigate('{place}')\nrobot.grab('{obj}')"

    return {
        "instruction": instruction,
        "output": output,
    }


def generate_dataset(target_count=50000, output_file="robot_dataset.json"):
    """Generate `target_count` samples and save them to `output_file`.

    The notebook hard-coded TARGET_COUNT = 50000 and
    OUTPUT_FILE = "robot_dataset.json"; here they are parameters.
    """
    print(f"Generating {target_count} samples... Please wait.")
    data = []

    for i in range(target_count):
        data.append(generate_sample())
        if (i + 1) % 10000 == 0:
            print(f"{i + 1} samples generated...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"\n[Complete] Saved to {output_file}")
    print(f"File Size: {file_size:.2f} MB")
    return data
